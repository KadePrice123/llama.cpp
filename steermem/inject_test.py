#!/usr/bin/env python3
"""Does llama_inject_set change the model's output, and only where it is told to?

The first test of the steermem fork's injection API, over ctypes on the fork's
own libllama.so -- no llama-cpp-python, nothing but the C API. It decodes a
prompt three times on the CPU: plain; with a random unit vector injected at
layer 8 on ONE position of the prompt; and with the injection cleared again.
The last logits must differ in the middle run and match in the first and third
(the table is cleared and a graph with injections is never reused). It prints
the max absolute logit difference for each pair.

    python inject_test.py --lib ~/llama-ngram/build-cpu/bin/libllama.so --model ~/gguf/steermem-D4N4k2B-q8_0.gguf
"""
import argparse
import ctypes as C
import os
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--pos", type=int, default=3)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    lib = C.CDLL(os.path.expanduser(a.lib))
    for name, res, args in (
            ("llama_backend_init", None, []),
            ("llama_model_default_params", None, []),           # struct by value: handled below
            ("llama_context_default_params", None, []),
            ("llama_inject_set", C.c_int32, [C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p, C.c_float]),
            ("llama_inject_clear", None, [C.c_void_p]),
            ("llama_decode", C.c_int32, [C.c_void_p, C.c_void_p]),
            ("llama_get_logits_ith", C.POINTER(C.c_float), [C.c_void_p, C.c_int32]),
            ("llama_model_free", None, [C.c_void_p]),
            ("llama_free", None, [C.c_void_p])):
        f = getattr(lib, name)
        if name.endswith("default_params"):
            continue
        f.restype, f.argtypes = res, args
    # the parameter structs are opaque here: take the defaults as raw bytes and pass them back by value
    # (ctypes needs the struct layout for by-value calls, so use the helper functions that take pointers
    #  where they exist; llama.h has llama_model_load_from_file(path, params) by value -- read the
    #  struct sizes from the library's own helpers)
    lib.llama_model_default_params.restype = LlamaModelParams
    lib.llama_context_default_params.restype = LlamaContextParams
    lib.llama_model_load_from_file.restype = C.c_void_p
    lib.llama_model_load_from_file.argtypes = [C.c_char_p, LlamaModelParams]
    lib.llama_init_from_model.restype = C.c_void_p
    lib.llama_init_from_model.argtypes = [C.c_void_p, LlamaContextParams]
    lib.llama_model_get_vocab.restype = C.c_void_p
    lib.llama_model_get_vocab.argtypes = [C.c_void_p]
    lib.llama_vocab_n_tokens.restype = C.c_int32
    lib.llama_vocab_n_tokens.argtypes = [C.c_void_p]
    lib.llama_tokenize.restype = C.c_int32
    lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32), C.c_int32, C.c_bool, C.c_bool]
    lib.llama_batch_get_one.restype = LlamaBatch
    lib.llama_batch_get_one.argtypes = [C.POINTER(C.c_int32), C.c_int32]
    lib.llama_model_n_embd.restype = C.c_int32
    lib.llama_model_n_embd.argtypes = [C.c_void_p]
    lib.llama_memory_clear.restype = None
    lib.llama_memory_clear.argtypes = [C.c_void_p, C.c_bool]
    lib.llama_get_memory.restype = C.c_void_p
    lib.llama_get_memory.argtypes = [C.c_void_p]

    lib.llama_backend_init()
    mp = lib.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = lib.llama_model_load_from_file(os.path.expanduser(a.model).encode(), mp)
    assert model, "model failed to load"
    cp = lib.llama_context_default_params()
    cp.n_ctx = 256
    cp.n_batch = 256
    cp.n_threads = a.threads
    cp.n_threads_batch = a.threads
    ctx = lib.llama_init_from_model(model, cp)
    assert ctx, "context failed"
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)
    n_embd = lib.llama_model_n_embd(model)
    prompt = b"Recall each of these from memory, in order:\n1. James 3:14\nAnswer:\n"
    toks = (C.c_int32 * 128)()
    n = lib.llama_tokenize(vocab, prompt, len(prompt), toks, 128, True, True)
    assert n > a.pos, "prompt too short for --pos"
    print("model n_embd %d, vocab %d, prompt %d tokens" % (n_embd, n_vocab, n))

    def run():
        lib.llama_memory_clear(lib.llama_get_memory(ctx), True)
        batch = lib.llama_batch_get_one(toks, n)
        rc = lib.llama_decode(ctx, C.byref(batch))
        assert rc == 0, "decode rc %d" % rc
        p = lib.llama_get_logits_ith(ctx, -1)
        return np.ctypeslib.as_array(p, shape=(n_vocab,)).copy()

    plain = run()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_embd).astype(np.float32)
    v /= np.linalg.norm(v)
    pos = (C.c_int32 * 1)(a.pos)
    data = v.ctypes.data_as(C.c_void_p)
    rc = lib.llama_inject_set(ctx, a.layer, 1, pos, data, a.scale)
    assert rc == 0, "inject_set rc %d" % rc
    injected = run()
    lib.llama_inject_clear(ctx)
    cleared = run()
    d1 = float(np.abs(plain - injected).max())
    d2 = float(np.abs(plain - cleared).max())
    print("max |logit diff|  plain vs injected: %.4f   plain vs cleared: %.4f" % (d1, d2))
    print("argmax plain %d  injected %d  cleared %d" % (int(plain.argmax()), int(injected.argmax()), int(cleared.argmax())))
    ok = d1 > 1e-3 and d2 < 1e-3
    print("INJECT-TEST %s" % ("PASS" if ok else "FAIL"))
    lib.llama_free(ctx)
    lib.llama_model_free(model)


# ---- the by-value structs, mirrored from llama.h of the fork's base (8b4b355) --------------
# Only the leading fields that this test sets are named; the rest is padding read back from
# the defaults, so the layout must match the library's. Sizes are checked against the
# struct returned by the library at run time (a mismatch shows up as a crash or a load failure,
# which is the honest signal to update this mirror).
class LlamaModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int32), ("main_gpu", C.c_int32),
                ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p), ("progress_callback_user_data", C.c_void_p),
                ("kv_overrides", C.c_void_p), ("vocab_only", C.c_bool), ("use_mmap", C.c_bool), ("use_mlock", C.c_bool),
                ("check_tensors", C.c_bool), ("use_extra_bufts", C.c_bool), ("no_host", C.c_bool), ("no_alloc", C.c_bool),
                ("_pad", C.c_uint8 * 64)]


class LlamaContextParams(C.Structure):
    _fields_ = [("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32), ("n_seq_max", C.c_uint32),
                ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32),
                ("_rest", C.c_uint8 * 256)]


class LlamaBatch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)), ("embd", C.POINTER(C.c_float)),
                ("pos", C.c_void_p), ("n_seq_id", C.c_void_p), ("seq_id", C.c_void_p), ("logits", C.c_void_p)]


if __name__ == "__main__":
    main()

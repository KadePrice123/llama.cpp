#!/usr/bin/env python3
"""twoside_export.py -- a twoside.py checkpoint as the side GGUF the fork serves in protocol mode.

Beyond the base model and the LoRA (lora_to_peft.py, then llama.cpp's convert_lora_to_gguf.py), the fork needs:
  memtag [d], seqtag [64, d]        block row embeddings: row j of a memory is memtag + seqtag[1 + min(j, 61)]
  marktag [2, d]                    the <|mem_start|>/<|mem_end|> rows, used only if the run trained with --marks 1
  nulltag [d], neartag [d]          the null block's first row, and the rows naming the nearest keys held
  nullrec.<L> [d]                   what the null row injects at layer L
  steermem.scale.<L>                the injection strength per layer, clamped at 0 the way the trainer's hook uses it
  steermem.emb_norm                 every block row is normalised to this -- computed exactly as twoside.py does:
                                    the mean norm of the first 4096 rows of the input embedding
and the settings the checkpoint trained with -- steermem.protocol 1 (the <recall> loop), layers, mem_seq, head, marks,
keyemb, span_from_key -- so the sidecar captures and delivers exactly as the trainer did.

    python twoside_export.py --ckpt twoside_pointer.s600.pt --model /path/to/qwen35-2b --out steermem-side-pointer-s600.gguf
"""
import argparse
import glob
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="the base model's Hugging Face folder, for emb_norm")
    ap.add_argument("--gguf-py", default="", help="llama.cpp's gguf-py folder, if the gguf package is not installed")
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf

    ck = torch.load(os.path.expanduser(a.ckpt), map_location="cpu", weights_only=False)
    args, side = dict(ck.get("args") or {}), ck.get("side") or {}
    if args.get("mem_from") != "summary":
        raise SystemExit("protocol mode serves --mem-from summary checkpoints; this one is %r" % args.get("mem_from"))
    if int(args.get("code_dim") or 0) or int(args.get("codec") or 0) or int(args.get("slots") or 0):
        raise SystemExit("codec and slot checkpoints have no fork path")
    for need in ("memtag", "seqtag", "marktag", "nulltag", "neartag"):
        if need not in side:
            raise SystemExit("the checkpoint's side lacks %s -- was it saved by a twoside.py with --save-every?" % need)
    layers = [int(x) for x in str(args.get("inject_layers") or "8,16").split(",") if x.strip()]

    from safetensors import safe_open
    rows = None
    for f in sorted(glob.glob(os.path.join(a.model, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.endswith("embed_tokens.weight"):
                    rows = st.get_slice(k)[0:4096].to(torch.float32)
                    break
        if rows is not None:
            break
    if rows is None:
        raise SystemExit("no embed_tokens.weight in %s" % a.model)
    emb_norm = float(rows.norm(dim=-1).mean())

    f32 = lambda t: t.detach().to(torch.float32).contiguous().numpy()
    w = gguf.GGUFWriter(os.path.expanduser(a.out), "steermem-side")
    w.add_string("steermem.checkpoint", os.path.basename(a.ckpt))
    w.add_uint32("steermem.step", int(ck.get("step") or 0))
    w.add_uint32("steermem.protocol", 1)
    w.add_array("steermem.layers", layers)
    w.add_float32("steermem.emb_norm", emb_norm)
    w.add_uint32("steermem.mem_seq", int(args.get("mem_seq") or 16))
    w.add_uint32("steermem.head", int(args.get("head") if args.get("head") is not None else 1))
    w.add_uint32("steermem.mem_marks", int(args.get("marks") or 0))
    w.add_uint32("steermem.keyemb", int(args.get("keyemb") if args.get("keyemb") is not None else 1))
    w.add_uint32("steermem.span_from_key", 1 if args.get("span_from") == "key" else 0)
    scales = {}
    for l in layers:
        scales[l] = max(0.0, float(side["scales.l%d" % l]))
        w.add_float32("steermem.scale.%d" % l, scales[l])
    w.add_tensor("memtag", f32(side["memtag"]))
    w.add_tensor("seqtag", f32(side["seqtag"]))
    w.add_tensor("marktag", f32(side["marktag"]))
    w.add_tensor("nulltag", f32(side["nulltag"]).reshape(-1))
    w.add_tensor("neartag", f32(side["neartag"]).reshape(-1))
    for l in layers:
        w.add_tensor("nullrec.%d" % l, f32(side["nullrec.l%d" % l]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s | step %s | layers %s | emb_norm %.4f | scales %s | mem_seq %s head %s marks %s keyemb %s span_from %s"
          % (a.out, ck.get("step"), layers, emb_norm, scales, args.get("mem_seq"), args.get("head"), args.get("marks"),
             args.get("keyemb"), args.get("span_from")))


if __name__ == "__main__":
    main()

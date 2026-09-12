# steermem: a memory that a model reads by key, for llama.cpp

This branch (`steermem`, on `8b4b355`) carries the llama.cpp side of
[steermem](https://github.com/KadePrice123): a lookup table the model addresses
by writing a key, whose cell arrives as the model's OWN hidden states on
dedicated positions right after the key. Nothing from the table is ever put in
the prompt as text. The trained side (a Qwen3.5-2B LoRA plus the delivery
modules) lives in the steermem trainer; this fork is what runs it as a GGUF.

## What is here today

| piece | state | where |
|---|---|---|
| exact-key n-gram bias sampler | done | `common/ngram-bias.{h,cpp}`, `common/sampling.cpp`, the `ngram_bias_*` server fields |
| position-tagged residual injection | done, tested | `llama_inject_set` / `llama_inject_clear` in `include/llama.h`; `build_inject` in `src/llama-graph.cpp`, applied after `build_cvec` in the Qwen3.5 / Qwen3-Next graphs |
| embedding-input batches for the MEM positions | already in llama.cpp | `llama_batch` with `embd` |
| hidden-state capture at the injection layers | next | over `cb_eval` |
| the merge, the key router, the sidecar, table hot-swap | next | see the plan below |

`llama_inject_set(ctx, il, n, pos, data, scale)` stores `n` unit vectors by
position for layer `il`; on the next `llama_decode` the residual stream after
that layer gets `scale * v[pos] * ||h[pos]||` at those positions and nothing
elsewhere. A graph with no injections is byte-for-byte the stock graph; a
graph with injections is never reused, so a stale node can never serve a new
table. `llama_inject_clear` removes them all. The test is
`steermem/inject_test.py` (ctypes on `libllama.so`: plain, injected, cleared;
the logits must differ in the middle run only).

## How a memory cell is delivered

1. The table is a file of `key, content` rows (csv or json). Loading a table
   is loading an index; nothing is computed until a key is used.
2. When a key is matched -- in the prompt, or in what the model writes -- the
   cell's content is run through the same GGUF model once and its hidden
   states at layers 8 and 16 are kept (the capture). One forward per cell,
   cached by content hash.
3. A MEM sequence is fed as an embedding batch after the key: a head position
   carrying the memory tag, the sequence code and the key's embedding, then
   one body position per content token carrying the tag and its rank. At
   layers 8 and 16 each body position gets that token's captured state
   injected (`llama_inject_set`). The model then attends to the cell as if it
   were in its context, and recites it.
4. Every key occurrence gets its own sequence; nothing is deduplicated, so a
   table updated between two mentions is read fresh the second time.

The capacity of this carrier was measured in the trainer: one position holds
one exact token, and a cell should hold at most about 256 tokens; longer
content is split into linked cells.

## Building

    cmake -B build-cpu -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j 8 --target llama-cli llama-server

The 2B runs on an AVX2 CPU at Q8_0; the bf16 GGUF is only for machines with
bf16 kernels.

## Plan

- capture: read `l_out-8` / `l_out-16` through `cb_eval` for a cell's forward
- merge: the trained rank-256 query/key maps and the null key as a few small
  ggml ops at the injected positions (weights from a side file), or, in the
  first version, a fixed weight
- router: the n-gram matcher already in the sampler fires when a key completes
- sidecar: holds the table, builds the sequences, drives the API, streams, and
  hot-swaps tables at runtime; ships with a small demo table (Strong's entries
  and verses)

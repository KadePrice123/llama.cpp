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
| hidden-state capture at the injection layers | done | `steermem/steermem.cpp`, over `cb_eval` in a second context with the LoRA scale at 0 |
| the sidecar: table load (csv / json / jsonl), blocks after every key, traces, mem-gen, greedy decode | done, reads memory | `steermem/steermem.cpp` (`steermem-cli`); the export of a checkpoint's side weights is `steermem/side_export.py` |
| the side-by-side with the PyTorch trainer on identical cells | done: the same output | the trainer's serve mode with `"raw": true` and pinned `"cells"` against `steermem-cli` on the same prompt: both recite James 3:14, both loop on one-line cells under the trainer's line rule (second result below) |
| the HTTP server and the chat page | done | `steermem-cli --serve PORT --tables DIR`: GET /, /ui, POST /preset, /table, /chat (NDJSON stream), /stop; the trainer's serve API |
| table hot-swap | done | POST /table (csv text or a cells list) or /preset replaces the cells; captures are lazy, so a swap costs nothing until a key is used |
| a Windows build on base Windows (no WSL, no Python) | done | cross-compiled from WSL with MinGW-w64, static, AVX2; see Building |
| the merge for K>1 | next | see the plan below |

First result (2026-09-12 16:31, the Bible table of 45,106 verse and Strong's
cells, the checkpoint gck_D4N4k2B.s250 as base + LoRA adapter + side weights):

    prompt   Recall each of these from memory, in order:
             1. James 3:14
             2. G26
             Answer:
    output   1. James 3:14:
             James 3:14 (ASV): But if ye have bitter jealousy and faction in your
             heart, glory not and lie not against the truth.
             Strong's: But [B1161] ; if [G1487] ; have [G2192] ; bitter [G4089] ; ...

the verse word for word out of the injected block, in 30 s on an AVX2 CPU.
One character of the Strong's line is wrong (B1161 for G1161). Keys match
on word boundaries: Qwen tokenizes digits singly, so "James 3:1" is a
token-prefix of "James 3:14", and a plain subsequence match delivered the
wrong verse on a full table before this rule. The checkpoint reads the
kinds of cells it trained on (verses, Strong's, file records); on an
invented parameter table it loops in both the trainer and here -- the
fine-tuning stage is what teaches those.

Second result (17:10): the same prompt through the trainer's serve mode and
through the sidecar, the two Strong's cells G26 and H1925 delivered, gave the
same loop in both ("1. G26:", "G26:", "G26:", ...). The cause is in the
trainer's rule for a row's "line" (the state pooled for the head and the
per-position trace): "after the second newline", which for a one-line cell is
the closing tag alone. `steermem-cli --line-rule 1` pools the content instead,
and with it the sidecar reads the Strong's pair exactly, the file cells, and
James 3:14 -- so the fork reproduces the trainer's behaviour under rule 2 and
does better under rule 1. The side GGUF records the rule a checkpoint trained
with (`steermem.line_rule`; absent means 2), and the sidecar follows it unless
the flag overrides. `--no-traces` switches the per-position trace off.

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

## Running it

    steermem-cli --model qwen35-2b-q8_0.gguf --lora steermem-STAGE-lora.gguf --side steermem-side-STAGE.gguf         --table tables/cells_demo_files.jsonl --tables tables --serve 8151
    # then open http://127.0.0.1:8151/ui -- presets, csv/json upload, MEM toggle, the reasoning and the answer

The page and the API are in the binary (cpp-httplib, vendored); nothing else is
needed at the venue. Messages go under Qwen's chat template with the opening
think tag as the generation prompt, so the model emits its reasoning and the
closing tag itself; keys it writes in that reasoning fire their blocks as it
writes them (mem-gen), a key counting as complete only once the next token
confirms a word boundary. One generation at a time; `/stop` ends it at the next
token. The trainer's Streamlit panel (`tools/memchat_ui.py`) speaks the same API.

## Building

Linux (the libraries, then the sidecar against them):

    cmake -B build-cpu -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j 8 --target llama ggml
    g++ -O2 -std=c++17 -I include -I ggml/include -I vendor steermem/steermem.cpp vendor/cpp-httplib/httplib.cpp         -L build-cpu/bin -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$PWD/build-cpu/bin -o build-cpu/bin/steermem-cli

Windows, cross-compiled from Linux/WSL with MinGW-w64 (`apt install mingw-w64`), a
static exe that runs on base Windows (AVX2; no GPU):

    cmake -B build-win -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc-posix         -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++-posix -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres         -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF         -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
    cmake --build build-win -j 12 --target llama ggml
    x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -I include -I ggml/include -I vendor steermem/steermem.cpp         vendor/cpp-httplib/httplib.cpp build-win/src/libllama.a build-win/ggml/src/ggml.a         build-win/ggml/src/ggml-cpu.a build-win/ggml/src/ggml-base.a -lws2_32 -static -static-libgcc         -static-libstdc++ -o build-win/steermem-cli.exe

The `-posix` compilers matter (std::thread for the server). The old stock targets:

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

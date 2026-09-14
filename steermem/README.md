# steermem: a memory that a model reads by key, for llama.cpp

This branch (`steermem`, on `8b4b355`) carries the llama.cpp side of
[steermem](https://github.com/KadePrice123): a lookup table the model addresses
by writing a key, whose cell arrives as the model's OWN hidden states on
dedicated positions right after the key. Nothing from the table is ever put in
the prompt as text. The trained side (a Qwen3.5-2B LoRA plus the delivery
modules) lives in the steermem trainer; this fork is what runs it as a GGUF.

It runs two kinds of checkpoint. The older ones fire a block when a key string
appears (the sidecar below). The two-sided checkpoints (`twoside.py`) run in
**protocol mode**: the model asks for a memory with `<recall>KEY<recall>` while
it reasons, decides what to do with what comes back (answer, or call a tool the
memory points to), and its memory grows as it is used. Memory can also come to
it unasked: listed in the system prompt when the message brings it to mind, or
injected right after a surprising word that names it.

## What is here today

| piece | state | where |
|---|---|---|
| exact-key n-gram bias sampler | done | `common/ngram-bias.{h,cpp}`, `common/sampling.cpp`, the `ngram_bias_*` server fields |
| position-tagged residual injection | done, tested | `llama_inject_set` / `llama_inject_clear` in `include/llama.h`; `build_inject` in `src/llama-graph.cpp`, applied after `build_cvec` in the Qwen3.5 / Qwen3-Next graphs |
| embedding-input batches for the MEM positions | already in llama.cpp | `llama_batch` with `embd` |
| hidden-state capture at the injection layers | done | `steermem/steermem.cpp`, over `cb_eval` in a second context |
| the sidecar: table load (csv / json / jsonl), blocks after every key, traces, mem-gen, greedy decode | done, reads memory | `steermem/steermem.cpp` (`steermem-cli`); the export of a checkpoint's side weights is `steermem/side_export.py` |
| the side-by-side with the PyTorch trainer on identical cells | done: the same output | the trainer's serve mode with `"raw": true` and pinned `"cells"` against `steermem-cli` on the same prompt |
| the HTTP server and the chat page (sidecar) | done | `steermem-cli --serve PORT --tables DIR`: GET /, /ui, POST /preset, /table, /chat (NDJSON stream), /stop |
| **protocol mode**: `<recall>KEY<recall>` in the model's own reasoning answered by the table on the spot, the null block with the nearest keys on a miss, Qwen3.5 tool calls, `<\|remember\|>` and `<\|alias\|>` | done, tested | `steermem/steermem_protocol.hpp`, edit `0018_protocol.py`; `test_protocol.sh` |
| **memory that builds as it is used**: the model's own `<\|remember\|>`, `POST /api/remember`, uploads (CSV / JSON / JSONL / text), a queue, sleep mode, `--memory-out` | done, tested | the same |
| **the auto menu**: the K memories a message brings to mind, found on the model's own hidden states, listed in the system prompt | done, tested | `proto_menu_*`, edit `0019_auto_menu.py` |
| **the saved menu index**: a restart reads it back and computes only new or changed memories | done, tested | `proto_menu_cache_*`, edit `0020_menu_cache.py` |
| **auto-inject**: a memory's block right after a surprising word in the message that names it, no `<recall>` needed | done, tested | `proto_plan_injects`, edit `0021_auto_inject.py` |
| **`--chat`**: a command-line conversation with every memory command | done, tested | `run_chat_repl` in the header |
| the harness API shared with the Python demo | done | `steermem/harness/`: `API.md`, `demo_ui.html`, `steermem_harness.py` (a standard-library CLI client) |
| protocol-mode export | done | `twoside_export.py` (the side GGUF); the adapter through `lora_to_peft.py` and `convert_lora_to_gguf.py` |
| steermem-cli as a CMake target | done | `-DLLAMA_BUILD_STEERMEM=ON`, edit `0022_cmake_target.py`, `steermem/CMakeLists.txt` |
| Windows builds | CPU: done (MinGW cross-compile); CUDA: the workflow below | `.github/workflows/steermem-windows-cuda.yml` |
| the merge for K>1 | next | see the plan below |

## Protocol mode

A two-sided checkpoint trains the sleep pass (the model writes a memory, `KEY is
SUMMARY`, and the hidden states of that writing ARE the memory) and the wake
pass (the model reads those states back) in one backward. Protocol mode serves
exactly that, rule for rule with the trainer:

- **capture**: the sleep prompt with the context, then `KEY is SUMMARY`, adapter
  on; the states after layers 8 and 16 from the key's first token through the
  summary's last.
- **the block**: `[head] rows j = memtag + seqtag[1+min(j,61)]` over evenly
  spaced states, rows normalised to the embedding norm, the states injected as
  unit vectors at the side layers.
- **a miss**: the null block (`nulltag`, injecting `nullrec`) plus up to three
  rows naming the nearest keys the table holds.
- **the loop**: greedy decoding; `<recall>KEY<recall>` in the text since the
  last block gets the table's answer right there. A turn that ends in a Qwen3.5
  `<tool_call>` runs the built-in tool (`tool_docs`, `lookup_verse`,
  `lookup_strongs`, `read_file` under `--files-dir`), and its
  `<tool_response>` starts the next turn.
- **memory in**: `<|remember|>KEY is ...<|remember|>` in an answer stores a
  memory (or queues it, `--remember-mode queue`); a harness stores with
  `POST /api/remember` or uploads a file; sleep (`POST /api/sleep`,
  `--sleep-at-start`, `--sleep-only`) has the model read each queued context,
  write the memory itself, capture its states and append it to `--memory-out`.
- **the auto menu** (`--auto-menu K`): every memory is also held as its
  content-token states at layer 6 over `KEY is SUMMARY`. Before an answer the
  user's message is matched token by token, each token's best match weighted by
  its rarity in the table, and the K best keys go into the system prompt in the
  key-menu format the checkpoint was trained on. The model still decides whether
  to recall one. The index is saved next to the memory file (`--menu-cache`),
  stamped with the model, adapter and layer that made it; on a T1200 it takes
  about 95 s for 580 memories the first time and a tenth of a second after.
- **auto-inject** (`--auto-inject N`): before the answer the model reads the
  message once with no memory. A word it finds surprising (its tokens' summed
  -log p at or above `--inject-surprise` nats, default 8) that names a stored key
  or alias -- or, with `--inject-match state` (the default), whose own layer-6
  states match one memory with a score of at least `--inject-min-score` and 0.05
  over the runner-up -- gets up to N rows of that memory's block placed right
  after it, the most surprising `--inject-max` words first (default 3). Measured
  on the step-600 checkpoint over 60 memories, with no `<recall>` at all:

  | rows after the word | answer loss (none: 1.989) | stored facts written back, of 19 |
  |---|---|---|
  | 2 | +0.088 | -- |
  | 8 | -0.432 | -- |
  | 16 | -0.738 | 0 |
  | 32 | -1.308 | 15 |
  | the whole block (43 rows on average) | -1.678 | 18 |
  | a real recall | -1.870 | 18 |
  | a wrong memory's whole block | +0.647 | -- |

  A couple of rows is not enough; the whole block is as good as a recall for the
  facts. A block is never longer than its memory, so `--auto-inject 64` gives
  the whole block for most memories.

Export a checkpoint, then run it:

    python twoside_export.py --ckpt twoside_pointer.s600.pt --model /path/to/Qwen3.5-2B --out steermem-side-pointer-s600.gguf
    python lora_to_peft.py --ckpt twoside_pointer.s600.pt --out peft_s600 --base /path/to/Qwen3.5-2B
    python convert_lora_to_gguf.py --base /path/to/Qwen3.5-2B --outfile steermem-pointer-s600-lora.gguf peft_s600

    steermem-cli --model qwen35-2b-q8_0.gguf --lora steermem-pointer-s600-lora.gguf --side steermem-side-pointer-s600.gguf \
        --table memories.jsonl --tool-docs tool_docs.jsonl --lookup bible_lookup.jsonl --memory-out learned.jsonl \
        --auto-menu 3 --auto-inject 64 -ngl 99 --chat

| flag | what it does |
|---|---|
| `--chat` | talk to the model at the command line, with the commands below |
| `--serve PORT --ui harness/demo_ui.html` | the HTTP API and the demo page instead |
| `--prompt TEXT` | one answer, every event printed, then exit |
| `--memory-out FILE` | where learned memories are appended; loaded at start |
| `--learn FILE` | queue a CSV / JSON / JSONL / text file (repeatable) |
| `--sleep-at-start`, `--sleep-only` | process the queue before serving, or process it and exit |
| `--remember-mode store\|queue` | what the model's own `<\|remember\|>` does |
| `--auto-menu K`, `--menu-layer L` | the auto menu, and the layer it matches on (default 6) |
| `--menu-cache FILE\|none` | where the menu index is saved (default: the `--memory-out` file + `.menu`) |
| `--auto-inject N` | up to N memory tokens after each surprising word that names (or clearly matches) a memory; 0 is off |
| `--inject-max K`, `--inject-surprise T` | how many words per message (3), and how surprising, in nats (8) |
| `--inject-match key\|state`, `--inject-min-score S` | only named keys, or also a clear hidden-state match (default), and that match's bar (0.6) |
| `--tool-docs`, `--lookup`, `--examples`, `--files-dir` | the data the built-in tools and the page read |
| `--system TEXT` | a system prompt, e.g. a key menu written by hand |

In `--chat`: `/upload FILE`, `/remember [KEY ::] TEXT`, `/remember-now [KEY ::] TEXT`,
`/sleep [N]`, `/queue`, `/memories [QUERY]`, `/alias BAD -> GOOD`, `/system TEXT`,
`/mode store|queue`, `/menu K`, `/inject N`, `/state`, `/new`, `/quit`; anything
else is a message to the model.

`steermem/test_protocol.sh` exercises all of it on a GPU: a one-shot prompt, a
batch sleep that learns a CSV, a scripted `--chat` session with every command,
the auto menu and its saved index, auto-inject, and every HTTP route. It checks
plumbing, not answer quality; that is the checkpoint's evaluation.

    MODEL=qwen35-2b-q8_0.gguf TABLES=demo/tables UI=steermem/harness/demo_ui.html BIN=build/bin/steermem-cli \
        bash steermem/test_protocol.sh steermem-side-pointer-s600.gguf steermem-pointer-s600-lora.gguf

`TABLES` is the steermem demo's `tables/` folder (memories, tool manuals, the
Bible lookup, examples). `steermem/harness/` holds the page, `API.md`, and
`steermem_harness.py`, a standard-library command-line client of the API.

## The sidecar (key-string checkpoints)

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

### How a sidecar cell is delivered

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

### Running the sidecar

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

With CMake, on any platform (edit 0022):

    cmake -B build -DLLAMA_BUILD_STEERMEM=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
    cmake --build build --config Release --target steermem-cli
    # add -DGGML_CUDA=ON for NVIDIA GPUs, then run with -ngl 99

**Windows with CUDA:** `.github/workflows/steermem-windows-cuda.yml` builds
steermem-cli on a `windows-2022` runner with CUDA 12.4, the same toolkit setup
and compiler as llama.cpp's own Windows CUDA release (GPU architectures 7.5,
8.0, 8.6 and 8.9), and publishes a release with the exe and the CUDA runtime
DLLs it needs. It runs on pushes to this branch that touch the sources, or by
hand from the Actions tab; on a fork, GitHub Actions has to be enabled for the
repository first.

By hand on Linux (the libraries, then the sidecar against them):

    cmake -B build-cpu -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j 8 --target llama ggml
    g++ -O2 -std=c++17 -I include -I ggml/include -I vendor steermem/steermem.cpp vendor/cpp-httplib/httplib.cpp         -L build-cpu/bin -lllama -lggml -lggml-base -lpthread -Wl,-rpath,$PWD/build-cpu/bin -o build-cpu/bin/steermem-cli

Windows on the CPU, cross-compiled from Linux/WSL with MinGW-w64 (`apt install
mingw-w64`), a static exe that runs on base Windows (AVX2; no GPU):

    cmake -B build-win -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc-posix         -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++-posix -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres         -DGGML_VULKAN=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF         -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
    cmake --build build-win -j 12 --target llama ggml
    x86_64-w64-mingw32-g++-posix -O2 -std=c++17 -I include -I ggml/include -I vendor steermem/steermem.cpp         vendor/cpp-httplib/httplib.cpp build-win/src/libllama.a build-win/ggml/src/ggml.a         build-win/ggml/src/ggml-cpu.a build-win/ggml/src/ggml-base.a -lws2_32 -static -static-libgcc         -static-libstdc++ -o build-win/steermem-cli.exe

The `-posix` compilers matter (std::thread for the server and the menu). The old stock targets:

    cmake --build build-cpu -j 8 --target llama-cli llama-server

The 2B runs on an AVX2 CPU at Q8_0; the bf16 GGUF is only for machines with
bf16 kernels.

## Plan

- done: capture through `cb_eval`; the sidecar with its table, sequences, API,
  stream and hot-swap; protocol mode with sleep, uploads, the auto menu and its
  saved index, and auto-inject; a CMake target and a Windows CUDA workflow
- next: calibrate auto-inject's hidden-state match on real conversations (the
  named-key match is the precise one; a wrong memory's block costs as much as the
  right one helps)
- merge: the trained rank-256 query/key maps and the null key as a few small
  ggml ops at the injected positions (weights from a side file), or, in the
  first version, a fixed weight

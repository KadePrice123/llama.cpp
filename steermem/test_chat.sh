#!/bin/bash
# test_chat.sh -- the fork's checks on a stage's GGUF set, in the reasoning format the
# model trains on (2026-09-12 17:30: "then convert to GGUF and run tests"). Each prompt
# is a user turn under Qwen's template ending in the opening think tag; mem-gen is on,
# so a key the model writes in its think gets its block there. Prints each generation.
#   bash test_chat.sh STAGE [MAX_NEW=160] [THREADS=6]
# Needs: ~/gguf/qwen35-2b-bf16.gguf, steermem-STAGE-lora.gguf, steermem-side-STAGE.gguf
# (export_stage.sh STAGE CKPT 128 1), /tmp/bible_all.jsonl (verses + Strong's), the demo
# tables in ~/steermem/data (cells_demo_files.jsonl, cells_demo_bad.jsonl).
ST=${1:?stage}; MAXNEW=${2:-160}; THREADS=${3:-6}
G=/home/kade/gguf; D=/home/kade/steermem/data; CLI=/home/kade/llama-ngram/build-cpu/bin/steermem-cli
[ -s /tmp/bible_all.jsonl ] || cat $D/cells_verses.jsonl $D/cells_strongs.jsonl > /tmp/bible_all.jsonl
mk() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n' "$1"; }
run() {   # name table prompt
  echo "=== $1"
  $CLI --model $G/qwen35-2b-bf16.gguf --lora $G/steermem-$ST-lora.gguf --side $G/steermem-side-$ST.gguf \
       --table "$2" --prompt "$3" --max-new $MAXNEW --threads $THREADS --memgen 1 2>&1 | grep -E '^(prompt|\{"blocks)' | cut -c1-900
}
run "bible: James 3:14 + G26"          /tmp/bible_all.jsonl          "$(mk $'Recall each of these from memory, in order:\n1. James 3:14\n2. G26')"
run "files: truck_queue.py + scale_calib.md" $D/cells_demo_files.jsonl "$(mk $'Recall each of these from memory, in order:\n1. truck_queue.py\n2. scale_calib.md')"
run "natural question"                 /tmp/bible_all.jsonl          "$(mk "What does Strong's G26 mean, and what does James 3:14 say?")"
run "garbled memory (James 3:14 nonsense; Psalms 23:1 fine)" $D/cells_demo_bad.jsonl "$(mk $'Recall each of these from memory, in order:\n1. James 3:14\n2. Psalms 23:1')"
run "no memory for the key (table lacks it)" $D/cells_demo_files.jsonl "$(mk $'Recall each of these from memory, in order:\n1. John 3:16')"

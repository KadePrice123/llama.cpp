#!/bin/bash
# A CHECKPOINT AS THE GGUF SET THE FORK SERVES (2026-09-12 16:30). Kade: "build a
# GGUF from it as the one to test so we can give it actual prompts."
#   bash export_stage.sh STAGE /path/gck_STAGE.sN.pt [MEMSEQ=256] [MEMGEN=1]
# Produces, in ~/gguf:
#   peft_STAGE/                       the LoRA as a PEFT adapter (lora_to_peft.py)
#   steermem-STAGE-lora.gguf          llama.cpp's LoRA adapter (convert_lora_to_gguf.py)
#   steermem-STAGE-bf16.gguf          base + adapter merged (llama-export-lora)
#   steermem-STAGE-q8_0.gguf          the merged model at Q8_0 for an AVX2 CPU
#   steermem-side-STAGE.gguf          the delivery side: memtag, seqtag, exptag, emb_norm,
#                                     scales, merge maps, and the run settings (side_export.py)
# The base bf16 GGUF (qwen35-2b-bf16.gguf) is made once and reused. The sidecar
# then runs: steermem-cli --model qwen35-2b-bf16.gguf --lora steermem-STAGE-lora.gguf
#   --side steermem-side-STAGE.gguf --table T --prompt P   (the adapter at run time,
# because the capture needs the adapter OFF and the generation needs it ON).
set -e
ST=${1:?stage}; CK=${2:?checkpoint path}; MEMSEQ=${3:-256}; MEMGEN=${4:-1}
cd ~/steermem/expert
PY=~/xpu-venv/bin/python; G=~/gguf; L=~/llama.cpp; B=~/llama.cpp/build/bin
mkdir -p $G
log() { echo "$(date +%H:%M:%S) $*"; }
[ -s $G/qwen35-2b-bf16.gguf ] || { log "base -> gguf"; nice -n 12 $PY $L/convert_hf_to_gguf.py /home/kade/hf/qwen35-2b --outfile $G/qwen35-2b-bf16.gguf --outtype bf16 2>&1 | tail -n 2; }
log "peft export of $(basename $CK)"
nice -n 12 $PY lora_to_peft.py --ckpt $CK --out $G/peft_$ST --base /home/kade/hf/qwen35-2b
log "lora -> gguf"
nice -n 12 $PY $L/convert_lora_to_gguf.py --base /home/kade/hf/qwen35-2b --outfile $G/steermem-$ST-lora.gguf $G/peft_$ST 2>&1 | tail -n 2
log "merge + quantize Q8_0"
nice -n 12 $B/llama-export-lora -m $G/qwen35-2b-bf16.gguf --lora $G/steermem-$ST-lora.gguf -o $G/steermem-$ST-bf16.gguf 2>&1 | tail -n 1
nice -n 12 $B/llama-quantize $G/steermem-$ST-bf16.gguf $G/steermem-$ST-q8_0.gguf Q8_0 2>&1 | tail -n 1
log "side weights (mem_seq $MEMSEQ, mem_gen $MEMGEN)"
nice -n 12 $PY side_export.py --ckpt $CK --out $G/steermem-side-$ST.gguf --mem-seq $MEMSEQ --mem-gen $MEMGEN 2>&1 | grep -v -i warn | tail -n 1
ls -la $G/steermem-$ST-lora.gguf $G/steermem-$ST-q8_0.gguf $G/steermem-side-$ST.gguf
log "export of $ST done"

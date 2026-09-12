#!/usr/bin/env python3
"""The delivery side of a checkpoint as a small GGUF, for the llama.cpp fork.

What the fork needs beyond the model and its LoRA (2026-09-12 16:20, from
reading mem_block / _apply_entry / row_states in the trainer):
  memtag [d]        the <|MEM|> tag added to every block position
  seqtag [64, d]    0 = head, 1..61 = body rank, 63 = pad
  exptag [8, d]     the expert group's tag on the head (group 1 = document)
  emb_norm          every block row is normalised to this (the mean input-
                    embedding row norm of the base model)
  scale.8, scale.16 moe_scales: 0.5 -- the injection at a block position is
                    scale * unit(record state) * mean ||h|| over the block
  merge.q/k/v, null the rank-256 merge maps, carried for the K>1 case (with
                    one cell the direction is the record's and they cancel)
plus the run settings the checkpoint was trained with (layers, mem_seq,
memtok, key_window, mem_seq_cells, mem_gen), so the sidecar serves it the
way the eval did. Written with gguf-py so the C++ side reads it with gguf.

    python side_export.py --ckpt ~/ladder_S/gck_D4N4k2B.s250.pt --out ~/gguf/steermem-side-D4N4k2B.gguf \\
        --mem-seq 128 --mem-gen 0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/home/kade/hf/qwen35-2b", help="the base, for emb_norm")
    ap.add_argument("--layers", default="8,16")
    ap.add_argument("--mem-seq", type=int, default=128)
    ap.add_argument("--mem-seq-cells", type=int, default=4)
    ap.add_argument("--mem-gen", type=int, default=0)
    ap.add_argument("--line-rule", type=int, default=1, help="the row-line rule the checkpoint trained with: 1 = the content span (patch 79 on), 2 = after the second newline (checkpoints before 79)")
    ap.add_argument("--mem-seq-every", type=int, default=1)
    ap.add_argument("--memtok", type=int, default=8)
    ap.add_argument("--key-window", type=int, default=96)
    ap.add_argument("--null-max", type=float, default=0.2)
    ap.add_argument("--egroup", type=int, default=1, help="the expert group of every table cell (1 = document)")
    a = ap.parse_args()
    ck = torch.load(os.path.expanduser(a.ckpt), map_location="cpu")
    layers = [int(x) for x in a.layers.split(",")]
    # emb_norm exactly as the trainer computes it: the mean row norm of the input embedding
    from safetensors import safe_open
    import glob
    norms = None
    for f in sorted(glob.glob(os.path.join(a.model, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.endswith("embed_tokens.weight"):
                    w = st.get_tensor(k).to(torch.float32)
                    norms = w.norm(dim=-1)
                    break
        if norms is not None:
            break
    assert norms is not None, "no embed_tokens.weight in the base model"
    emb_norm = float(norms.mean())
    w = gguf.GGUFWriter(os.path.expanduser(a.out), "steermem-side")
    w.add_string("steermem.checkpoint", os.path.basename(a.ckpt))
    w.add_array("steermem.layers", layers)
    w.add_float32("steermem.emb_norm", emb_norm)
    w.add_uint32("steermem.mem_seq", a.mem_seq)
    w.add_uint32("steermem.mem_seq_cells", a.mem_seq_cells)
    w.add_uint32("steermem.mem_gen", a.mem_gen)
    w.add_uint32("steermem.mem_seq_every", a.mem_seq_every)
    w.add_uint32("steermem.memtok", a.memtok)
    w.add_uint32("steermem.key_window", a.key_window)
    w.add_uint32("steermem.line_rule", a.line_rule)      # 1: the content span (trainer patch 79); 2: after the second newline (before it)
    w.add_float32("steermem.null_max", a.null_max)
    w.add_uint32("steermem.egroup", a.egroup)
    for l in layers:
        w.add_float32("steermem.scale.%d" % l, float(ck["moe_scales"]["l%d" % l]))
    f32 = lambda t: t.detach().to(torch.float32).contiguous().numpy()
    w.add_tensor("memtag", f32(ck["memtag"]))
    w.add_tensor("seqtag", f32(ck["seqtag"]))
    w.add_tensor("exptag", f32(ck["exptag"]))
    mm, mn = ck["moe_merge"], ck["moe_null"]
    for l in layers:
        w.add_tensor("merge.q%d" % l, f32(mm["q%d.weight" % l]))
        w.add_tensor("merge.k%d" % l, f32(mm["k%d.weight" % l]))
        w.add_tensor("merge.null%d" % l, f32(mn["n%d" % l]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s | emb_norm %.4f | scales %s | mem_seq %d mem_gen %d" % (a.out, emb_norm, {l: float(ck["moe_scales"]["l%d" % l]) for l in layers}, a.mem_seq, a.mem_gen))


if __name__ == "__main__":
    main()

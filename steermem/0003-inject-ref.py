#!/usr/bin/env python3
"""llama.cpp: the injection's reference norm is the MEAN over the injected positions.

Run inside the fork's worktree after 0002-inject.py. The trainer's rule
(_apply_entry): _ref = the mean of ||h|| over all the positions of one
injection entry, then h[pos] += scale * unit(c) * _ref. A MEM block is one
entry, so every position of the block is scaled by the block's mean norm, not
its own. 0002 used each position's own norm; this adds a mask input (1 where a
vector is set) and computes ref = sum(||h|| * mask) / sum(mask) over the ubatch.
A single injected position gets its own norm, as before.
"""
import io


def edit(path, old, new, count=1):
    s = io.open(path, encoding="utf-8").read()
    n = s.count(old)
    assert n == count, "%s: anchor found %d times, expected %d:\n%s" % (path, n, count, old[:120])
    s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("  edited", path)


edit("src/llama-graph.h",
     "    ggml_tensor * vec = nullptr; // F32 [n_embd, n_batch]\n\n    const llama_inject_table * table;\n",
     "    ggml_tensor * vec  = nullptr; // F32 [n_embd, n_batch]\n"
     "    ggml_tensor * mask = nullptr; // F32 [1, n_batch]: 1 where a vector is set\n\n    const llama_inject_table * table;\n")

edit("src/llama-graph.cpp",
     "    std::vector<float> buf((size_t) n_tokens * n_embd, 0.0f);\n"
     "    if (table != nullptr) {\n",
     "    std::vector<float> buf((size_t) n_tokens * n_embd, 0.0f);\n"
     "    std::vector<float> msk((size_t) n_tokens, 0.0f);\n"
     "    if (table != nullptr) {\n")
edit("src/llama-graph.cpp",
     "                    std::copy(vit->second.begin(), vit->second.end(), buf.begin() + i*n_embd);\n",
     "                    std::copy(vit->second.begin(), vit->second.end(), buf.begin() + i*n_embd);\n"
     "                    msk[i] = 1.0f;\n")
edit("src/llama-graph.cpp",
     "    ggml_backend_tensor_set(vec, buf.data(), 0, buf.size()*sizeof(float));\n}\n",
     "    ggml_backend_tensor_set(vec, buf.data(), 0, buf.size()*sizeof(float));\n"
     "    ggml_backend_tensor_set(mask, msk.data(), 0, msk.size()*sizeof(float));\n}\n")
edit("src/llama-graph.cpp",
     "    inp->vec = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);\n"
     "    ggml_set_input(inp->vec);\n"
     "    cb(inp->vec, \"inject_vec\", il);\n"
     "    // ||h|| per token: a unit vector lands at the residual's own scale (the trainer's rule)\n"
     "    ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));   // [1, n_tokens]\n"
     "    ggml_tensor * add = ggml_scale(ctx0, ggml_mul(ctx0, inp->vec, nrm), lit->second.scale);\n",
     "    inp->vec = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);\n"
     "    ggml_set_input(inp->vec);\n"
     "    cb(inp->vec, \"inject_vec\", il);\n"
     "    inp->mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);\n"
     "    ggml_set_input(inp->mask);\n"
     "    cb(inp->mask, \"inject_mask\", il);\n"
     "    // the trainer's rule: one reference norm per injection entry -- the MEAN of ||h|| over the injected\n"
     "    // positions of this ubatch (a MEM block is fed as one ubatch, so this is the block's mean) -- and\n"
     "    // h += scale * unit(v) * ref at each of them; a single injected position gets its own norm\n"
     "    ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));   // [1, n_tokens]\n"
     "    ggml_tensor * num = ggml_sum(ctx0, ggml_mul(ctx0, nrm, inp->mask));               // [1]\n"
     "    ggml_tensor * den = ggml_sum(ctx0, inp->mask);                                    // [1]\n"
     "    ggml_tensor * ref = ggml_div(ctx0, num, den);                                     // [1]\n"
     "    ggml_tensor * add = ggml_scale(ctx0, ggml_mul(ctx0, inp->vec, ref), lit->second.scale);\n")
print("inject-ref: applied")

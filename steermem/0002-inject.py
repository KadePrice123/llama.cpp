#!/usr/bin/env python3
"""llama.cpp: position-tagged residual injection (the steermem fork's second change).

Run inside the fork's worktree (~/llama-ngram). Edits, with exact anchors:
  include/llama.h        llama_inject_set / llama_inject_clear (public API)
  src/llama-graph.h      llama_inject_table; llm_graph_input_inject; the params
                         and context fields; build_inject decl
  src/llama-graph.cpp    set_input; build_inject; the ctor init; allow_reuse
  src/llama-context.h    the table member; inject_set / inject_clear methods
  src/llama-context.cpp  the params initialiser; the C API
  src/models/qwen35.cpp, qwen35moe.cpp, qwen3next.cpp   build_inject after build_cvec

What it does: for layer il and a set of positions, add s * v[pos] * ||h[pos]|| to
the residual stream after that layer -- the trainer's injection rule (a unit
vector at the residual's own scale, patch 58/64) -- with v[pos] set from the host
before llama_decode. Everything else in the model is untouched: a graph with no
injections is byte-for-byte the old graph, and one with injections is never
reused (the input's can_reuse is the base class's false), so a stale node can
never serve a new table. With llama_batch embd inputs (already in llama.cpp)
this is what a MEM sequence needs: tag embeddings at the block positions and the
record's captured states injected there at layers 8 and 16.
"""
import io
import os
import sys


def edit(path, old, new, count=1):
    s = io.open(path, encoding="utf-8").read()
    n = s.count(old)
    assert n == count, "%s: anchor found %d times, expected %d:\n%s" % (path, n, count, old[:120])
    s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("  edited", path)


assert os.path.exists("src/llama-graph.h"), "run this inside the llama.cpp worktree"

# ---- include/llama.h: the public API ------------------------------------------------------
edit("include/llama.h",
     "    LLAMA_API int32_t llama_set_adapter_cvec(\n",
     "    // steermem: position-tagged residual injection. For layer il, add\n"
     "    // scale * v[pos] * ||h[pos]|| to the residual stream after that layer at the\n"
     "    // given positions (v: n rows of n_embd floats, unit vectors); other positions\n"
     "    // are untouched. Set before llama_decode; llama_inject_clear removes all.\n"
     "    LLAMA_API int32_t llama_inject_set(\n"
     "            struct llama_context * ctx,\n"
     "                         int32_t   il,\n"
     "                         int32_t   n,\n"
     "                 const llama_pos * pos,\n"
     "                     const float * data,\n"
     "                           float   scale);\n"
     "    LLAMA_API void llama_inject_clear(struct llama_context * ctx);\n\n"
     "    LLAMA_API int32_t llama_set_adapter_cvec(\n")

# ---- src/llama-graph.h ---------------------------------------------------------------------
edit("src/llama-graph.h",
     "struct llm_graph_params {\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n",
     "// steermem: per-layer, per-position residual injections (see llama_inject_set)\n"
     "struct llama_inject_table {\n"
     "    struct layer {\n"
     "        float scale = 1.0f;\n"
     "        std::unordered_map<llama_pos, std::vector<float>> vec;   // position -> n_embd floats\n"
     "    };\n"
     "    std::map<int32_t, layer> layers;   // il -> its vectors\n"
     "    bool empty() const { return layers.empty(); }\n"
     "};\n\n"
     "struct llm_graph_params {\n    llm_arch arch = LLM_ARCH_UNKNOWN;\n")
edit("src/llama-graph.h",
     "    const llama_cross            * cross;\n\n    std::map<llama_seq_id, llama_sampler *> samplers;\n\n    static bool samplers_equal(\n",
     "    const llama_cross            * cross;\n    const llama_inject_table     * inject;   // steermem\n"
     "    bool inject_on = false;   // steermem: a SNAPSHOT of !inject->empty() at graph time -- the pointer is to the live\n"
     "                              // table, so comparing it could never tell an old graph without injection nodes from a new\n"
     "                              // one that needs them (the first build reused the plain graph and injected nothing)\n\n"
     "    std::map<llama_seq_id, llama_sampler *> samplers;\n\n    static bool samplers_equal(\n")
edit("src/llama-graph.h",
     "            cvec  == other.cvec  &&\n",
     "            cvec  == other.cvec  &&\n"
     "            inject_on == other.inject_on &&   // steermem\n")
edit("src/llama-graph.h",
     "// similar to llm_graph_input_embd but with an additional hidden state input\n",
     "// steermem: one layer's injection vectors, gathered by position for this ubatch\n"
     "struct llama_inject_table;   // defined below, beside llm_graph_params\n"
     "class llm_graph_input_inject : public llm_graph_input_i {\n"
     "public:\n"
     "    llm_graph_input_inject(const llama_inject_table * table, int32_t il, int64_t n_embd) : table(table), il(il), n_embd(n_embd) {}\n"
     "    virtual ~llm_graph_input_inject() = default;\n\n"
     "    void set_input(const llama_ubatch * ubatch) override;\n\n"
     "    ggml_tensor * vec = nullptr; // F32 [n_embd, n_batch]\n\n"
     "    const llama_inject_table * table;\n"
     "    const int32_t il;\n"
     "    const int64_t n_embd;\n"
     "};\n\n"
     "// similar to llm_graph_input_embd but with an additional hidden state input\n")
edit("src/llama-graph.h",
     "    const llama_cross            * cross;\n\n    std::map<llama_seq_id, llama_sampler *> samplers;\n\n    const llm_graph_cb & cb_func;\n",
     "    const llama_cross            * cross;\n    const llama_inject_table     * inject;   // steermem\n\n"
     "    std::map<llama_seq_id, llama_sampler *> samplers;\n\n    const llm_graph_cb & cb_func;\n")
edit("src/llama-graph.h",
     "    ggml_tensor * build_cvec(\n",
     "    // steermem: position-tagged residual injection after layer il\n"
     "    ggml_tensor * build_inject(\n"
     "             ggml_tensor * cur,\n"
     "                     int   il) const;\n\n"
     "    ggml_tensor * build_cvec(\n")
s = io.open("src/llama-graph.h", encoding="utf-8").read()
if "#include <unordered_map>" not in s:
    edit("src/llama-graph.h", "#include <map>\n", "#include <map>\n#include <unordered_map>\n")

# ---- src/llama-graph.cpp -------------------------------------------------------------------
edit("src/llama-graph.cpp",
     "    cross            (params.cross),\n",
     "    cross            (params.cross),\n    inject           (params.inject),\n")
edit("src/llama-graph.cpp",
     "ggml_tensor * llm_graph_context::build_cvec(\n",
     "// steermem\n"
     "void llm_graph_input_inject::set_input(const llama_ubatch * ubatch) {\n"
     "    const int64_t n_tokens = ubatch->n_tokens;\n"
     "    std::vector<float> buf((size_t) n_tokens * n_embd, 0.0f);\n"
     "    if (table != nullptr) {\n"
     "        const auto lit = table->layers.find(il);\n"
     "        if (lit != table->layers.end()) {\n"
     "            for (int64_t i = 0; i < n_tokens; ++i) {\n"
     "                const auto vit = lit->second.vec.find(ubatch->pos[i]);   // the temporal position (n_pos == 1, or the first dim)\n"
     "                if (vit != lit->second.vec.end() && (int64_t) vit->second.size() == n_embd) {\n"
     "                    std::copy(vit->second.begin(), vit->second.end(), buf.begin() + i*n_embd);\n"
     "                }\n"
     "            }\n"
     "        }\n"
     "    }\n"
     "    ggml_backend_tensor_set(vec, buf.data(), 0, buf.size()*sizeof(float));\n"
     "}\n\n"
     "ggml_tensor * llm_graph_context::build_inject(\n"
     "         ggml_tensor * cur,\n"
     "                 int   il) const {\n"
     "    if (inject == nullptr || inject->empty()) {\n"
     "        return cur;\n"
     "    }\n"
     "    const auto lit = inject->layers.find(il);\n"
     "    if (lit == inject->layers.end()) {\n"
     "        return cur;\n"
     "    }\n"
     "    auto inp = std::make_unique<llm_graph_input_inject>(inject, il, n_embd);\n"
     "    inp->vec = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);\n"
     "    ggml_set_input(inp->vec);\n"
     "    cb(inp->vec, \"inject_vec\", il);\n"
     "    // ||h|| per token: a unit vector lands at the residual's own scale (the trainer's rule)\n"
     "    ggml_tensor * nrm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));   // [1, n_tokens]\n"
     "    ggml_tensor * add = ggml_scale(ctx0, ggml_mul(ctx0, inp->vec, nrm), lit->second.scale);\n"
     "    cur = ggml_add(ctx0, cur, add);\n"
     "    cb(cur, \"inject_out\", il);\n"
     "    res->add_input(std::move(inp));\n"
     "    return cur;\n"
     "}\n\n"
     "ggml_tensor * llm_graph_context::build_cvec(\n")

# ---- src/llama-context.h -------------------------------------------------------------------
edit("src/llama-context.h",
     "    llama_adapter_cvec_ptr  cvec;\n",
     "    llama_adapter_cvec_ptr  cvec;\n    llama_inject_table      inject;   // steermem\n")
edit("src/llama-context.h",
     "    bool set_adapter_cvec(\n",
     "    // steermem: position-tagged residual injection (llama_inject_set); bodies in the .cpp,\n"
     "    // where llama_model is a complete type\n"
     "    int32_t inject_set(int32_t il, int32_t n, const llama_pos * pos, const float * data, float scale);\n"
     "    void    inject_clear();\n\n"
     "    bool set_adapter_cvec(\n")

# ---- src/llama-context.cpp -----------------------------------------------------------------
edit("src/llama-context.cpp",
     "        /*.cross       =*/ &cross,\n",
     "        /*.cross       =*/ &cross,\n        /*.inject      =*/ &inject,\n        /*.inject_on   =*/ !inject.empty(),\n")
edit("src/llama-context.cpp",
     "int32_t llama_set_adapter_cvec(\n",
     "// steermem\n"
     "int32_t llama_context::inject_set(int32_t il, int32_t n, const llama_pos * pos, const float * data, float scale) {\n"
     "    if (n <= 0 || il < 0 || pos == nullptr || data == nullptr) {\n"
     "        return -1;\n"
     "    }\n"
     "    auto & L = inject.layers[il];\n"
     "    L.scale = scale;\n"
     "    const size_t ne = model.hparams.n_embd;\n"
     "    for (int32_t i = 0; i < n; ++i) {\n"
     "        L.vec[pos[i]].assign(data + (size_t) i*ne, data + (size_t) (i + 1)*ne);\n"
     "    }\n"
     "    return 0;\n"
     "}\n\n"
     "void llama_context::inject_clear() {\n"
     "    inject.layers.clear();\n"
     "}\n\n"
     "int32_t llama_inject_set(llama_context * ctx, int32_t il, int32_t n, const llama_pos * pos, const float * data, float scale) {\n"
     "    return ctx->inject_set(il, n, pos, data, scale);\n"
     "}\n\n"
     "void llama_inject_clear(llama_context * ctx) {\n"
     "    ctx->inject_clear();\n"
     "}\n\n"
     "int32_t llama_set_adapter_cvec(\n")

# ---- the models ----------------------------------------------------------------------------
for f in ("src/models/qwen35.cpp", "src/models/qwen35moe.cpp", "src/models/qwen3next.cpp"):
    edit(f, "        cur = build_cvec(cur, il);\n", "        cur = build_cvec(cur, il);\n        cur = build_inject(cur, il);   // steermem\n")

print("inject: applied to %d files" % 8)

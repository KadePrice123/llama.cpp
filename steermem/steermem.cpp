// steermem sidecar: a memory table the model reads by key, on llama.cpp.
//
// Mirrors the trainer's delivery (qwenexp.py: row_states, mem_block, _seed_hits,
// _apply_entry, the key router) rule for rule:
//   cell row      "<cell: KEY | TABLE>\nCONTENT\n</cell>", at most row_len tokens
//   capture       the row through the BASE model (LoRA scale 0), hidden states after
//                 layers 8 and 16 kept per token; l0 = the index after the second
//                 newline token, and the head's "line" state is the mean over [l0, n)
//   key forms     enc(" " + key) and enc(key), each kept if it is 2+ tokens; the
//                 first form's mean input embedding rides on the block's head
//   hits          every occurrence of every key form in the prompt, sorted; the
//                 LAST mem_seq_cells kept (the FIRST when mem-gen is on)
//   block         [head, body_1..body_nb], nb = min(mem_seq, n): head vec = memtag +
//                 seqtag[0] + exptag[egroup] + keyemb, body_j = memtag + seqtag[1+min(j,61)],
//                 every row normalised to emb_norm; fed as an embedding batch right
//                 after the key; injected at layers 8 and 16 with unit(record state)
//                 at scale 0.5 of the block's mean residual norm (build_inject)
//   traces        at text positions 7, 15, 23, ... of the prompt and at every generated
//                 token: the most recent cell whose key occurs in the last 96 text
//                 tokens has its line state injected at that position, same scale
//   mem-gen       after a generated token that completes a key, that cell's block is
//                 fed before the next token is sampled
//   sampling      greedy, stop at end-of-generation or --max-new
//
// Build (inside the fork):
//   g++ -O2 -std=c++17 -I include -I ggml/include -I vendor steermem/steermem.cpp \
//       -L build-cpu/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-cpu/bin \
//       -o build-cpu/bin/steermem-cli
// Run:
//   steermem-cli --model base-q8_0.gguf --lora adapter.gguf --side side.gguf \
//       --table demo.jsonl --prompt "Recall each of these from memory, in order:\n1. KEY\nAnswer:\n"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------- the table
struct cell_t {
    std::string key, text, table;
    std::vector<std::vector<llama_token>> forms;   // key token forms, 2+ tokens each
    // capture (filled on first use)
    bool captured = false;
    std::vector<llama_token> ids;                  // the row's tokens
    int l0 = 0;
    std::map<int, std::vector<float>> st;          // layer -> [n * d]
    std::vector<float> keyemb;                     // [d]
};

static std::vector<cell_t> load_table(const std::string & path) {
    std::vector<cell_t> cells;
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open table %s\n", path.c_str()); exit(2); }
    if (path.size() > 4 && path.substr(path.size() - 4) == ".csv") {
        std::string line;
        std::vector<std::string> header;
        auto split = [](const std::string & s) {   // a small csv row parser: quotes, doubled quotes
            std::vector<std::string> out; std::string cur; bool q = false;
            for (size_t i = 0; i < s.size(); ++i) {
                char c = s[i];
                if (q) { if (c == '"') { if (i + 1 < s.size() && s[i + 1] == '"') { cur += '"'; ++i; } else { q = false; } } else { cur += c; } }
                else if (c == '"') { q = true; } else if (c == ',') { out.push_back(cur); cur.clear(); } else if (c != '\r') { cur += c; }
            }
            out.push_back(cur); return out;
        };
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto row = split(line);
            if (header.empty()) { header = row; continue; }
            cell_t c;
            for (size_t i = 0; i < header.size() && i < row.size(); ++i) {
                if (header[i] == "key") c.key = row[i];
                else if (header[i] == "content" || header[i] == "text") c.text = row[i];
                else if (header[i] == "table") c.table = row[i];
            }
            if (c.table.empty()) c.table = path.substr(path.find_last_of("/\\") + 1);
            if (!c.key.empty() && !c.text.empty()) cells.push_back(c);
        }
    } else {                                        // json (a list) or jsonl (one object per line)
        std::stringstream ss; ss << f.rdbuf();
        std::string all = ss.str();
        auto take = [&](const json & o) {
            cell_t c; c.key = o.value("key", ""); c.text = o.contains("text") ? o.value("text", "") : o.value("content", "");
            c.table = o.value("table", path.substr(path.find_last_of("/\\") + 1));
            if (!c.key.empty() && !c.text.empty()) cells.push_back(c);
        };
        try {
            json j = json::parse(all);
            if (j.is_array()) { for (auto & o : j) take(o); }
            else if (j.contains("cells")) { for (auto & o : j["cells"]) take(o); }
        } catch (...) {
            std::stringstream ls(all); std::string line;
            while (std::getline(ls, line)) { if (line.size() > 2) { try { take(json::parse(line)); } catch (...) {} } }
        }
    }
    return cells;
}

// ---------------------------------------------------------------- the side weights
struct side_t {
    int d = 0;
    float emb_norm = 0.0f;
    std::vector<int> layers;
    std::map<int, float> scale;
    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1;
    std::vector<float> memtag, seqtag, exptag;     // [d], [64*d], [8*d]
};

static side_t load_side(const std::string & path) {
    side_t s;
    ggml_context * gctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ false, &gctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) { fprintf(stderr, "cannot read side gguf %s\n", path.c_str()); exit(2); }
    auto f32 = [&](const char * k, float dflt) { int64_t i = gguf_find_key(g, k); return i < 0 ? dflt : gguf_get_val_f32(g, i); };
    auto u32 = [&](const char * k, int dflt) { int64_t i = gguf_find_key(g, k); return i < 0 ? dflt : (int) gguf_get_val_u32(g, i); };
    s.emb_norm = f32("steermem.emb_norm", 0.0f);
    s.mem_seq = u32("steermem.mem_seq", 128); s.mem_seq_cells = u32("steermem.mem_seq_cells", 4);
    s.mem_gen = u32("steermem.mem_gen", 0); s.memtok = u32("steermem.memtok", 8);
    s.key_window = u32("steermem.key_window", 96); s.egroup = u32("steermem.egroup", 1);
    int64_t li = gguf_find_key(g, "steermem.layers");
    if (li >= 0) {
        const int32_t * arr = (const int32_t *) gguf_get_arr_data(g, li);
        for (size_t i = 0; i < gguf_get_arr_n(g, li); ++i) s.layers.push_back(arr[i]);
    }
    for (int l : s.layers) s.scale[l] = f32(("steermem.scale." + std::to_string(l)).c_str(), 0.5f);
    auto tensor = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(gctx, name);
        if (!t) { fprintf(stderr, "side gguf lacks %s\n", name); exit(2); }
        std::vector<float> v(ggml_nelements(t));
        memcpy(v.data(), t->data, v.size() * sizeof(float));
        return v;
    };
    s.memtag = tensor("memtag"); s.d = (int) s.memtag.size();
    s.seqtag = tensor("seqtag"); s.exptag = tensor("exptag");
    gguf_free(g); ggml_free(gctx);
    return s;
}

// ---------------------------------------------------------------- capture over cb_eval
struct capture_t {
    std::vector<std::string> want;                  // tensor names to keep
    std::map<std::string, std::vector<float>> got;  // name -> data (f32, [ne0 * ne1])
    std::map<std::string, std::pair<int64_t, int64_t>> shape;
};

static bool cb_capture(ggml_tensor * t, bool ask, void * ud) {
    capture_t * c = (capture_t *) ud;
    bool hit = false;
    for (auto & w : c->want) if (strcmp(t->name, w.c_str()) == 0) hit = true;
    if (ask) return hit;
    if (!hit) return true;
    std::vector<float> v(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), v.data(), (int64_t) v.size());
    }
    c->got[t->name] = v;
    c->shape[t->name] = { t->ne[0], t->ne[1] };
    return true;
}

// ---------------------------------------------------------------- the engine
struct engine_t {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;      // the conversation
    llama_context * cap = nullptr;      // captures (adapter off), its own kv
    llama_adapter_lora * lora = nullptr;
    const llama_vocab * vocab = nullptr;
    side_t side;
    capture_t capd;
    std::vector<cell_t> cells;
    int d = 0, n_threads = 6;

    std::vector<llama_token> tok(const std::string & s, bool special = true) {
        std::vector<llama_token> out(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), out.data(), (int32_t) out.size(), false, special);
        if (n < 0) { out.resize(-n); n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), out.data(), (int32_t) out.size(), false, special); }
        out.resize(std::max(n, 0));
        return out;
    }
    std::string piece(llama_token t) {
        char buf[256]; int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : std::string();
    }
    // A KEY ENDS ON A BOUNDARY (16:35). Qwen tokenizes digits one at a time, so the
    // key "James 3:1" is a token-prefix of "James 3:14" and "G2" of "G26"; on a full
    // 45k-cell table a plain subsequence match delivered James 3:1 for a prompt that
    // asked for James 3:14. A hit that ends at position e (exclusive) counts only if
    // the text ends there or the next token's piece starts with a non-alphanumeric
    // character. (The trainer's _seed_hits has the same flaw; it rarely bites there
    // because an item's banks hold 16 rows, not the whole table.)
    bool boundary(const std::vector<llama_token> & ids, size_t e) {
        if (e >= ids.size()) return true;
        std::string p = piece(ids[e]);
        return p.empty() || !std::isalnum((unsigned char) p[0]);
    }
    void set_lora(float scale) {
        if (!lora) return;
        llama_set_adapters_lora(ctx, &lora, 1, &scale);
        llama_set_adapters_lora(cap, &lora, 1, &scale);
    }
    void prep_forms(cell_t & c) {
        c.forms.clear();
        for (auto & f : { tok(" " + c.key, false), tok(c.key, false) }) if (f.size() >= 2) c.forms.push_back(f);
    }
    // the row's states through the base model, once
    void capture(cell_t & c) {
        if (c.captured) return;
        std::string row = "<cell: " + c.key + " | " + c.table + ">\n" + c.text + "\n</cell>";
        c.ids = tok(row, true);
        if (c.ids.size() > 512) c.ids.resize(512);
        const int n = (int) c.ids.size();
        // l0: after the second newline-bearing token
        int nl = 0; c.l0 = 0;
        for (int i = 0; i < n; ++i) { if (piece(c.ids[i]).find('\n') != std::string::npos) { if (++nl == 2) { c.l0 = i + 1; break; } } }
        if (c.l0 >= n) c.l0 = 0;
        float zero = 0.0f;
        llama_set_adapters_lora(cap, &lora, 1, &zero);            // adapter OFF for the capture (model.disable_adapter)
        llama_inject_clear(cap);
        llama_memory_clear(llama_get_memory(cap), true);
        capd.want.clear(); capd.got.clear();
        for (int l : side.layers) capd.want.push_back("l_out-" + std::to_string(l));
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) { b.token[i] = c.ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        b.n_tokens = n;
        if (llama_decode(cap, b) != 0) { fprintf(stderr, "capture decode failed\n"); exit(2); }
        llama_batch_free(b);
        for (int l : side.layers) {
            auto & v = capd.got["l_out-" + std::to_string(l)];
            if ((int64_t) v.size() != (int64_t) n * d) { fprintf(stderr, "capture of l_out-%d has %zu floats, expected %d x %d\n", l, v.size(), n, d); exit(2); }
            c.st[l] = v;                                           // [n * d], token-major
        }
        // the key's mean input embedding (the first key form)
        c.keyemb.assign(d, 0.0f);
        if (!c.forms.empty()) {
            auto & kf = c.forms[0];
            capd.want.assign({ "inp_embd" }); capd.got.clear();
            llama_memory_clear(llama_get_memory(cap), true);
            llama_batch kb = llama_batch_init((int) kf.size(), 0, 1);
            for (int i = 0; i < (int) kf.size(); ++i) { kb.token[i] = kf[i]; kb.pos[i] = i; kb.n_seq_id[i] = 1; kb.seq_id[i][0] = 0; kb.logits[i] = 0; }
            kb.n_tokens = (int) kf.size();
            if (llama_decode(cap, kb) == 0) {
                auto & e = capd.got["inp_embd"];
                const int ne0 = (int) capd.shape["inp_embd"].first;
                for (size_t i = 0; i < kf.size(); ++i) for (int j = 0; j < d && j < ne0; ++j) c.keyemb[j] += e[i * ne0 + j] / (float) kf.size();
            }
            llama_batch_free(kb);
        }
        c.captured = true;
    }
    // one block: rows of vec [1+nb][d] and per-layer unit records [1+nb][d]
    struct block_t { std::vector<float> vec; std::map<int, std::vector<float>> rec; int n = 0; };
    block_t block(cell_t & c) {
        capture(c);
        const int n = (int) c.ids.size();
        const int nb = std::min(side.mem_seq, n);
        block_t b; b.n = 1 + nb;
        b.vec.assign((size_t) b.n * d, 0.0f);
        for (int l : side.layers) b.rec[l].assign((size_t) b.n * d, 0.0f);
        auto row = [&](int k) { return b.vec.data() + (size_t) k * d; };
        // head
        for (int j = 0; j < d; ++j) row(0)[j] = side.memtag[j] + side.seqtag[j] + side.exptag[(size_t) side.egroup * d + j] + c.keyemb[j];
        for (int l : side.layers) {
            float * r = b.rec[l].data();
            const float * st = c.st[l].data();
            const int span = std::max(1, n - c.l0);
            for (int t = c.l0; t < n; ++t) for (int j = 0; j < d; ++j) r[j] += st[(size_t) t * d + j] / (float) span;
        }
        // body
        for (int k = 0; k < nb; ++k) {
            const int pj = nb > 1 ? (int) std::lround((double) k * (n - 1) / (double) (nb - 1)) : n - 1;
            const int code = 1 + std::min(k, 61);
            for (int j = 0; j < d; ++j) row(1 + k)[j] = side.memtag[j] + side.seqtag[(size_t) code * d + j];
            for (int l : side.layers) memcpy(b.rec[l].data() + (size_t) (1 + k) * d, c.st[l].data() + (size_t) pj * d, d * sizeof(float));
        }
        // every vec row to emb_norm; every rec row to unit length (build_inject scales by the residual norm)
        for (int k = 0; k < b.n; ++k) {
            float nv = 0; for (int j = 0; j < d; ++j) nv += row(k)[j] * row(k)[j];
            nv = std::sqrt(nv) + 1e-6f; for (int j = 0; j < d; ++j) row(k)[j] = row(k)[j] / nv * side.emb_norm;
            for (int l : side.layers) {
                float * r = b.rec[l].data() + (size_t) k * d; float nr = 0; for (int j = 0; j < d; ++j) nr += r[j] * r[j];
                nr = std::sqrt(nr); if (nr > 1e-6f) for (int j = 0; j < d; ++j) r[j] /= nr;
            }
        }
        return b;
    }
    // the line state of a cell as a unit vector per layer (the trace)
    std::map<int, std::vector<float>> line_state(cell_t & c) {
        block_t b = block(c);
        std::map<int, std::vector<float>> out;
        for (int l : side.layers) out[l] = std::vector<float>(b.rec[l].begin(), b.rec[l].begin() + d);
        return out;
    }
    // feed helpers -------------------------------------------------------------------
    int kv_pos = 0;
    std::vector<llama_token> text;                  // every text token fed so far (the router's window)
    struct pending_inj { int layer; llama_pos pos; std::vector<float> v; };
    std::vector<pending_inj> inj;                   // injections for the next decode call
    int blocks_fed = 0, traces_fed = 0;

    void flush_inject() {
        llama_inject_clear(ctx);
        std::map<int, std::vector<pending_inj *>> by;
        for (auto & p : inj) by[p.layer].push_back(&p);
        for (auto & kv : by) {
            std::vector<llama_pos> pos; std::vector<float> data;
            for (auto * p : kv.second) { pos.push_back(p->pos); data.insert(data.end(), p->v.begin(), p->v.end()); }
            llama_inject_set(ctx, kv.first, (int32_t) pos.size(), pos.data(), data.data(), side.scale[kv.first]);
        }
        inj.clear();
    }
    // the most recent cell whose key occurs in the last key_window text tokens (inclusive of the end)
    cell_t * route(std::vector<cell_t *> & cands) {
        cell_t * best = nullptr; int best_end = -1;
        const int lo = std::max(0, (int) text.size() - side.key_window);
        for (auto * c : cands) for (auto & f : c->forms) {
            for (int e = (int) text.size(); e - (int) f.size() >= lo; --e) {
                if (std::equal(f.begin(), f.end(), text.begin() + e - f.size()) && boundary(text, (size_t) e)) { if (e > best_end) { best_end = e; best = c; } break; }
            }
        }
        return best;
    }
    void feed_text(const std::vector<llama_token> & seg, std::vector<cell_t *> & cands, bool want_logits, bool trace_last = false) {
        // traces at positions 7, 15, ... (text coordinates) inside this segment
        const int t0 = (int) text.size();
        std::vector<llama_token> tmp = text;
        for (int i = 0; i < (int) seg.size(); ++i) {
            tmp.push_back(seg[i]);
            const int pt = t0 + i;
            // the trainer's memtok points: every K-th text position, and always the last prompt token
            if (side.memtok > 0 && (pt % side.memtok == side.memtok - 1 || (trace_last && i + 1 == (int) seg.size()))) {
                std::swap(text, tmp);                   // route over the text up to and including pt
                cell_t * c = route(cands);
                std::swap(text, tmp);
                if (c) { auto ls = line_state(*c); for (int l : side.layers) inj.push_back({ l, kv_pos + i, ls[l] }); ++traces_fed; }
            }
        }
        flush_inject();
        llama_batch b = llama_batch_init((int) seg.size(), 0, 1);
        for (int i = 0; i < (int) seg.size(); ++i) { b.token[i] = seg[i]; b.pos[i] = kv_pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (want_logits && i + 1 == (int) seg.size()) ? 1 : 0; }
        b.n_tokens = (int) seg.size();
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at kv %d\n", kv_pos); exit(2); }
        llama_batch_free(b);
        llama_inject_clear(ctx);
        text.insert(text.end(), seg.begin(), seg.end());
        kv_pos += (int) seg.size();
    }
    void feed_block(cell_t & c, bool want_logits) {
        block_t b = block(c);
        for (int l : side.layers) for (int k = 0; k < b.n; ++k) inj.push_back({ l, kv_pos + k, std::vector<float>(b.rec[l].begin() + (size_t) k * d, b.rec[l].begin() + (size_t) (k + 1) * d) });
        flush_inject();
        llama_batch bb = llama_batch_init(b.n, d, 1);
        memcpy(bb.embd, b.vec.data(), b.vec.size() * sizeof(float));
        for (int k = 0; k < b.n; ++k) { bb.pos[k] = kv_pos + k; bb.n_seq_id[k] = 1; bb.seq_id[k][0] = 0; bb.logits[k] = (want_logits && k + 1 == b.n) ? 1 : 0; }
        bb.n_tokens = b.n;
        if (llama_decode(ctx, bb) != 0) { fprintf(stderr, "block decode failed at kv %d\n", kv_pos); exit(2); }
        llama_batch_free(bb);
        llama_inject_clear(ctx);
        kv_pos += b.n;
        ++blocks_fed;
    }
    // a key that ends exactly at the last text token
    std::vector<cell_t *> key_end(std::vector<cell_t *> & cands) {
        std::vector<cell_t *> out;
        for (auto * c : cands) for (auto & f : c->forms) {
            if (f.size() <= text.size() && std::equal(f.begin(), f.end(), text.end() - f.size())) { out.push_back(c); break; }
        }
        return out;
    }
};

int main(int argc, char ** argv) {
    std::string model_path, lora_path, side_path, table_path, prompt;
    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096;
    bool no_traces = false, quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "--model") model_path = next(); else if (a == "--lora") lora_path = next(); else if (a == "--side") side_path = next();
        else if (a == "--table") table_path = next(); else if (a == "--prompt") prompt = next(); else if (a == "--max-new") max_new = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str()); else if (a == "--memgen") memgen = atoi(next().c_str());
        else if (a == "--no-traces") no_traces = true; else if (a == "--quiet") quiet = true; else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || side_path.empty() || table_path.empty() || prompt.empty()) { fprintf(stderr, "need --model --side --table --prompt\n"); return 1; }
    // literal \n in the prompt argument
    for (size_t p; (p = prompt.find("\\n")) != std::string::npos;) prompt.replace(p, 2, "\n");

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    engine_t E;
    E.side = load_side(side_path);
    if (memgen >= 0) E.side.mem_gen = memgen;
    if (no_traces) E.side.memtok = 0;
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    E.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!E.model) { fprintf(stderr, "model failed\n"); return 2; }
    E.d = llama_model_n_embd(E.model);
    if (E.d != E.side.d) { fprintf(stderr, "side d %d != model d %d\n", E.side.d, E.d); return 2; }
    if (llama_model_n_embd_inp(E.model) != E.d) { fprintf(stderr, "n_embd_inp %d != n_embd %d: embedding batches need the input width\n", llama_model_n_embd_inp(E.model), E.d); return 2; }
    E.vocab = llama_model_get_vocab(E.model);
    E.n_threads = threads;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_threads = threads; cp.n_threads_batch = threads;
    E.ctx = llama_init_from_model(E.model, cp);
    llama_context_params cc = cp; cc.n_ctx = 1024; cc.cb_eval = cb_capture; cc.cb_eval_user_data = &E.capd;
    E.cap = llama_init_from_model(E.model, cc);
    if (!E.ctx || !E.cap) { fprintf(stderr, "context failed\n"); return 2; }
    if (!lora_path.empty()) {
        E.lora = llama_adapter_lora_init(E.model, lora_path.c_str());
        if (!E.lora) { fprintf(stderr, "lora failed\n"); return 2; }
        float one = 1.0f; llama_set_adapters_lora(E.ctx, &E.lora, 1, &one);
    }
    E.cells = load_table(table_path);
    for (auto & c : E.cells) E.prep_forms(c);
    if (!quiet) fprintf(stderr, "table %zu cells | mem_seq %d cells %d mem_gen %d memtok %d window %d | scale %s\n", E.cells.size(), E.side.mem_seq, E.side.mem_seq_cells, E.side.mem_gen, E.side.memtok, E.side.key_window, E.side.layers.empty() ? "?" : std::to_string(E.side.scale[E.side.layers[0]]).c_str());

    // ---- the prompt: hits, blocks in place ---------------------------------------------
    std::vector<llama_token> seed = E.tok(prompt, true);
    struct hit_t { int end; cell_t * c; };
    std::vector<hit_t> hits;
    for (auto & c : E.cells) for (auto & f : c.forms) {
        for (size_t i = 0; i + f.size() <= seed.size(); ++i) if (std::equal(f.begin(), f.end(), seed.begin() + i) && E.boundary(seed, i + f.size())) hits.push_back({ (int) (i + f.size()), &c });
    }
    std::sort(hits.begin(), hits.end(), [](const hit_t & a, const hit_t & b) { return a.end < b.end; });
    hits.erase(std::unique(hits.begin(), hits.end(), [](const hit_t & a, const hit_t & b) { return a.end == b.end && a.c == b.c; }), hits.end());
    if ((int) hits.size() > E.side.mem_seq_cells) {
        if (E.side.mem_gen) hits.resize(E.side.mem_seq_cells);
        else hits.erase(hits.begin(), hits.end() - E.side.mem_seq_cells);
    }
    std::vector<cell_t *> cands;
    for (auto & h : hits) if (std::find(cands.begin(), cands.end(), h.c) == cands.end()) cands.push_back(h.c);
    if (!quiet) { fprintf(stderr, "prompt %zu tokens | %zu key hits:", seed.size(), hits.size()); for (auto & h : hits) fprintf(stderr, " [%s @%d]", h.c->key.c_str(), h.end); fprintf(stderr, "\n"); }
    int done = 0;
    for (size_t k = 0; k < hits.size(); ++k) {
        const int cut = hits[k].end;
        if (cut > done) { std::vector<llama_token> seg(seed.begin() + done, seed.begin() + cut); E.feed_text(seg, cands, false, cut == (int) seed.size()); done = cut; }
        E.feed_block(*hits[k].c, done == (int) seed.size());
    }
    if (done < (int) seed.size()) { std::vector<llama_token> seg(seed.begin() + done, seed.end()); E.feed_text(seg, cands, true, true); }

    // ---- generation ----------------------------------------------------------------------
    std::vector<cell_t *> all; for (auto & c : E.cells) all.push_back(&c);
    std::string out;
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    for (int step = 0; step < max_new; ++step) {
        const float * lg = llama_get_logits_ith(E.ctx, -1);
        int best = 0; for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        if (llama_vocab_is_eog(E.vocab, best)) break;
        out += E.piece(best);
        if (!quiet) { fputs(E.piece(best).c_str(), stdout); fflush(stdout); }
        // the new token: its trace (the router over the text including it), then mem-gen blocks
        std::vector<llama_token> one = { best };
        std::vector<llama_token> tmp = E.text; tmp.push_back(best);
        std::swap(E.text, tmp);
        cell_t * rc = (E.side.memtok > 0) ? E.route(cands) : nullptr;
        std::vector<cell_t *> fired = E.side.mem_gen ? E.key_end(all) : std::vector<cell_t *>();
        std::swap(E.text, tmp);
        if (rc) { auto ls = E.line_state(*rc); for (int l : E.side.layers) E.inj.push_back({ l, E.kv_pos, ls[l] }); ++E.traces_fed; }
        {   // feed the token itself (feed_text would re-route at memtok points; generated tokens route every step)
            E.flush_inject();
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = best; b.pos[0] = E.kv_pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = fired.empty() ? 1 : 0; b.n_tokens = 1;
            if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 2; }
            llama_batch_free(b); llama_inject_clear(E.ctx);
            E.text.push_back(best); E.kv_pos += 1;
        }
        for (size_t i = 0; i < fired.size(); ++i) {
            if (std::find(cands.begin(), cands.end(), fired[i]) == cands.end()) cands.push_back(fired[i]);
            E.feed_block(*fired[i], i + 1 == fired.size());
        }
    }
    if (!quiet) fputs("\n", stdout);
    json summary = { { "generation", out }, { "prompt_tokens", (int) seed.size() }, { "blocks", E.blocks_fed }, { "traces", E.traces_fed }, { "kv_used", E.kv_pos },
                     { "cells", json::array() } };
    for (auto * c : cands) summary["cells"].push_back({ { "key", c->key }, { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 } });
    printf("%s\n", summary.dump().c_str());
    llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free();
    return 0;
}

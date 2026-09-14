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
#define CPPHTTPLIB_THREAD_POOL_COUNT 4
#include "cpp-httplib/httplib.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <mutex>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------- the table
struct cell_t {
    std::string key, text, table;
    std::string ctx;                               // 0016: text read before the row when its states are captured
    std::string summary;                           // 0018: protocol mode's stored memory text ("KEY is SUMMARY")
    std::vector<std::string> aliases;              // other spellings that name this cell (the core cell: "core memories")
    std::vector<std::vector<llama_token>> forms;   // key token forms, 2+ tokens each
    // capture (filled on first use)
    bool captured = false;
    std::vector<llama_token> ids;                  // the row's tokens
    int l0 = 0, l1 = 0;                            // the line span [l0, l1) pooled for the head and the trace
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
            c.ctx = o.value("context", "");     // 0016: a per-cell capture context
            c.summary = o.value("summary", "");  // 0018: protocol mode's stored memory text
            if (c.text.empty()) c.text = !c.summary.empty() ? c.summary : c.ctx;
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
    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2, mem_once = 0, key_min_chars = 0;
    std::vector<float> memtag, seqtag, exptag, marktag;   // [d], [64*d], [8*d], [2*d]
    int mem_marks = 0;                             // 0014: wrap every block in <|mem_start|> / <|mem_end|>
    int protocol = 0, head = 1, keyemb_on = 1, span_key = 0, step = 0;   // 0018: a twoside.py checkpoint (the <recall> loop)
    std::vector<float> nulltag, neartag;                                  // 0018: [d], [d]
    std::map<int, std::vector<float>> nullrec;                            // 0018: layer -> [d]
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
    s.mem_once = u32("steermem.mem_once", 0);          // 0012: one block per cell per turn (trainer patch 87)
    s.mem_marks = u32("steermem.mem_marks", 0);        // 0014: the block is wrapped in two learned marker positions (trainer patch 91)
    s.key_min_chars = u32("steermem.key_min_chars", 0); // 0013: a one-token key form counts when the name has N+ alphanumerics (trainer patch 88)
    s.key_window = u32("steermem.key_window", 96); s.egroup = u32("steermem.egroup", 1);
    s.line_rule = u32("steermem.line_rule", 2);   // absent: a checkpoint from before trainer patch 79
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
    s.protocol = u32("steermem.protocol", 0);            // 0018
    s.head = u32("steermem.head", 1); s.keyemb_on = u32("steermem.keyemb", 1);
    s.span_key = u32("steermem.span_from_key", 0); s.step = u32("steermem.step", 0);
    s.seqtag = tensor("seqtag");
    if (!s.protocol) {
        s.exptag = tensor("exptag");
    } else {
        s.nulltag = tensor("nulltag"); s.neartag = tensor("neartag");
        for (int l : s.layers) s.nullrec[l] = tensor(("nullrec." + std::to_string(l)).c_str());
        ggml_tensor * mt = ggml_get_tensor(gctx, "marktag");
        if (mt && (int) ggml_nelements(mt) == 2 * s.d) { s.marktag.resize(2 * s.d); memcpy(s.marktag.data(), mt->data, s.marktag.size() * sizeof(float)); }
        else s.mem_marks = 0;
    }
    if (s.mem_marks) {                                  // 0014: the marker vectors, or serve unmarked if the file predates them
        ggml_tensor * mt = ggml_get_tensor(gctx, "marktag");
        if (!mt || (int) ggml_nelements(mt) != 2 * s.d) { fprintf(stderr, "side gguf says mem_marks but lacks marktag [2,d]; serving unmarked\n"); s.mem_marks = 0; }
        else { s.marktag.resize(2 * s.d); memcpy(s.marktag.data(), mt->data, s.marktag.size() * sizeof(float)); }
    }
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
    int ngl = 0;                                   // 0015: layers offloaded to the GPU, for the info line
    std::string cell_ctx;                          // 0016: --cell-context, used for cells that carry none of their own
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
    // a fresh conversation: the kv, the injections, the router's text window
    void reset() {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_inject_clear(ctx);
        inj.clear(); text.clear(); kv_pos = 0; blocks_fed = 0; traces_fed = 0;
    }
    // a table swap: the cells are replaced; captures are recomputed lazily on first use
    void set_table(std::vector<cell_t> cells_) {
        cells = std::move(cells_);
        if (side.protocol) return;                     // 0018: protocol mode looks keys up by the tag the model writes: no core/index cells, no key forms
        add_core_cell(cells);
        for (auto & c : cells) prep_forms(c);
    }
    void set_lora(float scale) {
        if (!lora) return;
        llama_set_adapters_lora(ctx, &lora, 1, &scale);
        llama_set_adapters_lora(cap, &lora, 1, &scale);
    }
    void prep_forms(cell_t & c) {
        c.forms.clear();
        std::vector<std::string> names = { c.key };
        names.insert(names.end(), c.aliases.begin(), c.aliases.end());
        for (auto & nm : names) {
            int an = 0; for (unsigned char ch : nm) if (std::isalnum(ch)) ++an;     // 0013: one-token forms of a name with key_min_chars+ alphanumerics count
            for (auto & f : { tok(" " + nm, false), tok(nm, false) })
                if (f.size() >= 2 || (f.size() == 1 && side.key_min_chars > 0 && an >= side.key_min_chars)) c.forms.push_back(f);
        }
    }
    // THE CORE CELL (19:20): what the trainer's items always carry -- "Core memory. Memory tables
    // available: A; B; C." under the key "core memory" (and "core memories", the phrase the think
    // uses) -- added to every loaded table that does not bring its own.
    // THE INDEX CELLS (19:30): one per table, keyed by the table's name, listing its entries with a
    // blurb each -- what the training items carry, so the model can map a loose question to a key by
    // reading descriptions. A table that brings its own row under its name keeps it.
    static std::string blurb(const std::string & text) {
        std::string t = text; for (auto & ch : t) if (ch == '\n') ch = ' ';
        size_t n = 0, i = 0; while (i < t.size() && n < 8) { if (t[i] == ' ') ++n; ++i; }
        std::string b = t.substr(0, i); while (!b.empty() && (b.back() == ' ' || b.back() == ',' || b.back() == ';')) b.pop_back();
        return b + (i < t.size() ? "..." : "");
    }
    static void add_index_cells(std::vector<cell_t> & cells) {
        std::vector<std::string> names;
        for (auto & c : cells) if (c.key != "core memory" && std::find(names.begin(), names.end(), c.table) == names.end()) names.push_back(c.table);
        std::vector<cell_t> added;
        for (auto & nm : names) {
            bool has = false; for (auto & c : cells) if (c.key == nm) { has = true; break; }
            if (has) continue;
            std::vector<cell_t *> ents; for (auto & c : cells) if (c.table == nm && c.key != "core memory") ents.push_back(&c);
            const size_t show = ents.size() > 60 ? 40 : ents.size();
            std::string list; for (size_t i = 0; i < show; ++i) list += (i ? "; " : "") + ents[i]->key + " (" + blurb(ents[i]->text) + ")";
            if (show < ents.size()) list += "; ... and " + std::to_string(ents.size() - show) + " more";
            cell_t ix; ix.key = nm; ix.table = "index";
            ix.text = "Index of " + nm + ": " + std::to_string(ents.size()) + " entries. Keyed by the entry name. Entries: " + list + ".";
            added.push_back(ix);
        }
        cells.insert(cells.begin(), added.begin(), added.end());
    }
    static void add_core_cell(std::vector<cell_t> & cells) {
        add_index_cells(cells);
        for (auto & c : cells) if (c.key == "core memory") return;
        std::vector<std::string> names;
        for (auto & c : cells) if (c.table != "index" && std::find(names.begin(), names.end(), c.table) == names.end()) names.push_back(c.table);
        cell_t core; core.key = "core memory"; core.table = "index"; core.aliases = { "core memories" };
        std::string list;
        for (size_t i = 0; i < names.size(); ++i) {
            size_t n = 0; for (auto & c : cells) if (c.table == names[i]) ++n;
            list += (i ? "; " : "") + names[i] + " (" + std::to_string(n) + " entries)";
        }
        core.text = "Core memory. Memory tables available: " + list + ".";
        cells.insert(cells.begin(), core);
    }
    // the row's states through the base model, once
    void capture(cell_t & c) {
        if (c.captured) return;
        std::string row = "<cell: " + c.key + " | " + c.table + ">\n" + c.text + "\n</cell>";
        const std::string & ctx = c.ctx.empty() ? cell_ctx : c.ctx;      // 0016: read before the row, kept out of the block
        c.ids = tok(row, true);                                          // 0016: tok()'s flag is parse-special, not add-bos -- the row must tokenise identically with and without a context, or the A/B is not the same block
        if (c.ids.size() > 512) c.ids.resize(512);
        const int n = (int) c.ids.size();
        std::vector<llama_token> pre;
        if (!ctx.empty()) pre = tok(ctx + "\n", true);
        const int np = (int) pre.size();
        // the line span [l0, l1). Rule 2 (the trainer today): after the second newline-bearing token to the
        // end -- for a one-line cell that is "</cell>" alone. Rule 1: from after the header's newline through
        // the closing tag's newline-bearing token, i.e. the content.
        std::vector<int> nls;
        for (int i = 0; i < n; ++i) if (piece(c.ids[i]).find('\n') != std::string::npos) nls.push_back(i);
        c.l0 = 0; c.l1 = n;
        if (line_rule == 1) {
            if (!nls.empty()) c.l0 = nls[0] + 1;
            if (nls.size() >= 2 && nls.back() + 1 > c.l0) c.l1 = nls.back() + 1;
        } else if (nls.size() >= 2) {
            c.l0 = nls[1] + 1;
        }
        if (c.l0 >= n) { c.l0 = 0; c.l1 = n; }
        float zero = 0.0f;
        llama_set_adapters_lora(cap, &lora, 1, &zero);            // adapter OFF for the capture (model.disable_adapter)
        llama_inject_clear(cap);
        llama_memory_clear(llama_get_memory(cap), true);
        capd.want.clear(); capd.got.clear();
        for (int l : side.layers) capd.want.push_back("l_out-" + std::to_string(l));
        llama_batch b = llama_batch_init(np + n, 0, 1);
        for (int i = 0; i < np; ++i) { b.token[i] = pre[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        for (int i = 0; i < n; ++i) { b.token[np + i] = c.ids[i]; b.pos[np + i] = np + i; b.n_seq_id[np + i] = 1; b.seq_id[np + i][0] = 0; b.logits[np + i] = 0; }
        b.n_tokens = np + n;
        if (llama_decode(cap, b) != 0) { fprintf(stderr, "capture decode failed\n"); exit(2); }
        llama_batch_free(b);
        for (int l : side.layers) {
            auto & v = capd.got["l_out-" + std::to_string(l)];
            if ((int64_t) v.size() != (int64_t) (np + n) * d) { fprintf(stderr, "capture of l_out-%d has %zu floats, expected %d x %d\n", l, v.size(), np + n, d); exit(2); }
            c.st[l].assign(v.begin() + (size_t) np * d, v.end());  // 0016: the ROW's positions only; the context is read, not delivered
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
        const int mk = side.mem_marks ? 1 : 0;          // 0014: a marker row before the head and after the last body
        block_t b; b.n = 1 + nb + 2 * mk;
        b.vec.assign((size_t) b.n * d, 0.0f);
        for (int l : side.layers) b.rec[l].assign((size_t) b.n * d, 0.0f);
        auto row = [&](int k) { return b.vec.data() + (size_t) k * d; };
        // <|mem_start|> / head / ... / <|mem_end|>
        if (mk) for (int j = 0; j < d; ++j) row(0)[j] = side.marktag[j];
        for (int j = 0; j < d; ++j) row(mk)[j] = side.memtag[j] + side.seqtag[j] + side.exptag[(size_t) side.egroup * d + j] + c.keyemb[j];
        for (int l : side.layers) {
            float * r = b.rec[l].data() + (size_t) mk * d;
            const float * st = c.st[l].data();
            const int e1 = std::max(c.l0 + 1, std::min(n, c.l1));
            const int span = e1 - c.l0;
            for (int t = c.l0; t < e1; ++t) for (int j = 0; j < d; ++j) r[j] += st[(size_t) t * d + j] / (float) span;
        }
        // body
        for (int k = 0; k < nb; ++k) {
            const int pj = nb > 1 ? (int) std::lround((double) k * (n - 1) / (double) (nb - 1)) : n - 1;
            const int code = 1 + std::min(k, 61);
            for (int j = 0; j < d; ++j) row(mk + 1 + k)[j] = side.memtag[j] + side.seqtag[(size_t) code * d + j];
            for (int l : side.layers) memcpy(b.rec[l].data() + (size_t) (mk + 1 + k) * d, c.st[l].data() + (size_t) pj * d, d * sizeof(float));
        }
        if (mk) for (int j = 0; j < d; ++j) row(b.n - 1)[j] = side.marktag[d + j];   // <|mem_end|>
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
    int line_rule = 2;                              // 2: after the second newline to the end (the trainer today); 1: the content between the header and the closing tag
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


#include "steermem_protocol.hpp"                  // 0018: protocol mode

// ---------------------------------------------------------------- one prompt, start to finish
struct run_result_t { json summary; json start; };

// The prompt's key hits, the blocks in place, the greedy generation with traces and mem-gen.
// on_tok is called with every generated piece and returns false to stop (the /stop button);
// on_start receives the "start" event (the cells, the blocks after the prompt's keys) before the
// first token. This is the CLI's flow of 16:30, verified against the trainer on identical cells.
static run_result_t run_prompt(engine_t & E, const std::string & prompt, int max_new, bool quiet,
                               const std::function<bool(const std::string &)> & on_tok,
                               const std::function<void(const json &)> & on_start) {
    const auto t0 = std::chrono::steady_clock::now();
    E.reset();
    std::vector<llama_token> seed = E.tok(prompt, true);
    struct hit_t { int end; cell_t * c; };
    std::vector<hit_t> hits;
    for (auto & c : E.cells) for (auto & f : c.forms) {
        for (int i = 0; i + (int) f.size() <= (int) seed.size(); ++i)
            if (std::equal(f.begin(), f.end(), seed.begin() + i) && E.boundary(seed, (size_t) (i + f.size()))) hits.push_back({ i + (int) f.size(), &c });
    }
    std::sort(hits.begin(), hits.end(), [](const hit_t & a, const hit_t & b) { return a.end < b.end; });
    hits.erase(std::unique(hits.begin(), hits.end(), [](const hit_t & a, const hit_t & b) { return a.end == b.end && a.c == b.c; }), hits.end());
    if (E.side.mem_once) {
        // ONE BLOCK PER CELL PER TURN (0012; Kade 2026-09-12 20:30). Turns are the <|im_start|> segments of the
        // prompt: the first mention of a cell in a turn delivers its block, later mentions in that turn deliver
        // nothing, at most mem_seq_cells cells per turn. The trainer's _seed_hits (patch 87) applies the same rule.
        const std::vector<llama_token> im = E.tok("<|im_start|>", true);
        const llama_token im_start = im.size() == 1 ? im[0] : -1;
        std::vector<int> starts; for (int i = 0; i < (int) seed.size(); ++i) if (im_start >= 0 && seed[i] == im_start) starts.push_back(i);
        std::vector<hit_t> kept; std::set<std::pair<int, cell_t *>> seen; std::map<int, int> per;
        for (auto & h : hits) {
            int seg = 0; for (int st : starts) if (st < h.end) ++seg;
            if (seen.count({ seg, h.c })) continue;
            if (E.side.mem_seq_cells > 0 && per[seg] >= E.side.mem_seq_cells) continue;
            seen.insert({ seg, h.c }); ++per[seg]; kept.push_back(h);
        }
        hits.swap(kept);
    } else if ((int) hits.size() > E.side.mem_seq_cells) {
        if (E.side.mem_gen) hits.resize(E.side.mem_seq_cells);
        else hits.erase(hits.begin(), hits.end() - E.side.mem_seq_cells);
    }
    std::vector<cell_t *> cands;
    for (auto & h : hits) if (std::find(cands.begin(), cands.end(), h.c) == cands.end()) cands.push_back(h.c);
    if (!quiet) { fprintf(stderr, "prompt %zu tokens | %zu key hits:", seed.size(), hits.size()); for (auto & h : hits) fprintf(stderr, " [%s @%d]", h.c->key.c_str(), h.end); fprintf(stderr, "\n"); }
    json start = { { "start", true }, { "seed_tokens", (int) seed.size() }, { "n_exact", (int) hits.size() }, { "n_expanded", 0 }, { "n_fuzzy", 0 },
                   { "cells", json::array() }, { "seed_blocks", json::array() } };
    for (auto * c : cands) start["cells"].push_back({ { "key", c->key }, { "table", c->table }, { "chars", (int) c->text.size() } });
    for (auto & h : hits) start["seed_blocks"].push_back({ { "after_token", h.end - 1 }, { "key", h.c->key } });
    if (on_start) on_start(start);
    int done = 0, mem_positions = 0;
    for (size_t k = 0; k < hits.size(); ++k) {
        const int cut = hits[k].end;
        if (cut > done) { std::vector<llama_token> seg(seed.begin() + done, seed.begin() + cut); E.feed_text(seg, cands, false, cut == (int) seed.size()); done = cut; }
        E.feed_block(*hits[k].c, done == (int) seed.size());
        mem_positions += 1 + std::min(E.side.mem_seq, (int) hits[k].c->ids.size());
    }
    if (done < (int) seed.size()) { std::vector<llama_token> seg(seed.begin() + done, seed.end()); E.feed_text(seg, cands, true, true); }

    // ---- generation ----------------------------------------------------------------------
    std::vector<cell_t *> all; for (auto & c : E.cells) all.push_back(&c);
    std::string out;
    json fired_log = json::array();
    int blocks_fired = 0, n_tok = 0;
    std::set<cell_t *> seen_turn;                  // 0012: the cells delivered in this turn of the model's own
    bool stopped = false;
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    for (int step = 0; step < max_new; ++step) {
        const float * lg = llama_get_logits_ith(E.ctx, -1);
        int best = 0; for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        if (llama_vocab_is_eog(E.vocab, best)) break;
        const std::string pc = E.piece(best);
        out += pc; ++n_tok;
        if (!quiet) { fputs(pc.c_str(), stdout); fflush(stdout); }
        if (on_tok && !on_tok(pc)) { stopped = true; break; }
        // mem-gen: a key that ends at the text so far is complete only if the token about to be fed
        // starts a boundary (Qwen writes "G26" as G, 2, 6 -- "G2" must not fire inside it). Its block
        // goes in BEFORE that token: key, block, next token, the training layout.
        std::vector<cell_t *> fired;
        if (E.side.mem_gen && (pc.empty() || !std::isalnum((unsigned char) pc[0]))) fired = E.key_end(all);
        if (blocks_fired >= E.side.mem_seq_cells * 8) fired.clear();   // the trainer's cap on blocks fired in one generation (MEMGEN_N < mem_seq_cells * 8)
        if (E.side.mem_once && !fired.empty()) {   // 0012: a cell already delivered in this turn fires nothing more; at most mem_seq_cells cells
            std::vector<cell_t *> fresh;
            for (auto * c : fired) if (!seen_turn.count(c) && (E.side.mem_seq_cells <= 0 || (int) seen_turn.size() < E.side.mem_seq_cells)) { seen_turn.insert(c); fresh.push_back(c); }
            fired.swap(fresh);
        }
        if (!fired.empty()) {
            json keys = json::array(); for (auto * c : fired) keys.push_back(c->key);
            fired_log.push_back({ { "at_text_token", (int) E.text.size() }, { "keys", keys } });
            for (auto * c : fired) {
                if (std::find(cands.begin(), cands.end(), c) == cands.end()) cands.push_back(c);
                E.feed_block(*c, false);
                ++blocks_fired;
            }
        }
        std::vector<llama_token> tmp = E.text; tmp.push_back(best);
        std::swap(E.text, tmp);
        cell_t * rc = (E.side.memtok > 0) ? E.route(cands) : nullptr;
        std::swap(E.text, tmp);
        if (rc) { auto ls = E.line_state(*rc); for (int l : E.side.layers) E.inj.push_back({ l, E.kv_pos, ls[l] }); ++E.traces_fed; }
        {   // feed the token itself (feed_text would re-route at memtok points; generated tokens route every step)
            E.flush_inject();
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = best; b.pos[0] = E.kv_pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); break; }
            llama_batch_free(b); llama_inject_clear(E.ctx);
            E.text.push_back(best); E.kv_pos += 1;
        }
        if (E.kv_pos + 4 >= (int) llama_n_ctx(E.ctx)) { stopped = true; break; }   // the context is full
    }
    if (!quiet) fputs("\n", stdout);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    json summary = { { "done", true }, { "text", out }, { "generation", out }, { "tokens", n_tok }, { "seconds", secs }, { "stopped", stopped },
                     { "prompt_tokens", (int) seed.size() }, { "seed_tokens", (int) seed.size() }, { "blocks", E.blocks_fed }, { "traces", E.traces_fed },
                     { "kv_used", E.kv_pos }, { "mem_positions", mem_positions }, { "blocks_fired", blocks_fired }, { "fired", fired_log },
                     { "seed_blocks", start["seed_blocks"] }, { "n_exact", (int) hits.size() }, { "n_expanded", 0 }, { "n_fuzzy", 0 },
                     { "mem_gen", E.side.mem_gen }, { "mem_seq", E.side.mem_seq }, { "cells", json::array() } };
    for (auto * c : cands) summary["cells"].push_back({ { "key", c->key }, { "table", c->table }, { "chars", (int) c->text.size() }, { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 }, { "l1", c->l1 } });
    return { summary, start };
}

// Qwen's chat template as the reasoning-format items were built (RUNBOOK 17:15): every turn under
// <|im_start|>ROLE ... <|im_end|>, earlier assistant turns without their reasoning, and the
// generation prompt ending in the opening think tag -- the model emits the reasoning and the
// closing tag itself.
static std::string chat_prompt(const json & messages) {
    std::string p;
    for (const auto & m : messages) {
        std::string role = m.value("role", "user"), content = m.value("content", "");
        if (role == "assistant") {
            size_t a = content.find("<think>");
            size_t b = content.find("</think>");
            if (b != std::string::npos) { size_t e = b + 8; while (e < content.size() && content[e] == '\n') ++e; content = (a != std::string::npos && a < b ? content.substr(0, a) : std::string()) + content.substr(e); }
        }
        p += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    p += "<|im_start|>assistant\n<think>\n";
    return p;
}

static const char * UI_HTML = R"HTML(<!doctype html><html><head><meta charset="utf-8"><title>steermem</title>
<style>body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}header{background:#1f2a44;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header select,header input,header button{font:inherit;padding:4px 8px}main{max-width:900px;margin:0 auto;padding:16px}.msg{background:#fff;border-radius:8px;padding:10px 14px;margin:8px 0;box-shadow:0 1px 2px #0002;white-space:pre-wrap}
.user{background:#e8f0fe}.think{color:#555;font-size:.92em;border-left:3px solid #c9d1e0;padding-left:8px;margin:6px 0}.meta{color:#666;font-size:.85em}
#row{display:flex;gap:8px;margin-top:12px}#q{flex:1;font:inherit;padding:8px}button{font:inherit;padding:8px 12px}.mem{color:#7a4b00;font-size:.85em}</style></head><body>
<header><b>steermem</b> <span id="info" class="meta"></span>
<label>preset <select id="preset"></select></label><button onclick="loadPreset()">load</button>
<label>table <input type="file" id="file" accept=".csv,.json,.jsonl"></label>
<label><input type="checkbox" id="showmem" checked> show MEM</label>
<label>mem-gen <input type="checkbox" id="memgen" checked></label>
<label>max new <input id="maxnew" type="number" value="400" style="width:70px"></label>
<button onclick="clearChat()">new chat</button></header>
<main><div id="log"></div><div id="row"><input id="q" placeholder="ask; keys in the table are read from memory as the model writes them"><button id="send" onclick="send()">send</button><button onclick="stop()">stop</button></div></main>
<script>
let messages=[];const log=document.getElementById('log');
async function info(){try{const r=await (await fetch('/')).json();document.getElementById('info').textContent=`${r.cells} cells | mem_seq ${r.mem_seq} | ${r.model}`;
const s=document.getElementById('preset');s.innerHTML='';(r.presets||[]).forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;s.appendChild(o);});}catch(e){}}
async function loadPreset(){const p=document.getElementById('preset').value;const r=await (await fetch('/preset',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:p})})).json();note(`loaded preset ${p}: ${r.loaded} cells`);info();}
document.getElementById('file').addEventListener('change',async ev=>{const f=ev.target.files[0];if(!f)return;const txt=await f.text();let body;
if(f.name.endsWith('.csv'))body={name:f.name,csv:txt};else{let cells=[];try{const j=JSON.parse(txt);cells=Array.isArray(j)?j:(j.cells||[]);}catch(e){txt.split('\n').forEach(l=>{l=l.trim();if(l){try{cells.push(JSON.parse(l));}catch(e){}}});}body={name:f.name,cells:cells};}
const r=await (await fetch('/table',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)})).json();note(`loaded ${f.name}: ${r.loaded} cells`);info();});
function note(t){const d=document.createElement('div');d.className='meta';d.textContent=t;log.appendChild(d);}
function add(cls,html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d;}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(box,text,r){const i=text.indexOf('</think>');let think=i>=0?text.slice(0,i):'',ans=i>=0?text.slice(i+8):text;think=think.replace('<think>','').trim();
let h=think?`<details open><summary class="meta">reasoning</summary><div class="think">${esc(think)}</div></details>`:'';h+=`<div>${esc(ans.trim()||(i>=0?'(no answer after the reasoning)':''))}</div>`;
if(r&&document.getElementById('showmem').checked){const sb=(r.seed_blocks||[]).map(b=>'⟦'+b.key+'⟧').join(', ');const fi=(r.fired||[]).map(f=>`token ${f.at_text_token} → ${f.keys.join(', ')}`).join('; ');
h+=`<div class="mem">${r.mem_positions||0} MEM positions after the prompt's keys${sb?' ('+sb+')':''} | ${r.blocks_fired||0} blocks fired by written keys${fi?' ('+fi+')':''} | ${r.tokens} tokens in ${(r.seconds||0).toFixed(0)}s${r.stopped?' | STOPPED':''}</div>`;}
box.innerHTML=h;}
async function send(){const q=document.getElementById('q').value.trim();if(!q)return;document.getElementById('q').value='';messages.push({role:'user',content:q});add('user',esc(q));
const box=add('bot','…');let text='',r=null;document.getElementById('send').disabled=true;
try{const resp=await fetch('/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({messages,stream:true,memgen:document.getElementById('memgen').checked?1:0,max_new:+document.getElementById('maxnew').value})});
const rd=resp.body.getReader();const dec=new TextDecoder();let buf='';
while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let n;while((n=buf.indexOf('\n'))>=0){const line=buf.slice(0,n);buf=buf.slice(n+1);if(!line.trim())continue;const ev=JSON.parse(line);
if(ev.start){note(`${ev.n_exact} keys in the prompt${ev.seed_blocks.length?': '+ev.seed_blocks.map(b=>'⟦'+b.key+'⟧').join(', '):''}`);}else if(ev.tok!==undefined){text+=ev.tok;render(box,text,null);}else if(ev.done){r=ev;text=ev.text||text;}}}}
catch(e){box.textContent='error: '+e;}
render(box,text,r);messages.push({role:'assistant',content:text});document.getElementById('send').disabled=false;}
function stop(){fetch('/stop',{method:'POST'});}
function clearChat(){messages=[];log.innerHTML='';}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});info();
</script></body></html>)HTML";

static std::vector<std::string> list_presets(const std::string & dir) {
    std::vector<std::string> out;
    if (dir.empty() || !std::filesystem::is_directory(dir)) return out;
    for (auto & e : std::filesystem::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        if (e.is_regular_file() && (ext == ".jsonl" || ext == ".json" || ext == ".csv")) out.push_back(e.path().filename().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static int serve(engine_t & E, const std::string & host, int port, const std::string & tables_dir, const std::string & model_name, bool quiet) {
    httplib::Server srv;
    std::mutex mu;
    std::atomic<bool> stop_flag{ false };
    auto send_json = [](httplib::Response & res, const json & j, int status = 200) { res.status = status; res.set_content(j.dump(), "application/json"); };
    if (E.side.protocol) register_protocol_routes(srv, E, mu, stop_flag, model_name);   // 0018: registered first, so "/" is the protocol page
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        json j = { { "presets", list_presets(tables_dir) }, { "cells", (int) E.cells.size() }, { "mem_seq", E.side.mem_seq }, { "mem_gen", E.side.mem_gen },
                   { "mem_seq_cells", E.side.mem_seq_cells }, { "mem_once", E.side.mem_once }, { "mem_marks", E.side.mem_marks }, { "ngl", E.ngl }, { "cell_context", (int) E.cell_ctx.size() }, { "key_min_chars", E.side.key_min_chars }, { "memtok", E.side.memtok }, { "line_rule", E.line_rule }, { "model", model_name }, { "engine", "steermem-cli (llama.cpp fork)" } };
        send_json(res, j);
    });
    srv.Get("/ui", [&](const httplib::Request &, httplib::Response & res) { res.set_content(UI_HTML, "text/html; charset=utf-8"); });
    srv.Post("/stop", [&](const httplib::Request &, httplib::Response & res) { stop_flag = true; send_json(res, { { "ok", true } }); });
    srv.Post("/preset", [&](const httplib::Request & req, httplib::Response & res) {
        std::string name; try { name = json::parse(req.body).value("name", ""); } catch (...) {}
        std::string path = tables_dir + "/" + name;
        if (name.empty() || name.find("..") != std::string::npos || !std::filesystem::is_regular_file(path)) { send_json(res, { { "error", "no such preset" } }, 404); return; }
        std::lock_guard<std::mutex> lk(mu);
        E.set_table(load_table(path));
        send_json(res, { { "loaded", (int) E.cells.size() }, { "name", name } });
    });
    srv.Post("/cellvec", [&](const httplib::Request & req, httplib::Response & res) {   // 0017
        std::string key; try { key = json::parse(req.body).value("key", ""); } catch (...) {}
        std::lock_guard<std::mutex> lk(mu);
        cell_t * c = nullptr;
        for (auto & x : E.cells) if (x.key == key) { c = &x; break; }
        if (!c) { send_json(res, { { "error", "no such cell" } }, 404); return; }
        auto b = E.block(*c);                              // exactly what a delivered block carries
        const int d = E.d, mk = E.side.mem_marks ? 1 : 0;
        json out = { { "key", c->key }, { "table", c->table }, { "context_chars", (int) (c->ctx.empty() ? E.cell_ctx.size() : c->ctx.size()) },
                     { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 }, { "l1", c->l1 }, { "block_rows", b.n },
                     { "layers", json::object() } };
        const int nb = b.n - 1 - 2 * mk;                    // body rows
        std::vector<std::pair<std::string, int>> want = { { "head", mk }, { "body0", mk + 1 },
                                                          { "bodymid", mk + 1 + nb / 2 }, { "bodylast", mk + nb } };
        for (int l : E.side.layers) {
            json rows = json::object();
            for (auto & w : want) {
                if (w.second < 0 || w.second >= b.n) continue;
                const float * r = b.rec[l].data() + (size_t) w.second * d;
                rows[w.first] = std::vector<float>(r, r + d);
            }
            out["layers"][std::to_string(l)] = rows;
        }
        send_json(res, out);
    });
    srv.Post("/table", [&](const httplib::Request & req, httplib::Response & res) {
        json body; try { body = json::parse(req.body); } catch (...) { send_json(res, { { "error", "bad json" } }, 400); return; }
        std::string name = body.value("name", "upload");
        std::vector<cell_t> cells;
        if (body.contains("csv")) {
            std::string tmp = std::filesystem::temp_directory_path().string() + "/steermem_upload.csv";
            std::ofstream f(tmp); f << body["csv"].get<std::string>(); f.close();
            cells = load_table(tmp);
            for (auto & c : cells) if (c.table == "steermem_upload.csv") c.table = name;
        } else if (body.contains("cells")) {
            for (auto & o : body["cells"]) {
                cell_t c; c.key = o.value("key", ""); c.text = o.contains("text") ? o.value("text", "") : o.value("content", ""); c.table = o.value("table", name); c.ctx = o.value("context", "");   // 0016
                if (!c.key.empty() && !c.text.empty()) cells.push_back(c);
            }
        }
        std::lock_guard<std::mutex> lk(mu);
        E.set_table(std::move(cells));
        send_json(res, { { "loaded", (int) E.cells.size() }, { "name", name } });
    });
    srv.Post("/chat", [&](const httplib::Request & req, httplib::Response & res) {
        json body; try { body = json::parse(req.body); } catch (...) { send_json(res, { { "error", "bad json" } }, 400); return; }
        std::string prompt;
        if (body.value("raw", false)) prompt = body["messages"].empty() ? std::string() : body["messages"][0].value("content", "");
        else if (body.contains("prompt")) prompt = body.value("prompt", "");
        else prompt = chat_prompt(body.value("messages", json::array()));
        if (prompt.empty()) { send_json(res, { { "error", "no messages" } }, 400); return; }
        const int max_new = body.value("max_new", 400);
        const int memgen = body.value("memgen", -1);
        const int memseq = body.value("memseq", 0);
        const bool stream = body.value("stream", false);
        auto apply = [&]() {   // per-request switches (the checkpoint's values are the defaults)
            if (memgen >= 0) E.side.mem_gen = memgen;
            if (memseq > 0) E.side.mem_seq = memseq;
        };
        if (!stream) {
            std::lock_guard<std::mutex> lk(mu);
            const int mg = E.side.mem_gen, ms = E.side.mem_seq; apply(); stop_flag = false;
            auto r = run_prompt(E, prompt, max_new, true, [&](const std::string &) { return !stop_flag.load(); }, nullptr);
            E.side.mem_gen = mg; E.side.mem_seq = ms;
            send_json(res, r.summary);
            return;
        }
        // the provider runs after this handler has returned: capture nothing of the handler's frame
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, prompt, max_new, memgen, memseq](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            const int mg = E.side.mem_gen, ms = E.side.mem_seq;
            if (memgen >= 0) E.side.mem_gen = memgen;
            if (memseq > 0) E.side.mem_seq = memseq;
            stop_flag = false;
            auto line = [&](const json & j) { std::string s = j.dump() + "\n"; return sink.write(s.data(), s.size()); };
            auto r = run_prompt(E, prompt, max_new, true,
                                [&](const std::string & pc) { return line({ { "tok", pc } }) && !stop_flag.load(); },
                                [&](const json & st) { line(st); });
            E.side.mem_gen = mg; E.side.mem_seq = ms;
            line(r.summary);
            sink.done();
            return true;
        });
    });
    fprintf(stderr, "steermem server on http://%s:%d/ui | %zu cells | presets in %s\n", host.c_str(), port, E.cells.size(), tables_dir.empty() ? "(none)" : tables_dir.c_str());
    if (!srv.listen(host, port)) { fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 3; }
    return 0;
}

int main(int argc, char ** argv) {
    std::string model_path, lora_path, side_path, table_path, prompt, tables_dir, host = "127.0.0.1";
    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1, key_min_chars = -1, mem_marks = -1, ngl = 0;   // line_rule 0 / mem_once -1 / key_min_chars -1: follow the side file
    std::string cell_ctx;                          // 0016: read before every row at capture time
    std::string tool_docs, lookup_path, examples_path, ui_path, memory_out, remember_mode = "store", system_prompt;   // 0018
    std::vector<std::string> learn;                                                                                  // 0018
    bool chat_repl = false, sleep_at_start = false, sleep_only = false;                                              // 0018
    std::string menu_cache;                                                                                          // 0020
    bool no_traces = false, quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "--model") model_path = next(); else if (a == "--lora") lora_path = next(); else if (a == "--side") side_path = next();
        else if (a == "--table") table_path = next(); else if (a == "--prompt") prompt = next(); else if (a == "--max-new") max_new = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str()); else if (a == "--memgen") memgen = atoi(next().c_str());
        else if (a == "--no-traces") no_traces = true; else if (a == "--quiet") quiet = true; else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--line-rule") line_rule = atoi(next().c_str());
        else if (a == "--mem-once") mem_once = atoi(next().c_str());
        else if (a == "--key-min-chars") key_min_chars = atoi(next().c_str());
        else if (a == "--mem-marks") mem_marks = atoi(next().c_str());
        else if (a == "-ngl" || a == "--n-gpu-layers") ngl = atoi(next().c_str());
        else if (a == "--cell-context") cell_ctx = next();
        else if (a == "--tool-docs") tool_docs = next(); else if (a == "--lookup") lookup_path = next();          // 0018
        else if (a == "--examples") examples_path = next(); else if (a == "--ui") ui_path = next();
        else if (a == "--memory-out") memory_out = next(); else if (a == "--learn") learn.push_back(next());
        else if (a == "--remember-mode") remember_mode = next(); else if (a == "--files-dir") PROTO.files_dir = next();
        else if (a == "--system") system_prompt = next();
        else if (a == "--auto-menu") MENU.k = atoi(next().c_str()); else if (a == "--menu-layer") MENU.layer = atoi(next().c_str());   // 0019
        else if (a == "--menu-cache") menu_cache = next();                                                           // 0020
        else if (a == "--auto-inject") INJECT.rows = atoi(next().c_str()); else if (a == "--inject-max") INJECT.max = atoi(next().c_str());   // 0021
        else if (a == "--inject-surprise") INJECT.surprise = atof(next().c_str()); else if (a == "--inject-min-score") INJECT.min_score = atof(next().c_str());
        else if (a == "--inject-match") { INJECT.match = next(); if (INJECT.match != "key" && INJECT.match != "state") { fprintf(stderr, "--inject-match takes key or state\n"); return 1; } }
        else if (a == "--chat") chat_repl = true; else if (a == "--sleep-at-start") sleep_at_start = true; else if (a == "--sleep-only") sleep_only = true;
        else if (a == "--serve") port = atoi(next().c_str()); else if (a == "--tables") tables_dir = next(); else if (a == "--host") host = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || side_path.empty() || (port == 0 && !chat_repl && !sleep_only && (table_path.empty() || prompt.empty()))) {
        fprintf(stderr, "usage: steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --table T --prompt P [--max-new N] [--memgen 0|1] [--line-rule 1|2] [--no-traces]\n"
                        "       steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --serve PORT [--tables DIR] [--table T] [--host 0.0.0.0]   (then open http://127.0.0.1:PORT/ui)\n"
                        "  protocol mode (a side file from twoside_export.py):\n"
                        "       steermem-cli --model BASE.gguf --lora ADAPTER.gguf --side SIDE.gguf [--table memories.jsonl] [--memory-out learned.jsonl]\n"
                        "           [--tool-docs tool_docs.jsonl] [--lookup lookup.jsonl] [--examples examples.json] [--learn FILE ...] [--remember-mode store|queue]\n"
                        "           --chat | --serve PORT --ui demo_ui.html | --sleep-only | --prompt TEXT   [--sleep-at-start] [--system TEXT] [-ngl N]\n"
                        "           [--auto-menu K] [--menu-layer L]   (0019: put the K memories a message brings to mind in the system prompt)\n"
                        "           [--menu-cache FILE|none]   (0020: where that index is saved; default the --memory-out file + .menu)\n"
                        "           [--auto-inject N] [--inject-max K] [--inject-surprise T] [--inject-match key|state] [--inject-min-score S]   (0021: memory after surprising words)\n");
        return 1;
    }
    for (size_t p; (p = prompt.find("\\n")) != std::string::npos;) prompt.replace(p, 2, "\n");   // literal \n in the prompt argument

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    engine_t E;
    E.side = load_side(side_path);
    E.line_rule = line_rule > 0 ? line_rule : E.side.line_rule;
    if (mem_once >= 0) E.side.mem_once = mem_once;       // 0012: the command line overrides the side file
    if (key_min_chars >= 0) E.side.key_min_chars = key_min_chars;   // 0013: the command line overrides the side file
    if (mem_marks >= 0) E.side.mem_marks = (mem_marks && !E.side.marktag.empty()) ? 1 : 0;   // 0014: only if the file carries the vectors
    if (memgen >= 0) E.side.mem_gen = memgen;
    if (no_traces) E.side.memtok = 0;
    E.ngl = ngl;
    E.cell_ctx = cell_ctx;                        // 0016
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;   // 0015: -ngl N puts N layers on the GPU (CUDA/SYCL builds; 0 = CPU, the old behaviour)
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
    if (E.side.protocol) { cc.n_ctx = 2048; cc.n_batch = 2048; cc.n_ubatch = 2048; }   // 0018: a capture must be one ubatch
    E.cap = llama_init_from_model(E.model, cc);
    if (!E.ctx || !E.cap) { fprintf(stderr, "context failed\n"); return 2; }
    if (!lora_path.empty()) {
        E.lora = llama_adapter_lora_init(E.model, lora_path.c_str());
        if (!E.lora) { fprintf(stderr, "lora failed\n"); return 2; }
        float one = 1.0f; llama_set_adapters_lora(E.ctx, &E.lora, 1, &one);
    }
    if (!table_path.empty()) E.set_table(load_table(table_path));
    if (E.side.protocol) {                                // 0018
        PROTO.memory_out = memory_out;
        {                                                 // 0020: the saved menu index, and what must match to reuse it
            auto fsize = [](const std::string & path) { std::error_code ec; const auto n = path.empty() ? 0 : std::filesystem::file_size(path, ec); return ec ? 0ULL : (unsigned long long) n; };
            MENU.cache_path = menu_cache == "none" ? std::string() : (!menu_cache.empty() ? menu_cache : (memory_out.empty() ? std::string() : memory_out + ".menu"));
            MENU.fingerprint = "steermem-menu layer=" + std::to_string(MENU.layer) + " d=" + std::to_string(E.d) + " step=" + std::to_string(E.side.step) +
                               " model=" + std::to_string(fsize(model_path)) + " lora=" + std::to_string(fsize(lora_path)) + " side=" + std::to_string(fsize(side_path));
        }
        PROTO.remember_mode = remember_mode == "queue" ? "queue" : "store";
        proto_load_files(tool_docs, lookup_path, examples_path, ui_path);
        if (!memory_out.empty() && std::filesystem::exists(memory_out)) { auto more = load_table(memory_out); E.cells.insert(E.cells.end(), more.begin(), more.end()); }
        for (auto & path : learn) {
            const std::string name = path.substr(path.find_last_of("/\\") + 1);
            try { json r = proto_learn(E, name, proto_read_file(path), ""); fprintf(stderr, "learn %s: %d queued, %d stored, %d skipped\n", path.c_str(), r["queued"].get<int>(), r["stored"].get<int>(), r["skipped"].get<int>()); }
            catch (const std::exception & e) { fprintf(stderr, "learn %s: %s\n", path.c_str(), e.what()); }
        }
        if (sleep_at_start || sleep_only) proto_sleep(E, 0, [](const json & ev) { proto_print_event(ev); return true; }, nullptr);
        if (sleep_only) { llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free(); return 0; }
        if (MENU.k > 0 || (INJECT.rows > 0 && INJECT.match == "state")) proto_menu_build(E);          // 0019: every memory's states at the menu layer
        if (!quiet) fprintf(stderr, "protocol mode | checkpoint step %d | %zu memories, %zu queued | %zu tool manuals, %zu lookup entries | remember %s\n",
                            E.side.step, E.cells.size(), PROTO.queue.size(), PROTO.docs.size(), PROTO.lookup.size(), PROTO.remember_mode.c_str());
        if (chat_repl) {
            const std::string mname = model_path.substr(model_path.find_last_of("/\\") + 1);
            return run_chat_repl(E, max_new > 64 ? max_new : 700, system_prompt, mname);
        }
    }
    if (!quiet) fprintf(stderr, "table %zu cells | mem_seq %d cells %d mem_gen %d memtok %d window %d line_rule %d | scale %s\n", E.cells.size(), E.side.mem_seq, E.side.mem_seq_cells, E.side.mem_gen, E.side.memtok, E.side.key_window, E.line_rule, std::to_string(E.side.scale.begin()->second).c_str());

    if (port > 0) {
        std::string mname = model_path.substr(model_path.find_last_of("/\\") + 1);
        if (!lora_path.empty()) mname += " + " + lora_path.substr(lora_path.find_last_of("/\\") + 1);
        return serve(E, host, port, tables_dir, mname, quiet);
    }
    if (E.side.protocol) {                                // 0018: one question through the protocol loop
        json msgs = json::array({ { { "role", "user" }, { "content", prompt } } });
        auto printer = [](const json & ev) { proto_print_event(ev); return true; };
        const std::string sys = proto_apply_menu(E, system_prompt, prompt, MENU.k, printer);   // 0019
        std::vector<proto_inject_t> injects = proto_plan_injects(E, prompt, INJECT.rows, printer);   // 0021
        json done = run_protocol(E, proto_chat_prompt(msgs, sys), prompt, max_new > 64 ? max_new : 700, printer, nullptr, &injects);
        printf("%s\n", proto_dump(done).c_str());
    } else {
        auto r = run_prompt(E, prompt, max_new, quiet, nullptr, nullptr);
        printf("%s\n", r.summary.dump().c_str());
    }
    llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free();
    return 0;
}

// steermem_protocol.hpp -- PROTOCOL MODE (fork edit 0018, 2026-09-14): serve a twoside.py checkpoint the way it trained,
// let memory build as the model is used, and let a harness (or a person at the command line) control all of it.
//
// Included by steermem.cpp after engine_t. Active when the side GGUF says steermem.protocol = 1
// (llama/twoside_export.py writes it). Mirrors expert/twoside.py and demo/steermem_demo.py rule for rule:
//
//   capture     the sleep pass: "<|im_start|>user\nCONTEXT\n\nDefine: KEY<|im_end|>\n<|im_start|>assistant\n
//               <think>\n\n</think>\n\n" + KEY + " is SUMMARY" + <|im_end|>, each piece tokenised separately as the
//               trainer does, the ADAPTER ON, states kept after the side layers from the key's first token
//               (span_from_key) through the summary's last
//   sleep write the model reads a context and writes "KEY is ...", greedy, first line after " is "
//   block       [marks] [head] rows j = memtag + seqtag[1+min(j,61)] over min(mem_seq, n) evenly spaced states,
//               rows normalised to emb_norm, the states injected as unit vectors at the side layers (build_inject)
//   null block  nulltag row injecting nullrec, then up to 3 rows neartag + mean key embedding for the nearest keys
//   the loop    greedy; <recall>KEY<recall> in the text since the last block -> the table's answer right there;
//               a turn ending in a Qwen3.5 <tool_call> -> the built-in tool runs, its <tool_response> starts the
//               next turn; <|remember|>KEY is ...<|remember|> stores (or queues, --remember-mode queue);
//               <|alias|>bad -> good<|alias|> adds a name
//   memory in   the model's own <|remember|>; POST /api/remember; POST /api/upload (CSV / JSON / JSONL / text);
//               --learn FILE; everything waiting is learned by sleep (POST /api/sleep, --sleep-at-start,
//               --sleep-only), which writes each memory, captures its states and appends it to --memory-out
//   API         the routes and NDJSON events of demo/API.md, so demo_ui.html and steermem_harness.py run on either
//   --chat      the same, at the command line: talk to the model and control memory with slash commands
//   auto-inject a surprising word in the user's message that names a memory gets that memory's block right after it,
//               before the answer starts (0021: --auto-inject N rows, 0 off)
#pragma once
#include <iostream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX                                   // 0021: MSVC's windows.h would otherwise turn std::min and std::max into macros
#endif
#include <windows.h>
#undef near                                        // windef.h's empty 16-bit-era macros would erase these identifiers
#undef far
#endif

struct proto_cfg_t {
    std::unordered_map<std::string, std::string> docs, lookup, alias;
    json examples = json::array();
    std::vector<json> queue;                       // waiting for sleep: {key, context, summary, source}
    std::string memory_out, ui_html, files_dir, remember_mode = "store";
};
static proto_cfg_t PROTO;

static std::string proto_trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) ++a;
    while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

static std::string proto_dump(const json & j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

static std::string proto_read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static void proto_each_jsonl(const std::string & path, const std::function<void(const json &)> & fn) {
    if (path.empty()) return;
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) { if (line.size() > 2) { try { fn(json::parse(line)); } catch (...) {} } }
}

static void proto_load_files(const std::string & docs, const std::string & lookup, const std::string & examples, const std::string & ui) {
    proto_each_jsonl(docs, [](const json & o) { if (o.contains("name")) PROTO.docs[o.value("name", "")] = o.value("manual", ""); });
    proto_each_jsonl(lookup, [](const json & o) { if (o.contains("key")) PROTO.lookup[o.value("key", "")] = o.value("text", ""); });
    if (!examples.empty()) { try { PROTO.examples = json::parse(proto_read_file(examples)); } catch (...) {} }
    if (!ui.empty()) PROTO.ui_html = proto_read_file(ui);
}

// ---------------------------------------------------------------- uploads
static std::string proto_default_key(const std::string & context) {
    std::string first = context.substr(0, context.find('\n'));
    std::istringstream ws(first); std::string w, out; int n = 0;
    while (ws >> w && n < 6) { out += (n ? " " : "") + w; ++n; }
    if (out.size() > 48) out = out.substr(0, 48);
    while (!out.empty() && std::strchr(" .,;:", out.back())) out.pop_back();
    return out.empty() ? std::string("note") : out;
}

static std::vector<std::vector<std::string>> proto_csv(const std::string & s) {
    std::vector<std::vector<std::string>> rows(1);
    std::string cur; bool q = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (q) {
            if (c == '"') { if (i + 1 < s.size() && s[i + 1] == '"') { cur += '"'; ++i; } else q = false; }
            else cur += c;
        } else if (c == '"') q = true;
        else if (c == ',') { rows.back().push_back(cur); cur.clear(); }
        else if (c == '\n') { rows.back().push_back(cur); cur.clear(); rows.emplace_back(); }
        else if (c != '\r') cur += c;
    }
    if (!cur.empty() || !rows.back().empty()) rows.back().push_back(cur);
    while (!rows.empty() && (rows.back().empty() || (rows.back().size() == 1 && rows.back()[0].empty()))) rows.pop_back();
    return rows;
}

// rows to learn: {key, context, summary, source}; a row with a summary is stored as it is, the rest wait for sleep
static std::vector<json> proto_parse_upload(const std::string & name, const std::string & content, std::string fmt, int & skipped) {
    std::vector<json> items;
    skipped = 0;
    if (fmt.empty()) { size_t dot = name.find_last_of('.'); fmt = dot == std::string::npos ? "" : name.substr(dot + 1); }
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
    auto take = [&](const json & o0) {
        if (!o0.is_object()) { ++skipped; return; }
        json o = json::object();
        for (auto it = o0.begin(); it != o0.end(); ++it) { std::string k = it.key(); std::transform(k.begin(), k.end(), k.begin(), ::tolower); o[k] = it.value(); }
        auto str = [&](const char * k) { return o.contains(k) && o[k].is_string() ? proto_trim(o[k].get<std::string>()) : std::string(); };
        std::string ctx = str("context"); if (ctx.empty()) ctx = str("text"); if (ctx.empty()) ctx = str("content");
        std::string summ = str("summary"), key = str("key"); if (key.empty()) key = str("name");
        if (key.empty() && !ctx.empty()) key = proto_default_key(ctx);
        if (!key.empty() && (!ctx.empty() || !summ.empty())) items.push_back({ { "key", key }, { "context", ctx }, { "summary", summ }, { "source", name } });
        else ++skipped;
    };
    if (fmt == "csv") {
        auto rows = proto_csv(content);
        if (rows.empty()) return items;
        std::vector<std::string> head = rows[0];
        for (auto & h : head) { h = proto_trim(h); std::transform(h.begin(), h.end(), h.begin(), ::tolower); }
        for (size_t r = 1; r < rows.size(); ++r) {
            json o = json::object();
            for (size_t c = 0; c < head.size() && c < rows[r].size(); ++c) o[head[c]] = rows[r][c];
            take(o);
        }
    } else if (fmt == "json") {
        json j = json::parse(content);
        json list = j.is_array() ? j : (j.contains("memories") ? j["memories"] : (j.contains("cells") ? j["cells"] : json::array({ j })));
        for (auto & o : list) take(o);
    } else if (fmt == "jsonl") {
        std::istringstream ls(content); std::string line;
        while (std::getline(ls, line)) { if (proto_trim(line).empty()) continue; try { take(json::parse(line)); } catch (...) { ++skipped; } }
    } else {
        std::vector<std::string> chunks; std::string cur, para;
        std::istringstream ls(content); std::string line;
        auto flush_para = [&]() {
            para = proto_trim(para);
            if (para.empty()) return;
            if (!cur.empty() && cur.size() + para.size() > 1200) { chunks.push_back(cur); cur.clear(); }
            cur = cur.empty() ? para : cur + "\n\n" + para;
            para.clear();
        };
        while (std::getline(ls, line)) { if (proto_trim(line).empty()) flush_para(); else para += line + "\n"; }
        flush_para();
        if (!cur.empty()) chunks.push_back(cur);
        for (size_t i = 0; i < chunks.size(); ++i)
            items.push_back({ { "key", chunks.size() == 1 ? name : name + " part " + std::to_string(i + 1) }, { "context", chunks[i] }, { "summary", "" }, { "source", name } });
    }
    return items;
}

// ---------------------------------------------------------------- sleep mode: writing a memory and its states
static std::string sleep_prompt(const std::string & ctx, const std::string & key) {
    return "<|im_start|>user\n" + ctx + "\n\nDefine: " + key + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

static void proto_cap_ready(engine_t & E) {
    float one = 1.0f;
    if (E.lora) llama_set_adapters_lora(E.cap, &E.lora, 1, &one);     // the sleep side writes with the adapter ON
    llama_inject_clear(E.cap);
    llama_memory_clear(llama_get_memory(E.cap), true);
    E.capd.want.clear(); E.capd.got.clear();
}

static bool proto_decode(llama_context * lctx, const std::vector<llama_token> & ids, int pos0, bool logits_last, int n_batch) {
    for (size_t s = 0; s < ids.size(); s += (size_t) n_batch) {
        size_t e = std::min(ids.size(), s + (size_t) n_batch);
        llama_batch b = llama_batch_init((int) (e - s), 0, 1);
        for (size_t i = s; i < e; ++i) {
            b.token[i - s] = ids[i]; b.pos[i - s] = pos0 + (int) i; b.n_seq_id[i - s] = 1; b.seq_id[i - s][0] = 0;
            b.logits[i - s] = (logits_last && i + 1 == ids.size()) ? 1 : 0;
        }
        b.n_tokens = (int) (e - s);
        const int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) return false;
    }
    return true;
}

static std::vector<float> proto_token_mean(engine_t & E, const std::vector<llama_token> & ids) {
    std::vector<float> out((size_t) E.d, 0.0f);
    if (ids.empty()) return out;
    proto_cap_ready(E);
    E.capd.want.assign({ "inp_embd" });
    if (proto_decode(E.cap, ids, 0, false, (int) ids.size()) && E.capd.got.count("inp_embd")) {
        auto & e = E.capd.got["inp_embd"];
        const int ne0 = (int) E.capd.shape["inp_embd"].first;
        for (size_t i = 0; i < ids.size(); ++i)
            for (int j = 0; j < E.d && j < ne0; ++j) out[(size_t) j] += e[i * (size_t) ne0 + (size_t) j] / (float) ids.size();
    }
    E.capd.want.clear();
    return out;
}

// ---------------------------------------------------------------- what comes to mind: the auto key menu (0019)
// Every memory is also held as the model's own token states at one layer, over "KEY is SUMMARY" (adapter on, no chat
// template). Before an answer the user's message is matched token by token: each message token takes its best token
// in a memory, weighted by how rare that token is in the table -- expert/assoc_probe.py's best retrieval of quoted
// content, and demo/steermem_demo.py's MenuIndex. The K best keys become a key menu in the system prompt, written the
// way the checkpoint's menu items were (expert/chain_items.py: "- KEY: what it is. When to recall it.").
#include <thread>

struct proto_menu_t {
    int k = 0, layer = 6;                           // --auto-menu K; --menu-layer L counts like hidden_states: l_out-(L-1)
    bool built = false, fitted = false, building = false;
    std::vector<float> mu, sd;                      // z-score over every state held, fitted once
    std::vector<std::string> keys;
    std::vector<std::vector<float>> rows;           // per memory: n x d, z-scored and unit length once fitted
    std::vector<std::vector<llama_token>> ids;      // the content tokens those rows came from
    std::vector<uint64_t> hashes;                   // 0020: FNV-1a of "KEY is SUMMARY", so a changed memory is recomputed
    std::unordered_map<std::string, size_t> at;
    std::string cache_path, fingerprint;            // 0020: --menu-cache, and what made it (layer, width, step, file sizes)
};
static proto_menu_t MENU;

static const std::set<std::string> & proto_stop() {
    static const std::set<std::string> S = [] {
        std::set<std::string> s;
        std::istringstream ws("a an the of to in on at by for with from and or but is are was were be been being it its this that "
                              "these those as into than then so such not no nor do does did has have had i you he she we they them his her their our "
                              "your my me us who whom which what when where why how all any both each few more most other some own same can will "
                              "just should now over under again further once here there about above below up down out off very thee thou thy ye "
                              "unto shall hath doth also get give please");
        std::string w;
        while (ws >> w) s.insert(w);
        return s;
    }();
    return S;
}

// content-token states of a text at the menu layer: a real word, not a stopword, never the first token
static bool proto_token_states(engine_t & E, const std::string & text, std::vector<float> & st, std::vector<llama_token> & kept) {
    st.clear(); kept.clear();
    std::vector<llama_token> ids = E.tok(text, true);
    const int cap_n = std::min((int) llama_n_ctx(E.cap), (int) llama_n_ubatch(E.cap));
    if ((int) ids.size() > cap_n) ids.resize((size_t) cap_n);
    if (ids.empty() || MENU.layer < 1) return false;
    const std::string name = "l_out-" + std::to_string(MENU.layer - 1);
    proto_cap_ready(E);
    E.capd.want.assign({ name });
    const bool ok = proto_decode(E.cap, ids, 0, false, (int) ids.size()) && E.capd.got.count(name) && E.capd.got[name].size() == ids.size() * (size_t) E.d;
    if (!ok) { E.capd.want.clear(); return false; }
    const std::vector<float> & v = E.capd.got[name];
    std::vector<size_t> keep;
    for (size_t i = 1; i < ids.size(); ++i) {
        std::string w = proto_trim(E.piece(ids[i]));
        std::transform(w.begin(), w.end(), w.begin(), ::tolower);
        bool word = false;
        for (unsigned char ch : w) if (std::isalnum(ch)) word = true;
        if (word && !proto_stop().count(w)) keep.push_back(i);
    }
    if (keep.empty()) for (size_t i = ids.size() > 1 ? 1 : 0; i < ids.size(); ++i) keep.push_back(i);
    for (size_t i : keep) {
        st.insert(st.end(), v.begin() + (ptrdiff_t) (i * (size_t) E.d), v.begin() + (ptrdiff_t) ((i + 1) * (size_t) E.d));
        kept.push_back(ids[i]);
    }
    E.capd.want.clear();
    return true;
}

static void proto_menu_norm(std::vector<float> & st, int d) {
    for (size_t o = 0; o + (size_t) d <= st.size(); o += (size_t) d) {
        float * x = st.data() + o;
        double nn = 0.0;
        for (int j = 0; j < d; ++j) { x[j] = (x[j] - MENU.mu[(size_t) j]) / MENU.sd[(size_t) j]; nn += (double) x[j] * x[j]; }
        const float inv = 1.0f / ((float) std::sqrt(nn) + 1e-6f);
        for (int j = 0; j < d; ++j) x[j] *= inv;
    }
}

static void proto_menu_fit(int d) {
    std::vector<double> m((size_t) d, 0.0), s((size_t) d, 0.0);
    size_t n = 0;
    for (auto & r : MENU.rows)
        for (size_t o = 0; o + (size_t) d <= r.size(); o += (size_t) d, ++n)
            for (int j = 0; j < d; ++j) { const double x = r[o + (size_t) j]; m[(size_t) j] += x; s[(size_t) j] += x * x; }
    if (n < 2) return;
    MENU.mu.assign((size_t) d, 0.0f); MENU.sd.assign((size_t) d, 0.0f);
    for (int j = 0; j < d; ++j) {
        const double mean = m[(size_t) j] / (double) n;
        MENU.mu[(size_t) j] = (float) mean;
        MENU.sd[(size_t) j] = (float) std::sqrt(std::max(0.0, s[(size_t) j] / (double) n - mean * mean)) + 1e-3f;
    }
    for (auto & r : MENU.rows) proto_menu_norm(r, d);
    MENU.fitted = true;
}

static uint64_t proto_fnv1a(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

static void proto_menu_put(const std::string & key, uint64_t hash, std::vector<float> && rows, std::vector<llama_token> && ids) {
    const auto it = MENU.at.find(key);
    if (it != MENU.at.end()) { MENU.rows[it->second] = std::move(rows); MENU.ids[it->second] = std::move(ids); MENU.hashes[it->second] = hash; return; }
    MENU.at[key] = MENU.keys.size();
    MENU.keys.push_back(key); MENU.rows.push_back(std::move(rows)); MENU.ids.push_back(std::move(ids)); MENU.hashes.push_back(hash);
}

// ---- 0020: the saved index. "SMENU01\n", the fingerprint, d, mu[d], sd[d] (f32), then records until the end of the
// file: key, hash, n, ids[n] (i32), rows[n x d] (f16, normalised). A later record for a key replaces an earlier one.
static const char PROTO_MENU_MAGIC[9] = "SMENU01\n";
struct proto_menu_saved_t { uint64_t hash = 0; std::vector<float> rows; std::vector<llama_token> ids; };

static void proto_menu_write_record(std::ofstream & f, size_t i, int d) {
    const std::string & key = MENU.keys[i];
    const uint32_t kl = (uint32_t) key.size(), n = (uint32_t) MENU.ids[i].size();
    f.write((const char *) &kl, 4); f.write(key.data(), kl);
    f.write((const char *) &MENU.hashes[i], 8);
    f.write((const char *) &n, 4);
    for (llama_token t : MENU.ids[i]) { const int32_t v = (int32_t) t; f.write((const char *) &v, 4); }
    std::vector<ggml_fp16_t> half((size_t) n * (size_t) d);
    ggml_fp32_to_fp16_row(MENU.rows[i].data(), half.data(), (int64_t) half.size());
    f.write((const char *) half.data(), (std::streamsize) (half.size() * sizeof(ggml_fp16_t)));
}

static void proto_menu_cache_write(int d) {
    if (MENU.cache_path.empty() || !MENU.fitted) return;
    const std::string tmp = MENU.cache_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { fprintf(stderr, "menu cache: cannot write %s\n", tmp.c_str()); return; }
        const uint32_t fl = (uint32_t) MENU.fingerprint.size(), dd = (uint32_t) d;
        f.write(PROTO_MENU_MAGIC, 8);
        f.write((const char *) &fl, 4); f.write(MENU.fingerprint.data(), fl);
        f.write((const char *) &dd, 4);
        f.write((const char *) MENU.mu.data(), (std::streamsize) d * 4);
        f.write((const char *) MENU.sd.data(), (std::streamsize) d * 4);
        for (size_t i = 0; i < MENU.keys.size(); ++i) proto_menu_write_record(f, i, d);
    }
    std::error_code ec;
    std::filesystem::rename(tmp, MENU.cache_path, ec);
    if (ec) { std::filesystem::remove(MENU.cache_path, ec); std::filesystem::rename(tmp, MENU.cache_path, ec); }   // where rename will not replace a file
    if (ec) fprintf(stderr, "menu cache: cannot replace %s: %s\n", MENU.cache_path.c_str(), ec.message().c_str());
}

static void proto_menu_cache_append(int d, size_t i) {
    if (MENU.cache_path.empty() || !std::filesystem::exists(MENU.cache_path)) return;
    std::ofstream f(MENU.cache_path, std::ios::binary | std::ios::app);
    if (f) proto_menu_write_record(f, i, d);
}

// the saved index, if this model at this layer made it: sets mu/sd and returns every record (a torn last record is dropped)
static bool proto_menu_cache_load(int d, std::unordered_map<std::string, proto_menu_saved_t> & out) {
    if (MENU.cache_path.empty()) return false;
    std::ifstream f(MENU.cache_path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    uint32_t fl = 0, dd = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, PROTO_MENU_MAGIC, 8) != 0 || !f.read((char *) &fl, 4) || fl > 4096) return false;
    std::string fp(fl, '\0');
    if (!f.read(&fp[0], fl) || fp != MENU.fingerprint || !f.read((char *) &dd, 4) || (int) dd != d) {
        fprintf(stderr, "menu cache %s was made by another model, adapter or layer: computing the index again\n", MENU.cache_path.c_str());
        return false;
    }
    std::vector<float> mu((size_t) d), sd((size_t) d);
    if (!f.read((char *) mu.data(), (std::streamsize) d * 4) || !f.read((char *) sd.data(), (std::streamsize) d * 4)) return false;
    while (true) {
        uint32_t kl = 0, n = 0;
        uint64_t h = 0;
        if (!f.read((char *) &kl, 4) || kl > 4096) break;
        std::string key(kl, '\0');
        if (!f.read(&key[0], kl) || !f.read((char *) &h, 8) || !f.read((char *) &n, 4) || n > (1u << 16)) break;
        std::vector<int32_t> ids(n);
        std::vector<ggml_fp16_t> half((size_t) n * (size_t) d);
        if (!f.read((char *) ids.data(), (std::streamsize) n * 4) || !f.read((char *) half.data(), (std::streamsize) (half.size() * sizeof(ggml_fp16_t)))) break;
        proto_menu_saved_t s;
        s.hash = h;
        s.ids.assign(ids.begin(), ids.end());
        s.rows.resize(half.size());
        ggml_fp16_to_fp32_row(half.data(), s.rows.data(), (int64_t) half.size());
        out[key] = std::move(s);
    }
    MENU.mu = std::move(mu); MENU.sd = std::move(sd); MENU.fitted = true;
    return true;
}

// index one memory (a key stored again replaces its rows); a no-op until the index is built
static void proto_menu_add(engine_t & E, const std::string & key, const std::string & summary) {
    if (!MENU.built || key.empty() || summary.empty()) return;
    const std::string text = key + " is " + summary;
    std::vector<float> st; std::vector<llama_token> kept;
    if (!proto_token_states(E, text, st, kept)) return;
    if (MENU.fitted) proto_menu_norm(st, E.d);
    proto_menu_put(key, proto_fnv1a(text), std::move(st), std::move(kept));
    if (MENU.fitted && !MENU.building) proto_menu_cache_append(E.d, MENU.at[key]);   // 0020: a memory stored mid-session is saved as it arrives
}

static void proto_menu_build(engine_t & E) {
    const auto t0 = std::chrono::steady_clock::now();
    MENU.built = true;
    std::unordered_map<std::string, proto_menu_saved_t> saved;
    const bool loaded = proto_menu_cache_load(E.d, saved);
    std::unordered_map<std::string, size_t> newest;
    for (size_t i = 0; i < E.cells.size(); ++i) newest[E.cells[i].key] = i;
    size_t reused = 0, computed = 0;
    MENU.building = true;
    for (size_t i = 0; i < E.cells.size(); ++i) {
        const cell_t & c = E.cells[i];
        if (newest[c.key] != i) continue;
        const std::string summary = c.summary.empty() ? c.text : c.summary;
        const auto it = saved.find(c.key);
        if (loaded && it != saved.end() && it->second.hash == proto_fnv1a(c.key + " is " + summary)) {
            proto_menu_put(c.key, it->second.hash, std::move(it->second.rows), std::move(it->second.ids));
            ++reused;
            continue;
        }
        proto_menu_add(E, c.key, summary);
        if (++computed % 100 == 0) fprintf(stderr, "  menu index: %zu memories computed\n", computed);
    }
    MENU.building = false;
    if (!MENU.fitted) proto_menu_fit(E.d);
    if (computed > 0 || !loaded) proto_menu_cache_write(E.d);
    fprintf(stderr, "menu index: %zu memories at layer %d in %.1fs (%zu read from %s, %zu computed)\n", MENU.keys.size(), MENU.layer,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), reused,
            MENU.cache_path.empty() ? "no saved index" : MENU.cache_path.c_str(), computed);
}

struct proto_menu_hit_t { std::vector<std::string> keys; std::vector<double> scores; };

// every memory ranked against query states (z-scored, unit length) and their token ids: the k best (0021: shared by
// the menu and auto-inject's state match)
static proto_menu_hit_t proto_menu_rank(engine_t & E, const std::vector<float> & q, const std::vector<llama_token> & qids, int k) {
    proto_menu_hit_t out;
    if (!MENU.fitted || MENU.keys.empty() || k <= 0 || qids.empty()) return out;
    const size_t d = (size_t) E.d, nq = qids.size(), nm = MENU.keys.size();
    std::unordered_map<llama_token, int> df;
    for (auto & ids : MENU.ids) { std::set<llama_token> u(ids.begin(), ids.end()); for (llama_token t : u) df[t]++; }
    std::vector<double> w(nq);
    double wsum = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        const auto it = df.find(qids[i]);
        w[i] = std::log((nm + 1.0) / ((it == df.end() ? 0 : it->second) + 1.0)) + 1.0;
        wsum += w[i];
    }
    std::vector<std::pair<double, size_t>> sc(nm);
    auto work = [&](size_t a, size_t b) {
        for (size_t m = a; m < b; ++m) {
            const std::vector<float> & R = MENU.rows[m];
            const size_t nr = R.size() / d;
            double s = 0.0;
            for (size_t i = 0; i < nq && nr; ++i) {
                const float * x = q.data() + i * d;
                float best = -2.0f;
                for (size_t r = 0; r < nr; ++r) {
                    const float * y = R.data() + r * d;
                    float dot = 0.0f;
                    for (size_t j = 0; j < d; ++j) dot += x[j] * y[j];
                    best = std::max(best, dot);
                }
                s += w[i] * best;
            }
            sc[m] = { nr ? s / wsum : -1.0, m };
        }
    };
    const size_t nt = (size_t) std::max(1, std::min(E.n_threads, 16));
    std::vector<std::thread> th;
    for (size_t t = 0; t < nt; ++t) th.emplace_back(work, nm * t / nt, nm * (t + 1) / nt);
    for (auto & t : th) t.join();
    const size_t top = std::min((size_t) k, nm);
    std::partial_sort(sc.begin(), sc.begin() + (ptrdiff_t) top, sc.end(),
                      [](const std::pair<double, size_t> & a, const std::pair<double, size_t> & b) { return a.first > b.first; });
    for (size_t i = 0; i < top; ++i) { out.keys.push_back(MENU.keys[sc[i].second]); out.scores.push_back(std::round(sc[i].first * 1000.0) / 1000.0); }
    return out;
}

static proto_menu_hit_t proto_menu_query(engine_t & E, const std::string & message, int k) {
    if (!MENU.fitted) proto_menu_fit(E.d);
    std::vector<float> q; std::vector<llama_token> qids;
    if (!MENU.fitted || k <= 0 || !proto_token_states(E, message, q, qids) || qids.empty()) return proto_menu_hit_t();
    proto_menu_norm(q, E.d);
    return proto_menu_rank(E, q, qids, k);
}

// one menu line, written the way the menu items were: what the memory is, then when to recall it
static std::string proto_menu_line(const std::string & key, const std::string & summary) {
    auto words = [](const std::string & s, int n, bool & more) {
        std::istringstream ws(s); std::string w, out; int i = 0; more = false;
        while (ws >> w) { if (i == n) { more = true; break; } out += (i ? " " : "") + w; ++i; }
        while (!out.empty() && std::strchr(",;:.", out.back())) out.pop_back();
        return out;
    };
    auto starts = [&](const char * p) { return summary.rfind(p, 0) == 0; };
    bool more = false;
    std::string desc, when;
    if (starts("an ASV Bible verse in ")) {
        const size_t c = summary.find(": ");
        std::string body = c == std::string::npos ? summary : summary.substr(c + 2);
        const size_t sg = body.find(" Strong's:");
        if (sg != std::string::npos) body = body.substr(0, sg);
        desc = "the verse that begins \"" + words(body, 5, more) + "...\"";
        when = "Recall it before quoting it.";
    } else if (starts("a Strong's ")) {
        const size_t c = summary.find(": ");
        desc = std::string("the ") + (summary.find("Greek") != std::string::npos ? "Greek" : "Hebrew") + " word for \"" +
               words(c == std::string::npos ? summary : summary.substr(c + 2), 4, more) + "\"";
        when = "Recall it when that word's original meaning matters.";
    } else if (starts("a tool that ")) {
        desc = "the tool " + key;
        when = "Recall it before calling that tool, to see how it worked last time.";
    } else if (summary.find(" file that ") != std::string::npos && summary.find(" touched") != std::string::npos) {
        const size_t a = summary.find(" file that ") + 11, b = summary.find(" touched");
        desc = "the file " + (b > a ? summary.substr(a, b - a) : std::string("a tool")) + " touched";
        when = "Recall it to find where it lives before reading it.";
    } else {
        desc = words(summary, 10, more) + (more ? "..." : "");
        when = "Recall it if the question needs it.";
    }
    return "- " + key + ": " + desc + ". " + when;
}

static std::string proto_menu_text(engine_t & E, const std::vector<std::string> & keys) {
    std::string out = "Memory you can use (recall with <recall>KEY<recall>):";
    for (auto & key : keys)
        for (int i = (int) E.cells.size() - 1; i >= 0; --i) {
            const cell_t & c = E.cells[(size_t) i];
            if (c.key != key) continue;
            out += "\n" + proto_menu_line(key, c.summary.empty() ? c.text : c.summary);
            break;
        }
    return out + "\nIf none of these fit, just answer normally.";
}

// the system prompt with what the message brings to mind added; emits menu {keys, scores} first
static std::string proto_apply_menu(engine_t & E, const std::string & system, const std::string & message, int k,
                                    const std::function<bool(const json &)> & emit) {
    if (k <= 0 || E.cells.empty() || message.empty()) return system;
    if (!MENU.built) proto_menu_build(E);
    const proto_menu_hit_t h = proto_menu_query(E, message, k);
    if (h.keys.empty()) return system;
    emit({ { "type", "menu" }, { "keys", h.keys }, { "scores", h.scores } });
    const std::string menu = proto_menu_text(E, h.keys);
    return system.empty() ? menu : system + "\n\n" + menu;
}

static std::string sleep_write(engine_t & E, const std::string & key, const std::string & ctx, int n_new = 160) {
    std::string cs = ctx.empty() ? key : ctx;
    const int cap_n = (int) llama_n_ctx(E.cap);
    std::vector<llama_token> ids = E.tok(sleep_prompt(cs, key), true);
    while ((int) ids.size() + n_new + 8 > cap_n && cs.size() > 16) { cs = cs.substr(0, cs.size() * 3 / 4); ids = E.tok(sleep_prompt(cs, key), true); }
    proto_cap_ready(E);
    if (!proto_decode(E.cap, ids, 0, true, (int) llama_n_batch(E.cap))) return std::string();
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    std::string out;
    int pos = (int) ids.size();
    for (int i = 0; i < n_new; ++i) {
        const float * lg = llama_get_logits_ith(E.cap, -1);
        int best = 0; for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
        if (llama_vocab_is_eog(E.vocab, best)) break;
        const std::string pc = E.piece(best);
        const size_t nl = pc.find('\n');
        if (nl != std::string::npos) { out += pc.substr(0, nl); if (!proto_trim(out).empty()) break; }
        else out += pc;
        if (!proto_decode(E.cap, { best }, pos, true, 1)) break;
        ++pos;
    }
    const size_t is = out.find(" is ");
    return proto_trim(is == std::string::npos ? out : out.substr(is + 4));
}

static bool capture_twoside(engine_t & E, cell_t & c) {
    if (c.captured) return true;
    const std::string ctxs = !c.ctx.empty() ? c.ctx : c.text;
    if (c.summary.empty()) c.summary = sleep_write(E, c.key, ctxs);
    if (c.summary.empty()) return false;
    std::vector<llama_token> kid = E.tok(c.key, true), sid = E.tok(" is " + c.summary, true), end = E.tok("<|im_end|>", true);
    const int body = (int) (kid.size() + sid.size() + end.size());
    const int cap_n = std::min((int) llama_n_ctx(E.cap), (int) llama_n_ubatch(E.cap));   // one ubatch, or cb_eval keeps only the last chunk
    std::string cs = ctxs.empty() ? c.summary : ctxs;
    std::vector<llama_token> pre = E.tok(sleep_prompt(cs, c.key), true);
    while ((int) pre.size() + body > cap_n && cs.size() > 16) { cs = cs.substr(0, cs.size() * 3 / 4); pre = E.tok(sleep_prompt(cs, c.key), true); }
    if ((int) pre.size() + body > cap_n) return false;
    std::vector<llama_token> all = pre;
    all.insert(all.end(), kid.begin(), kid.end());
    all.insert(all.end(), sid.begin(), sid.end());
    all.insert(all.end(), end.begin(), end.end());
    proto_cap_ready(E);
    for (int l : E.side.layers) E.capd.want.push_back("l_out-" + std::to_string(l));
    if (!proto_decode(E.cap, all, 0, false, (int) all.size())) { E.capd.want.clear(); return false; }
    const int np = (int) pre.size(), nk = (int) kid.size(), ns = (int) sid.size();
    const int s0 = np + (E.side.span_key ? 0 : nk), n = ns + (E.side.span_key ? nk : 0);
    for (int l : E.side.layers) {
        auto & v = E.capd.got["l_out-" + std::to_string(l)];
        if ((int64_t) v.size() != (int64_t) all.size() * E.d) { fprintf(stderr, "protocol capture of l_out-%d has %zu floats, expected %zu x %d\n", l, v.size(), all.size(), E.d); E.capd.want.clear(); return false; }
        c.st[l].assign(v.begin() + (size_t) s0 * E.d, v.begin() + (size_t) (s0 + n) * E.d);
    }
    c.ids.assign(all.begin() + s0, all.begin() + s0 + n);
    E.capd.want.clear();
    c.keyemb = E.side.keyemb_on ? proto_token_mean(E, kid) : std::vector<float>((size_t) E.d, 0.0f);
    c.captured = true;
    return true;
}

static void proto_normalise(engine_t & E, engine_t::block_t & b) {
    const int d = E.d;
    for (int k = 0; k < b.n; ++k) {
        float * row = b.vec.data() + (size_t) k * d;
        float nv = 0; for (int j = 0; j < d; ++j) nv += row[j] * row[j];
        nv = std::sqrt(nv) + 1e-6f; for (int j = 0; j < d; ++j) row[j] = row[j] / nv * E.side.emb_norm;
        for (int l : E.side.layers) {
            float * r = b.rec[l].data() + (size_t) k * d; float nr = 0; for (int j = 0; j < d; ++j) nr += r[j] * r[j];
            nr = std::sqrt(nr); if (nr > 1e-6f) for (int j = 0; j < d; ++j) r[j] /= nr;
        }
    }
}

static engine_t::block_t block_twoside(engine_t & E, cell_t & c) {
    engine_t::block_t b;
    if (!capture_twoside(E, c)) return b;
    const int d = E.d, n = (int) c.ids.size();
    const int nb = std::min(E.side.mem_seq, n);
    const int mk = E.side.mem_marks ? 1 : 0, hd = E.side.head ? 1 : 0;
    b.n = mk + hd + nb + mk;
    b.vec.assign((size_t) b.n * d, 0.0f);
    for (int l : E.side.layers) b.rec[l].assign((size_t) b.n * d, 0.0f);
    int r0 = 0;
    if (mk) { for (int j = 0; j < d; ++j) b.vec[(size_t) j] = E.side.marktag[(size_t) j]; r0 = 1; }
    if (hd) {
        for (int j = 0; j < d; ++j) b.vec[(size_t) r0 * d + j] = E.side.memtag[(size_t) j] + E.side.seqtag[(size_t) j] + c.keyemb[(size_t) j];
        for (int l : E.side.layers) {
            float * r = b.rec[l].data() + (size_t) r0 * d;
            for (int t = 0; t < n; ++t) for (int j = 0; j < d; ++j) r[j] += c.st[l][(size_t) t * d + j] / (float) n;
        }
        ++r0;
    }
    for (int k = 0; k < nb; ++k) {
        const int pj = nb > 1 ? (int) std::nearbyint((double) k * (n - 1) / (double) (nb - 1)) : n - 1;   // Python's round: half to even
        const int code = 1 + std::min(k, 61);
        for (int j = 0; j < d; ++j) b.vec[(size_t) (r0 + k) * d + j] = E.side.memtag[(size_t) j] + E.side.seqtag[(size_t) code * d + j];
        for (int l : E.side.layers) memcpy(b.rec[l].data() + (size_t) (r0 + k) * d, c.st[l].data() + (size_t) pj * d, (size_t) d * sizeof(float));
    }
    if (mk) for (int j = 0; j < d; ++j) b.vec[(size_t) (b.n - 1) * d + j] = E.side.marktag[(size_t) d + j];
    proto_normalise(E, b);
    return b;
}

static engine_t::block_t null_block(engine_t & E, const std::vector<std::string> & near) {
    engine_t::block_t b;
    const int d = E.d, nn = (int) std::min<size_t>(3, near.size());
    b.n = 1 + nn;
    b.vec.assign((size_t) b.n * d, 0.0f);
    for (int l : E.side.layers) b.rec[l].assign((size_t) b.n * d, 0.0f);
    for (int j = 0; j < d; ++j) b.vec[(size_t) j] = E.side.nulltag[(size_t) j];
    for (int l : E.side.layers) memcpy(b.rec[l].data(), E.side.nullrec[l].data(), (size_t) d * sizeof(float));
    for (int i = 0; i < nn; ++i) {
        std::vector<float> ke = proto_token_mean(E, E.tok(near[(size_t) i], true));
        for (int j = 0; j < d; ++j) b.vec[(size_t) (1 + i) * d + j] = E.side.neartag[(size_t) j] + ke[(size_t) j];
        for (int l : E.side.layers) memcpy(b.rec[l].data() + (size_t) (1 + i) * d, ke.data(), (size_t) d * sizeof(float));
    }
    proto_normalise(E, b);
    return b;
}

static double proto_similarity(const std::string & a, const std::string & b) {   // 1 - edit distance / longer length
    const size_t n = a.size(), m = b.size();
    if (!n || !m) return 0.0;
    std::vector<size_t> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = j;
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= m; ++j) cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1) });
        std::swap(prev, cur);
    }
    return 1.0 - (double) prev[m] / (double) std::max(n, m);
}

static std::vector<std::string> proto_nearest(engine_t & E, const std::string & key, int n = 3) {
    std::vector<std::pair<double, std::string>> sc;
    std::vector<std::string> keys;
    std::set<std::string> seen;
    for (auto & c : E.cells) {
        if (c.key == key || seen.count(c.key)) continue;
        seen.insert(c.key); keys.push_back(c.key);
        const double s = proto_similarity(key, c.key);
        if (s >= 0.45) sc.push_back({ s, c.key });
    }
    std::sort(sc.begin(), sc.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
    std::vector<std::string> out;
    for (auto & p : sc) { if ((int) out.size() >= n) break; out.push_back(p.second); }
    const size_t h = std::hash<std::string>{}(key);             // padded, as in training, with other keys the table holds
    for (size_t t = 0; (int) out.size() < n && !keys.empty() && t < 64; ++t) {
        const std::string & k = keys[(h + t * 2654435761u) % keys.size()];
        if (std::find(out.begin(), out.end(), k) == out.end()) out.push_back(k);
    }
    return out;
}

struct proto_hit_t { bool hit = false; std::string key, summary; std::vector<std::string> near; engine_t::block_t b; };

static proto_hit_t proto_lookup(engine_t & E, const std::string & asked) {
    proto_hit_t h;
    std::string k = proto_trim(asked);
    auto al = PROTO.alias.find(k);
    if (al != PROTO.alias.end()) k = al->second;
    if (!k.empty() && k.back() == ')' && k.find('(') != std::string::npos) {   // 0023: a key written in call form, KEY(args)
        const std::string base = proto_trim(k.substr(0, k.find('(')));
        bool as_written = false, named = false;
        for (auto & c : E.cells) { if (c.key == k) as_written = true; if (c.key == base) named = true; }
        if (named && !as_written) k = base;
    }
    for (int i = (int) E.cells.size() - 1; i >= 0; --i) {             // the newest mention of the key
        if (E.cells[(size_t) i].key != k) continue;
        h.b = block_twoside(E, E.cells[(size_t) i]);
        if (h.b.n > 0) { h.hit = true; h.key = k; h.summary = E.cells[(size_t) i].summary; return h; }
        break;
    }
    h.key = asked;
    h.near = proto_nearest(E, k);
    h.b = null_block(E, h.near);
    return h;
}

static bool proto_feed_block(engine_t & E, engine_t::block_t & b) {
    for (int l : E.side.layers)
        for (int k = 0; k < b.n; ++k)
            E.inj.push_back({ l, E.kv_pos + k, std::vector<float>(b.rec[l].begin() + (size_t) k * E.d, b.rec[l].begin() + (size_t) (k + 1) * E.d) });
    E.flush_inject();
    llama_batch bb = llama_batch_init(b.n, E.d, 1);
    memcpy(bb.embd, b.vec.data(), b.vec.size() * sizeof(float));
    for (int k = 0; k < b.n; ++k) { bb.pos[k] = E.kv_pos + k; bb.n_seq_id[k] = 1; bb.seq_id[k][0] = 0; bb.logits[k] = (k + 1 == b.n) ? 1 : 0; }
    bb.n_tokens = b.n;
    const int rc = llama_decode(E.ctx, bb);
    llama_batch_free(bb);
    llama_inject_clear(E.ctx);
    if (rc != 0) return false;
    E.kv_pos += b.n;
    ++E.blocks_fed;
    return true;
}

static bool proto_feed_tokens(engine_t & E, const std::vector<llama_token> & ids, bool logits_last) {
    llama_inject_clear(E.ctx);
    if (!proto_decode(E.ctx, ids, E.kv_pos, logits_last, (int) llama_n_batch(E.ctx))) return false;
    E.kv_pos += (int) ids.size();
    return true;
}

static bool proto_recall_end(const std::string & seg, std::string & key) {
    static const std::string T = "<recall>";
    size_t e = seg.size();
    while (e > 0 && std::isspace((unsigned char) seg[e - 1])) --e;
    if (e < 2 * T.size() + 1 || seg.compare(e - T.size(), T.size(), T) != 0) return false;
    const size_t close = e - T.size();
    const size_t open = seg.rfind(T, close - T.size());
    if (open == std::string::npos || open + T.size() >= close) return false;
    const std::string k = seg.substr(open + T.size(), close - open - T.size());
    if (k.size() > 80 || k.find('<') != std::string::npos || k.find('\n') != std::string::npos) return false;
    key = proto_trim(k);
    return !key.empty();
}

static bool proto_parse_tool(const std::string & said, std::string & name, std::map<std::string, std::string> & params) {
    const size_t a = said.find("<tool_call>");
    if (a == std::string::npos) return false;
    const size_t z = said.find("</tool_call>", a);
    if (z == std::string::npos) return false;
    const std::string body = said.substr(a, z - a);
    const size_t f = body.find("<function=");
    if (f == std::string::npos) return false;
    const size_t fe = body.find('>', f);
    if (fe == std::string::npos) return false;
    name = proto_trim(body.substr(f + 10, fe - f - 10));
    size_t p = fe;
    while ((p = body.find("<parameter=", p)) != std::string::npos) {
        const size_t pe = body.find('>', p);
        if (pe == std::string::npos) break;
        const size_t ve = body.find("</parameter>", pe);
        if (ve == std::string::npos) break;
        params[proto_trim(body.substr(p + 11, pe - p - 11))] = proto_trim(body.substr(pe + 1, ve - pe - 1));
        p = ve;
    }
    return !name.empty();
}

static std::string proto_run_tool(const std::string & name, const std::map<std::string, std::string> & params) {
    auto get = [&](const char * k) { auto it = params.find(k); return it == params.end() ? std::string() : it->second; };
    if (name == "tool_docs") {
        auto it = PROTO.docs.find(get("name"));
        return it != PROTO.docs.end() ? it->second : proto_dump({ { "error", "no documentation for " + get("name") } });
    }
    if (name == "lookup_verse" || name == "lookup_strongs") {
        std::string ref = get("reference"); if (ref.empty()) ref = get("number");
        auto it = PROTO.lookup.find(ref);
        return it != PROTO.lookup.end() ? it->second : proto_dump({ { "error", "nothing found for " + ref } });
    }
    if (name == "read_file") {
        const std::string path = get("path");
        if (!PROTO.files_dir.empty() && path.find("..") == std::string::npos) {
            std::string text = proto_read_file(PROTO.files_dir + "/" + path);
            if (!text.empty()) return text.substr(0, 8000);
        }
        return proto_dump({ { "note", "no file system is connected for this path" }, { "path", path } });
    }
    return proto_dump({ { "note", "this build does not connect " + name + " to a real service" }, { "arguments", params } });
}

// ---------------------------------------------------------------- storing, queuing, sleeping
static void proto_add_cell(engine_t & E, const std::string & key, const std::string & context, const std::string & summary, const std::string & source) {
    cell_t c; c.key = key; c.ctx = context; c.summary = summary; c.text = summary.empty() ? context : summary; c.table = source;
    E.cells.push_back(c);
    if (!PROTO.memory_out.empty() && !summary.empty()) {
        std::ofstream f(PROTO.memory_out, std::ios::app);
        f << proto_dump({ { "key", key }, { "context", context }, { "summary", summary }, { "source", source } }) << "\n";
    }
    proto_menu_add(E, key, summary);                  // 0019: it can come to mind from the next message on
}

// store now, and capture its states now so the first recall does not pay for them
static bool proto_store(engine_t & E, const std::string & key, const std::string & context, const std::string & summary, const std::string & source) {
    if (summary.empty()) return false;
    proto_add_cell(E, key, context, summary, source);
    return capture_twoside(E, E.cells.back());
}

static json proto_sleep(engine_t & E, int limit, const std::function<bool(const json &)> & emit, std::atomic<bool> * stop) {
    const auto t0 = std::chrono::steady_clock::now();
    int stored = 0;
    emit({ { "type", "sleep" }, { "queued", (int) PROTO.queue.size() } });
    while (!PROTO.queue.empty() && (limit <= 0 || stored < limit)) {
        if (stop && stop->load()) break;
        json item = PROTO.queue.front();
        PROTO.queue.erase(PROTO.queue.begin());
        const std::string key = item.value("key", ""), ctx = item.value("context", ""), src = item.value("source", "sleep");
        std::string summary = item.value("summary", "");
        if (summary.empty()) summary = sleep_write(E, key, ctx);
        if (summary.empty() || !proto_store(E, key, ctx, summary, src)) { emit({ { "type", "failed" }, { "key", key }, { "reason", summary.empty() ? "the model wrote nothing" : "the capture failed" } }); continue; }
        ++stored;
        emit({ { "type", "wrote" }, { "key", key }, { "summary", summary } });
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    json done = { { "type", "done" }, { "stored", stored }, { "left", (int) PROTO.queue.size() }, { "seconds", std::round(secs * 10.0) / 10.0 } };
    emit(done);
    return done;
}

static json proto_learn(engine_t & E, const std::string & name, const std::string & content, const std::string & fmt) {
    int skipped = 0, stored = 0;
    std::vector<json> items = proto_parse_upload(name, content, fmt, skipped);
    for (auto & it : items) {
        if (!it.value("summary", "").empty()) { if (proto_store(E, it["key"], it["context"], it["summary"], it["source"])) ++stored; }
        else PROTO.queue.push_back(it);
    }
    return { { "queued", (int) items.size() - stored }, { "stored", stored }, { "skipped", skipped }, { "waiting", (int) PROTO.queue.size() } };
}

static json proto_remember_request(engine_t & E, std::string key, const std::string & context, std::string summary, bool now) {
    if (context.empty() && summary.empty()) return { { "error", "nothing to remember: send a context or a summary" } };
    if (key.empty()) key = proto_default_key(context.empty() ? summary : context);
    if (summary.empty() && !now) {
        PROTO.queue.push_back({ { "key", key }, { "context", context }, { "summary", "" }, { "source", "remember" } });
        return { { "status", "queued" }, { "key", key }, { "queued", (int) PROTO.queue.size() } };
    }
    const bool wrote = summary.empty();
    if (wrote) summary = sleep_write(E, key, context);
    if (summary.empty() || !proto_store(E, key, context, summary, "remember")) return { { "error", "the model wrote nothing for that" } };
    return { { "status", "stored" }, { "key", key }, { "summary", summary }, { "wrote", wrote ? key + " is " + summary : std::string() } };
}

static void proto_marks(engine_t & E, const std::string & answer, const std::string & context, const std::function<bool(const json &)> & emit) {
    static const std::string R = "<|remember|>", A = "<|alias|>";
    for (size_t p = 0; (p = answer.find(R, p)) != std::string::npos;) {
        const size_t q = answer.find(R, p + R.size());
        if (q == std::string::npos) break;
        const std::string body = proto_trim(answer.substr(p + R.size(), q - p - R.size()));
        const size_t is = body.find(" is ");
        if (is != std::string::npos && !proto_trim(body.substr(0, is)).empty()) {
            const std::string key = proto_trim(body.substr(0, is)), summary = proto_trim(body.substr(is + 4));
            if (PROTO.remember_mode == "queue") {
                PROTO.queue.push_back({ { "key", key }, { "context", context }, { "summary", summary }, { "source", "remembered" } });
                emit({ { "type", "remember" }, { "key", key }, { "summary", summary }, { "status", "queued" } });
            } else {
                proto_add_cell(E, key, context, summary, "remembered");
                emit({ { "type", "remember" }, { "key", key }, { "summary", summary }, { "status", "stored" } });
            }
        }
        p = q + R.size();
    }
    for (size_t p = 0; (p = answer.find(A, p)) != std::string::npos;) {
        const size_t q = answer.find(A, p + A.size());
        if (q == std::string::npos) break;
        const std::string body = answer.substr(p + A.size(), q - p - A.size());
        const size_t arrow = body.find("->");
        if (arrow != std::string::npos) {
            const std::string bad = proto_trim(body.substr(0, arrow)), good = proto_trim(body.substr(arrow + 2));
            if (!bad.empty() && !good.empty()) { PROTO.alias[bad] = good; emit({ { "type", "alias" }, { "from", bad }, { "to", good } }); }
        }
        p = q + A.size();
    }
}

static std::string proto_chat_prompt(const json & messages, const std::string & system) {
    std::string p = system.empty() ? std::string() : "<|im_start|>system\n" + system + "<|im_end|>\n";
    for (size_t i = 0; i + 1 < messages.size(); ++i)
        p += "<|im_start|>" + messages[i].value("role", "user") + "\n" + messages[i].value("content", "") + "<|im_end|>\n";
    p += "<|im_start|>user\n" + (messages.empty() ? std::string() : messages.back().value("content", "")) + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    return p;
}

// ---------------------------------------------------------------- auto-inject: memory right after the surprising word (0021)
// expert/assoc_insert.py on the step-600 checkpoint: a memory's block placed right after the word in the user's message,
// with no <recall>, is read -- the whole block almost as well as a recall, 8 rows partly, 2 rows not at all.
struct proto_inject_cfg_t {
    int rows = 0, max = 3;                          // --auto-inject N rows per injected memory (0 = off), --inject-max words per message
    double surprise = 8.0, min_score = 0.6, margin = 0.05;
    std::string match = "state";                    // key: a stored key named in the message; state: also a clear hidden-state match
};
static proto_inject_cfg_t INJECT;
struct proto_inject_t { size_t at = 0; std::string word, key, match; double surprise = 0.0, score = 0.0; engine_t::block_t b; };

static bool proto_word_char(unsigned char c) { return std::isalnum(c) || c >= 0x80 || (c && std::strchr("_./:@-", c)); }

// a block of at most `rows` rows (head and marks included): evenly spaced states, as assoc_insert.py measured
static engine_t::block_t block_twoside_rows(engine_t & E, cell_t & c, int rows) {
    const int extra = (E.side.head ? 1 : 0) + (E.side.mem_marks ? 2 : 0), saved = E.side.mem_seq;
    if (rows > 0) E.side.mem_seq = std::min(saved, std::max(1, rows - extra));
    engine_t::block_t b = block_twoside(E, c);
    E.side.mem_seq = saved;
    return b;
}

// where memory goes into the last user message: the most surprising words that name a stored key, or (match state)
// whose own states match one memory clearly, each with up to `rows` rows of that memory's block; in message order
static std::vector<proto_inject_t> proto_plan_injects(engine_t & E, const std::string & message, int rows, const std::function<bool(const json &)> & emit) {
    std::vector<proto_inject_t> plan;
    if (rows <= 0 || INJECT.max <= 0 || message.empty() || E.cells.empty()) return plan;
    const bool by_state = INJECT.match == "state" && MENU.layer >= 1;
    if (by_state && !MENU.built) proto_menu_build(E);
    const std::string open = "<|im_start|>user\n";
    std::vector<llama_token> ids = E.tok(open + message, true);
    if (ids.size() > 256) ids.resize(256);
    proto_cap_ready(E);                                   // one pass: every token's logits, and the menu layer's states
    const std::string lname = "l_out-" + std::to_string(MENU.layer - 1);
    if (by_state) E.capd.want.assign({ lname });
    {
        llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) { b.token[i] = ids[i]; b.pos[i] = (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = (int) ids.size();
        const int rc = llama_decode(E.cap, b);
        llama_batch_free(b);
        if (rc != 0) { E.capd.want.clear(); return plan; }
    }
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    std::vector<double> nll(ids.size(), 0.0);
    std::vector<size_t> start(ids.size() + 1, 0);        // byte offset of each token in open + message
    for (size_t i = 0; i < ids.size(); ++i) start[i + 1] = start[i] + E.piece(ids[i]).size();
    for (size_t i = 1; i < ids.size(); ++i) {
        const float * lg = llama_get_logits_ith(E.cap, (int) i - 1);
        if (!lg) continue;
        double mx = lg[0];
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > mx) mx = lg[v];
        double z = 0.0;
        for (int v = 0; v < n_vocab; ++v) z += std::exp((double) lg[v] - mx);
        nll[i] = mx + std::log(z) - (double) lg[ids[i]];
    }
    std::vector<float> hs;
    if (by_state && E.capd.got.count(lname) && E.capd.got[lname].size() == ids.size() * (size_t) E.d) hs = E.capd.got[lname];
    E.capd.want.clear();
    const size_t p0 = open.size();
    std::vector<size_t> ts;
    auto span = [&](size_t a, size_t b) {                 // the tokens overlapping message bytes [a, b), and their summed -log p
        ts.clear();
        double s = 0.0;
        for (size_t i = 1; i < ids.size(); ++i) if (start[i + 1] > start[i] && start[i] < p0 + b && start[i + 1] > p0 + a) { ts.push_back(i); s += nll[i]; }
        return s;
    };
    std::vector<proto_inject_t> cands;
    std::vector<std::pair<size_t, size_t>> taken;
    auto overlaps = [&](size_t a, size_t b) { for (auto & t : taken) if (a < t.second && b > t.first) return true; return false; };
    std::vector<std::string> names;                       // 1. words that name a stored key or an alias, longest first
    {
        std::set<std::string> s;
        for (auto & c : E.cells) s.insert(c.key);
        for (auto & kv : PROTO.alias) s.insert(kv.first);
        names.assign(s.begin(), s.end());
    }
    std::sort(names.begin(), names.end(), [](const std::string & x, const std::string & y) { return x.size() > y.size(); });
    auto wc = [](unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; };
    for (auto & name : names) {
        if (name.size() < 2) continue;
        for (size_t j = message.find(name); j != std::string::npos; j = message.find(name, j + 1)) {
            const size_t e = j + name.size();
            const bool before_ok = j == 0 || !wc((unsigned char) message[j - 1]);
            const bool after_ok = e >= message.size() || (!wc((unsigned char) message[e]) && !(message[e] == '.' && e + 1 < message.size() && std::isalnum((unsigned char) message[e + 1])));
            if (!before_ok || !after_ok || overlaps(j, e)) continue;
            taken.push_back({ j, e });
            const double s = span(j, e);
            if (ts.empty() || s < INJECT.surprise) continue;
            proto_inject_t c;
            c.at = e; c.word = name; c.key = name; c.match = "key"; c.surprise = s; c.score = 1.0;
            auto al = PROTO.alias.find(name);
            if (al != PROTO.alias.end()) c.key = al->second;
            cands.push_back(c);
        }
    }
    if (by_state && !hs.empty() && MENU.fitted) {         // 2. other surprising words whose states match one memory clearly
        size_t i = 0;
        while (i < message.size()) {
            if (!proto_word_char((unsigned char) message[i])) { ++i; continue; }
            size_t a = i;
            while (i < message.size() && proto_word_char((unsigned char) message[i])) ++i;
            size_t b = i;
            while (b > a && std::strchr(".,;:-/", message[b - 1])) --b;
            while (a < b && std::strchr(".,;:-/", message[a])) ++a;
            if (b - a < 2 || overlaps(a, b)) continue;
            std::string lw = message.substr(a, b - a);
            std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
            if (proto_stop().count(lw)) continue;
            const double s = span(a, b);
            if (ts.empty() || s < INJECT.surprise) continue;
            std::vector<float> q;
            std::vector<llama_token> qids;
            for (size_t t : ts) { q.insert(q.end(), hs.begin() + (ptrdiff_t) (t * (size_t) E.d), hs.begin() + (ptrdiff_t) ((t + 1) * (size_t) E.d)); qids.push_back(ids[t]); }
            proto_menu_norm(q, E.d);
            const proto_menu_hit_t h = proto_menu_rank(E, q, qids, 2);
            if (h.keys.empty() || h.scores[0] < INJECT.min_score || (h.scores.size() > 1 && h.scores[0] - h.scores[1] < INJECT.margin)) continue;
            taken.push_back({ a, b });
            proto_inject_t c;
            c.at = b; c.word = message.substr(a, b - a); c.key = h.keys[0]; c.match = "state"; c.surprise = s; c.score = h.scores[0];
            cands.push_back(c);
        }
    }
    std::sort(cands.begin(), cands.end(), [](const proto_inject_t & x, const proto_inject_t & y) { return x.surprise > y.surprise; });
    for (auto & c : cands) {
        if ((int) plan.size() >= INJECT.max) break;
        for (int i = (int) E.cells.size() - 1; i >= 0; --i) {          // the newest mention of the key
            if (E.cells[(size_t) i].key != c.key) continue;
            c.b = block_twoside_rows(E, E.cells[(size_t) i], rows);
            break;
        }
        if (c.b.n <= 0) continue;
        emit({ { "type", "inject" }, { "word", c.word }, { "key", c.key }, { "rows", c.b.n }, { "match", c.match },
               { "surprise", std::round(c.surprise * 100.0) / 100.0 }, { "score", std::round(c.score * 1000.0) / 1000.0 } });
        plan.push_back(c);
    }
    std::sort(plan.begin(), plan.end(), [](const proto_inject_t & x, const proto_inject_t & y) { return x.at < y.at; });
    return plan;
}

// ONE ANSWER: the loop the checkpoint was trained and evaluated with. injects (0021): blocks that go right after words
// of the last user message, which sits just before the assistant's opening at the end of `prompt`
static json run_protocol(engine_t & E, const std::string & prompt, const std::string & last_user, int max_new,
                         const std::function<bool(const json &)> & emit, std::atomic<bool> * stop,
                         std::vector<proto_inject_t> * injects = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    E.reset();
    const std::vector<llama_token> im_end_ids = E.tok("<|im_end|>", true);
    const llama_token IM_END = im_end_ids.size() == 1 ? im_end_ids[0] : -1;
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    emit({ { "type", "start" } });
    std::string seg, turn, answer, context = last_user;
    int ntok = 0, rounds = 0;
    bool finished = false, ok = true;
    {                                                     // 0021: the prompt in pieces, a memory block after each injected word
        static const std::string tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        size_t pos = 0;
        const size_t base = prompt.size() >= tail.size() + last_user.size() ? prompt.size() - tail.size() - last_user.size() : std::string::npos;
        if (injects && base != std::string::npos && prompt.compare(base, last_user.size(), last_user) == 0) {
            for (auto & inj : *injects) {
                const size_t at = base + inj.at;
                if (!ok || at <= pos || at > base + last_user.size()) continue;
                ok = proto_feed_tokens(E, E.tok(prompt.substr(pos, at - pos), true), false) && proto_feed_block(E, inj.b);
                pos = at;
            }
        }
        ok = ok && proto_feed_tokens(E, E.tok(prompt.substr(pos), true), true);
    }
    for (int step = 0; ok && step < max_new; ++step) {
        if (stop && stop->load()) break;
        const float * lg = llama_get_logits_ith(E.ctx, -1);
        int best = 0; for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        ++ntok;
        if (llama_vocab_is_eog(E.vocab, best) || best == IM_END) {
            turn += seg; seg.clear();
            const size_t tp = turn.rfind("</think>");
            const std::string said = tp == std::string::npos ? turn : turn.substr(tp + 8);
            std::string name; std::map<std::string, std::string> params;
            if (rounds < 3 && proto_parse_tool(said, name, params)) {
                ++rounds;
                const std::string result = proto_run_tool(name, params);
                context = result;
                emit({ { "type", "tool" }, { "name", name }, { "args", params }, { "result", result } });
                std::vector<llama_token> rep = { IM_END };
                const auto more = E.tok("\n<|im_start|>user\n<tool_response>\n" + result + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n", true);
                rep.insert(rep.end(), more.begin(), more.end());
                ok = proto_feed_tokens(E, rep, true);
                emit({ { "type", "turn" } });
                turn.clear();
                continue;
            }
            answer = said; finished = true;
            break;
        }
        const std::string pc = E.piece(best);
        seg += pc;
        if (!emit({ { "type", "token" }, { "text", pc } })) break;
        std::string key;
        if (proto_recall_end(seg, key)) {
            ok = proto_feed_tokens(E, { best }, false);
            if (!ok) break;
            proto_hit_t h = proto_lookup(E, key);
            json ev = { { "type", "recall" }, { "asked", key }, { "hit", h.hit }, { "rows", h.b.n } };
            if (h.hit) { ev["key"] = h.key; ev["summary"] = h.summary; } else { ev["near"] = h.near; }
            emit(ev);
            ok = h.b.n > 0 && proto_feed_block(E, h.b);
            turn += seg; seg.clear();
        } else {
            ok = proto_feed_tokens(E, { best }, true);
        }
        if (E.kv_pos + 8 >= (int) llama_n_ctx(E.ctx)) break;
    }
    if (!finished) { turn += seg; const size_t tp = turn.rfind("</think>"); answer = tp == std::string::npos ? std::string() : turn.substr(tp + 8); }
    proto_marks(E, answer, context, emit);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    json done = { { "type", "done" }, { "tokens", ntok }, { "seconds", std::round(secs * 10.0) / 10.0 }, { "answer", proto_trim(answer) }, { "blocks", E.blocks_fed } };
    emit(done);
    return done;
}

static json proto_state(engine_t & E, const std::string & model_name) {
    std::set<std::string> keys; for (auto & c : E.cells) keys.insert(c.key);
    return { { "step", E.side.step }, { "keys", (int) keys.size() }, { "memories", (int) E.cells.size() }, { "queued", (int) PROTO.queue.size() },
             { "manuals", (int) PROTO.docs.size() }, { "lookup", (int) PROTO.lookup.size() }, { "examples", PROTO.examples },
             { "device", E.ngl > 0 ? "llama.cpp, " + std::to_string(E.ngl) + " GPU layers" : std::string("llama.cpp, CPU") },
             { "dtype", model_name }, { "layers", E.side.layers }, { "remember_mode", PROTO.remember_mode }, { "auto_menu", MENU.k },
             { "auto_inject", INJECT.rows }, { "inject_max", INJECT.max }, { "inject_match", INJECT.match }, { "inject_surprise", INJECT.surprise } };
}

static json proto_memories(engine_t & E, std::string q, size_t limit = 150) {
    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
    json out = json::array(); std::set<std::string> seen;
    for (int i = (int) E.cells.size() - 1; i >= 0 && out.size() < limit; --i) {
        auto & c = E.cells[(size_t) i];
        if (seen.count(c.key)) continue;
        seen.insert(c.key);
        std::string hay = c.key + " " + c.summary; std::transform(hay.begin(), hay.end(), hay.begin(), ::tolower);
        if (q.empty() || hay.find(q) != std::string::npos) out.push_back({ { "key", c.key }, { "summary", c.summary }, { "source", c.table } });
    }
    json al = json::object(); for (auto & kv : PROTO.alias) al[kv.first] = kv.second;
    return { { "memories", out }, { "aliases", al } };
}

// ---------------------------------------------------------------- the HTTP API (demo/API.md)
static void register_protocol_routes(httplib::Server & srv, engine_t & E, std::mutex & mu, std::atomic<bool> & stop_flag, const std::string & model_name) {
    auto js = [](httplib::Response & res, const json & j, int status = 200) { res.status = status; res.set_content(proto_dump(j), "application/json"); };
    auto parse = [](const httplib::Request & req, json & body) { try { body = json::parse(req.body.empty() ? std::string("{}") : req.body); return true; } catch (...) { return false; } };
    srv.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(PROTO.ui_html.empty() ? std::string("<p>steermem protocol mode is running. Start it with --ui demo_ui.html for the page.</p>") : PROTO.ui_html, "text/html; charset=utf-8");
    });
    srv.Get("/api/state", [&E, &mu, js, model_name](const httplib::Request &, httplib::Response & res) { std::lock_guard<std::mutex> lk(mu); js(res, proto_state(E, model_name)); });
    srv.Get("/api/memories", [&E, &mu, js](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(mu);
        js(res, proto_memories(E, req.has_param("q") ? req.get_param_value("q") : std::string()));
    });
    srv.Get("/api/queue", [&mu, js](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(mu);
        json q = json::array();
        for (auto & it : PROTO.queue) q.push_back({ { "key", it.value("key", "") }, { "chars", (int) it.value("context", "").size() }, { "source", it.value("source", "") } });
        js(res, { { "queued", q } });
    });
    srv.Post("/api/stop", [&stop_flag, js](const httplib::Request &, httplib::Response & res) { stop_flag = true; js(res, { { "ok", true } }); });
    auto remember = [&E, &mu, js, parse](const httplib::Request & req, httplib::Response & res, bool force_now) {
        json body; if (!parse(req, body)) { js(res, { { "error", "the request body is not JSON" } }, 400); return; }
        std::lock_guard<std::mutex> lk(mu);
        json r = proto_remember_request(E, proto_trim(body.value("key", "")), proto_trim(body.value("context", "")), proto_trim(body.value("summary", "")), force_now || body.value("now", false));
        js(res, r, r.contains("error") ? 400 : 200);
    };
    srv.Post("/api/remember", [remember](const httplib::Request & req, httplib::Response & res) { remember(req, res, false); });
    srv.Post("/api/memory", [remember](const httplib::Request & req, httplib::Response & res) { remember(req, res, true); });
    srv.Post("/api/upload", [&E, &mu, js, parse](const httplib::Request & req, httplib::Response & res) {
        json body; if (!parse(req, body)) { js(res, { { "error", "the request body is not JSON" } }, 400); return; }
        std::lock_guard<std::mutex> lk(mu);
        try { js(res, proto_learn(E, body.value("name", "upload.txt"), body.value("content", ""), body.value("format", ""))); }
        catch (const std::exception & e) { js(res, { { "error", std::string("could not read that file: ") + e.what() } }, 400); }
    });
    srv.Post("/api/sleep", [&E, &mu, &stop_flag, parse](const httplib::Request & req, httplib::Response & res) {
        json body; parse(req, body);
        const int limit = body.value("limit", 0);
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, limit](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            stop_flag = false;
            auto emit = [&](const json & j) { const std::string s = proto_dump(j) + "\n"; return sink.write(s.data(), s.size()); };
            proto_sleep(E, limit, emit, &stop_flag);
            sink.done();
            return true;
        });
    });
    srv.Post("/api/chat", [&E, &mu, &stop_flag, js, parse](const httplib::Request & req, httplib::Response & res) {
        json body; if (!parse(req, body)) { js(res, { { "error", "the request body is not JSON" } }, 400); return; }
        json msgs = json::array();
        for (auto & m : body.value("messages", json::array())) if (!m.value("content", "").empty()) msgs.push_back(m);
        if (msgs.empty() || msgs.back().value("role", "") != "user") { js(res, { { "error", "the last message has to be the user's" } }, 400); return; }
        const std::string system = body.value("system", "");
        const std::string last = msgs.back().value("content", "");
        const int max_new = body.value("max_new", 700);
        const int auto_menu = body.contains("auto_menu") && body["auto_menu"].is_number() ? body["auto_menu"].get<int>() : -1;   // 0019: -1 = the --auto-menu default
        const int auto_inject = body.contains("auto_inject") && body["auto_inject"].is_number() ? body["auto_inject"].get<int>() : -1;   // 0021: -1 = --auto-inject
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, msgs, system, last, max_new, auto_menu, auto_inject](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            stop_flag = false;
            auto emit = [&](const json & j) { const std::string s = proto_dump(j) + "\n"; return sink.write(s.data(), s.size()); };
            const std::string sys = proto_apply_menu(E, system, last, auto_menu < 0 ? MENU.k : auto_menu, emit);   // 0019: the menu goes out first
            std::vector<proto_inject_t> injects = proto_plan_injects(E, last, auto_inject < 0 ? INJECT.rows : auto_inject, emit);   // 0021
            run_protocol(E, proto_chat_prompt(msgs, sys), last, max_new, emit, &stop_flag, &injects);
            sink.done();
            return true;
        });
    });
}

// ---------------------------------------------------------------- --chat: the same, at the command line
static void proto_print_event(const json & ev) {
    const std::string t = ev.value("type", "");
    if (t == "token") { fputs(ev.value("text", "").c_str(), stdout); fflush(stdout); return; }
    if (t == "menu") {                                  // 0019
        std::string ks;
        for (auto & k : ev.value("keys", json::array())) ks += (ks.empty() ? "" : ", ") + k.get<std::string>();
        printf("  [came to mind: %s]\n", ks.c_str());
    } else if (t == "inject") {                         // 0021
        printf("  [injected after \"%s\": %s, %d memory tokens (surprise %.1f nats, %s match)]\n", ev.value("word", "").c_str(), ev.value("key", "").c_str(),
               ev.value("rows", 0), ev.value("surprise", 0.0), ev.value("match", "").c_str());
    } else if (t == "recall") {
        if (ev.value("hit", false)) printf("\n  [recall %s -> %s]\n", ev.value("asked", "").c_str(), ev.value("summary", "").substr(0, 160).c_str());
        else { std::string near; for (auto & k : ev.value("near", json::array())) near += (near.empty() ? "" : ", ") + k.get<std::string>(); printf("\n  [recall %s -> nothing stored; nearest: %s]\n", ev.value("asked", "").c_str(), near.c_str()); }
    } else if (t == "tool") {
        std::string r = ev.value("result", ""); for (auto & ch : r) if (ch == '\n') ch = ' ';
        printf("\n  [tool %s(%s) -> %s]\n", ev.value("name", "").c_str(), proto_dump(ev.value("args", json::object())).c_str(), r.substr(0, 200).c_str());
    } else if (t == "remember") printf("\n  [%s: %s is %s]\n", ev.value("status", "stored").c_str(), ev.value("key", "").c_str(), ev.value("summary", "").c_str());
    else if (t == "alias") printf("\n  [alias: %s -> %s]\n", ev.value("from", "").c_str(), ev.value("to", "").c_str());
    else if (t == "wrote") printf("  learned: %s is %s\n", ev.value("key", "").c_str(), ev.value("summary", "").c_str());
    else if (t == "failed") printf("  could not learn %s: %s\n", ev.value("key", "").c_str(), ev.value("reason", "").c_str());
    else if (t == "done" && ev.contains("tokens")) printf("\n  (%d tokens, %.1fs, %d blocks)\n", ev.value("tokens", 0), ev.value("seconds", 0.0), ev.value("blocks", 0));
    else if (t == "done") printf("slept: %d stored, %d left, %.1fs\n", ev.value("stored", 0), ev.value("left", 0), ev.value("seconds", 0.0));
    fflush(stdout);
}

static const char * PROTO_CHAT_HELP =
    "  anything else            a message to the model (it recalls, reads and uses tools on its own)\n"
    "  /upload FILE             learn a CSV, JSON, JSONL or text file at the next sleep\n"
    "  /remember [KEY ::] TEXT  queue something to remember at the next sleep\n"
    "  /remember-now [KEY ::] TEXT   the model writes that memory now\n"
    "  /sleep [N]               sleep: write and store what is queued (at most N)\n"
    "  /queue                   what is waiting for sleep\n"
    "  /memories [QUERY]        what is stored\n"
    "  /alias BAD -> GOOD       another name for a stored key\n"
    "  /system TEXT             a system prompt (empty to clear), e.g. the keys worth recalling and when\n"
    "  /mode store|queue        what the model's own <|remember|> does\n"
    "  /menu K                  put the K memories a message brings to mind in the system prompt (0 is off)\n"
    "  /inject N                up to N memory tokens after each surprising word that names a memory (0 is off)\n"
    "  /new                     a new conversation\n"
    "  /state                   the table, the queue, the settings\n"
    "  /quit\n";

static int run_chat_repl(engine_t & E, int max_new, std::string system, const std::string & model_name) {
#ifdef _WIN32
    SetConsoleOutputCP(65001); SetConsoleCP(65001);
#endif
    json state = proto_state(E, model_name);
    printf("steermem chat | checkpoint step %d | %d keys, %d waiting for sleep | remember mode %s | /help for commands\n",
           state.value("step", 0), state.value("keys", 0), state.value("queued", 0), PROTO.remember_mode.c_str());
    json history = json::array();
    std::string line;
    auto split_key = [](const std::string & rest, std::string & key, std::string & text) {
        const size_t p = rest.find("::");
        if (p == std::string::npos) { key.clear(); text = proto_trim(rest); } else { key = proto_trim(rest.substr(0, p)); text = proto_trim(rest.substr(p + 2)); }
    };
    auto printer = [](const json & ev) { proto_print_event(ev); return true; };
    while (true) {
        printf("\n> "); fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        line = proto_trim(line);
        if (line.empty()) continue;
        const size_t sp = line.find(' ');
        const std::string cmd = line[0] == '/' ? line.substr(0, sp) : std::string();
        const std::string rest = sp == std::string::npos ? std::string() : proto_trim(line.substr(sp + 1));
        if (cmd == "/quit" || cmd == "/exit") break;
        else if (cmd == "/help") fputs(PROTO_CHAT_HELP, stdout);
        else if (cmd == "/new") { history = json::array(); printf("new conversation\n"); }
        else if (cmd == "/system") { system = rest; printf("system prompt %s\n", system.empty() ? "cleared" : "set"); }
        else if (cmd == "/mode") { if (rest == "store" || rest == "queue") { PROTO.remember_mode = rest; printf("remember mode %s\n", rest.c_str()); } else printf("/mode store or /mode queue\n"); }
        else if (cmd == "/state") { json s = proto_state(E, model_name); s.erase("examples"); printf("%s\n", s.dump(2, ' ', false, json::error_handler_t::replace).c_str()); }
        else if (cmd == "/upload") {
            const std::string content = proto_read_file(rest);
            if (content.empty()) { printf("cannot read %s\n", rest.c_str()); continue; }
            const std::string name = rest.substr(rest.find_last_of("/\\") + 1);
            try { json r = proto_learn(E, name, content, ""); printf("%s: %d to learn at sleep, %d stored now, %d skipped\n", name.c_str(), r["queued"].get<int>(), r["stored"].get<int>(), r["skipped"].get<int>()); }
            catch (const std::exception & e) { printf("could not read that file: %s\n", e.what()); }
        }
        else if (cmd == "/remember" || cmd == "/remember-now") {
            std::string key, text; split_key(rest, key, text);
            json r = proto_remember_request(E, key, text, "", cmd == "/remember-now");
            if (r.contains("error")) printf("%s\n", r["error"].get<std::string>().c_str());
            else printf("%s: %s%s\n", r["status"].get<std::string>().c_str(), r["key"].get<std::string>().c_str(), r.contains("summary") ? (" is " + r["summary"].get<std::string>()).c_str() : "");
        }
        else if (cmd == "/sleep") proto_sleep(E, rest.empty() ? 0 : atoi(rest.c_str()), printer, nullptr);
        else if (cmd == "/queue") { printf("%zu waiting for sleep\n", PROTO.queue.size()); for (size_t i = 0; i < PROTO.queue.size() && i < 30; ++i) printf("  %s (%zu chars, from %s)\n", PROTO.queue[i].value("key", "").c_str(), PROTO.queue[i].value("context", "").size(), PROTO.queue[i].value("source", "").c_str()); }
        else if (cmd == "/memories") { json m = proto_memories(E, rest, 30); for (auto & it : m["memories"]) printf("  %s: %s\n", it.value("key", "").c_str(), it.value("summary", "").substr(0, 120).c_str()); printf("(%zu shown)\n", m["memories"].size()); }
        else if (cmd == "/alias") {
            const size_t arrow = rest.find("->");
            if (arrow == std::string::npos) { printf("/alias BAD -> GOOD\n"); continue; }
            PROTO.alias[proto_trim(rest.substr(0, arrow))] = proto_trim(rest.substr(arrow + 2));
            printf("alias stored\n");
        }
        else if (cmd == "/menu") { MENU.k = std::max(0, atoi(rest.c_str())); printf("auto menu %d\n", MENU.k); }   // 0019
        else if (cmd == "/inject") { INJECT.rows = std::max(0, atoi(rest.c_str())); printf("auto inject %d\n", INJECT.rows); }   // 0021
        else if (!cmd.empty()) printf("unknown command %s -- /help\n", cmd.c_str());
        else {
            history.push_back({ { "role", "user" }, { "content", line } });
            const std::string sys = proto_apply_menu(E, system, line, MENU.k, printer);   // 0019: what this message brings to mind
            std::vector<proto_inject_t> injects = proto_plan_injects(E, line, INJECT.rows, printer);   // 0021
            json done = run_protocol(E, proto_chat_prompt(history, sys), line, max_new, printer, nullptr, &injects);
            history.push_back({ { "role", "assistant" }, { "content", done.value("answer", "") } });
        }
    }
    return 0;
}

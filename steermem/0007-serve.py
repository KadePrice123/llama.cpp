#!/usr/bin/env python3
"""steermem sidecar: `--serve PORT` -- a resident HTTP server with the chat panel's API and a built-in page.

Kade, 2026-09-12 18:45: "set up my llama.cpp on my actual desktop with the chat panel
so I can test the models separate from the AI server" and "at work I can't use WSL,
so I need a Windows fork that runs on base Windows". So the server lives in the C++
binary (no Python at the venue): the same endpoints the trainer's serve mode offers
(GET /, POST /preset, /table, /chat with NDJSON streaming, /stop), plus GET /ui, a
single-page chat client. One generation at a time (a mutex); a table swap replaces
the cells (captures are recomputed lazily, so a swap costs nothing until a key is used).

    python 0007-serve.py steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

# 1. includes
old = '#include "nlohmann/json.hpp"\n\n#include <algorithm>\n'
new = ('#include "nlohmann/json.hpp"\n'
       '#define CPPHTTPLIB_THREAD_POOL_COUNT 4\n'
       '#include "cpp-httplib/httplib.h"\n\n'
       '#include <algorithm>\n#include <atomic>\n#include <chrono>\n#include <filesystem>\n#include <functional>\n#include <mutex>\n')
assert s.count(old) == 1, "includes"
s = s.replace(old, new)

# 2. the engine: reset between requests, a table swap
old = "    void set_lora(float scale) {\n"
new = '''    // a fresh conversation: the kv, the injections, the router's text window
    void reset() {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_inject_clear(ctx);
        inj.clear(); text.clear(); kv_pos = 0; blocks_fed = 0; traces_fed = 0;
    }
    // a table swap: the cells are replaced; captures are recomputed lazily on first use
    void set_table(std::vector<cell_t> cells_) {
        cells = std::move(cells_);
        for (auto & c : cells) prep_forms(c);
    }
    void set_lora(float scale) {
'''
assert s.count(old) == 1, "engine reset"
s = s.replace(old, new)

# 3. main() -> run_prompt() + cli main + serve
i = s.find("int main(int argc, char ** argv) {")
assert i >= 0, "main"
s = s[:i]
s += r'''
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
    if ((int) hits.size() > E.side.mem_seq_cells) {
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
            if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); break; }
            llama_batch_free(b); llama_inject_clear(E.ctx);
            E.text.push_back(best); E.kv_pos += 1;
        }
        if (!fired.empty()) {
            json keys = json::array(); for (auto * c : fired) keys.push_back(c->key);
            fired_log.push_back({ { "at_text_token", (int) E.text.size() }, { "keys", keys } });
        }
        for (size_t i = 0; i < fired.size(); ++i) {
            if (std::find(cands.begin(), cands.end(), fired[i]) == cands.end()) cands.push_back(fired[i]);
            E.feed_block(*fired[i], i + 1 == fired.size());
            ++blocks_fired;
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
    srv.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        json j = { { "presets", list_presets(tables_dir) }, { "cells", (int) E.cells.size() }, { "mem_seq", E.side.mem_seq }, { "mem_gen", E.side.mem_gen },
                   { "mem_seq_cells", E.side.mem_seq_cells }, { "memtok", E.side.memtok }, { "line_rule", E.line_rule }, { "model", model_name }, { "engine", "steermem-cli (llama.cpp fork)" } };
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
                cell_t c; c.key = o.value("key", ""); c.text = o.contains("text") ? o.value("text", "") : o.value("content", ""); c.table = o.value("table", name);
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
        res.set_chunked_content_provider("application/x-ndjson", [&, prompt, max_new, memgen, memseq](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            const int mg = E.side.mem_gen, ms = E.side.mem_seq; apply(); stop_flag = false;
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
    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0;   // line_rule 0: follow the side file
    bool no_traces = false, quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if (a == "--model") model_path = next(); else if (a == "--lora") lora_path = next(); else if (a == "--side") side_path = next();
        else if (a == "--table") table_path = next(); else if (a == "--prompt") prompt = next(); else if (a == "--max-new") max_new = atoi(next().c_str());
        else if (a == "--threads") threads = atoi(next().c_str()); else if (a == "--memgen") memgen = atoi(next().c_str());
        else if (a == "--no-traces") no_traces = true; else if (a == "--quiet") quiet = true; else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--line-rule") line_rule = atoi(next().c_str());
        else if (a == "--serve") port = atoi(next().c_str()); else if (a == "--tables") tables_dir = next(); else if (a == "--host") host = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || side_path.empty() || (port == 0 && (table_path.empty() || prompt.empty()))) {
        fprintf(stderr, "usage: steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --table T --prompt P [--max-new N] [--memgen 0|1] [--line-rule 1|2] [--no-traces]\n"
                        "       steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --serve PORT [--tables DIR] [--table T] [--host 0.0.0.0]   (then open http://127.0.0.1:PORT/ui)\n");
        return 1;
    }
    for (size_t p; (p = prompt.find("\\n")) != std::string::npos;) prompt.replace(p, 2, "\n");   // literal \n in the prompt argument

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    engine_t E;
    E.side = load_side(side_path);
    E.line_rule = line_rule > 0 ? line_rule : E.side.line_rule;
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
    if (!table_path.empty()) E.set_table(load_table(table_path));
    if (!quiet) fprintf(stderr, "table %zu cells | mem_seq %d cells %d mem_gen %d memtok %d window %d line_rule %d | scale %s\n", E.cells.size(), E.side.mem_seq, E.side.mem_seq_cells, E.side.mem_gen, E.side.memtok, E.side.key_window, E.line_rule, std::to_string(E.side.scale.begin()->second).c_str());

    if (port > 0) {
        std::string mname = model_path.substr(model_path.find_last_of("/\\") + 1);
        if (!lora_path.empty()) mname += " + " + lora_path.substr(lora_path.find_last_of("/\\") + 1);
        return serve(E, host, port, tables_dir, mname, quiet);
    }
    auto r = run_prompt(E, prompt, max_new, quiet, nullptr, nullptr);
    printf("%s\n", r.summary.dump().c_str());
    llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free();
    return 0;
}
'''
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: --serve PORT (GET /, /ui; POST /preset, /table, /chat, /stop)")

"""fork edit 0018 (2026-09-14): PROTOCOL MODE -- serve twoside.py checkpoints the way they trained.

The protocol itself lives in llama/steermem_protocol.hpp (capture with the adapter on over "KEY is SUMMARY", blocks,
the null block, the <recall> splice loop, tools, remember/alias, uploads, sleep, the demo/API.md routes, --chat).
This script wires it into steermem.cpp:
  - cell_t carries the memory's own text (summary); load_table reads it and accepts rows without a text field
  - side_t reads steermem.protocol / head / keyemb / span_from_key / step, and nulltag, neartag, nullrec.<L>
    (exptag only for the old checkpoints)
  - set_table adds no core/index cells and no key forms in protocol mode
  - the header is included before run_prompt; serve() registers its routes first, so "/" is the protocol page
  - main: --tool-docs --lookup --examples --ui --memory-out --learn --remember-mode --files-dir --system --chat
    --sleep-at-start --sleep-only; the capture context gets 2048 tokens in one ubatch (cb_eval keeps only the last
    chunk of a split batch); a one-shot --prompt runs through the protocol loop

  python 0018_protocol.py path/to/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
if "0018" in s:
    raise SystemExit("0018 is already applied to %s" % p)


def once(old, new, tag):
    global s
    assert s.count(old) == 1, "%s: %d matches" % (tag, s.count(old))
    s = s.replace(old, new)


once('''    std::string ctx;                               // 0016: text read before the row when its states are captured''',
     '''    std::string ctx;                               // 0016: text read before the row when its states are captured
    std::string summary;                           // 0018: protocol mode's stored memory text ("KEY is SUMMARY")''',
     "cell_t")

once('''            c.ctx = o.value("context", "");     // 0016: a per-cell capture context
            if (!c.key.empty() && !c.text.empty()) cells.push_back(c);''',
     '''            c.ctx = o.value("context", "");     // 0016: a per-cell capture context
            c.summary = o.value("summary", "");  // 0018: protocol mode's stored memory text
            if (c.text.empty()) c.text = !c.summary.empty() ? c.summary : c.ctx;
            if (!c.key.empty() && !c.text.empty()) cells.push_back(c);''',
     "load_table")

once('''    int mem_marks = 0;                             // 0014: wrap every block in <|mem_start|> / <|mem_end|>''',
     '''    int mem_marks = 0;                             // 0014: wrap every block in <|mem_start|> / <|mem_end|>
    int protocol = 0, head = 1, keyemb_on = 1, span_key = 0, step = 0;   // 0018: a twoside.py checkpoint (the <recall> loop)
    std::vector<float> nulltag, neartag;                                  // 0018: [d], [d]
    std::map<int, std::vector<float>> nullrec;                            // 0018: layer -> [d]''',
     "side_t")

once('''    s.seqtag = tensor("seqtag"); s.exptag = tensor("exptag");''',
     '''    s.protocol = u32("steermem.protocol", 0);            // 0018
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
    }''',
     "load_side")

once('''        cells = std::move(cells_);
        add_core_cell(cells);''',
     '''        cells = std::move(cells_);
        if (side.protocol) return;                     // 0018: protocol mode looks keys up by the tag the model writes: no core/index cells, no key forms
        add_core_cell(cells);''',
     "set_table")

once('''// ---------------------------------------------------------------- one prompt, start to finish''',
     '''#include "steermem_protocol.hpp"                  // 0018: protocol mode

// ---------------------------------------------------------------- one prompt, start to finish''',
     "include")

once('''    auto send_json = [](httplib::Response & res, const json & j, int status = 200) { res.status = status; res.set_content(j.dump(), "application/json"); };''',
     '''    auto send_json = [](httplib::Response & res, const json & j, int status = 200) { res.status = status; res.set_content(j.dump(), "application/json"); };
    if (E.side.protocol) register_protocol_routes(srv, E, mu, stop_flag, model_name);   // 0018: registered first, so "/" is the protocol page''',
     "serve routes")

once('''    std::string cell_ctx;                          // 0016: read before every row at capture time''',
     '''    std::string cell_ctx;                          // 0016: read before every row at capture time
    std::string tool_docs, lookup_path, examples_path, ui_path, memory_out, remember_mode = "store", system_prompt;   // 0018
    std::vector<std::string> learn;                                                                                  // 0018
    bool chat_repl = false, sleep_at_start = false, sleep_only = false;                                              // 0018''',
     "main vars")

once('''        else if (a == "--cell-context") cell_ctx = next();''',
     '''        else if (a == "--cell-context") cell_ctx = next();
        else if (a == "--tool-docs") tool_docs = next(); else if (a == "--lookup") lookup_path = next();          // 0018
        else if (a == "--examples") examples_path = next(); else if (a == "--ui") ui_path = next();
        else if (a == "--memory-out") memory_out = next(); else if (a == "--learn") learn.push_back(next());
        else if (a == "--remember-mode") remember_mode = next(); else if (a == "--files-dir") PROTO.files_dir = next();
        else if (a == "--system") system_prompt = next();
        else if (a == "--chat") chat_repl = true; else if (a == "--sleep-at-start") sleep_at_start = true; else if (a == "--sleep-only") sleep_only = true;''',
     "main args")

once('''    if (model_path.empty() || side_path.empty() || (port == 0 && (table_path.empty() || prompt.empty()))) {''',
     '''    if (model_path.empty() || side_path.empty() || (port == 0 && !chat_repl && !sleep_only && (table_path.empty() || prompt.empty()))) {''',
     "usage condition")

once('''                        "       steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --serve PORT [--tables DIR] [--table T] [--host 0.0.0.0]   (then open http://127.0.0.1:PORT/ui)\\n");''',
     '''                        "       steermem-cli --model BASE.gguf [--lora ADAPTER.gguf] --side SIDE.gguf --serve PORT [--tables DIR] [--table T] [--host 0.0.0.0]   (then open http://127.0.0.1:PORT/ui)\\n"
                        "  protocol mode (a side file from twoside_export.py):\\n"
                        "       steermem-cli --model BASE.gguf --lora ADAPTER.gguf --side SIDE.gguf [--table memories.jsonl] [--memory-out learned.jsonl]\\n"
                        "           [--tool-docs tool_docs.jsonl] [--lookup lookup.jsonl] [--examples examples.json] [--learn FILE ...] [--remember-mode store|queue]\\n"
                        "           --chat | --serve PORT --ui demo_ui.html | --sleep-only | --prompt TEXT   [--sleep-at-start] [--system TEXT] [-ngl N]\\n");''',
     "usage text")

once('''    llama_context_params cc = cp; cc.n_ctx = 1024; cc.cb_eval = cb_capture; cc.cb_eval_user_data = &E.capd;''',
     '''    llama_context_params cc = cp; cc.n_ctx = 1024; cc.cb_eval = cb_capture; cc.cb_eval_user_data = &E.capd;
    if (E.side.protocol) { cc.n_ctx = 2048; cc.n_batch = 2048; cc.n_ubatch = 2048; }   // 0018: a capture must be one ubatch''',
     "capture context")

once('''    if (!table_path.empty()) E.set_table(load_table(table_path));''',
     '''    if (!table_path.empty()) E.set_table(load_table(table_path));
    if (E.side.protocol) {                                // 0018
        PROTO.memory_out = memory_out;
        PROTO.remember_mode = remember_mode == "queue" ? "queue" : "store";
        proto_load_files(tool_docs, lookup_path, examples_path, ui_path);
        if (!memory_out.empty() && std::filesystem::exists(memory_out)) { auto more = load_table(memory_out); E.cells.insert(E.cells.end(), more.begin(), more.end()); }
        for (auto & path : learn) {
            const std::string name = path.substr(path.find_last_of("/\\\\") + 1);
            try { json r = proto_learn(E, name, proto_read_file(path), ""); fprintf(stderr, "learn %s: %d queued, %d stored, %d skipped\\n", path.c_str(), r["queued"].get<int>(), r["stored"].get<int>(), r["skipped"].get<int>()); }
            catch (const std::exception & e) { fprintf(stderr, "learn %s: %s\\n", path.c_str(), e.what()); }
        }
        if (sleep_at_start || sleep_only) proto_sleep(E, 0, [](const json & ev) { proto_print_event(ev); return true; }, nullptr);
        if (sleep_only) { llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free(); return 0; }
        if (!quiet) fprintf(stderr, "protocol mode | checkpoint step %d | %zu memories, %zu queued | %zu tool manuals, %zu lookup entries | remember %s\\n",
                            E.side.step, E.cells.size(), PROTO.queue.size(), PROTO.docs.size(), PROTO.lookup.size(), PROTO.remember_mode.c_str());
        if (chat_repl) {
            const std::string mname = model_path.substr(model_path.find_last_of("/\\\\") + 1);
            return run_chat_repl(E, max_new > 64 ? max_new : 700, system_prompt, mname);
        }
    }''',
     "protocol setup")

once('''    auto r = run_prompt(E, prompt, max_new, quiet, nullptr, nullptr);
    printf("%s\\n", r.summary.dump().c_str());''',
     '''    if (E.side.protocol) {                                // 0018: one question through the protocol loop
        json msgs = json::array({ { { "role", "user" }, { "content", prompt } } });
        json done = run_protocol(E, proto_chat_prompt(msgs, system_prompt), prompt, max_new > 64 ? max_new : 700,
                                 [](const json & ev) { proto_print_event(ev); return true; }, nullptr);
        printf("%s\\n", proto_dump(done).c_str());
    } else {
        auto r = run_prompt(E, prompt, max_new, quiet, nullptr, nullptr);
        printf("%s\\n", r.summary.dump().c_str());
    }''',
     "one-shot prompt")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0018 applied to %s: protocol mode wired in" % p)

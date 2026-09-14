import io
import sys

# 0016 -- CAPTURE WITH CONTEXT (2026-09-13 15:10). Kade: "if we give the AI a sentence as a memory and have the
# same sentence be stored but with different hidden states what happens? Say the VFD parameters are ... as the
# sample, but when the model stores those hidden states we have in context what the VFD is and all that -- does
# the model's response change?"
# The delivered block is built from the model's own states over the row. Those states depend on everything the
# model read before the row -- which today is nothing: capture() decodes the row alone. --cell-context TEXT (or
# a per-cell "context" field in the table) puts TEXT in front of the row during capture and keeps only the ROW's
# positions, so the block carries the same tokens, the same span, the same number of positions -- and different
# vectors. It is the cleanest test of whether the states do semantic work or only carry the letters.
p = sys.argv[1] if len(sys.argv) > 1 else "steermem.cpp"
s = io.open(p, encoding="utf-8").read()

old = '''struct cell_t {
    std::string key, text, table;
'''
new = '''struct cell_t {
    std::string key, text, table;
    std::string ctx;                               // 0016: text read before the row when its states are captured
'''
assert s.count(old) == 1, "cell_t"
s = s.replace(old, new)

old = '''    int ngl = 0;                                   // 0015: layers offloaded to the GPU, for the info line
'''
new = '''    int ngl = 0;                                   // 0015: layers offloaded to the GPU, for the info line
    std::string cell_ctx;                          // 0016: --cell-context, used for cells that carry none of their own
'''
assert s.count(old) == 1, "engine field"
s = s.replace(old, new)

old = '''        std::string row = "<cell: " + c.key + " | " + c.table + ">\\n" + c.text + "\\n</cell>";
        c.ids = tok(row, true);
        if (c.ids.size() > 512) c.ids.resize(512);
        const int n = (int) c.ids.size();
'''
new = '''        std::string row = "<cell: " + c.key + " | " + c.table + ">\\n" + c.text + "\\n</cell>";
        const std::string & ctx = c.ctx.empty() ? cell_ctx : c.ctx;      // 0016: read before the row, kept out of the block
        c.ids = tok(row, true);                                          // 0016: tok()'s flag is parse-special, not add-bos -- the row must tokenise identically with and without a context, or the A/B is not the same block
        if (c.ids.size() > 512) c.ids.resize(512);
        const int n = (int) c.ids.size();
        std::vector<llama_token> pre;
        if (!ctx.empty()) pre = tok(ctx + "\\n", true);
        const int np = (int) pre.size();
'''
assert s.count(old) == 1, "row tokens"
s = s.replace(old, new)

old = '''        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) { b.token[i] = c.ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        b.n_tokens = n;
        if (llama_decode(cap, b) != 0) { fprintf(stderr, "capture decode failed\\n"); exit(2); }
        llama_batch_free(b);
        for (int l : side.layers) {
            auto & v = capd.got["l_out-" + std::to_string(l)];
            if ((int64_t) v.size() != (int64_t) n * d) { fprintf(stderr, "capture of l_out-%d has %zu floats, expected %d x %d\\n", l, v.size(), n, d); exit(2); }
            c.st[l] = v;                                           // [n * d], token-major
        }
'''
new = '''        llama_batch b = llama_batch_init(np + n, 0, 1);
        for (int i = 0; i < np; ++i) { b.token[i] = pre[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        for (int i = 0; i < n; ++i) { b.token[np + i] = c.ids[i]; b.pos[np + i] = np + i; b.n_seq_id[np + i] = 1; b.seq_id[np + i][0] = 0; b.logits[np + i] = 0; }
        b.n_tokens = np + n;
        if (llama_decode(cap, b) != 0) { fprintf(stderr, "capture decode failed\\n"); exit(2); }
        llama_batch_free(b);
        for (int l : side.layers) {
            auto & v = capd.got["l_out-" + std::to_string(l)];
            if ((int64_t) v.size() != (int64_t) (np + n) * d) { fprintf(stderr, "capture of l_out-%d has %zu floats, expected %d x %d\\n", l, v.size(), np + n, d); exit(2); }
            c.st[l].assign(v.begin() + (size_t) np * d, v.end());  // 0016: the ROW's positions only; the context is read, not delivered
        }
'''
assert s.count(old) == 1, "capture decode"
s = s.replace(old, new)

old = '''        else if (a == "-ngl" || a == "--n-gpu-layers") ngl = atoi(next().c_str());
'''
new = '''        else if (a == "-ngl" || a == "--n-gpu-layers") ngl = atoi(next().c_str());
        else if (a == "--cell-context") cell_ctx = next();
'''
assert s.count(old) == 1, "cli parse"
s = s.replace(old, new)

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1, key_min_chars = -1, mem_marks = -1, ngl = 0;"
assert s.count(old) == 1, "cli vars"
s = s.replace(old, old + "\n    std::string cell_ctx;                          // 0016: read before every row at capture time")

old = "    E.ngl = ngl;\n"
assert s.count(old) == 1, "apply"
s = s.replace(old, "    E.ngl = ngl;\n    E.cell_ctx = cell_ctx;                        // 0016\n")

old = '{ "mem_marks", E.side.mem_marks }, { "ngl", E.ngl },'
assert s.count(old) == 1, "info"
s = s.replace(old, '{ "mem_marks", E.side.mem_marks }, { "ngl", E.ngl }, { "cell_context", (int) E.cell_ctx.size() },')

# the table loader: an optional per-cell context
old = '''            cell_t c; c.key = o.value("key", ""); c.text = o.contains("text") ? o.value("text", "") : o.value("content", ""); c.table = o.value("table", name);
'''
if s.count(old) == 1:
    s = s.replace(old, old.rstrip("\n") + ' c.ctx = o.value("context", "");   // 0016\n')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0016: --cell-context TEXT (and a per-cell \"context\" field) -- the same row, states captured after reading TEXT")

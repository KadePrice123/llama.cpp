#!/usr/bin/env python3
"""steermem sidecar: `--line-rule 1|2` -- which span of a cell the head and the trace pool over.

Rule 2 is the trainer's today (row_states: "the row's own line starts after the
second newline"): the mean over the states from the second newline-bearing token
to the end of the row. For a one-line cell (a Strong's entry, a file record) the
second newline is the one before the closing tag, so the trace is the mean over
"</cell>" alone -- a content-free vector injected at every generated position.
Found 2026-09-12 17:06: with that trace the s250 checkpoint loops on one-line
cells in the trainer AND in the sidecar; with no trace the sidecar recites them.
Rule 1 pools the content: from after the header's newline to the closing tag's
newline. The flag lets the sidecar mirror either trainer rule; the default stays
2 until the trainer changes.

    python 0005-line-rule.py ~/llama-ngram/steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "    int l0 = 0;\n    std::map<int, std::vector<float>> st;          // layer -> [n * d]\n"
new = "    int l0 = 0, l1 = 0;                            // the line span [l0, l1) pooled for the head and the trace\n    std::map<int, std::vector<float>> st;          // layer -> [n * d]\n"
assert s.count(old) == 1, "cell struct"
s = s.replace(old, new)

old = "    int kv_pos = 0;\n"
new = "    int line_rule = 2;                              // 2: after the second newline to the end (the trainer today); 1: the content between the header and the closing tag\n    int kv_pos = 0;\n"
assert s.count(old) == 1, "engine member"
s = s.replace(old, new)

old = """        // l0: after the second newline-bearing token
        int nl = 0; c.l0 = 0;
        for (int i = 0; i < n; ++i) { if (piece(c.ids[i]).find('\\n') != std::string::npos) { if (++nl == 2) { c.l0 = i + 1; break; } } }
        if (c.l0 >= n) c.l0 = 0;
"""
new = """        // the line span [l0, l1). Rule 2 (the trainer today): after the second newline-bearing token to the
        // end -- for a one-line cell that is "</cell>" alone. Rule 1: from after the header's newline through
        // the closing tag's newline-bearing token, i.e. the content.
        std::vector<int> nls;
        for (int i = 0; i < n; ++i) if (piece(c.ids[i]).find('\\n') != std::string::npos) nls.push_back(i);
        c.l0 = 0; c.l1 = n;
        if (line_rule == 1) {
            if (!nls.empty()) c.l0 = nls[0] + 1;
            if (nls.size() >= 2 && nls.back() + 1 > c.l0) c.l1 = nls.back() + 1;
        } else if (nls.size() >= 2) {
            c.l0 = nls[1] + 1;
        }
        if (c.l0 >= n) { c.l0 = 0; c.l1 = n; }
"""
assert s.count(old) == 1, "l0 rule"
s = s.replace(old, new)

old = """            const int span = std::max(1, n - c.l0);
            for (int t = c.l0; t < n; ++t) for (int j = 0; j < d; ++j) r[j] += st[(size_t) t * d + j] / (float) span;
"""
new = """            const int e1 = std::max(c.l0 + 1, std::min(n, c.l1));
            const int span = e1 - c.l0;
            for (int t = c.l0; t < e1; ++t) for (int j = 0; j < d; ++j) r[j] += st[(size_t) t * d + j] / (float) span;
"""
assert s.count(old) == 1, "head mean"
s = s.replace(old, new)

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096;\n"
new = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 2;\n"
assert s.count(old) == 1, "main locals"
s = s.replace(old, new)

old = '        else if (a == "--no-traces") no_traces = true; else if (a == "--quiet") quiet = true; else if (a == "--ctx") n_ctx = atoi(next().c_str());\n'
new = ('        else if (a == "--no-traces") no_traces = true; else if (a == "--quiet") quiet = true; else if (a == "--ctx") n_ctx = atoi(next().c_str());\n'
       '        else if (a == "--line-rule") line_rule = atoi(next().c_str());\n')
assert s.count(old) == 1, "flag parse"
s = s.replace(old, new)

old = "    engine_t E;\n    E.side = load_side(side_path);\n"
new = "    engine_t E;\n    E.line_rule = line_rule;\n    E.side = load_side(side_path);\n"
assert s.count(old) == 1, "engine setup"
s = s.replace(old, new)

old = '    for (auto * c : cands) summary["cells"].push_back({ { "key", c->key }, { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 } });\n'
new = '    for (auto * c : cands) summary["cells"].push_back({ { "key", c->key }, { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 }, { "l1", c->l1 } });\n'
assert s.count(old) == 1, "summary"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: --line-rule 1|2")

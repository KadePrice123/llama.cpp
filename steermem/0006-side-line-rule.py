#!/usr/bin/env python3
"""steermem sidecar: the line rule comes from the side GGUF (steermem.line_rule), the flag overrides.

A checkpoint must be run under the rule it trained with: side_export.py writes the
rule (1 after trainer patch 79), and a side file without the key is an older
checkpoint, rule 2. `--line-rule 0` (the default now) follows the side file.

    python 0006-side-line-rule.py ~/llama-ngram/steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1;\n"
new = "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2;\n"
assert s.count(old) == 1, "side struct"
s = s.replace(old, new)

old = '    s.key_window = u32("steermem.key_window", 96); s.egroup = u32("steermem.egroup", 1);\n'
new = ('    s.key_window = u32("steermem.key_window", 96); s.egroup = u32("steermem.egroup", 1);\n'
       '    s.line_rule = u32("steermem.line_rule", 2);   // absent: a checkpoint from before trainer patch 79\n')
assert s.count(old) == 1, "side read"
s = s.replace(old, new)

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 2;\n"
new = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0;   // line_rule 0: follow the side file\n"
assert s.count(old) == 1, "main default"
s = s.replace(old, new)

old = "    engine_t E;\n    E.line_rule = line_rule;\n    E.side = load_side(side_path);\n"
new = "    engine_t E;\n    E.side = load_side(side_path);\n    E.line_rule = line_rule > 0 ? line_rule : E.side.line_rule;\n"
assert s.count(old) == 1, "engine setup"
s = s.replace(old, new)

old = 'fprintf(stderr, "table %zu cells | mem_seq %d cells %d mem_gen %d memtok %d window %d | scale %s\\n", E.cells.size(), E.side.mem_seq, E.side.mem_seq_cells, E.side.mem_gen, E.side.me'
if s.count(old) == 1:
    pass  # the report line keeps its shape; the rule is visible in the summary json (l0/l1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: line rule from the side file, --line-rule overrides")

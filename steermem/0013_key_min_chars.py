import io
import sys

# 0013 -- ONE-TOKEN KEYS (2026-09-12 20:55; with trainer patch 88 and the builders' --key-min-chars). Every key form
# here had to be 2+ tokens, so "dog", "cat", "car", "kid" never delivered on either side. steermem.key_min_chars in the
# side file (side_export.py --key-min-chars N; --key-min-chars on the command line overrides): a one-token form counts
# when the key's name has at least N alphanumeric characters; 0 keeps the old rule. Applies after 0012.
p = sys.argv[1] if len(sys.argv) > 1 else "steermem.cpp"
s = io.open(p, encoding="utf-8").read()

old = "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2, mem_once = 0;\n"
assert s.count(old) == 1, "side struct (needs 0012)"
s = s.replace(old, "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2, mem_once = 0, key_min_chars = 0;\n")

old = '    s.mem_once = u32("steermem.mem_once", 0);          // 0012: one block per cell per turn (trainer patch 87)\n'
assert s.count(old) == 1, "side reader"
s = s.replace(old, old + '    s.key_min_chars = u32("steermem.key_min_chars", 0); // 0013: a one-token key form counts when the name has N+ alphanumerics (trainer patch 88)\n')

old = '        for (auto & nm : names) for (auto & f : { tok(" " + nm, false), tok(nm, false) }) if (f.size() >= 2) c.forms.push_back(f);\n'
new = '''        for (auto & nm : names) {
            int an = 0; for (unsigned char ch : nm) if (std::isalnum(ch)) ++an;     // 0013: one-token forms of a name with key_min_chars+ alphanumerics count
            for (auto & f : { tok(" " + nm, false), tok(nm, false) })
                if (f.size() >= 2 || (f.size() == 1 && side.key_min_chars > 0 && an >= side.key_min_chars)) c.forms.push_back(f);
        }
'''
assert s.count(old) == 1, "forms"
s = s.replace(old, new)

old = '{ "mem_once", E.side.mem_once }, { "memtok", E.side.memtok },'
assert s.count(old) == 1, "info"
s = s.replace(old, '{ "mem_once", E.side.mem_once }, { "key_min_chars", E.side.key_min_chars }, { "memtok", E.side.memtok },')

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1;   // line_rule 0 / mem_once -1: follow the side file\n"
assert s.count(old) == 1, "cli vars"
s = s.replace(old, "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1, key_min_chars = -1;   // line_rule 0 / mem_once -1 / key_min_chars -1: follow the side file\n")

old = '        else if (a == "--mem-once") mem_once = atoi(next().c_str());\n'
assert s.count(old) == 1, "cli parse"
s = s.replace(old, old + '        else if (a == "--key-min-chars") key_min_chars = atoi(next().c_str());\n')

old = "    if (mem_once >= 0) E.side.mem_once = mem_once;       // 0012: the command line overrides the side file\n"
assert s.count(old) == 1, "apply"
s = s.replace(old, old + "    if (key_min_chars >= 0) E.side.key_min_chars = key_min_chars;   // 0013: the command line overrides the side file\n")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0013: steermem.key_min_chars / --key-min-chars: one-token key forms of names with N+ alphanumerics deliver")

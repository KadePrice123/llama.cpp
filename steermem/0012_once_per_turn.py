import io
import sys

# 0012 -- ONE BLOCK PER CELL PER TURN (2026-09-12 20:30). Kade, reading the step-100 battery's MEM lines (the
# same keys fired 19 times inside one answer, every fire restarting the "X says: ..." recite the training data
# taught): "each assistant turn can call each cell once whenever it chooses, but if it calls the same key
# twice no 2nd key is given"; the next turn resets. The trainer's patch 87 (--mem-once 1) applies the identical
# rule; the side file carries it as steermem.mem_once so serving follows the checkpoint's training, and
# --mem-once on the command line overrides. Turns are the <|im_start|> segments of the prompt; in the prompt
# the first mention of a cell in a turn delivers its block and later mentions in that turn deliver nothing, at
# most mem_seq_cells cells per turn; in generation (the model's own turn, which starts with nothing delivered)
# a cell fires once, at most mem_seq_cells cells.
p = sys.argv[1] if len(sys.argv) > 1 else "steermem.cpp"
s = io.open(p, encoding="utf-8").read()

old = "#include <map>\n"
assert s.count(old) == 1, "include"
if "#include <set>" not in s:
    s = s.replace(old, "#include <map>\n#include <set>\n")

old = "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2;\n"
assert s.count(old) == 1, "side struct"
s = s.replace(old, "    int mem_seq = 128, mem_seq_cells = 4, mem_gen = 0, memtok = 8, key_window = 96, egroup = 1, line_rule = 2, mem_once = 0;\n")

old = '    s.mem_gen = u32("steermem.mem_gen", 0); s.memtok = u32("steermem.memtok", 8);\n'
assert s.count(old) == 1, "side reader"
s = s.replace(old, old + '    s.mem_once = u32("steermem.mem_once", 0);          // 0012: one block per cell per turn (trainer patch 87)\n')

old = '''    if ((int) hits.size() > E.side.mem_seq_cells) {
        if (E.side.mem_gen) hits.resize(E.side.mem_seq_cells);
        else hits.erase(hits.begin(), hits.end() - E.side.mem_seq_cells);
    }
'''
new = '''    if (E.side.mem_once) {
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
'''
assert s.count(old) == 1, "seed cap"
s = s.replace(old, new)

old = "    int blocks_fired = 0, n_tok = 0;\n"
assert s.count(old) == 1, "gen vars"
s = s.replace(old, old + "    std::set<cell_t *> seen_turn;                  // 0012: the cells delivered in this turn of the model's own\n")

old = "        if (blocks_fired >= E.side.mem_seq_cells * 8) fired.clear();   // the trainer's cap on blocks fired in one generation (MEMGEN_N < mem_seq_cells * 8)\n"
assert s.count(old) == 1, "fire cap"
s = s.replace(old, old + '''        if (E.side.mem_once && !fired.empty()) {   // 0012: a cell already delivered in this turn fires nothing more; at most mem_seq_cells cells
            std::vector<cell_t *> fresh;
            for (auto * c : fired) if (!seen_turn.count(c) && (E.side.mem_seq_cells <= 0 || (int) seen_turn.size() < E.side.mem_seq_cells)) { seen_turn.insert(c); fresh.push_back(c); }
            fired.swap(fresh);
        }
''')

old = '{ "mem_seq_cells", E.side.mem_seq_cells }, { "memtok", E.side.memtok }, { "line_rule", E.line_rule },'
assert s.count(old) == 1, "info"
s = s.replace(old, '{ "mem_seq_cells", E.side.mem_seq_cells }, { "mem_once", E.side.mem_once }, { "memtok", E.side.memtok }, { "line_rule", E.line_rule },')

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0;   // line_rule 0: follow the side file\n"
assert s.count(old) == 1, "cli vars"
s = s.replace(old, "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1;   // line_rule 0 / mem_once -1: follow the side file\n")

old = '        else if (a == "--line-rule") line_rule = atoi(next().c_str());\n'
assert s.count(old) == 1, "cli parse"
s = s.replace(old, old + '        else if (a == "--mem-once") mem_once = atoi(next().c_str());\n')

old = "    E.line_rule = line_rule > 0 ? line_rule : E.side.line_rule;\n"
assert s.count(old) == 1, "apply"
s = s.replace(old, old + "    if (mem_once >= 0) E.side.mem_once = mem_once;       // 0012: the command line overrides the side file\n")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0012: one block per cell per turn (steermem.mem_once / --mem-once), prompt segments by <|im_start|>, generation dedup")

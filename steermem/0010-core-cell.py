#!/usr/bin/env python3
"""steermem sidecar: a core memory cell for every table, with both key forms.

Kade, 2026-09-12 19:20, on "what memories do you have?" answering as a plain chatbot:
"this should have called core memories and searched through its memories". The trainer's
items always carry a core cell -- key "core memory", text "Core memory. Memory tables
available: A; B; C." -- but a table loaded into the sidecar had none, so nothing could
fire. And the think phrases say "core memories" (plural) while the key is singular; the
two tokenize differently, so the plural must be a key form too. On every table load the
sidecar now adds the core cell (unless the table brings its own) with the forms of
"core memory" and "core memories", built exactly as the trainer builds it.

    python 0010-core-cell.py steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "    std::string key, text, table;\n    std::vector<std::vector<llama_token>> forms;   // key token forms, 2+ tokens each\n"
new = ("    std::string key, text, table;\n    std::vector<std::string> aliases;              // other spellings that name this cell (the core cell: \"core memories\")\n"
       "    std::vector<std::vector<llama_token>> forms;   // key token forms, 2+ tokens each\n")
assert s.count(old) == 1, "cell struct"
s = s.replace(old, new)

old = '''    void prep_forms(cell_t & c) {
        c.forms.clear();
        for (auto & f : { tok(" " + c.key, false), tok(c.key, false) }) if (f.size() >= 2) c.forms.push_back(f);
    }
'''
new = '''    void prep_forms(cell_t & c) {
        c.forms.clear();
        std::vector<std::string> names = { c.key };
        names.insert(names.end(), c.aliases.begin(), c.aliases.end());
        for (auto & nm : names) for (auto & f : { tok(" " + nm, false), tok(nm, false) }) if (f.size() >= 2) c.forms.push_back(f);
    }
    // THE CORE CELL (19:20): what the trainer's items always carry -- "Core memory. Memory tables
    // available: A; B; C." under the key "core memory" (and "core memories", the phrase the think
    // uses) -- added to every loaded table that does not bring its own.
    static void add_core_cell(std::vector<cell_t> & cells) {
        for (auto & c : cells) if (c.key == "core memory") return;
        std::vector<std::string> names;
        for (auto & c : cells) if (std::find(names.begin(), names.end(), c.table) == names.end()) names.push_back(c.table);
        cell_t core; core.key = "core memory"; core.table = "index"; core.aliases = { "core memories" };
        std::string list; for (size_t i = 0; i < names.size(); ++i) list += (i ? "; " : "") + names[i];
        core.text = "Core memory. Memory tables available: " + list + ".";
        cells.insert(cells.begin(), core);
    }
'''
assert s.count(old) == 1, "prep_forms"
s = s.replace(old, new)

old = '''    void set_table(std::vector<cell_t> cells_) {
        cells = std::move(cells_);
        for (auto & c : cells) prep_forms(c);
    }
'''
new = '''    void set_table(std::vector<cell_t> cells_) {
        cells = std::move(cells_);
        add_core_cell(cells);
        for (auto & c : cells) prep_forms(c);
    }
'''
assert s.count(old) == 1, "set_table"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: a core memory cell (both key forms) on every table load")

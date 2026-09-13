#!/usr/bin/env python3
"""steermem sidecar: an index cell per table on load, with a blurb per entry, and the core cell describing tables.

Kade, 2026-09-12 19:30: "every table needs a description / index the AI can remember to use each
table" and "the model has to figure out what key is what using the tables' descriptions". The
training items (memory_items.py) carry, per table, an INDEX cell keyed by the table's name:
"Index of TABLE: DESCRIPTION. Keyed by the entry name. Entries: key (blurb); ...". A table
uploaded to the sidecar has no such cell, so the sidecar builds one on load: the blurb is the
first ~8 words of the entry, the description "N entries" (a table file may carry its own index
row under the table's name, which then wins). Tables above 60 entries get the first 40 keys and
"... and N more" -- the whole Bible cannot fit a row; its own book/chapter cells are the index
there. The core cell now lists tables with their entry counts, as the training core cells do
when no description is given.

    python 0011-index-cells.py steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = '''    static void add_core_cell(std::vector<cell_t> & cells) {
        for (auto & c : cells) if (c.key == "core memory") return;
        std::vector<std::string> names;
        for (auto & c : cells) if (std::find(names.begin(), names.end(), c.table) == names.end()) names.push_back(c.table);
        cell_t core; core.key = "core memory"; core.table = "index"; core.aliases = { "core memories" };
        std::string list; for (size_t i = 0; i < names.size(); ++i) list += (i ? "; " : "") + names[i];
        core.text = "Core memory. Memory tables available: " + list + ".";
        cells.insert(cells.begin(), core);
    }
'''
new = '''    // THE INDEX CELLS (19:30): one per table, keyed by the table's name, listing its entries with a
    // blurb each -- what the training items carry, so the model can map a loose question to a key by
    // reading descriptions. A table that brings its own row under its name keeps it.
    static std::string blurb(const std::string & text) {
        std::string t = text; for (auto & ch : t) if (ch == '\\n') ch = ' ';
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
'''
assert s.count(old) == 1, "core cell"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: an index cell per table on load; the core cell lists tables with counts")

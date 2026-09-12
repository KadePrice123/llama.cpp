#!/usr/bin/env python3
"""steermem sidecar: a trace at the LAST prompt token, as the trainer places one.

The trainer's memtok points are every K-th seed position (7, 15, ...) AND the
last seed position (pts += range(cut - 1, L - 1, ae): the answer's stride
starts at cut - 1, the token whose logits produce the first answer token), then
every generated token. The sidecar traced only the K-th positions of the prompt,
so the first generated token was produced without the routed cell's trace.
Found 2026-09-12 16:55 in the side-by-side on one-line cells.

    python 0004-trace-last.py ~/llama-ngram/steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "    void feed_text(const std::vector<llama_token> & seg, std::vector<cell_t *> & cands, bool want_logits) {\n"
new = "    void feed_text(const std::vector<llama_token> & seg, std::vector<cell_t *> & cands, bool want_logits, bool trace_last = false) {\n"
assert s.count(old) == 1, "feed_text signature"
s = s.replace(old, new)

old = "            if (side.memtok > 0 && pt % side.memtok == side.memtok - 1) {\n"
new = ("            // the trainer's memtok points: every K-th text position, and always the last prompt token\n"
       "            if (side.memtok > 0 && (pt % side.memtok == side.memtok - 1 || (trace_last && i + 1 == (int) seg.size()))) {\n")
assert s.count(old) == 1, "memtok point"
s = s.replace(old, new)

old = "        if (cut > done) { std::vector<llama_token> seg(seed.begin() + done, seed.begin() + cut); E.feed_text(seg, cands, false); done = cut; }\n"
new = "        if (cut > done) { std::vector<llama_token> seg(seed.begin() + done, seed.begin() + cut); E.feed_text(seg, cands, false, cut == (int) seed.size()); done = cut; }\n"
assert s.count(old) == 1, "segment before a block"
s = s.replace(old, new)

old = "    if (done < (int) seed.size()) { std::vector<llama_token> seg(seed.begin() + done, seed.end()); E.feed_text(seg, cands, true); }\n"
new = "    if (done < (int) seed.size()) { std::vector<llama_token> seg(seed.begin() + done, seed.end()); E.feed_text(seg, cands, true, true); }\n"
assert s.count(old) == 1, "final segment"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: a trace at the last prompt token")

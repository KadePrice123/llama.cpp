#!/usr/bin/env python3
"""steermem sidecar: a written key fires its block only once the NEXT token confirms the boundary.

Seen on the first server test (2026-09-12 18:45, the Strong's table, mem-gen on): the
model wrote "G26" and the sidecar fired the block for "G2" as soon as the text ended in
"G2", then "H1" inside "H1925", again and again -- key_end matched a key form at the
text's end with no boundary check, and Qwen writes digits one at a time. A key ending at
the text's end is complete only if the token about to be appended starts with a
non-alphanumeric character (the prompt path already applies this rule to the token after
a hit). So the check moves to just before the new token is fed: if the text so far ends
in a key and the new token is a boundary, that cell's block is fed first, then the token
-- the training layout (key, block, next token). The trainer's own _key_end has the same
flaw; a trainer patch follows between stages.

    python 0008-keyend-boundary.py steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = '''        std::vector<llama_token> tmp = E.text; tmp.push_back(best);
        std::swap(E.text, tmp);
        cell_t * rc = (E.side.memtok > 0) ? E.route(cands) : nullptr;
        std::vector<cell_t *> fired = E.side.mem_gen ? E.key_end(all) : std::vector<cell_t *>();
        std::swap(E.text, tmp);
        if (rc) { auto ls = E.line_state(*rc); for (int l : E.side.layers) E.inj.push_back({ l, E.kv_pos, ls[l] }); ++E.traces_fed; }
        {   // feed the token itself (feed_text would re-route at memtok points; generated tokens route every step)
            E.flush_inject();
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = best; b.pos[0] = E.kv_pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = fired.empty() ? 1 : 0; b.n_tokens = 1;
            if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\\n"); break; }
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
'''
new = '''        // mem-gen: a key that ends at the text so far is complete only if the token about to be fed
        // starts a boundary (Qwen writes "G26" as G, 2, 6 -- "G2" must not fire inside it). Its block
        // goes in BEFORE that token: key, block, next token, the training layout.
        std::vector<cell_t *> fired;
        if (E.side.mem_gen && (pc.empty() || !std::isalnum((unsigned char) pc[0]))) fired = E.key_end(all);
        if (!fired.empty()) {
            json keys = json::array(); for (auto * c : fired) keys.push_back(c->key);
            fired_log.push_back({ { "at_text_token", (int) E.text.size() }, { "keys", keys } });
            for (auto * c : fired) {
                if (std::find(cands.begin(), cands.end(), c) == cands.end()) cands.push_back(c);
                E.feed_block(*c, false);
                ++blocks_fired;
            }
        }
        std::vector<llama_token> tmp = E.text; tmp.push_back(best);
        std::swap(E.text, tmp);
        cell_t * rc = (E.side.memtok > 0) ? E.route(cands) : nullptr;
        std::swap(E.text, tmp);
        if (rc) { auto ls = E.line_state(*rc); for (int l : E.side.layers) E.inj.push_back({ l, E.kv_pos, ls[l] }); ++E.traces_fed; }
        {   // feed the token itself (feed_text would re-route at memtok points; generated tokens route every step)
            E.flush_inject();
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = best; b.pos[0] = E.kv_pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(E.ctx, b) != 0) { fprintf(stderr, "decode failed\\n"); break; }
            llama_batch_free(b); llama_inject_clear(E.ctx);
            E.text.push_back(best); E.kv_pos += 1;
        }
'''
assert s.count(old) == 1, "generation loop"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: a written key fires only once the next token confirms its boundary; the block precedes that token")

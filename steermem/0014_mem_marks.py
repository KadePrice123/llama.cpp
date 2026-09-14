import io
import sys

# 0014 -- MARKED MEMORY IN THE FORK (2026-09-13 13:10), the serving half of trainer patch 91. Kade: "the model
# uses its own hidden states for the inserted token items, it probably can't tell that from its own thinking...
# the first and last token in every cell will be labelled <|mem_start|> <|mem_end|>". The side file carries the
# two learned vectors as the tensor `marktag` [2, d] plus steermem.mem_marks; when it is set, every block the
# sidecar feeds becomes [<|mem_start|>, head, body_1..body_nb, <|mem_end|>] -- the marker rows are normalised to
# emb_norm like every other row and carry no recorded state (their rec stays zero), so they add the boundary and
# no content, exactly as the trainer builds them. A side file without the tensor serves as before.
p = sys.argv[1] if len(sys.argv) > 1 else "steermem.cpp"
s = io.open(p, encoding="utf-8").read()

old = "    std::vector<float> memtag, seqtag, exptag;     // [d], [64*d], [8*d]\n"
assert s.count(old) == 1, "side struct"
s = s.replace(old, "    std::vector<float> memtag, seqtag, exptag, marktag;   // [d], [64*d], [8*d], [2*d]\n    int mem_marks = 0;                             // 0014: wrap every block in <|mem_start|> / <|mem_end|>\n")

old = '    s.mem_once = u32("steermem.mem_once", 0);          // 0012: one block per cell per turn (trainer patch 87)\n'
assert s.count(old) == 1, "side reader (needs 0012)"
s = s.replace(old, old + '    s.mem_marks = u32("steermem.mem_marks", 0);        // 0014: the block is wrapped in two learned marker positions (trainer patch 91)\n')

old = '''    s.memtag = tensor("memtag"); s.d = (int) s.memtag.size();
    s.seqtag = tensor("seqtag"); s.exptag = tensor("exptag");
'''
new = '''    s.memtag = tensor("memtag"); s.d = (int) s.memtag.size();
    s.seqtag = tensor("seqtag"); s.exptag = tensor("exptag");
    if (s.mem_marks) {                                  // 0014: the marker vectors, or serve unmarked if the file predates them
        ggml_tensor * mt = ggml_get_tensor(gctx, "marktag");
        if (!mt || (int) ggml_nelements(mt) != 2 * s.d) { fprintf(stderr, "side gguf says mem_marks but lacks marktag [2,d]; serving unmarked\\n"); s.mem_marks = 0; }
        else { s.marktag.resize(2 * s.d); memcpy(s.marktag.data(), mt->data, s.marktag.size() * sizeof(float)); }
    }
'''
assert s.count(old) == 1, "tensor read"
s = s.replace(old, new)

old = '''        block_t b; b.n = 1 + nb;
'''
new = '''        const int mk = side.mem_marks ? 1 : 0;          // 0014: a marker row before the head and after the last body
        block_t b; b.n = 1 + nb + 2 * mk;
'''
assert s.count(old) == 1, "block size"
s = s.replace(old, new)

old = '''        // head
        for (int j = 0; j < d; ++j) row(0)[j] = side.memtag[j] + side.seqtag[j] + side.exptag[(size_t) side.egroup * d + j] + c.keyemb[j];
'''
new = '''        // <|mem_start|> / head / ... / <|mem_end|>
        if (mk) for (int j = 0; j < d; ++j) row(0)[j] = side.marktag[j];
        for (int j = 0; j < d; ++j) row(mk)[j] = side.memtag[j] + side.seqtag[j] + side.exptag[(size_t) side.egroup * d + j] + c.keyemb[j];
'''
assert s.count(old) == 1, "head row"
s = s.replace(old, new)

old = '''            float * r = b.rec[l].data();
'''
new = '''            float * r = b.rec[l].data() + (size_t) mk * d;
'''
assert s.count(old) == 1, "head rec"
s = s.replace(old, new)

old = '''            for (int j = 0; j < d; ++j) row(1 + k)[j] = side.memtag[j] + side.seqtag[(size_t) code * d + j];
            for (int l : side.layers) memcpy(b.rec[l].data() + (size_t) (1 + k) * d, c.st[l].data() + (size_t) pj * d, d * sizeof(float));
'''
new = '''            for (int j = 0; j < d; ++j) row(mk + 1 + k)[j] = side.memtag[j] + side.seqtag[(size_t) code * d + j];
            for (int l : side.layers) memcpy(b.rec[l].data() + (size_t) (mk + 1 + k) * d, c.st[l].data() + (size_t) pj * d, d * sizeof(float));
'''
assert s.count(old) == 1, "body rows"
s = s.replace(old, new)

old = '''        // every vec row to emb_norm; every rec row to unit length (build_inject scales by the residual norm)
'''
new = '''        if (mk) for (int j = 0; j < d; ++j) row(b.n - 1)[j] = side.marktag[d + j];   // <|mem_end|>
        // every vec row to emb_norm; every rec row to unit length (build_inject scales by the residual norm)
'''
assert s.count(old) == 1, "end marker"
s = s.replace(old, new)

old = '{ "mem_once", E.side.mem_once }, { "key_min_chars", E.side.key_min_chars },'
assert s.count(old) == 1, "info"
s = s.replace(old, '{ "mem_once", E.side.mem_once }, { "mem_marks", E.side.mem_marks }, { "key_min_chars", E.side.key_min_chars },')

old = '        else if (a == "--key-min-chars") key_min_chars = atoi(next().c_str());\n'
assert s.count(old) == 1, "cli parse"
s = s.replace(old, old + '        else if (a == "--mem-marks") mem_marks = atoi(next().c_str());\n')

old = "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1, key_min_chars = -1;"
assert s.count(old) == 1, "cli vars"
s = s.replace(old, "    int max_new = 64, threads = 6, memgen = -1, n_ctx = 4096, line_rule = 0, port = 0, mem_once = -1, key_min_chars = -1, mem_marks = -1;")

old = "    if (key_min_chars >= 0) E.side.key_min_chars = key_min_chars;   // 0013: the command line overrides the side file\n"
assert s.count(old) == 1, "apply"
s = s.replace(old, old + "    if (mem_marks >= 0) E.side.mem_marks = (mem_marks && !E.side.marktag.empty()) ? 1 : 0;   // 0014: only if the file carries the vectors\n")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0014: blocks are fed as <|mem_start|>, head, bodies, <|mem_end|> when the side file carries marktag")

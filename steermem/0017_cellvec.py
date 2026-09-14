import io
import sys

# 0017 -- READ THE BLOCK THE SERVER WOULD DELIVER (2026-09-13 16:05). Companion to 0016. The behaviour probe
# answers "does the response change?"; this answers "did the vectors change, and by how much?" -- the two
# together separate "capture context is not a lever" from "the adapter ignores what the lever moved".
# POST /cellvec {"key": "..."} builds the SAME block the delivery path builds (E.block(c), so markers, pooling
# and unit-normalisation are whatever the checkpoint says) and returns the head row and three body rows per
# layer, unit vectors, so a client can cosine them across two captures of the identical row.
p = sys.argv[1] if len(sys.argv) > 1 else "steermem.cpp"
s = io.open(p, encoding="utf-8").read()

old = '''    srv.Post("/table", [&](const httplib::Request & req, httplib::Response & res) {'''
new = '''    srv.Post("/cellvec", [&](const httplib::Request & req, httplib::Response & res) {   // 0017
        std::string key; try { key = json::parse(req.body).value("key", ""); } catch (...) {}
        std::lock_guard<std::mutex> lk(mu);
        cell_t * c = nullptr;
        for (auto & x : E.cells) if (x.key == key) { c = &x; break; }
        if (!c) { send_json(res, { { "error", "no such cell" } }, 404); return; }
        auto b = E.block(*c);                              // exactly what a delivered block carries
        const int d = E.d, mk = E.side.mem_marks ? 1 : 0;
        json out = { { "key", c->key }, { "table", c->table }, { "context_chars", (int) (c->ctx.empty() ? E.cell_ctx.size() : c->ctx.size()) },
                     { "row_tokens", (int) c->ids.size() }, { "l0", c->l0 }, { "l1", c->l1 }, { "block_rows", b.n },
                     { "layers", json::object() } };
        const int nb = b.n - 1 - 2 * mk;                    // body rows
        std::vector<std::pair<std::string, int>> want = { { "head", mk }, { "body0", mk + 1 },
                                                          { "bodymid", mk + 1 + nb / 2 }, { "bodylast", mk + nb } };
        for (int l : E.side.layers) {
            json rows = json::object();
            for (auto & w : want) {
                if (w.second < 0 || w.second >= b.n) continue;
                const float * r = b.rec[l].data() + (size_t) w.second * d;
                rows[w.first] = std::vector<float>(r, r + d);
            }
            out["layers"][std::to_string(l)] = rows;
        }
        send_json(res, out);
    });
    srv.Post("/table", [&](const httplib::Request & req, httplib::Response & res) {'''
assert s.count(old) == 1, "route anchor"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0017: POST /cellvec {key} -> the head and three body rows of the block, per layer, as unit vectors")

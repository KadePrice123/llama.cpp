"""fork edit 0021 (2026-09-14): AUTO-INJECT -- a memory's block right after the surprising word that names it.

expert/assoc_insert.py measured it on the step-600 checkpoint: a stored memory's block placed right after the word in
the user's message, with no <recall>, is read -- the whole block almost as well as a recall, 8 rows partly, 2 rows
not at all. Kade: "allow a setting that allows the user to enable or disable auto injection and set the max tokens
to auto inject".

In llama/steermem_protocol.hpp:
  - proto_plan_injects: one pass over the message on the capture context (every token's logits, and the menu layer's
    states); a word whose tokens' summed -log p is at or above --inject-surprise nats and that names a stored key (or
    an alias) -- or, with --inject-match state, whose own states match one memory with at least --inject-min-score and
    0.05 over the runner-up -- gets up to --auto-inject N rows of that memory's block (block_twoside_rows); the most
    surprising --inject-max words win; an `inject` event names each one
  - run_protocol feeds the prompt in pieces, a block after each injected word of the last user message
  - proto_menu_rank: the menu's ranking over precomputed states, shared by the menu and the state match
  - /api/chat takes auto_inject per request; /api/state reports the settings; --chat has /inject N; NOMINMAX before
    windows.h for MSVC
In steermem.cpp: --auto-inject N --inject-max K --inject-surprise T --inject-match key|state --inject-min-score S, the
menu index built at start when the state match needs it, and the one-shot prompt planned the same way.

  python 0021_auto_inject.py path/to/steermem_protocol.hpp path/to/steermem.cpp
"""
import io
import sys

hp, cp = sys.argv[1], sys.argv[2]
H = io.open(hp, encoding="utf-8").read()
C = io.open(cp, encoding="utf-8").read()
if "0021" in H or "0021" in C:
    raise SystemExit("0021 is already applied")
if "0020" not in C or "proto_menu_build" not in H:
    raise SystemExit("apply 0019 and 0020 first")


def once(text, old, new, tag):
    assert text.count(old) == 1, "%s: %d matches" % (tag, text.count(old))
    return text.replace(old, new)


# ---------------------------------------------------------------- the header
H = once(H, r'''//   --chat      the same, at the command line: talk to the model and control memory with slash commands
''', r'''//   --chat      the same, at the command line: talk to the model and control memory with slash commands
//   auto-inject a surprising word in the user's message that names a memory gets that memory's block right after it,
//               before the answer starts (0021: --auto-inject N rows, 0 off)
''', "doc")

H = once(H, r'''#ifdef _WIN32
#include <windows.h>
''', r'''#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX                                   // 0021: MSVC's windows.h would otherwise turn std::min and std::max into macros
#endif
#include <windows.h>
''', "nominmax")

H = once(H, r'''static proto_menu_hit_t proto_menu_query(engine_t & E, const std::string & message, int k) {
    proto_menu_hit_t out;
    if (!MENU.fitted) proto_menu_fit(E.d);
    if (!MENU.fitted || MENU.keys.empty() || k <= 0) return out;
    std::vector<float> q; std::vector<llama_token> qids;
    if (!proto_token_states(E, message, q, qids) || qids.empty()) return out;
    proto_menu_norm(q, E.d);
''', r'''// every memory ranked against query states (z-scored, unit length) and their token ids: the k best (0021: shared by
// the menu and auto-inject's state match)
static proto_menu_hit_t proto_menu_rank(engine_t & E, const std::vector<float> & q, const std::vector<llama_token> & qids, int k) {
    proto_menu_hit_t out;
    if (!MENU.fitted || MENU.keys.empty() || k <= 0 || qids.empty()) return out;
''', "rank")

H = once(H, r'''// one menu line, written the way the menu items were: what the memory is, then when to recall it
''', r'''static proto_menu_hit_t proto_menu_query(engine_t & E, const std::string & message, int k) {
    if (!MENU.fitted) proto_menu_fit(E.d);
    std::vector<float> q; std::vector<llama_token> qids;
    if (!MENU.fitted || k <= 0 || !proto_token_states(E, message, q, qids) || qids.empty()) return proto_menu_hit_t();
    proto_menu_norm(q, E.d);
    return proto_menu_rank(E, q, qids, k);
}

// one menu line, written the way the menu items were: what the memory is, then when to recall it
''', "query")

H = once(H, r'''// ONE ANSWER: the loop the checkpoint was trained and evaluated with
static json run_protocol(engine_t & E, const std::string & prompt, const std::string & last_user, int max_new,
                         const std::function<bool(const json &)> & emit, std::atomic<bool> * stop) {
''', r'''// ---------------------------------------------------------------- auto-inject: memory right after the surprising word (0021)
// expert/assoc_insert.py on the step-600 checkpoint: a memory's block placed right after the word in the user's message,
// with no <recall>, is read -- the whole block almost as well as a recall, 8 rows partly, 2 rows not at all.
struct proto_inject_cfg_t {
    int rows = 0, max = 3;                          // --auto-inject N rows per injected memory (0 = off), --inject-max words per message
    double surprise = 8.0, min_score = 0.6, margin = 0.05;
    std::string match = "state";                    // key: a stored key named in the message; state: also a clear hidden-state match
};
static proto_inject_cfg_t INJECT;
struct proto_inject_t { size_t at = 0; std::string word, key, match; double surprise = 0.0, score = 0.0; engine_t::block_t b; };

static bool proto_word_char(unsigned char c) { return std::isalnum(c) || c >= 0x80 || (c && std::strchr("_./:@-", c)); }

// a block of at most `rows` rows (head and marks included): evenly spaced states, as assoc_insert.py measured
static engine_t::block_t block_twoside_rows(engine_t & E, cell_t & c, int rows) {
    const int extra = (E.side.head ? 1 : 0) + (E.side.mem_marks ? 2 : 0), saved = E.side.mem_seq;
    if (rows > 0) E.side.mem_seq = std::min(saved, std::max(1, rows - extra));
    engine_t::block_t b = block_twoside(E, c);
    E.side.mem_seq = saved;
    return b;
}

// where memory goes into the last user message: the most surprising words that name a stored key, or (match state)
// whose own states match one memory clearly, each with up to `rows` rows of that memory's block; in message order
static std::vector<proto_inject_t> proto_plan_injects(engine_t & E, const std::string & message, int rows, const std::function<bool(const json &)> & emit) {
    std::vector<proto_inject_t> plan;
    if (rows <= 0 || INJECT.max <= 0 || message.empty() || E.cells.empty()) return plan;
    const bool by_state = INJECT.match == "state" && MENU.layer >= 1;
    if (by_state && !MENU.built) proto_menu_build(E);
    const std::string open = "<|im_start|>user\n";
    std::vector<llama_token> ids = E.tok(open + message, true);
    if (ids.size() > 256) ids.resize(256);
    proto_cap_ready(E);                                   // one pass: every token's logits, and the menu layer's states
    const std::string lname = "l_out-" + std::to_string(MENU.layer - 1);
    if (by_state) E.capd.want.assign({ lname });
    {
        llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) { b.token[i] = ids[i]; b.pos[i] = (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = (int) ids.size();
        const int rc = llama_decode(E.cap, b);
        llama_batch_free(b);
        if (rc != 0) { E.capd.want.clear(); return plan; }
    }
    const int n_vocab = llama_vocab_n_tokens(E.vocab);
    std::vector<double> nll(ids.size(), 0.0);
    std::vector<size_t> start(ids.size() + 1, 0);        // byte offset of each token in open + message
    for (size_t i = 0; i < ids.size(); ++i) start[i + 1] = start[i] + E.piece(ids[i]).size();
    for (size_t i = 1; i < ids.size(); ++i) {
        const float * lg = llama_get_logits_ith(E.cap, (int) i - 1);
        if (!lg) continue;
        double mx = lg[0];
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > mx) mx = lg[v];
        double z = 0.0;
        for (int v = 0; v < n_vocab; ++v) z += std::exp((double) lg[v] - mx);
        nll[i] = mx + std::log(z) - (double) lg[ids[i]];
    }
    std::vector<float> hs;
    if (by_state && E.capd.got.count(lname) && E.capd.got[lname].size() == ids.size() * (size_t) E.d) hs = E.capd.got[lname];
    E.capd.want.clear();
    const size_t p0 = open.size();
    std::vector<size_t> ts;
    auto span = [&](size_t a, size_t b) {                 // the tokens overlapping message bytes [a, b), and their summed -log p
        ts.clear();
        double s = 0.0;
        for (size_t i = 1; i < ids.size(); ++i) if (start[i + 1] > start[i] && start[i] < p0 + b && start[i + 1] > p0 + a) { ts.push_back(i); s += nll[i]; }
        return s;
    };
    std::vector<proto_inject_t> cands;
    std::vector<std::pair<size_t, size_t>> taken;
    auto overlaps = [&](size_t a, size_t b) { for (auto & t : taken) if (a < t.second && b > t.first) return true; return false; };
    std::vector<std::string> names;                       // 1. words that name a stored key or an alias, longest first
    {
        std::set<std::string> s;
        for (auto & c : E.cells) s.insert(c.key);
        for (auto & kv : PROTO.alias) s.insert(kv.first);
        names.assign(s.begin(), s.end());
    }
    std::sort(names.begin(), names.end(), [](const std::string & x, const std::string & y) { return x.size() > y.size(); });
    auto wc = [](unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; };
    for (auto & name : names) {
        if (name.size() < 2) continue;
        for (size_t j = message.find(name); j != std::string::npos; j = message.find(name, j + 1)) {
            const size_t e = j + name.size();
            const bool before_ok = j == 0 || !wc((unsigned char) message[j - 1]);
            const bool after_ok = e >= message.size() || (!wc((unsigned char) message[e]) && !(message[e] == '.' && e + 1 < message.size() && std::isalnum((unsigned char) message[e + 1])));
            if (!before_ok || !after_ok || overlaps(j, e)) continue;
            taken.push_back({ j, e });
            const double s = span(j, e);
            if (ts.empty() || s < INJECT.surprise) continue;
            proto_inject_t c;
            c.at = e; c.word = name; c.key = name; c.match = "key"; c.surprise = s; c.score = 1.0;
            auto al = PROTO.alias.find(name);
            if (al != PROTO.alias.end()) c.key = al->second;
            cands.push_back(c);
        }
    }
    if (by_state && !hs.empty() && MENU.fitted) {         // 2. other surprising words whose states match one memory clearly
        size_t i = 0;
        while (i < message.size()) {
            if (!proto_word_char((unsigned char) message[i])) { ++i; continue; }
            size_t a = i;
            while (i < message.size() && proto_word_char((unsigned char) message[i])) ++i;
            size_t b = i;
            while (b > a && std::strchr(".,;:-/", message[b - 1])) --b;
            while (a < b && std::strchr(".,;:-/", message[a])) ++a;
            if (b - a < 2 || overlaps(a, b)) continue;
            std::string lw = message.substr(a, b - a);
            std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
            if (proto_stop().count(lw)) continue;
            const double s = span(a, b);
            if (ts.empty() || s < INJECT.surprise) continue;
            std::vector<float> q;
            std::vector<llama_token> qids;
            for (size_t t : ts) { q.insert(q.end(), hs.begin() + (ptrdiff_t) (t * (size_t) E.d), hs.begin() + (ptrdiff_t) ((t + 1) * (size_t) E.d)); qids.push_back(ids[t]); }
            proto_menu_norm(q, E.d);
            const proto_menu_hit_t h = proto_menu_rank(E, q, qids, 2);
            if (h.keys.empty() || h.scores[0] < INJECT.min_score || (h.scores.size() > 1 && h.scores[0] - h.scores[1] < INJECT.margin)) continue;
            taken.push_back({ a, b });
            proto_inject_t c;
            c.at = b; c.word = message.substr(a, b - a); c.key = h.keys[0]; c.match = "state"; c.surprise = s; c.score = h.scores[0];
            cands.push_back(c);
        }
    }
    std::sort(cands.begin(), cands.end(), [](const proto_inject_t & x, const proto_inject_t & y) { return x.surprise > y.surprise; });
    for (auto & c : cands) {
        if ((int) plan.size() >= INJECT.max) break;
        for (int i = (int) E.cells.size() - 1; i >= 0; --i) {          // the newest mention of the key
            if (E.cells[(size_t) i].key != c.key) continue;
            c.b = block_twoside_rows(E, E.cells[(size_t) i], rows);
            break;
        }
        if (c.b.n <= 0) continue;
        emit({ { "type", "inject" }, { "word", c.word }, { "key", c.key }, { "rows", c.b.n }, { "match", c.match },
               { "surprise", std::round(c.surprise * 100.0) / 100.0 }, { "score", std::round(c.score * 1000.0) / 1000.0 } });
        plan.push_back(c);
    }
    std::sort(plan.begin(), plan.end(), [](const proto_inject_t & x, const proto_inject_t & y) { return x.at < y.at; });
    return plan;
}

// ONE ANSWER: the loop the checkpoint was trained and evaluated with. injects (0021): blocks that go right after words
// of the last user message, which sits just before the assistant's opening at the end of `prompt`
static json run_protocol(engine_t & E, const std::string & prompt, const std::string & last_user, int max_new,
                         const std::function<bool(const json &)> & emit, std::atomic<bool> * stop,
                         std::vector<proto_inject_t> * injects = nullptr) {
''', "plan")

H = once(H, r'''    bool finished = false, ok = proto_feed_tokens(E, E.tok(prompt, true), true);
''', r'''    bool finished = false, ok = true;
    {                                                     // 0021: the prompt in pieces, a memory block after each injected word
        static const std::string tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        size_t pos = 0;
        const size_t base = prompt.size() >= tail.size() + last_user.size() ? prompt.size() - tail.size() - last_user.size() : std::string::npos;
        if (injects && base != std::string::npos && prompt.compare(base, last_user.size(), last_user) == 0) {
            for (auto & inj : *injects) {
                const size_t at = base + inj.at;
                if (!ok || at <= pos || at > base + last_user.size()) continue;
                ok = proto_feed_tokens(E, E.tok(prompt.substr(pos, at - pos), true), false) && proto_feed_block(E, inj.b);
                pos = at;
            }
        }
        ok = ok && proto_feed_tokens(E, E.tok(prompt.substr(pos), true), true);
    }
''', "feed")

H = once(H, r'''{ "remember_mode", PROTO.remember_mode }, { "auto_menu", MENU.k } };''',
         r'''{ "remember_mode", PROTO.remember_mode }, { "auto_menu", MENU.k },
             { "auto_inject", INJECT.rows }, { "inject_max", INJECT.max }, { "inject_match", INJECT.match }, { "inject_surprise", INJECT.surprise } };''', "state")

H = once(H, r'''        const int auto_menu = body.contains("auto_menu") && body["auto_menu"].is_number() ? body["auto_menu"].get<int>() : -1;   // 0019: -1 = the --auto-menu default
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, msgs, system, last, max_new, auto_menu](size_t, httplib::DataSink & sink) {
''', r'''        const int auto_menu = body.contains("auto_menu") && body["auto_menu"].is_number() ? body["auto_menu"].get<int>() : -1;   // 0019: -1 = the --auto-menu default
        const int auto_inject = body.contains("auto_inject") && body["auto_inject"].is_number() ? body["auto_inject"].get<int>() : -1;   // 0021: -1 = --auto-inject
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, msgs, system, last, max_new, auto_menu, auto_inject](size_t, httplib::DataSink & sink) {
''', "chat-body")

H = once(H, r'''            run_protocol(E, proto_chat_prompt(msgs, sys), last, max_new, emit, &stop_flag);
''', r'''            std::vector<proto_inject_t> injects = proto_plan_injects(E, last, auto_inject < 0 ? INJECT.rows : auto_inject, emit);   // 0021
            run_protocol(E, proto_chat_prompt(msgs, sys), last, max_new, emit, &stop_flag, &injects);
''', "chat-run")

H = once(H, r'''    } else if (t == "recall") {
''', r'''    } else if (t == "inject") {                         // 0021
        printf("  [injected after \"%s\": %s, %d memory tokens (surprise %.1f nats, %s match)]\n", ev.value("word", "").c_str(), ev.value("key", "").c_str(),
               ev.value("rows", 0), ev.value("surprise", 0.0), ev.value("match", "").c_str());
    } else if (t == "recall") {
''', "print")

H = once(H, r'''    "  /menu K                  put the K memories a message brings to mind in the system prompt (0 is off)\n"
''', r'''    "  /menu K                  put the K memories a message brings to mind in the system prompt (0 is off)\n"
    "  /inject N                up to N memory tokens after each surprising word that names a memory (0 is off)\n"
''', "help")

H = once(H, r'''        else if (cmd == "/menu") { MENU.k = std::max(0, atoi(rest.c_str())); printf("auto menu %d\n", MENU.k); }   // 0019
''', r'''        else if (cmd == "/menu") { MENU.k = std::max(0, atoi(rest.c_str())); printf("auto menu %d\n", MENU.k); }   // 0019
        else if (cmd == "/inject") { INJECT.rows = std::max(0, atoi(rest.c_str())); printf("auto inject %d\n", INJECT.rows); }   // 0021
''', "repl-cmd")

H = once(H, r'''            json done = run_protocol(E, proto_chat_prompt(history, sys), line, max_new, printer, nullptr);
''', r'''            std::vector<proto_inject_t> injects = proto_plan_injects(E, line, INJECT.rows, printer);   // 0021
            json done = run_protocol(E, proto_chat_prompt(history, sys), line, max_new, printer, nullptr, &injects);
''', "repl-run")

# ---------------------------------------------------------------- steermem.cpp
C = once(C, r'''        else if (a == "--chat") chat_repl = true;''', r'''        else if (a == "--auto-inject") INJECT.rows = atoi(next().c_str()); else if (a == "--inject-max") INJECT.max = atoi(next().c_str());   // 0021
        else if (a == "--inject-surprise") INJECT.surprise = atof(next().c_str()); else if (a == "--inject-min-score") INJECT.min_score = atof(next().c_str());
        else if (a == "--inject-match") { INJECT.match = next(); if (INJECT.match != "key" && INJECT.match != "state") { fprintf(stderr, "--inject-match takes key or state\n"); return 1; } }
        else if (a == "--chat") chat_repl = true;''', "args")

C = once(C, r'''(0020: where that index is saved; default the --memory-out file + .menu)\n");''',
         r'''(0020: where that index is saved; default the --memory-out file + .menu)\n"
                        "           [--auto-inject N] [--inject-max K] [--inject-surprise T] [--inject-match key|state] [--inject-min-score S]   (0021: memory after surprising words)\n");''', "usage")

C = once(C, r'''if (MENU.k > 0) proto_menu_build(E);''', r'''if (MENU.k > 0 || (INJECT.rows > 0 && INJECT.match == "state")) proto_menu_build(E);''', "build")

C = once(C, r'''        json done = run_protocol(E, proto_chat_prompt(msgs, sys), prompt, max_new > 64 ? max_new : 700, printer, nullptr);''',
         r'''        std::vector<proto_inject_t> injects = proto_plan_injects(E, prompt, INJECT.rows, printer);   // 0021
        json done = run_protocol(E, proto_chat_prompt(msgs, sys), prompt, max_new > 64 ? max_new : 700, printer, nullptr, &injects);''', "one-shot")

io.open(hp, "w", encoding="utf-8", newline="\n").write(H)
io.open(cp, "w", encoding="utf-8", newline="\n").write(C)
print("0021 applied to %s and %s" % (hp, cp))

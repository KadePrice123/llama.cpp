"""fork edit 0019 (2026-09-14): THE AUTO MENU -- what a message brings to mind, found on the model's own states.

The index and the matching live in llama/steermem_protocol.hpp (proto_menu_*: every memory's content-token states at
one layer, z-scored; the message matched token by token, each token's best match weighted by its rarity in the table;
the K best keys written into the system prompt as the key menu the checkpoint was trained to read). This script wires
it into steermem.cpp:
  - main: --auto-menu K (0 = off) and --menu-layer L (hidden_states numbering: L reads l_out-(L-1), default 6)
  - the index is built after any sleep at start, before the chat, the server or a one-shot prompt
  - a one-shot --prompt gets the menu too, and prints what came to mind

  python 0019_auto_menu.py path/to/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
if "0019" in s:
    raise SystemExit("0019 is already applied to %s" % p)
if "0018" not in s:
    raise SystemExit("apply 0018 first: the auto menu is part of protocol mode")


def once(old, new, tag):
    global s
    assert s.count(old) == 1, "%s: %d matches" % (tag, s.count(old))
    s = s.replace(old, new)


once('''        else if (a == "--system") system_prompt = next();
''',
     '''        else if (a == "--system") system_prompt = next();
        else if (a == "--auto-menu") MENU.k = atoi(next().c_str()); else if (a == "--menu-layer") MENU.layer = atoi(next().c_str());   // 0019
''',
     "args")

once('''                        "           --chat | --serve PORT --ui demo_ui.html | --sleep-only | --prompt TEXT   [--sleep-at-start] [--system TEXT] [-ngl N]\\n");''',
     '''                        "           --chat | --serve PORT --ui demo_ui.html | --sleep-only | --prompt TEXT   [--sleep-at-start] [--system TEXT] [-ngl N]\\n"
                        "           [--auto-menu K] [--menu-layer L]   (0019: put the K memories a message brings to mind in the system prompt)\\n");''',
     "usage")

once('''        if (sleep_only) { llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free(); return 0; }
''',
     '''        if (sleep_only) { llama_free(E.ctx); llama_free(E.cap); llama_model_free(E.model); llama_backend_free(); return 0; }
        if (MENU.k > 0) proto_menu_build(E);          // 0019: every memory's states at the menu layer
''',
     "build")

once('''        json done = run_protocol(E, proto_chat_prompt(msgs, system_prompt), prompt, max_new > 64 ? max_new : 700,
                                 [](const json & ev) { proto_print_event(ev); return true; }, nullptr);''',
     '''        auto printer = [](const json & ev) { proto_print_event(ev); return true; };
        const std::string sys = proto_apply_menu(E, system_prompt, prompt, MENU.k, printer);   // 0019
        json done = run_protocol(E, proto_chat_prompt(msgs, sys), prompt, max_new > 64 ? max_new : 700, printer, nullptr);''',
     "one-shot")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0019 applied to %s" % p)

"""fork edit 0020 (2026-09-14): THE MENU INDEX IS SAVED -- a restart reads it instead of recomputing it.

Building the auto menu's index costs one forward pass per memory: 117 s for 583 memories on a T1200, and 3.6 s per
memory on a busy CPU, at every start. The index now lives in a file (llama/steermem_protocol.hpp, proto_menu_cache_*):
the z-score and every memory's normalised states in float16, stamped with what made them (menu layer, width,
checkpoint step, and the sizes of the model, adapter and side files). At build time a memory whose "KEY is SUMMARY"
hash matches its saved record is read back; only new or changed memories are computed, and the file is rewritten
when anything was. Memories stored during a session are appended to it as they arrive. This script wires it into
steermem.cpp:
  - main: --menu-cache FILE (default: the --memory-out file + ".menu"; "none" turns it off)
  - the protocol setup sets the cache path and the fingerprint before anything can build the index

  python 0020_menu_cache.py path/to/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
if "0020" in s:
    raise SystemExit("0020 is already applied to %s" % p)
if "0019" not in s:
    raise SystemExit("apply 0019 first: the cache saves the auto menu's index")


def once(old, new, tag):
    global s
    assert s.count(old) == 1, "%s: %d matches" % (tag, s.count(old))
    s = s.replace(old, new)


once('''    bool chat_repl = false, sleep_at_start = false, sleep_only = false;                                              // 0018
''',
     '''    bool chat_repl = false, sleep_at_start = false, sleep_only = false;                                              // 0018
    std::string menu_cache;                                                                                          // 0020
''',
     "declare")

once('''        else if (a == "--auto-menu") MENU.k = atoi(next().c_str()); else if (a == "--menu-layer") MENU.layer = atoi(next().c_str());   // 0019
''',
     '''        else if (a == "--auto-menu") MENU.k = atoi(next().c_str()); else if (a == "--menu-layer") MENU.layer = atoi(next().c_str());   // 0019
        else if (a == "--menu-cache") menu_cache = next();                                                           // 0020
''',
     "args")

once('''                        "           [--auto-menu K] [--menu-layer L]   (0019: put the K memories a message brings to mind in the system prompt)\\n");''',
     '''                        "           [--auto-menu K] [--menu-layer L]   (0019: put the K memories a message brings to mind in the system prompt)\\n"
                        "           [--menu-cache FILE|none]   (0020: where that index is saved; default the --memory-out file + .menu)\\n");''',
     "usage")

once('''        PROTO.memory_out = memory_out;
''',
     '''        PROTO.memory_out = memory_out;
        {                                                 // 0020: the saved menu index, and what must match to reuse it
            auto fsize = [](const std::string & path) { std::error_code ec; const auto n = path.empty() ? 0 : std::filesystem::file_size(path, ec); return ec ? 0ULL : (unsigned long long) n; };
            MENU.cache_path = menu_cache == "none" ? std::string() : (!menu_cache.empty() ? menu_cache : (memory_out.empty() ? std::string() : memory_out + ".menu"));
            MENU.fingerprint = "steermem-menu layer=" + std::to_string(MENU.layer) + " d=" + std::to_string(E.d) + " step=" + std::to_string(E.side.step) +
                               " model=" + std::to_string(fsize(model_path)) + " lora=" + std::to_string(fsize(lora_path)) + " side=" + std::to_string(fsize(side_path));
        }
''',
     "setup")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("0020 applied to %s" % p)

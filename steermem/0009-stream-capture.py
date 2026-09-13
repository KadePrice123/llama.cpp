#!/usr/bin/env python3
"""steermem sidecar: the streaming provider must not reference the request handler's locals.

Seen in the browser (2026-09-12 19:02): "hello?" -> net::ERR_CONNECTION_RESET on /chat.
With "stream": true the chunked content provider runs AFTER the handler lambda has
returned, and it captured the handler's local `apply` lambda by reference -- a dangling
reference, undefined behaviour, the process died mid-answer. The non-streaming path ran
inside the handler and passed the curl test. The provider now captures only what outlives
the handler (the engine, the mutex, the stop flag -- all in serve()'s frame) and copies of
the request's values, and applies the per-request switches inline.

    python 0009-stream-capture.py steermem/steermem.cpp
"""
import io
import sys

p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = '''        res.set_chunked_content_provider("application/x-ndjson", [&, prompt, max_new, memgen, memseq](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            const int mg = E.side.mem_gen, ms = E.side.mem_seq; apply(); stop_flag = false;
'''
new = '''        // the provider runs after this handler has returned: capture nothing of the handler's frame
        res.set_chunked_content_provider("application/x-ndjson", [&E, &mu, &stop_flag, prompt, max_new, memgen, memseq](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(mu);
            const int mg = E.side.mem_gen, ms = E.side.mem_seq;
            if (memgen >= 0) E.side.mem_gen = memgen;
            if (memseq > 0) E.side.mem_seq = memseq;
            stop_flag = false;
'''
assert s.count(old) == 1, "provider capture"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sidecar: the streaming provider captures only what outlives the handler")

#!/usr/bin/env python3
"""steermem_harness.py -- the smallest harness: chat with a steermem server, teach it files, tell it what to remember,
and put it to sleep. Standard library only; works against steermem-cli --serve (protocol mode) and against
demo/steermem_demo.py, because both speak API.md.

  python steermem_harness.py --url http://127.0.0.1:8160

Commands (anything else is a message to the model):
  /upload FILE                 learn a CSV, JSON, JSONL or text file at the next sleep
  /remember [KEY ::] TEXT      queue something to remember at the next sleep
  /remember-now [KEY ::] TEXT  the model writes the memory now
  /sleep                       sleep: the model writes and stores everything queued
  /queue                       what is waiting for sleep
  /memories [QUERY]            what is stored
  /system TEXT                 a system prompt, e.g. the keys worth recalling and when (empty to clear)
  /menu K                      list the K memories each message brings to mind in the system prompt (0 off)
  /inject N                    up to N memory tokens after each surprising word that names a memory (0 off)
  /new                         start a new conversation
  /quit
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def call(url, path, body=None, stream=False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read().decode("utf-8")).get("error", str(e))}
        except ValueError:
            return {"error": str(e)}
    if stream:
        return resp
    return json.loads(resp.read().decode("utf-8"))


def events(resp):
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if line:
            yield json.loads(line)


def split_key(text):
    if "::" in text:
        k, t = text.split("::", 1)
        return k.strip(), t.strip()
    return "", text.strip()


def chat(url, history, system, menu=None, inject=None):
    body = {"messages": history, "system": system, "max_new": 700}
    if menu is not None:
        body["auto_menu"] = menu
    if inject is not None:
        body["auto_inject"] = inject
    resp = call(url, "/api/chat", body, stream=True)
    if isinstance(resp, dict):
        print("error:", resp["error"])
        return None
    answer = None
    for ev in events(resp):
        t = ev["type"]
        if t == "token":
            sys.stdout.write(ev["text"])
            sys.stdout.flush()
        elif t == "menu":
            print("  [came to mind: %s]" % ", ".join(ev.get("keys") or []))
        elif t == "inject":
            print("  [injected after \"%s\": %s, %s memory tokens (surprise %s nats, %s match)]"
                  % (ev.get("word"), ev.get("key"), ev.get("rows"), ev.get("surprise"), ev.get("match")))
        elif t == "recall":
            if ev.get("hit"):
                print("\n  [recall %s -> %s]" % (ev["asked"], ev.get("summary", "")[:120]))
            else:
                print("\n  [recall %s -> nothing stored; nearest: %s]" % (ev["asked"], ", ".join(ev.get("near") or [])))
        elif t == "tool":
            print("\n  [tool %s(%s) -> %s]" % (ev["name"], json.dumps(ev["args"]), ev["result"][:160].replace("\n", " ")))
        elif t == "remember":
            print("\n  [%s: %s is %s]" % (ev.get("status", "stored"), ev["key"], ev["summary"]))
        elif t == "alias":
            print("\n  [alias: %s -> %s]" % (ev["from"], ev["to"]))
        elif t == "done":
            answer = ev.get("answer", "")
            print("\n  (%s tokens, %ss)" % (ev.get("tokens"), ev.get("seconds")))
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8160")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    url = a.url.rstrip("/")
    s = call(url, "/api/state")
    if "error" in s:
        raise SystemExit("cannot reach %s: %s" % (url, s["error"]))
    print("connected: checkpoint step %s, %s keys, %s waiting for sleep, remember mode %s, auto menu %s, auto inject %s"
          % (s.get("step"), s.get("keys"), s.get("queued"), s.get("remember_mode"), s.get("auto_menu") or "off",
             s.get("auto_inject") or "off"))
    history, system, menu, inject = [], "", None, None
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        if cmd == "/quit":
            break
        if cmd == "/new":
            history = []
            print("new conversation")
        elif cmd == "/system":
            system = rest.strip()
            print("system prompt %s" % ("set" if system else "cleared"))
        elif cmd in ("/menu", "/inject"):
            if not rest.strip().isdigit():
                print("%s N -- a number, 0 for off" % cmd)
                continue
            if cmd == "/menu":
                menu = int(rest.strip())
                print("auto menu %s" % (menu or "off"))
            else:
                inject = int(rest.strip())
                print("auto inject %s" % (("%d tokens" % inject) if inject else "off"))
        elif cmd == "/upload":
            path = os.path.expanduser(rest.strip())
            if not os.path.isfile(path):
                print("no such file:", path)
                continue
            with open(path, encoding="utf-8") as f:
                r = call(url, "/api/upload", {"name": os.path.basename(path), "content": f.read()})
            print(r if "error" in r else "%s: %s to learn at sleep, %s stored now, %s skipped" % (os.path.basename(path), r["queued"], r["stored"], r["skipped"]))
        elif cmd in ("/remember", "/remember-now"):
            key, text = split_key(rest)
            r = call(url, "/api/remember", {"key": key, "context": text, "now": cmd == "/remember-now"})
            print(r if "error" in r else ("%s: %s%s" % (r["status"], r["key"], (" is " + r["summary"]) if r.get("summary") else "")))
        elif cmd == "/sleep":
            resp = call(url, "/api/sleep", {}, stream=True)
            if isinstance(resp, dict):
                print("error:", resp["error"])
                continue
            for ev in events(resp):
                if ev["type"] == "wrote":
                    print("  learned: %s is %s" % (ev["key"], ev["summary"]))
                elif ev["type"] == "failed":
                    print("  could not learn %s: %s" % (ev["key"], ev["reason"]))
                elif ev["type"] == "done":
                    print("slept: %s stored, %s left, %ss" % (ev["stored"], ev.get("left", 0), ev["seconds"]))
        elif cmd == "/queue":
            q = call(url, "/api/queue").get("queued", [])
            print("%d waiting for sleep" % len(q))
            for item in q[:30]:
                print("  %s (%s chars, from %s)" % (item["key"], item["chars"], item["source"]))
        elif cmd == "/memories":
            m = call(url, "/api/memories?q=" + urllib.request.quote(rest.strip())).get("memories", [])
            for item in m[:30]:
                print("  %s: %s" % (item["key"], item["summary"][:120]))
            print("(%d shown)" % min(len(m), 30))
        else:
            history.append({"role": "user", "content": line})
            answer = chat(url, history, system, menu, inject)
            if answer is None:
                history.pop()
            else:
                history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

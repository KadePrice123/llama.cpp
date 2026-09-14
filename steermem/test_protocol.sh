#!/bin/bash
# Protocol mode end to end on a GPU (or the CPU with NGL=0): a one-shot prompt, a batch sleep that learns a CSV, a
# scripted --chat session with every memory command, the auto menu (0019) and its saved index (0020), auto-inject
# (0021), and the HTTP API (state, chat stream, upload, queue, sleep, memories). It checks plumbing, not answer
# quality -- that is the checkpoint's job.
#   MODEL=qwen35-2b-q8_0.gguf TABLES=demo/tables UI=demo/demo_ui.html bash wsl_test_protocol.sh SIDE.gguf LORA.gguf [NGL]
# TABLES holds memories.jsonl, tool_docs.jsonl, bible_lookup.jsonl and examples.json (the steermem demo's tables).
# BIN defaults to build-cuda/bin/steermem-cli under the current directory.
set -u
SIDE=${1:?side gguf}; LORA=${2:?lora gguf}; NGL=${3:-99}
BIN=${BIN:-$PWD/build-cuda/bin/steermem-cli}
MODEL=${MODEL:?set MODEL to a Qwen3.5-2B GGUF}
T=${TABLES:?set TABLES to the demo tables folder}
UI=${UI:?set UI to demo_ui.html}
W=/tmp/steermem_test; rm -rf $W; mkdir -p $W
COMMON="--model $MODEL --lora $LORA --side $SIDE -ngl $NGL --table $T/memories.jsonl --tool-docs $T/tool_docs.jsonl --lookup $T/bible_lookup.jsonl --examples $T/examples.json --memory-out $W/learned.jsonl"
cat > $W/learn.csv <<'CSV'
key,context,summary
pump_7,"The pump in bay 7 is a Grundfos CR 10. It runs at 40 psi and was serviced in March.",
valve_2,,a shutoff valve on line 2 that stays closed during cleaning
CSV
pass=0; fail=0
check() { if [ "$2" = "0" ]; then echo "PASS  $1"; pass=$((pass+1)); else echo "FAIL  $1"; fail=$((fail+1)); fi; }

echo "=== 1. one-shot prompt through the protocol loop"
$BIN $COMMON --max-new 120 --prompt "What is Genesis 1:1?" > $W/oneshot.out 2> $W/oneshot.err
grep -q '"type":"done"' $W/oneshot.out; check "one-shot answer ends with a done event" $?
tail -c 400 $W/oneshot.out; echo

echo "=== 2. batch sleep: learn a CSV and exit"
$BIN $COMMON --learn $W/learn.csv --sleep-only > $W/sleep.out 2> $W/sleep.err
cat $W/sleep.err | grep -E "learn|protocol" ; cat $W/sleep.out
grep -q "slept: 1 stored" $W/sleep.out; check "sleep wrote and stored the context-only row" $?
grep -q '"key": *"valve_2"' $W/learned.jsonl; check "the row that carried a summary was stored as it is" $?
grep -q '"key": *"pump_7"' $W/learned.jsonl; check "the learned memory was appended to --memory-out" $?

echo "=== 3. --chat with every command"
printf '/state\n/remember-now boiler_3 :: Boiler 3 is a Cleaver-Brooks CB 200 in the basement, fed from the north gas line.\n/remember lathe_1 :: The lathe in shop 1 needs a new belt.\n/queue\n/sleep\n/memories boiler\n/alias boiler three -> boiler_3\n/mode queue\n/system Memory you can use: boiler_3.\nWhat is boiler_3?\n/menu 3\nWhere is the boiler that is fed from the north gas line?\n/inject 16\n/quit\n' \
  | $BIN $COMMON --chat --max-new 120 > $W/chat.out 2> $W/chat.err
cat $W/chat.out | head -70
grep -q "stored: boiler_3 is" $W/chat.out; check "/remember-now had the model write and store a memory" $?
grep -q "queued: lathe_1" $W/chat.out; check "/remember queued a memory for sleep" $?
grep -q "learned: lathe_1 is" $W/chat.out; check "/sleep learned the queued memory" $?
grep -q "boiler_3:" $W/chat.out; check "/memories finds what was stored" $?
grep -q "remember mode queue" $W/chat.out; check "/mode switched the remember mode" $?
grep -q "tokens," $W/chat.out; check "a message ran through the protocol loop" $?
grep -q "auto menu 3" $W/chat.out; check "/menu turned the auto menu on" $?
grep -q "came to mind:.*boiler_3" $W/chat.out; check "a memory stored in this session comes to mind for a paraphrase" $?
grep -q "auto inject 16" $W/chat.out; check "/inject set the memory tokens to inject" $?

echo "=== 4. the auto menu: the message matched against every memory on the model's own states"
$BIN $COMMON --auto-menu 3 --max-new 60 --prompt 'Quote the verse that starts "Attai the sixth, Eliel the seventh" for me.' > $W/menu.out 2> $W/menu.err
grep -E "menu index" $W/menu.err; grep -E "came to mind" $W/menu.out
grep -q "menu index: [0-9]* memories at layer 6" $W/menu.err; check "--auto-menu builds the index at start" $?
grep -q "came to mind:.*1 Chronicles 12:11" $W/menu.out; check "the quoted verse comes to mind" $?
$BIN $COMMON --auto-menu 3 --max-new 60 --prompt 'Quote the verse that starts "Attai the sixth, Eliel the seventh" for me.' > $W/menu2.out 2> $W/menu2.err
grep -E "menu index" $W/menu2.err; ls -la $W/learned.jsonl.menu
grep -q ", 0 computed)" $W/menu2.err; check "a restart reads the saved index (--memory-out + .menu) and computes nothing" $?
grep -q "came to mind:.*1 Chronicles 12:11" $W/menu2.out; check "the saved index brings up the same verse" $?

echo "=== 5. auto-inject: a surprising word that names a memory gets its block before the answer"
$BIN $COMMON --auto-inject 16 --inject-surprise 0 --inject-match key --max-new 60 --prompt 'What is calculate_averages?' > $W/inject.out 2> $W/inject.err
grep -E "injected after" $W/inject.out; tail -c 300 $W/inject.out; echo
grep -q 'injected after "calculate_averages": calculate_averages, 16 memory tokens' $W/inject.out; check "--auto-inject 16 puts 16 rows of the named memory after the word" $?
grep -q '"type":"done"' $W/inject.out; check "the answer runs with the block spliced into the message" $?
! grep -q 'recall calculate_averages(.* -> nothing stored' $W/inject.out; check "a recall written in call form reaches the tool's memory (0023)" $?
$BIN $COMMON --auto-inject 16 --inject-surprise 1000 --inject-match key --max-new 30 --prompt 'What is calculate_averages?' > $W/inject2.out 2> $W/inject2.err
! grep -q "injected after" $W/inject2.out; check "nothing is injected when no word is surprising enough" $?
$BIN $COMMON --auto-inject 16 --max-new 60 --prompt 'What does the Hebrew name Attai mean in 1 Chronicles 12:11?' > $W/inject3.out 2> $W/inject3.err
grep -E "injected after|menu index" $W/inject3.out $W/inject3.err
grep -q '"type":"done"' $W/inject3.out; check "the state match (the default) plans and answers" $?

echo "=== 6. the HTTP API"
$BIN $COMMON --serve 8171 --ui $UI --inject-surprise 0 --inject-match key > $W/serve.out 2> $W/serve.err &
SPID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8171/api/state && break; sleep 1; done
curl -s http://127.0.0.1:8171/api/state | head -c 300; echo
curl -s http://127.0.0.1:8171/ | grep -q "<title>steermem</title>"; check "GET / serves the page" $?
curl -s -X POST http://127.0.0.1:8171/api/upload -d '{"name":"notes.txt","content":"The compressor in room 12 trips when the oil is low.\n\nIts reset button is behind the left panel."}' > $W/upload.json; cat $W/upload.json; echo
grep -q '"queued":1' $W/upload.json; check "upload queued a text file" $?
curl -s http://127.0.0.1:8171/api/queue | grep -q "notes.txt"; check "the queue lists it" $?
curl -s -N -X POST http://127.0.0.1:8171/api/sleep -d '{}' > $W/sleep_api.out; cat $W/sleep_api.out
grep -q '"type":"wrote"' $W/sleep_api.out; check "POST /api/sleep streamed a written memory" $?
curl -s -X POST http://127.0.0.1:8171/api/remember -d '{"key":"forklift_4","summary":"a forklift whose battery is swapped every Friday"}' | grep -q '"status":"stored"'; check "POST /api/remember with a summary stores it" $?
curl -s -N -X POST http://127.0.0.1:8171/api/chat -d '{"messages":[{"role":"user","content":"What is forklift_4?"}],"max_new":80}' > $W/chat_api.out
grep -q '"type":"done"' $W/chat_api.out; check "POST /api/chat streams to a done event" $?
! grep -q '"type":"inject"' $W/chat_api.out; check "with auto-inject off by default, nothing is injected" $?
curl -s "http://127.0.0.1:8171/api/memories?q=forklift" | grep -q "forklift_4"; check "GET /api/memories finds it" $?
curl -s -N -X POST http://127.0.0.1:8171/api/chat -d '{"messages":[{"role":"user","content":"Where does the file visualize_trading_strategy touched live?"}],"max_new":60,"auto_menu":3}' > $W/menu_api.out
head -1 $W/menu_api.out
head -1 $W/menu_api.out | grep -q '"type":"menu"'; check "POST /api/chat with auto_menu streams the menu first" $?
curl -s -N -X POST http://127.0.0.1:8171/api/chat -d '{"messages":[{"role":"user","content":"What is forklift_4?"}],"max_new":40,"auto_inject":16}' > $W/inject_api.out
head -1 $W/inject_api.out
head -1 $W/inject_api.out | grep '"type":"inject"' | grep -q '"key":"forklift_4"'; check "POST /api/chat with auto_inject streams the injection before start" $?   # the JSON keys come out sorted
curl -s http://127.0.0.1:8171/api/state | grep -q '"auto_inject":0'; check "per-request settings leave the defaults alone" $?
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "=== $pass passed, $fail failed"
[ "$fail" = "0" ] && echo STEERMEM-PROTOCOL-TEST-PASS || echo STEERMEM-PROTOCOL-TEST-FAIL

# steermem harness API

Both backends speak it: `demo/steermem_demo.py` (Python, the trainer's own code) and `steermem-cli --serve` in
protocol mode (the llama.cpp fork). `demo_ui.html` and `steermem_harness.py` are clients of it, and so can be anything
else.

## How memory gets built

There are three ways in, and one sleep:

1. **The model decides.** While answering, it can write `<|remember|>KEY is ...<|remember|>`, typically after the
   user agrees to keep something. With `--remember-mode store` (the default) the memory is stored at once. With
   `queue` it waits for the next sleep.
2. **The harness stores.** `POST /api/remember` with a `summary` stores it now, as written. With only a `context`, the
   model writes the memory itself: at the next sleep, or immediately when `now` is true.
3. **Files.** `POST /api/upload` (or `--learn FILE` at start) queues CSV, JSON, JSONL or plain text for learning.
4. **Sleep.** `POST /api/sleep` (or `--sleep-at-start`, or `--sleep-only` for a batch job that exits) goes through
   the queue. For each item the model reads the context and writes `KEY is ...`, the states it produced are captured,
   and the memory is stored.

Every stored memory is appended to the memory file (`--memory-out`), which is loaded again at start.

## How memory gets found

Three ways, and they work together:

1. **By name.** The model writes `<recall>KEY<recall>` and the table answers right there: the memory's states on a
   hit, or the null block and the nearest keys it holds on a miss.
2. **By what the message brings to mind** (`--auto-menu K`). Every memory is also held as the model's own token
   states at one layer (`--menu-layer`, default 6). Before an answer, the user's message is matched against them
   token by token: each message token takes its best match in a memory, weighted by how rare that token is in the
   table. The K best keys go into the system prompt as a key menu, in the format the model was trained to read:

   ```
   Memory you can use (recall with <recall>KEY<recall>):
   - Matthew 24:44: the verse that begins "Therefore be ye also ready...". Recall it before quoting it.
   - TSLA_Trade_Data.csv: the file visualize_trading_strategy touched. Recall it to find where it lives before reading it.
   If none of these fit, just answer normally.
   ```

   The model still decides whether to recall one. The index is built when the backend starts (one forward pass per
   memory), or on the first request that asks for a menu, and every memory stored after that joins it at once. It is
   saved next to the memory file (`--menu-cache`), stamped with the model that made it, so a restart reads it back and
   computes only the memories that are new or changed.
3. **Injected after a surprising word** (`--auto-inject N`). Before the answer starts, the model reads the message
   with no memory. A word it finds surprising (its tokens' summed -log p at or above `--inject-surprise` nats, default
   8) that names a stored key, or, with `--inject-match state` (the default), whose own hidden states match one memory
   clearly, gets up to N rows of that memory's block placed right after it in the message: the same rows a recall
   delivers, with no `<recall>` needed. Measured on the step-600 checkpoint over 60 memories (answer loss with no
   memory 1.989): 2 rows made it worse (+0.09), 8 rows -0.43, 16 rows -0.74, 32 rows -1.31, the whole block (43 rows
   on average) -1.68, against -1.87 for a real recall -- and on the stored fact words the whole block matched the
   recall exactly. A wrong memory's block costs +0.65, which is why a word must name a key or match one clearly.
   A block is never longer than its memory, so N = 64 gives the whole block for most memories. `--inject-max` limits
   how many words per message get one (default 3, the most surprising first).

## Routes

| route | body | answer |
|---|---|---|
| `GET /api/state` | | `step, keys, memories, queued, manuals, lookup, examples, device, dtype, layers, remember_mode, auto_menu, auto_inject, inject_max, inject_match, inject_surprise` |
| `GET /api/memories?q=TEXT` | | `memories: [{key, summary, source}]`, `aliases: {bad: good}` |
| `POST /api/chat` | `messages: [{role, content}]`, `system?`, `max_new?`, `auto_menu?`, `auto_inject?` | NDJSON events, below |
| `POST /api/remember` | `key?`, `context?`, `summary?`, `now?` | `status: stored \| queued`, `key`, `summary?` |
| `POST /api/upload` | `name`, `content` (the file's text), `format?` (`csv`, `json`, `jsonl`, `text`) | `queued`, `stored`, `skipped` |
| `GET /api/queue` | | `queued: [{key, chars, source}]` |
| `POST /api/sleep` | `limit?` | NDJSON: `sleep {queued}`, `wrote {key, summary}`, `failed {key, reason}`, `done {stored, seconds}` |
| `POST /api/stop` | | stops the answer or the sleep in progress |

A key is optional for `/api/remember` and in uploads. Without one, the first words of the context become the key.
`auto_menu` and `auto_inject` in a chat body override `--auto-menu` and `--auto-inject` for that request only (`0`
turns it off).

### Upload formats

- **CSV:** a header row with `key` and one of `context`, `text` or `content`; `summary` is optional.
- **JSON:** a list of objects with those fields, or `{"memories": [...]}` / `{"cells": [...]}`.
- **JSONL:** one object per line.
- **text** (anything else): split at blank lines into pieces of up to about 1,200 characters. The keys are the file
  name, or `NAME part N` when there are several pieces.

A row that already carries a `summary` is stored as it is. A row with only a context waits for sleep.

### Chat events

| event | fields |
|---|---|
| `menu` | `keys`, `scores`: what the message brought to mind (only with an auto menu) |
| `inject` | `word`, `key`, `rows`, `match` (`key` or `state`), `surprise`, `score`: a memory placed after a word (only with auto-inject) |
| `start` | |
| `token` | `text` |
| `recall` | `asked`, `hit`, `rows`, then `key`, `summary` on a hit or `near` (the nearest keys held) on a miss |
| `tool` | `name`, `args`, `result`; the tool's response starts the next assistant turn |
| `turn` | a new assistant turn begins |
| `remember` | `key`, `summary`, `status` (`stored` or `queued`) |
| `alias` | `from`, `to` |
| `done` | `tokens`, `seconds`, `answer` |

`menu` and `inject` events come before `start`.

## Flags

| flag | what it does |
|---|---|
| `--memory-out FILE` | where learned memories are appended; loaded at start |
| `--learn FILE` | queue a CSV / JSON / JSONL / text file at start (repeatable) |
| `--sleep-at-start` | process the queue before serving |
| `--sleep-only` | process the queue, store, and exit |
| `--remember-mode store\|queue` | what a `<|remember|>` in an answer does |
| `--auto-menu K` | put the K memories the message brings to mind in the system prompt (see above); 0 is off |
| `--menu-layer L` | the layer whose token states the menu matches, counted like Hugging Face `hidden_states` (default 6) |
| `--menu-cache FILE` | where the menu index is saved (default: the memory file + `.menu`, or `.menu.pt` in the Python demo); `none` turns it off |
| `--auto-inject N` | up to N memory tokens after each surprising word that names (or clearly matches) a stored memory; 0 is off |
| `--inject-max K` | the most words per message that get a memory injected (default 3) |
| `--inject-surprise T` | how surprising a word must be, in nats (default 8) |
| `--inject-match key\|state` | `key`: only words that name a stored key or alias; `state` (default): also a clear hidden-state match |
| `--inject-min-score S` | with `state`: the least match score (default 0.6), and 0.05 over the runner-up |

## The fork's command line (`steermem-cli --chat`)

The same features without a harness: type a message to talk to the model, or a command.

| command | what it does |
|---|---|
| `/upload FILE` | learn a CSV, JSON, JSONL or text file at the next sleep |
| `/remember [KEY ::] TEXT` | queue something to remember at the next sleep |
| `/remember-now [KEY ::] TEXT` | the model writes that memory now |
| `/sleep [N]` | write and store what is queued (at most N) |
| `/queue`, `/memories [QUERY]`, `/state` | what is waiting, what is stored, the settings |
| `/alias BAD -> GOOD` | another name for a stored key |
| `/system TEXT` | a system prompt (empty clears it) |
| `/mode store\|queue` | what the model's own `<|remember|>` does |
| `/menu K` | the auto menu for the next messages (0 is off) |
| `/inject N` | auto-inject up to N memory tokens per surprising word for the next messages (0 is off) |
| `/new`, `/quit` | a new conversation; leave |

# harness

A minimal agentic coding harness in Go: a plain-text, line-oriented CLI that
drives a tool-using LLM loop against local files, shell commands, and git.

## What it is

- **Small and legible.** The whole system is meant to be readable in an
  afternoon — one purpose per package, no framework.
- **Zero third-party dependencies.** Go standard library only. SSE parsing, diff
  application, HTML-to-text reduction, and retries are all small enough to own.
- **Generic over providers.** One internal message/streaming model with two HTTP
  dialects: **Anthropic Messages** and **OpenAI Chat Completions**. The
  OpenAI-style path is the ecosystem standard — the same code works against
  OpenAI, vLLM, Ollama, OpenRouter, and llama.cpp via a configurable base URL.
- **No sandbox, no permission prompts.** The harness assumes it is launched
  inside an already-sandboxed environment; tools run with the process's
  privileges, immediately.
- **First-class git.** A dedicated `git` tool plus a git summary in the system
  prompt.

The full behavioral specification lives in [`docs/design.md`](docs/design.md).
The end-to-end verification matrix is in [`docs/smoke.md`](docs/smoke.md).

## Build

```sh
go build ./cmd/harness     # produces ./harness
```

Requires Go 1.24+ (the toolchain uses range-over-func). Verify a checkout with:

```sh
go build ./... && go vet ./... && go test ./...
```

## Quickstart

API keys are read from the **environment only** — never from flags or the config
file, because both leak into shell history and committed dotfiles.

### Anthropic

```sh
export ANTHROPIC_API_KEY=sk-ant-...
./harness -model claude-opus-4-8                       # interactive REPL
./harness -model claude-opus-4-8 -p "summarize README.md"   # one-shot
```

### OpenAI

```sh
export OPENAI_API_KEY=sk-...
./harness -model gpt-5.5
./harness -model gpt-5.5 -p "what files are in internal/?"
```

### A local OpenAI-compatible server (Ollama)

No key is needed when the base URL is non-default (local servers need none):

```sh
ollama serve &
ollama pull llama3.2
./harness -model llama3.2 -base-url http://localhost:11434/v1
```

The base URL supplies scheme/host/prefix only; the dialect appends its standard
path (`/chat/completions` or `/messages`).

### Provider selection

`-model` is primary. The provider is **inferred** from the model name: anything
starting with `claude` uses the Anthropic dialect, everything else uses the
OpenAI-compatible dialect (the right fallback for arbitrary local model names).
An explicit `-provider` overrides the inference.

### One-shot mode

In one-shot mode (`-p`) the **assistant's text goes to stdout** while tool
summaries, the usage line, notices, and errors go to stderr — so you can capture
exactly the answer:

```sh
./harness -model gpt-5.5 -p "explain this repo in one paragraph" > answer.txt
```

`-p -`, or piping into stdin, reads the prompt from stdin; with both a flag value
and piped stdin they are concatenated, so `harness -p "summarize:" < notes.txt`
works. Exit codes: `0` completed, `1` runtime error, `2` usage error, `130`
interrupted.

## Flags

```
-p <prompt|->     one-shot mode; "-" or piped stdin reads the prompt from stdin
-provider <name>  openai | anthropic (default: inferred from -model)
-model <id>       model id (required)
-base-url <url>   provider base URL (e.g. http://localhost:11434/v1 for Ollama)
-system <text|@file>           append to the system prompt (project notes)
-system-override <text|@file>  replace the builtin instructions
-no-env           omit the environment context block (cwd/os/date/git)
-resume <file>    load a session transcript and continue
-session <file>   explicit session save path
-max-steps <n>    model round-trips per user turn (default 50)
-context-window <n>   override the model's context window (tokens)
-v                show tool result snippets (first ~5 lines, dimmed)
-no-color         disable color (also: NO_COLOR env var; color is TTY-only anyway)
-config <file>    alternate config path
-h, --help        print this usage screen and exit 0
```

`-system`/`-system-override` accept a `@file` reference (a literal leading `@` is
escaped as `@@`).

### Configuration and environment

Precedence is **flags > environment > config file > built-in defaults**.

- Environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, plus `HARNESS_*` equivalents for most flags
  (`HARNESS_MODEL`, `HARNESS_MAX_STEPS`, …). Environment API keys override keys
  from provider config files.
- Optional config file at `~/.config/harness/config.json` (override with
  `-config`): `provider`, `model`, `provider_configs`, and flag defaults. Provider
  config paths are resolved relative to the config file and may define `api_type`,
  `base_url`, `api_key`, models, context windows, and pricing. See
  `examples/config/` for sample files.

## Meta-commands (REPL)

Lines starting with `/` are commands; `//` sends a literal leading slash.

| command | effect |
|---|---|
| `/help` | list commands |
| `/exit`, `/quit` | save and exit |
| `/clear` | reset the conversation; rotate to a fresh session file |
| `/compact` | force compaction now |
| `/usage` | cumulative session tokens and cost |
| `/save [file]` | force save (optionally elsewhere) |
| `/model` | print provider, model, and base URL |

## Sessions

- The transcript is **saved after every turn**, atomically (write a `.tmp`, then
  rename). It auto-saves to `~/.local/state/harness/sessions/<timestamp>.json`
  (honoring `$XDG_STATE_HOME`); the path is printed at startup.
- `-session <file>` chooses an explicit path. `-resume <file>` loads any prior
  transcript and continues; `/clear` rotates to a fresh file.
- Transcripts are **provider-neutral**, so a session started against Anthropic
  resumes against an OpenAI-compatible server and vice versa. When flags disagree
  with a resumed file's provider/model, the flags win with a warning.
- A session saved mid-turn (a dangling `tool_use`) is repaired on load by
  synthesizing an `interrupted` tool result, so the resumed transcript is always
  valid for both APIs.

## Compaction

When a turn's reported input tokens reach **78%** of the model's context window
(or on `/compact`), the harness summarizes the conversation to free context. It
keeps the system prompt and the **last 4 turns verbatim**, sends everything older
to the model with a summarization instruction, and replaces the old messages with
a single summary message. The summary call's tokens and cost are folded into the
session totals and reported:

```
[compacted: 38 messages → summary · 9.1k in / 0.4k out · $0.05]
```

If the transcript is still over budget it degrades further (keep only the last
turn, then hard-truncate the largest tool results) — it never wedges. If the
summary call itself fails, compaction is aborted and the full transcript is kept,
so a visible context-length error is preferred over silent data loss.

## Interrupts

- **Ctrl-C during a turn** cancels the turn (aborting the HTTP stream and killing
  any `run_command` process group); streamed partial text is kept and
  un-executed tool calls are stripped. Prints `[cancelled]` and returns to the
  prompt.
- **A second Ctrl-C within ~1s, or Ctrl-C at the idle prompt** saves and exits
  130. **Ctrl-D** at the prompt saves and exits 0.

## Tools

`read_file`, `list_dir`, `grep`, `edit`, `write_file`, `apply_patch`,
`run_command`, `git`, `web_fetch`. See [`docs/design.md`](docs/design.md) §9 for
each tool's schema and exact behavior.

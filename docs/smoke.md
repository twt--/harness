# Smoke / verification matrix

This document records the manual smoke matrix for `harness` (design §13) and how
to re-run each leg. It complements — it does not replace — the unit and golden
suites (`go test ./...`, 278 test functions across 14 packages).

The legs split in two groups:

- **Hermetic legs** drive the real, freshly-built binary as a subprocess against
  a throwaway OpenAI-compatible mock server bound to `127.0.0.1` (no network, no
  API keys). They are automated in `cmd/harness/integration_test.go` and run as
  part of `go test ./...`. The mock lives only in `_test.go`, so it is never
  compiled into the shipped binary.
- **Real-API legs** require provider credentials and are **BLOCKED** in this
  environment (no `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, no local Ollama). They
  are documented below with the exact commands to run them by hand.

## Environment at time of writing

- Go: `go1.26.4 darwin/arm64`
- `ANTHROPIC_API_KEY`: not set
- `OPENAI_API_KEY`: not set
- Ollama: not installed / not running

## Hermetic legs (automated, PASS)

Run them all:

```sh
go test ./cmd/harness/ -run TestSmoke -v
# or under the race detector:
go test -race ./cmd/harness/ -run TestSmoke
```

Last run: all three PASS, no data races.

| Leg | Test | What it asserts |
|---|---|---|
| Local OpenAI-compatible server, tool round-trip | `TestSmokeToolRoundTrip` | The mock streams a `read_file` tool call, the harness executes it, and a **second** request to the mock carries the `role:"tool"` result with the file's content. The assistant's final text lands on **stdout**; a session file is written and passes `ValidateTranscript`. |
| `^C` during a stream | `TestSmokeInterruptMidStream` | The mock streams `partial answer` then stalls (300 ms/line). After the partial text reaches stdout, the test sends `SIGINT` to the subprocess. The process exits **130**; the saved session keeps the partial assistant text and passes `ValidateTranscript` (the §4 cancel-repair: keep streamed text, strip un-executed tool calls). |
| Resume of an interrupted session | `TestSmokeResumeInterrupted` | A crafted session whose transcript ends in a **dangling `tool_use`** is resumed with `-resume`. `session.Load` repairs it with a synthesized `tool_result` (`is_error`, text `interrupted`). The mock's single request is verified to contain that `role:"tool"` / `tool_call_id` message, and the run completes against the mock's text turn. |

### How the mock works

`recordingMock.ServeHTTP` decodes each `/v1/chat/completions` request body,
records it, and replies with a scripted SSE stream (OpenAI chunk shape:
`choices[].delta` for text, `choices[].delta.tool_calls[]` fragments for a tool
call, `finish_reason`, a trailing usage chunk, then `data: [DONE]`). The model
name `mock-model` infers the OpenAI dialect (design §7); the non-default
`-base-url` makes an empty API key acceptable.

## Real-API legs (BLOCKED — run by hand once credentials exist)

These exercise the live provider dialects end to end. The harness reads keys from
the environment only (design §2/§7); never pass them as flags.

### Anthropic Messages API

```sh
export ANTHROPIC_API_KEY=sk-ant-...
go build ./cmd/harness

# One-shot, assistant text captured to a file (tool summaries/usage go to stderr):
./harness -model claude-opus-4-8 -p "list the Go files in this directory using your tools" > answer.txt

# Interactive REPL (try /help, a prompt that needs a tool, then /usage, /exit):
./harness -model claude-opus-4-8
```

Expect: a per-turn usage line on stderr with a dollar cost (claude models are in
the price registry), tool one-liners on stderr, the final answer on stdout, and a
session auto-saved under `~/.local/state/harness/sessions/`.

### OpenAI Chat Completions API

```sh
export OPENAI_API_KEY=sk-...
go build ./cmd/harness

./harness -model gpt-5.5 -p "read README.md and summarize it in two sentences" > answer.txt
./harness -model gpt-5.5            # interactive
```

Expect: same behavior as above. Cost appears only if the model is in the price
registry; unknown model names show token counts without a dollar figure.

### Local Ollama (OpenAI-compatible, no key)

```sh
ollama serve &                 # if not already running
ollama pull llama3.2

go build ./cmd/harness
./harness -model llama3.2 -base-url http://localhost:11434/v1 \
  -p "what files are in this directory?"
```

Expect: provider inferred as `openai` (non-`claude*` model), empty API key
accepted because the base URL is non-default, token counts with no dollar figure
(local model not in the price registry). Tool reliability depends on the local
model's tool-calling support.

### Interrupt and resume against a real provider

To reproduce the interrupt/resume legs against a live API rather than the mock:

```sh
# Start a turn that will take a while, then press Ctrl-C once mid-stream:
./harness -model claude-opus-4-8 -session /tmp/s.json
> write a very long essay about distributed systems
# ^C  -> [cancelled], partial text kept; ^C again (or at the idle prompt) -> exit 130

# Resume the saved session and continue:
./harness -model claude-opus-4-8 -resume /tmp/s.json -p "continue"
```

Expect: the resumed transcript is re-sent intact; if the prior run was saved
mid-tool-call, the dangling `tool_use` is repaired with an `interrupted`
`tool_result` before the next request (design §4, §11).

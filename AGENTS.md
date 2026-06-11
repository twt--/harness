# AGENTS.md — harness

An agentic coding harness in Go: plain-text, line-oriented CLI that drives a tool-using LLM loop against local files, shell commands, and git.

## Project philosophy

- **Small and legible.** The whole system should be readable in an afternoon. One purpose per package, no framework.
- **Zero third-party dependencies.** Go stdlib only. SSE parsing, diff application, HTML-to-text reduction, and retries are all small enough to own.
- **Provider-agnostic.** Two HTTP dialects (Anthropic Messages + OpenAI Chat Completions), same internal model.
- **No sandbox, no permission prompts.** Tools run with process privileges, immediately. Assume the harness itself runs sandboxed.
- **First-class git.** Dedicated `git` tool plus git summary in the system prompt.

## Build, test, lint

```sh
go build ./cmd/harness
make                          # same thing, shorter
make test                     # go test ./...
```

Verify a full checkout:

```sh
go build ./... && go vet ./... && go test ./...
```

There is no `go fmt` or `golangci-lint` step in CI — `go vet` is the only automated check, and the Go version is 1.24+ (range-over-func, `iter`).

## Architecture

```
cmd/harness/main.go          flags, config load, wiring, signals, REPL-vs-oneshot
internal/llm                 provider-agnostic types, Provider interface, model registry, factory
internal/llm/openai          Chat Completions wire structs, request builder, stream decode
internal/llm/anthropic       Messages wire structs, request builder, stream decode
internal/sse                 generic SSE frame reader
internal/retry               backoff + jitter + Retry-After parsing
internal/agent               turn loop, interrupt state machine, compaction
internal/tools               Tool interface, registry, dispatch, the 10 tools
internal/session             transcript persistence (atomic save/load)
internal/config              flags > env > config-file resolution
internal/ui                  REPL, streaming renderer, tool summaries, usage line
internal/sysprompt           builtin instructions + environment context
```

`internal/llm` is the shared contract — import only it in the agent loop. The factory (`internal/llm/factory`) lives in its own package to avoid import cycles with both dialects.

The internal message model is Anthropic-shaped (content-block list) because it's a lossless superset of OpenAI's flat fields.

## Code style

- Go idioms: `errors.Is`, `fmt.Errorf` with `%w`, structured errors over string matching.
- One purpose per package. If a package has a single natural responsibility, keep it cohesive.
- Small functions, explicit returns, no panic in library paths.
- No `fmt` prints in `internal/` — return errors and let the caller (`ui`, `cmd/harness`) render.
- No colored output outside of `internal/ui` — ANSI is rendered there only when stdout is a TTY; honors `NO_COLOR` and `-no-color`.
- Session files use provider-neutral JSON tags (`kind`, `tool_use_id`, `result_for_id`, …) so transcripts resume across providers.
- `ToolInput` is `json.RawMessage`, not `map[string]any` — the tool layer decodes into typed structs on demand.
- The system prompt lives on `Request.System`, not in the message list, so compaction cannot accidentally summarize it.

## Adding a tool

Tools live in `internal/tools`. Each tool:

1. Implements the `Tool` interface (name, description, parameter JSON schema, `Invoke`).
2. Is its own file for readability (`grep.go`, `edit.go`, `applypatch.go`, …).
3. Decodes `ToolInput` (a `json.RawMessage`) into a typed struct locally.
4. Returns `(string, error)`. Errors become tool-result errors surfaced to the model; do not return `fmt.Errorf` for expected failures when plain text suffices.
5. Is registered in `internal/tools` (the dispatch/registry layer).

The dispatch layer adds context recovery and a central truncation pass over long results — tools return the full result and let the framework trim.

## Adding a provider dialect

A new provider follows the OpenAI / Anthropic structure:

1. New package `internal/llm/<dialect>`.
2. Own wire structs, own request builder, own streaming decoder.
3. Implement `internal/llm.Provider` and register in `internal/llm/factory.New`.
4. Update `cmd/harness/main.go` inference if `-model` should pick it automatically.

Keep provider state in the dialect package; the loop imports only `internal/llm`.

## Testing

- Unit tests live next to the file they test (`foo.go` → `foo_test.go`), standard Go layout.
- Integration / end-to-end tests live in `cmd/harness/integration_test.go` and `cmd/harness/main_test.go` — they spin up a mock OpenAI server in-process and exercise the full REPL / one-shot / session / compaction matrix.
- Test the contract (`internal/llm` types, tool result shape, session round-trip), not wire JSON.
- Avoid flaky tests: do not sleep waiting on goroutines; use channels or `sync.WaitGroup`.
- Do not hit the network from unit tests. Use `httptest.NewServer` for provider behavior.

## Conventions worth calling out

- **Atomic saves.** `internal/session` writes to `.tmp` then renames. Any new persistence site should do the same.
- **Interrupt state machine.** `internal/agent/interrupt.go` owns Ctrl-C semantics — one Ctrl-C cancels the turn, a second within ~1s exits. Do not duplicate signal handling elsewhere.
- **Compaction.** Keeps the system prompt and last 4 turns verbatim; everything older summarizes. Budget checks use the model's context window; if unknown, `default_context_window` (default 256000) is the fallback. The summary call's tokens/cost are folded into session totals — do not drop them on error paths.
- **Retries.** All provider HTTP calls go through `internal/retry` (backoff, jitter, Retry-After parsing). Do not implement custom sleep loops.
- **SSE.** Streaming from both dialects uses `internal/sse`. If you add a new dialect, reuse it.

## What not to do

- Do not add a dependency. If stdlib is missing something, we've probably written it already — check sibling packages first.
- Do not add sub-agents, MCP, or markdown rendering. Explicit non-goals (see `docs/design.md` §1). Parallel read-only tool dispatch and gitignore-aware grep (via `git ls-files`) were adopted in v1.1 — see `docs/superpowers/specs/2026-06-11-roadmap-items-design.md`.
- Do not sandbox or permission-prompt. If the caller wants safety, they sandbox the process.
- Do not let `internal/llm` import a dialect or the factory. That direction would create a cycle and break the provider-agnostic loop.

## Before submitting

- `go build ./... && go vet ./... && go test ./...` — all three, in that order.
- Any public-facing flag changes → update `README.md` Flags table and the usage screen in `cmd/harness/main.go`.
- Any behavioral changes to tools → update `docs/design.md` §9.
- Any change to the system prompt → update `internal/sysprompt` and consider compaction behavior.

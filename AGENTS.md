# AGENTS.md - harness

Harness is a small Go CLI for running a tool-using LLM loop over local files,
shell commands, web fetches, and git. Keep it simple, stdlib-only, terminal
first, and provider-neutral.

## Hard Rules

- No third-party Go module dependencies unless explicitly approved.
- Assume no backwards compatibility or migration requirement unless stated.
- Bug fixes need regression tests.
- Use conventional commit messages. Do not open draft PRs.
- Never pipe output to `head` or `tail` unless `tee` also saves the full output.
- Do not revert or overwrite user changes unless explicitly asked.
- Do not add sandboxing, permission prompts, markdown rendering, MCP, or
  sub-agent orchestration unless explicitly requested.

## Verify

- Quick build: `go build ./cmd/harness` or `make`.
- Tests: `make test` (`go test ./...`).
- Before submitting: `go build ./... && go vet ./... && go test ./...`.
- Go version is 1.24+. CI uses `go vet`, not `golangci-lint`.

## Package Map

- `cmd/harness/main.go`: flags, config/setup, provider wiring, signals, REPL vs one-shot.
- `internal/llm`: provider-neutral types, validation, model/reasoning/pricing metadata. Must not import dialects or factory.
- `internal/llm/openai`, `internal/llm/anthropic`: HTTP dialects, request builders, stream decoders, tool-call assembly.
- `internal/llm/factory`: selects dialects; keep separate to avoid import cycles.
- `internal/agent`: turn loop, tool orchestration, interrupts, compaction.
- `internal/tools`: `Tool` interface, registry/subsets, dispatch recovery/truncation, built-ins. Inputs are `json.RawMessage`.
- `internal/config`, `internal/mode`, `internal/modelsdev`: config precedence, run modes, models.dev setup/catalog metadata.
- `internal/session`: transcripts, replay logs, compaction archives, tool artifacts. New persistence should write temp-file then rename.
- `internal/ui`, `internal/term`, `internal/logging`: REPL/one-shot rendering, terminal behavior, plaintext slog. ANSI belongs here only.
- `internal/sysprompt`, `internal/skills`: built-in prompt/env context and skill discovery/disclosure.
- `internal/sse`, `internal/retry`: shared SSE reader and provider HTTP retry/backoff.

## Code Patterns

- Keep packages cohesive and functions small. Return errors from library code; only UI/logging should print.
- Use `errors.Is`/`errors.As` and `fmt.Errorf("%w")`; avoid string matching.
- Keep the system prompt on `llm.Request.System`, never in message history.
- Preserve provider neutrality: agent code depends on `internal/llm` contracts, not dialect packages.
- Hand-write tool JSON schemas; decode inputs into typed private structs; tolerate unknown JSON keys.
- Prefer argv-style tools (`exec`, `git`, `grep`, `rg`) when shell quoting is risky; use shell commands only for shell features.

## Tests

- Unit tests live next to code; integration tests live in `cmd/harness/*_test.go`.
- Avoid network in tests except `httptest.Server`; use fake providers, fixtures, and temp dirs.
- Avoid sleeps for goroutine coordination; use channels or `sync.WaitGroup`.
- Preserve `ValidateTranscript` invariants after transcript mutations.
- Behavioral tool changes need focused tests under `internal/tools`.

## Keep Docs In Sync

- Public flags/usage: `README.md` and `cmd/harness/main.go` usage text.
- Tool behavior/schemas: `docs/design.md` section 9.
- System prompt behavior: `internal/sysprompt` tests/docs; consider compaction impact.
- Run modes: `README.md` and `docs/design.md` section 14.
- Smoke workflow changes: `docs/smoke.md`.

## Adding Things

- Tool: add one file in `internal/tools`, implement `Tool`, register it, test it, document its model-facing contract.
- Provider dialect: add `internal/llm/<dialect>`, implement `llm.Provider`, register in `internal/llm/factory`, keep dialect details out of `internal/llm`.
- Config field or flag: follow `flags > env > config > defaults`; update examples when useful.

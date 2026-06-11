# Roadmap

Prioritized improvements to the agent loop and the default command set,
from a June 2026 review of `internal/agent`, `internal/llm`, and
`internal/tools`. Items reference the code as of that review; re-verify
line-level claims before starting one.

## Done

1. **`exec` tool + `stdin` on `run_command`.** Shell quoting was the dominant
   model failure when generated content (commit messages, `python -c`
   one-liners, sed programs, JSON) traveled through `run_command` as part of a
   command line. `exec` (argv array, no shell — design §9.8) eliminates the
   failure class for arguments; `stdin` on both tools eliminates it for
   documents (`git commit -F -`, `python -`, `tee file`).

4. **Per-tool-call timeout ceiling in `Dispatch`.** Every tool call gets an
   11-minute ceiling (largest self-limit + grace); Dispatch unblocks even for
   tools that ignore ctx, returning a timed-out is_error result.

## High value — loop reliability

2. **Mid-stream retry.** `internal/retry` protects only connection setup: once
   the first byte streams, any failure (truncated body, mid-stream error
   frame) is turn-fatal (`anthropic/provider.go`, `openai/provider.go`).
   Because partial output is never committed to the transcript until a step
   completes, a failed step can be re-requested from scratch with a capped
   attempt budget. Cost: re-paying for the partial generation.

3. **Proactive compaction.** `MaybeCompact` fires *after* a turn using the
   last step's reported input tokens (`agent.go`, `compact.go`); a turn whose
   tool results balloon the context can overflow the next request before
   compaction ever runs. Add a pre-request estimate check inside the step
   loop, reusing the existing `estimateTokens` heuristic.

## Smaller wins

4. **`maxSteps` auto-continue.** Exhausting the 50-step cap stops with a "say
   continue" notice. Optionally summarize-and-continue, or make the behavior
   configurable.
6. **Defensive usage accounting.** `Agent.stream` overwrites usage on each
   `EventUsage`, trusting providers to send cumulative numbers. Correct for
   both current dialects; brittle for new OpenAI-compatible servers that send
   deltas. Accumulate defensively or normalize per dialect.
7. **Anthropic cache-breakpoint tuning.** Only two breakpoints today (system
   block + last message, `anthropic/wire.go`). A breakpoint after the static
   tool-schema array could improve cache hit rates in long sessions.

## Flagged — conflicts with documented v1 non-goals

Deliberate non-goals (AGENTS.md, design §1). Revisit the stance explicitly
before implementing; do not slip these in as incidental changes.

8. **Parallel dispatch of read-only tool calls.** The loop serializes
   parallel tool calls emitted in one step; independent reads (grep,
   read_file, list_dir) are the obvious latency win.
9. **`.gitignore`-aware grep.** The fixed denylist is predictable and
   stdlib-trivial; a correct `.gitignore` parser is a real subsystem.

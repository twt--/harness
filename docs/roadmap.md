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
2. **Mid-stream retry.** The agent loop re-requests a step from scratch when
   its stream fails after the first byte (truncated body, retryable error
   frame, transport reset), 2 retries per step with backoff. Failed-attempt
   usage still counts toward the turn total.
3. **Defensive usage accounting.** Usage events are documented as cumulative
   snapshots and merged element-wise (max) in the loop and compaction, so a
   zeroed or partial late frame cannot erase earlier numbers.

## High value — loop reliability

1. **Proactive compaction.** `MaybeCompact` fires *after* a turn using the
   last step's reported input tokens (`agent.go`, `compact.go`); a turn whose
   tool results balloon the context can overflow the next request before
   compaction ever runs. Add a pre-request estimate check inside the step
   loop, reusing the existing `estimateTokens` heuristic.

## Smaller wins

2. **`maxSteps` auto-continue.** Exhausting the 50-step cap stops with a "say
   continue" notice. Optionally summarize-and-continue, or make the behavior
   configurable.
3. **Per-tool-call timeout ceiling in `Dispatch`.** `run_command`/`exec`
   self-limit, but a hanging `web_fetch` (or future tool) blocks the turn
   until ^C. A default ceiling in the dispatch layer covers every tool.
4. **Anthropic cache-breakpoint tuning.** Only two breakpoints today (system
   block + last message, `anthropic/wire.go`). A breakpoint after the static
   tool-schema array could improve cache hit rates in long sessions.

## Flagged — conflicts with documented v1 non-goals

Deliberate non-goals (AGENTS.md, design §1). Revisit the stance explicitly
before implementing; do not slip these in as incidental changes.

5. **Parallel dispatch of read-only tool calls.** The loop serializes
   parallel tool calls emitted in one step; independent reads (grep,
   read_file, list_dir) are the obvious latency win.
6. **`.gitignore`-aware grep.** The fixed denylist is predictable and
   stdlib-trivial; a correct `.gitignore` parser is a real subsystem.

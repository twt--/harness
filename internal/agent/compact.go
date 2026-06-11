package agent

import (
	"context"
	"fmt"
	"strings"

	"harness/internal/llm"
)

// keepTurns is how many whole turns compaction preserves verbatim; everything
// older is summarized into one message (design §12).
const keepTurns = 4

// compactThresholdPct is the fraction of the context window at which the
// post-turn trigger fires: reported input tokens ≥ 78% leaves headroom for the
// summary call plus the next turn (design §12).
const compactThresholdPct = 78

// bytesPerToken is a coarse token estimate used only by the degradation ladder,
// which must decide whether a compacted transcript still overflows without a
// tokenizer or another model round-trip (design §12).
const bytesPerToken = 4

// summaryInstruction is the system prompt for the summarization call. It asks
// the model to preserve everything the loop needs to continue and to invent
// nothing (design §12).
const summaryInstruction = `You are compacting an agentic coding session to save context. ` +
	`Summarize the conversation so far so the assistant can continue seamlessly. Preserve: ` +
	`the task or goal; decisions made and their rationale; files created or modified and their current state; ` +
	`key facts learned; and open TODOs. Be specific and concise. Do not invent details that are not present.`

// summaryHeader prefixes the replacement message so the model recognizes the
// collapsed history (design §12).
const summaryHeader = "=== Summary of earlier conversation ===\n"

// MaybeCompact compacts the transcript when lastInputTokens (the input tokens
// the final step of the just-finished turn reported) is at least
// compactThresholdPct of the model's context window; otherwise it is a no-op
// (design §12, §8.1). It returns the summary call's usage (zero when no
// compaction ran) so the caller can fold it into the session totals.
func (a *Agent) MaybeCompact(ctx context.Context, lastInputTokens int, sink EventSink) (llm.Usage, error) {
	window := a.window()
	if lastInputTokens*100 < window*compactThresholdPct {
		return llm.Usage{}, nil
	}
	return a.Compact(ctx, sink)
}

// Compact collapses every turn older than the last keepTurns into a single
// model-written summary message, keeping the system prompt (it lives on
// Request.System) and the recent turns verbatim (design §12). The summary call's
// usage is returned for the session totals. On a summary-call error the
// transcript is left fully intact and the error is returned, with a warning
// reported via the sink — a visible context-length failure beats silent data
// loss. The result always satisfies the §4 invariant: kept turns are whole, so
// no tool_use/tool_result pair is ever split.
func (a *Agent) Compact(ctx context.Context, sink EventSink) (llm.Usage, error) {
	starts := turnStarts(a.transcript)
	if len(starts) <= keepTurns {
		// Nothing older than the kept turns to summarize.
		return llm.Usage{}, nil
	}

	boundary := starts[len(starts)-keepTurns]
	older := a.transcript[:boundary]
	kept := a.transcript[boundary:]

	summary, usage, err := a.summarize(ctx, older)
	if err != nil {
		sink.Notice(fmt.Sprintf("[compact failed: %v; keeping full transcript]", err))
		return llm.Usage{}, err
	}

	collapsed := len(older)
	compacted := make([]llm.Message, 0, 1+len(kept))
	compacted = append(compacted, summaryMessage(summary))
	compacted = append(compacted, kept...)

	// Degradation ladder: shrink further while the estimate still overflows
	// (design §12). Never wedge.
	compacted = a.degrade(compacted, starts)

	a.transcript = compacted
	sink.Notice(compactionReport(a.registry, a.model, collapsed, usage))
	return usage, nil
}

// summarize runs one tool-less model call over the older messages and returns
// the summary text and the call's usage.
func (a *Agent) summarize(ctx context.Context, older []llm.Message) (string, llm.Usage, error) {
	req := llm.Request{
		Model:     a.model,
		System:    summaryInstruction,
		Messages:  older,
		Reasoning: a.reasoning,
	}
	var text []byte
	var usage llm.Usage
	for ev, err := range a.provider.Stream(ctx, req) {
		if err != nil {
			return "", llm.Usage{}, err
		}
		switch ev.Kind {
		case llm.EventTextDelta:
			text = append(text, ev.Text...)
		case llm.EventUsage:
			if ev.Usage != nil {
				usage = *ev.Usage
			}
		case llm.EventDone:
			if ev.Usage != nil {
				usage = *ev.Usage
			}
		}
	}
	return string(text), usage, nil
}

// degrade applies the lower rungs of the ladder when the compacted transcript's
// estimate still exceeds budget: first drop to only the last turn, then
// hard-truncate the largest tool results in place (design §12). compacted is
// [summary, ...keptTurns]; starts indexes the pre-compaction transcript so the
// last turn's start can be located.
func (a *Agent) degrade(compacted []llm.Message, starts []int) []llm.Message {
	budget := a.window() * compactThresholdPct / 100
	if estimateTokens(compacted) <= budget {
		return compacted
	}

	// Rung 2: keep only the last turn.
	lastStart := starts[len(starts)-1]
	lastTurn := a.transcript[lastStart:]
	compacted = append([]llm.Message{compacted[0]}, lastTurn...)
	if estimateTokens(compacted) <= budget {
		return compacted
	}

	// Rung 3: hard-truncate the largest tool results in place until it fits.
	// Each pass removes the current overage from the single largest result; a
	// pass that cannot shrink anything further stops the loop so we never wedge.
	for estimateTokens(compacted) > budget {
		excessBytes := (estimateTokens(compacted) - budget) * bytesPerToken
		if !truncateLargestResult(compacted, excessBytes) {
			break
		}
	}
	return compacted
}

// turnStarts returns the indices in msgs where a turn begins: every user message
// that carries genuine user content (not solely tool_result blocks). A
// tool_result-only user message continues the current turn — it answers the
// preceding assistant's tool calls — so it never starts a new one. Keeping turns
// whole this way guarantees the §4 invariant survives compaction.
func turnStarts(msgs []llm.Message) []int {
	var starts []int
	for i, m := range msgs {
		if m.Role == llm.RoleUser && hasNonResult(m) {
			starts = append(starts, i)
		}
	}
	return starts
}

func hasNonResult(m llm.Message) bool {
	for _, b := range m.Content {
		if b.Kind != llm.BlockToolResult {
			return true
		}
	}
	return len(m.Content) == 0
}

func summaryMessage(summary string) llm.Message {
	return llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: summaryHeader + summary}},
	}
}

// minTruncResult is the smallest tool_result worth shrinking; below it the saving
// is not worth a truncation marker and the ladder stops to avoid spinning.
const minTruncResult = 256

// truncateLargestResult removes at least dropBytes from the single largest
// tool_result block, replacing its tail with a marker. It returns false when no
// tool_result is large enough to shrink usefully, so the caller stops rather than
// loops forever (never wedge, design §12).
func truncateLargestResult(msgs []llm.Message, dropBytes int) bool {
	bi, bj, bestLen := -1, -1, 0
	for i := range msgs {
		for j := range msgs[i].Content {
			b := msgs[i].Content[j]
			if b.Kind == llm.BlockToolResult && len(b.ResultText) > bestLen {
				bi, bj, bestLen = i, j, len(b.ResultText)
			}
		}
	}
	if bi < 0 || bestLen < minTruncResult {
		return false
	}
	orig := msgs[bi].Content[bj].ResultText
	keep := len(orig) - dropBytes
	if keep < minTruncResult {
		keep = minTruncResult // floor: always leave a usable head
	}
	marker := fmt.Sprintf("\n[truncated: %d of %d bytes shown after compaction]", keep, len(orig))
	replacement := orig[:keep] + marker
	if len(replacement) >= len(orig) {
		return false // already at the floor; shrinking further is not worthwhile
	}
	msgs[bi].Content[bj].ResultText = replacement
	return true
}

// estimateTokens approximates the token footprint of a message list by its byte
// size. Coarse by design: it only gates the degradation ladder (design §12).
func estimateTokens(msgs []llm.Message) int {
	bytes := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			bytes += len(b.Text) + len(b.ResultText) + len(b.ToolInput) + len(b.ToolName)
		}
	}
	return bytes / bytesPerToken
}

// compactionReport is the exact post-compaction notice (design §12):
//
//	[compacted: 38 messages → summary · 9.1k in / 0.4k out · $0.05]
//
// The cost segment is omitted for models with no price entry.
func compactionReport(registry *llm.Registry, model string, collapsed int, u llm.Usage) string {
	var b strings.Builder
	fmt.Fprintf(&b, "[compacted: %d messages → summary · %s in / %s out",
		collapsed, kiloTokens(u.InputTokens), kiloTokens(u.OutputTokens))
	if registry != nil {
		if usd, known := registry.Cost(model, u); known {
			fmt.Fprintf(&b, " · $%.2f", usd)
		}
	}
	b.WriteString("]")
	return b.String()
}

// kiloTokens renders a token count in thousands with one decimal, matching the
// design's compaction report (9100 -> "9.1k", 400 -> "0.4k").
func kiloTokens(n int) string {
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}

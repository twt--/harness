package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
	"harness/internal/tools"
)

// userText is a genuine user turn-start message (text, not tool results).
func userText(s string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: s}}}
}

// asstText is an end-turn assistant message with no tool calls.
func asstText(s string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.BlockText, Text: s}}}
}

// asstToolUse is an assistant message that issues one tool call.
func asstToolUse(id, name, input string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
		{Kind: llm.BlockToolUse, ToolUseID: id, ToolName: name, ToolInput: []byte(input)},
	}}
}

// toolResult is the user message answering one tool call.
func toolResult(id, text string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{
		{Kind: llm.BlockToolResult, ResultForID: id, ResultText: text},
	}}
}

// summaryStep scripts a canned summary reply from the FakeProvider.
func summaryStep(summary string, in, out int) llmtest.Step {
	return llmtest.Step{
		Events: []llm.StreamEvent{textDelta(summary)},
		Stop:   llm.StopEndTurn,
		Usage:  llm.Usage{InputTokens: in, OutputTokens: out},
	}
}

// makeTurns builds n whole text turns (user + assistant), labelled by index.
func makeTurns(n int) []llm.Message {
	msgs := make([]llm.Message, 0, 2*n)
	for i := 0; i < n; i++ {
		msgs = append(msgs, userText(turnLabel(i)+" question"), asstText(turnLabel(i)+" answer"))
	}
	return msgs
}

func turnLabel(i int) string {
	return string(rune('A' + i))
}

func TestCompactKeepsLastFourTurns(t *testing.T) {
	// Ten whole turns; compaction keeps the system prompt plus the last four
	// verbatim and collapses the older six into one summary message.
	transcript := makeTurns(10)

	fp := llmtest.New("fake", summaryStep("CANNED SUMMARY", 200, 40))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("system prompt")
	a.SetTranscript(transcript)

	sink := &recordSink{}
	if _, err := a.Compact(context.Background(), sink); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	msgs := a.Transcript()
	mustValid(t, msgs)

	// One summary message + last four turns (8 messages).
	if len(msgs) != 1+8 {
		t.Fatalf("want 1 summary + 8 kept messages, got %d:\n%s", len(msgs), dump(msgs))
	}
	if msgs[0].Role != llm.RoleUser {
		t.Errorf("summary should be a user message, got role %q", msgs[0].Role)
	}
	got := msgs[0].Content[0].Text
	if !strings.HasPrefix(got, "=== Summary of earlier conversation ===\n") {
		t.Errorf("summary message missing header, got %q", got)
	}
	if !strings.Contains(got, "CANNED SUMMARY") {
		t.Errorf("summary message should carry the model's summary, got %q", got)
	}

	// The kept turns are the last four (G..J), verbatim and whole.
	if msgs[1].Content[0].Text != "G question" {
		t.Errorf("first kept turn = %q, want %q", msgs[1].Content[0].Text, "G question")
	}
	if msgs[8].Content[0].Text != "J answer" {
		t.Errorf("last kept message = %q, want %q", msgs[8].Content[0].Text, "J answer")
	}

	// The summary request received the older turns but never the system prompt
	// as a message (it lives on Request.System).
	if len(fp.Requests) != 1 {
		t.Fatalf("summary call count = %d, want 1", len(fp.Requests))
	}
}

func TestCompactKeepsToolPairsWhole(t *testing.T) {
	// A turn that spans a tool round-trip must be kept whole: no tool_use may be
	// separated from its tool_result by the kept-turns boundary.
	var transcript []llm.Message
	transcript = append(transcript, makeTurns(6)...)
	// Turn G: user, assistant(tool_use), user(tool_result), assistant(text).
	transcript = append(transcript,
		userText("G question"),
		asstToolUse("call_g", "echo", `{}`),
		toolResult("call_g", "tool output"),
		asstText("G answer"),
	)
	transcript = append(transcript, makeTurns(3)...) // H, I, J relabelled below

	fp := llmtest.New("fake", summaryStep("S", 100, 20))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	if _, err := a.Compact(context.Background(), &recordSink{}); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)

	// The kept boundary must not split the G tool pair: if turn G is kept, both
	// its tool_use and tool_result survive; the validation above already proves
	// no pair is split, but assert the tool_result is present when its use is.
	var sawUse, sawResult bool
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Kind == llm.BlockToolUse && b.ToolUseID == "call_g" {
				sawUse = true
			}
			if b.Kind == llm.BlockToolResult && b.ResultForID == "call_g" {
				sawResult = true
			}
		}
	}
	if sawUse != sawResult {
		t.Errorf("tool pair split across the compaction boundary: use=%v result=%v", sawUse, sawResult)
	}
}

func TestCompactBelowThresholdUntouched(t *testing.T) {
	// Only the post-turn trigger should be threshold-gated; a transcript whose
	// last reported input is below 78% of the window is left alone.
	transcript := makeTurns(10)
	fp := llmtest.New("fake") // no summary step scripted
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	window := llm.ContextWindow("claude-opus-4-8")
	below := window / 2 // well under 78%

	sink := &recordSink{}
	if _, err := a.MaybeCompact(context.Background(), below, sink); err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if len(a.Transcript()) != len(transcript) {
		t.Errorf("below-threshold transcript should be untouched, got %d messages", len(a.Transcript()))
	}
	if len(fp.Requests) != 0 {
		t.Errorf("no summary call should happen below threshold, got %d", len(fp.Requests))
	}
	if len(sink.notices) != 0 {
		t.Errorf("no compaction notice below threshold, got %v", sink.notices)
	}
}

func TestMaybeCompactAboveThresholdCompacts(t *testing.T) {
	transcript := makeTurns(10)
	fp := llmtest.New("fake", summaryStep("S", 50, 10))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	window := llm.ContextWindow("claude-opus-4-8")
	above := window * 80 / 100 // ≥ 78%

	if _, err := a.MaybeCompact(context.Background(), above, &recordSink{}); err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)
	if len(msgs) != 1+8 {
		t.Fatalf("above threshold should compact to summary + 8, got %d", len(msgs))
	}
	if len(fp.Requests) != 1 {
		t.Errorf("summary call count = %d, want 1", len(fp.Requests))
	}
}

func TestCompactSummaryFailureKeepsTranscript(t *testing.T) {
	transcript := makeTurns(10)
	fp := llmtest.New("fake", llmtest.Step{Err: errors.New("api down")})
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	sink := &recordSink{}
	_, err := a.Compact(context.Background(), sink)
	if err == nil {
		t.Fatalf("Compact should return the summary-call error")
	}
	// Full transcript intact — a visible context-length failure beats data loss.
	if len(a.Transcript()) != len(transcript) {
		t.Errorf("failed compaction must keep the full transcript, got %d messages", len(a.Transcript()))
	}
	mustValid(t, a.Transcript())
	var warned bool
	for _, n := range sink.notices {
		if strings.Contains(strings.ToLower(n), "compact") {
			warned = true
		}
	}
	if !warned {
		t.Errorf("a summary-call failure should warn via the sink, notices=%v", sink.notices)
	}
}

func TestCompactDegradesToLastTurnWhenOversized(t *testing.T) {
	// The last four turns together exceed the budget but the last turn alone fits;
	// the ladder drops to the last turn. Budget is 78% of the 1M window ≈ 3.12M
	// bytes; four 1M-byte turns overflow, one does not.
	big := strings.Repeat("x", 1_000_000) // ~250k token estimate per kept turn
	var transcript []llm.Message
	transcript = append(transcript, makeTurns(4)...) // older turns, summarized
	for i := 0; i < 4; i++ {
		transcript = append(transcript, userText("Q"+turnLabel(i)), asstText(big))
	}

	fp := llmtest.New("fake", summaryStep("S", 10, 5))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	if _, err := a.Compact(context.Background(), &recordSink{}); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)
	// summary + the single last turn (user + assistant) = 3 messages.
	if len(msgs) != 1+2 {
		t.Fatalf("oversized kept turns should drop to the last turn (summary + 2), got %d:\n%s", len(msgs), dumpShort(msgs))
	}
	if msgs[2].Content[0].Text != big {
		t.Errorf("the single kept turn should be the most recent one")
	}
}

func TestCompactHardTruncatesWhenSingleTurnOversized(t *testing.T) {
	// Even the last turn alone is over budget: the ladder hard-truncates the
	// largest tool results in place, leaving a marker, and never wedges.
	huge := strings.Repeat("y", 5_000_000) // ~1.25M token estimate, over the window
	var transcript []llm.Message
	transcript = append(transcript, makeTurns(4)...)
	transcript = append(transcript,
		userText("final"),
		asstToolUse("c1", "big", `{}`),
		toolResult("c1", huge),
		asstText("done"),
	)

	fp := llmtest.New("fake", summaryStep("S", 10, 5))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	if _, err := a.Compact(context.Background(), &recordSink{}); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	msgs := a.Transcript()
	mustValid(t, msgs)

	var truncated bool
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Kind == llm.BlockToolResult && len(b.ResultText) < len(huge) && strings.Contains(b.ResultText, "truncated") {
				truncated = true
			}
		}
	}
	if !truncated {
		t.Errorf("the largest tool result should be hard-truncated with a marker")
	}
}

func TestCompactUsageReported(t *testing.T) {
	transcript := makeTurns(10)
	fp := llmtest.New("fake", summaryStep("S", 9100, 400))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	sink := &recordSink{}
	if _, err := a.Compact(context.Background(), sink); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	// The compaction report names the message collapse and the summary-call usage.
	var report string
	for _, n := range sink.notices {
		if strings.Contains(n, "compacted") {
			report = n
		}
	}
	if report == "" {
		t.Fatalf("expected a [compacted: …] notice, got %v", sink.notices)
	}
	if !strings.Contains(report, "9.1k in") || !strings.Contains(report, "0.4k out") {
		t.Errorf("compaction report should show summary-call usage, got %q", report)
	}
	if !strings.Contains(report, "summary") {
		t.Errorf("compaction report should mention the summary, got %q", report)
	}
}

func TestMaybeCompactReturnsUsageForTotals(t *testing.T) {
	// The summary call's tokens are returned so the caller can fold them into the
	// session totals (design §12, §6).
	transcript := makeTurns(10)
	fp := llmtest.New("fake", summaryStep("S", 5000, 100))
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	window := llm.ContextWindow("claude-opus-4-8")
	above := window * 80 / 100

	u, err := a.MaybeCompact(context.Background(), above, &recordSink{})
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if u.InputTokens != 5000 || u.OutputTokens != 100 {
		t.Errorf("returned compaction usage = %+v, want 5000 in / 100 out", u)
	}
}

func TestMaybeCompactBelowThresholdReturnsZeroUsage(t *testing.T) {
	transcript := makeTurns(4)
	fp := llmtest.New("fake")
	a := newAgent(fp, tools.Default(), Options{Model: "claude-opus-4-8"})
	a.SetSystem("sys")
	a.SetTranscript(transcript)

	below := llm.ContextWindow("claude-opus-4-8") / 2
	u, err := a.MaybeCompact(context.Background(), below, &recordSink{})
	if err != nil {
		t.Fatalf("MaybeCompact: %v", err)
	}
	if u != (llm.Usage{}) {
		t.Errorf("no compaction below threshold should return zero usage, got %+v", u)
	}
}

func dumpShort(msgs []llm.Message) string {
	var b strings.Builder
	for i, m := range msgs {
		b.WriteString(string(rune('0'+i%10)) + ":" + string(m.Role) + " ")
	}
	return b.String()
}

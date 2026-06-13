package llm

import (
	"context"
	"encoding/json"
	"iter"
)

// Provider runs one model call as a stream of events. Concrete implementations
// live in internal/llm/anthropic and internal/llm/openai.
type Provider interface {
	Name() string // "openai" | "responses" | "anthropic"

	// Stream runs one model call. The iterator yields events until a Done
	// event or a terminal error (yielded at most once, last). Consumer break
	// or ctx cancellation aborts the underlying HTTP request.
	//
	// Usage events carry cumulative snapshots of the whole call, never
	// deltas; consumers may merge them with element-wise max.
	Stream(ctx context.Context, req Request) iter.Seq2[StreamEvent, error]
}

// Request is one model call's worth of input, provider-neutral.
type Request struct {
	Model       string          `json:"model"`
	System      string          `json:"system,omitempty"`
	Messages    []Message       `json:"messages,omitempty"`
	Tools       []ToolSchema    `json:"tools,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`  // 0 = provider policy (see design §5.4)
	Temperature *float64        `json:"temperature,omitempty"` // nil = omit
	Reasoning   ReasoningConfig `json:"reasoning,omitempty"`
	StopSeqs    []string        `json:"stop_seqs,omitempty"`
}

// ToolSchema is the model-facing declaration of one tool. Parameters is the raw
// JSON Schema object owned by the tool layer; it is passed through unchanged.
type ToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema object, owned by the tool layer
}

// EventKind discriminates the StreamEvent union.
type EventKind int

const (
	EventTextDelta     EventKind = iota // incremental assistant text
	EventToolCallStart                  // tool_use began: ID + Name known
	EventToolCallDelta                  // partial JSON args (rendering only)
	EventToolCallDone                   // one call fully assembled
	EventUsage                          // usage snapshot (may arrive >1x)
	EventDone                           // turn end: StopReason + final Usage
)

// StreamEvent is one event in a provider stream. Which fields are populated
// depends on Kind.
type StreamEvent struct {
	Kind EventKind `json:"kind"`

	Text string `json:"text,omitempty"` // EventTextDelta

	// EventToolCall*; Index disambiguates parallel calls within one turn.
	Index     int             `json:"index,omitempty"`
	ToolID    string          `json:"tool_id,omitempty"`    // Start/Done
	ToolName  string          `json:"tool_name,omitempty"`  // Start/Done
	ArgsDelta string          `json:"args_delta,omitempty"` // Delta
	ToolInput json.RawMessage `json:"tool_input,omitempty"` // Done only: complete, valid JSON

	Usage      *Usage     `json:"usage,omitempty"`       // EventUsage / EventDone
	StopReason StopReason `json:"stop_reason,omitempty"` // EventDone
}

// StopReason is the normalized reason a model turn ended.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopMaxTokens StopReason = "max_tokens"
	StopStop      StopReason = "stop" // stop sequence matched
)

// Usage is the normalized token accounting for a model call. After
// normalization InputTokens means the same thing on both dialects: uncached
// input billed at full rate (see design §6).
type Usage struct {
	InputTokens      int `json:"input_tokens"` // uncached input, billed at full rate
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens"`
}

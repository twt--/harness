package anthropic

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"harness/internal/llm"
)

func floatPtr(f float64) *float64 { return &f }

// basicRequest is the canonical request exercised by the golden test and reused
// by request-building unit assertions.
func basicRequest() llm.Request {
	return llm.Request{
		Model:  "claude-opus-4-8",
		System: "You are a helpful coding assistant.",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.ContentBlock{
					{Kind: llm.BlockText, Text: "What is the weather in SF and NYC?"},
				},
			},
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					{Kind: llm.BlockText, Text: "Let me check both cities."},
					{
						Kind:      llm.BlockToolUse,
						ToolUseID: "toolu_01A",
						ToolName:  "get_weather",
						ToolInput: json.RawMessage(`{"location": "San Francisco, CA"}`),
					},
					{
						Kind:      llm.BlockToolUse,
						ToolUseID: "toolu_01B",
						ToolName:  "get_weather",
						ToolInput: json.RawMessage(`{"location": "New York, NY"}`),
					},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentBlock{
					{
						Kind:        llm.BlockToolResult,
						ResultForID: "toolu_01A",
						ResultText:  "59F and foggy",
					},
					{
						Kind:        llm.BlockToolResult,
						ResultForID: "toolu_01B",
						ResultText:  "could not reach weather service",
						ResultError: true,
					},
				},
			},
		},
		Tools: []llm.ToolSchema{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a location.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
					},
					"required": ["location"]
				}`),
			},
			{
				Name:        "list_dir",
				Description: "List directory entries.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {"path": {"type": "string"}}
				}`),
			},
		},
		// MaxTokens unset: default policy min(8192, contextWindow/4).
		// Temperature nil: omitted.
	}
}

func TestBuildRequestGolden(t *testing.T) {
	req := basicRequest()
	if err := llm.ValidateTranscript(req.Messages); err != nil {
		t.Fatalf("transcript invariant violated: %v", err)
	}

	// claude-opus-4-8 window is 1,000,000; quarter (250,000) > 8192, so the
	// default cap of 8192 applies.
	const contextWindow = 1_000_000
	got, err := json.Marshal(buildRequest(req, contextWindow))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	want, err := os.ReadFile("testdata/basic_request.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}

	if !jsonEqual(t, got, want) {
		t.Errorf("request JSON mismatch.\n got: %s\nwant: %s", canonical(t, got), canonical(t, want))
	}
}

func TestBuildRequestMaxTokensDefaultSmallWindow(t *testing.T) {
	req := basicRequest()
	// A small window makes contextWindow/4 the binding default.
	w := buildRequest(req, 20_000)
	if w.MaxTokens != 5_000 {
		t.Errorf("max_tokens = %d, want 5000 (window/4)", w.MaxTokens)
	}
}

func TestBuildRequestMaxTokensUserSet(t *testing.T) {
	req := basicRequest()
	req.MaxTokens = 333
	w := buildRequest(req, 1_000_000)
	if w.MaxTokens != 333 {
		t.Errorf("max_tokens = %d, want 333 (user-set)", w.MaxTokens)
	}
}

func TestBuildRequestTemperatureOmittedWhenNil(t *testing.T) {
	req := basicRequest()
	b, err := json.Marshal(buildRequest(req, 1_000_000))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(b, []byte("temperature")) {
		t.Errorf("temperature present in body though Temperature is nil: %s", b)
	}

	req.Temperature = floatPtr(0)
	b, err = json.Marshal(buildRequest(req, 1_000_000))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"temperature":0`)) {
		t.Errorf("temperature 0 not sent though Temperature is non-nil: %s", b)
	}
}

func TestBuildRequestReasoningEffort(t *testing.T) {
	req := basicRequest()
	req.Reasoning = llm.ReasoningConfig{Effort: "xhigh"}
	w := buildRequest(req, 1_000_000)
	if w.OutputConfig == nil || w.OutputConfig.Effort != "xhigh" {
		t.Fatalf("output_config = %+v, want effort xhigh", w.OutputConfig)
	}

	b, err := json.Marshal(w)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"output_config":{"effort":"xhigh"}`)) {
		t.Fatalf("output_config effort missing from JSON: %s", b)
	}
}

func TestBuildRequestNoSystemOmitsSystem(t *testing.T) {
	req := basicRequest()
	req.System = ""
	w := buildRequest(req, 1_000_000)
	if w.System != nil {
		t.Errorf("system block list present though System is empty")
	}
}

// jsonEqual reports whether two JSON documents are semantically equal.
func jsonEqual(t *testing.T, a, b []byte) bool {
	t.Helper()
	var av, bv any
	if err := json.Unmarshal(a, &av); err != nil {
		t.Fatalf("unmarshal a: %v\n%s", err, a)
	}
	if err := json.Unmarshal(b, &bv); err != nil {
		t.Fatalf("unmarshal b: %v\n%s", err, b)
	}
	ab, _ := json.Marshal(av)
	bb, _ := json.Marshal(bv)
	return bytes.Equal(ab, bb)
}

func canonical(t *testing.T, b []byte) string {
	t.Helper()
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return string(b)
	}
	out, _ := json.MarshalIndent(v, "", "  ")
	return string(out)
}

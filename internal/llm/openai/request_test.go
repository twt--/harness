package openai

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"harness/internal/llm"
)

func floatPtr(f float64) *float64 { return &f }

// basicRequest is the canonical request exercised by the golden test. It
// includes a second assistant message that issues a no-text, no-arg tool call,
// so the golden covers content omission and the "{}" empty-args serialization.
func basicRequest() llm.Request {
	return llm.Request{
		Model:  "gpt-5.4",
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
						ToolUseID: "call_01A",
						ToolName:  "get_weather",
						ToolInput: json.RawMessage(`{"location": "San Francisco, CA"}`),
					},
					{
						Kind:      llm.BlockToolUse,
						ToolUseID: "call_01B",
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
						ResultForID: "call_01A",
						ResultText:  "59F and foggy",
					},
					{
						Kind:        llm.BlockToolResult,
						ResultForID: "call_01B",
						ResultText:  "could not reach weather service",
						ResultError: true,
					},
				},
			},
			{
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{
					{
						Kind:      llm.BlockToolUse,
						ToolUseID: "call_01C",
						ToolName:  "list_dir",
						// No ToolInput: must serialize as "{}", never "".
					},
				},
			},
			{
				Role: llm.RoleUser,
				Content: []llm.ContentBlock{
					{
						Kind:        llm.BlockToolResult,
						ResultForID: "call_01C",
						ResultText:  "main.go",
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
		// MaxTokens unset: omitted entirely. Temperature nil: omitted.
	}
}

func TestBuildRequestGolden(t *testing.T) {
	req := basicRequest()
	if err := llm.ValidateTranscript(req.Messages); err != nil {
		t.Fatalf("transcript invariant violated: %v", err)
	}

	got, err := json.Marshal(buildRequest(req))
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

func TestBuildRequestMaxTokensOmittedWhenUnset(t *testing.T) {
	req := basicRequest()
	b, err := json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(b, []byte("max_tokens")) {
		t.Errorf("max_tokens present though MaxTokens is unset: %s", b)
	}
}

func TestBuildRequestMaxTokensUserSet(t *testing.T) {
	req := basicRequest()
	req.MaxTokens = 333
	w := buildRequest(req)
	if w.MaxTokens == nil || *w.MaxTokens != 333 {
		t.Errorf("max_tokens = %v, want 333 (user-set)", w.MaxTokens)
	}
}

func TestBuildRequestTemperatureOmittedWhenNil(t *testing.T) {
	req := basicRequest()
	b, err := json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(b, []byte("temperature")) {
		t.Errorf("temperature present though Temperature is nil: %s", b)
	}

	req.Temperature = floatPtr(0)
	b, err = json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"temperature":0`)) {
		t.Errorf("temperature 0 not sent though Temperature is non-nil: %s", b)
	}
}

func TestBuildRequestStreamOptionsAlwaysPresent(t *testing.T) {
	req := basicRequest()
	b, err := json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"stream_options":{"include_usage":true}`)) {
		t.Errorf("stream_options.include_usage missing: %s", b)
	}
}

func TestBuildRequestNoSystemOmitsSystemMessage(t *testing.T) {
	req := basicRequest()
	req.System = ""
	w := buildRequest(req)
	if len(w.Messages) == 0 || w.Messages[0].Role == "system" {
		t.Errorf("leading system message present though System is empty: %+v", w.Messages[0])
	}
}

func TestBuildRequestStopSequences(t *testing.T) {
	req := basicRequest()
	req.StopSeqs = []string{"STOP", "END"}
	w := buildRequest(req)
	if len(w.Stop) != 2 || w.Stop[0] != "STOP" || w.Stop[1] != "END" {
		t.Errorf("stop = %v, want [STOP END]", w.Stop)
	}

	req.StopSeqs = nil
	b, err := json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(b, []byte(`"stop"`)) {
		t.Errorf("stop present though StopSeqs is empty: %s", b)
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

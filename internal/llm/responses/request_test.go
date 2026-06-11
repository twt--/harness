package responses

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"harness/internal/llm"
)

func floatPtr(f float64) *float64 { return &f }

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

func TestBuildRequestMaxTokensUsesMaxOutputTokens(t *testing.T) {
	req := basicRequest()
	req.MaxTokens = 333
	w := buildRequest(req)
	if w.MaxOutputTokens == nil || *w.MaxOutputTokens != 333 {
		t.Errorf("max_output_tokens = %v, want 333", w.MaxOutputTokens)
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

func TestBuildRequestReasoningEffort(t *testing.T) {
	req := basicRequest()
	req.Reasoning = llm.ReasoningConfig{Effort: "high"}
	w := buildRequest(req)
	if w.Reasoning == nil || w.Reasoning.Effort != "high" {
		t.Fatalf("reasoning = %+v, want effort high", w.Reasoning)
	}
}

func TestBuildRequestStreamAndStore(t *testing.T) {
	w := buildRequest(basicRequest())
	if !w.Stream {
		t.Fatal("stream = false, want true")
	}
	if w.Store {
		t.Fatal("store = true, want false")
	}
}

func TestBuildRequestToolsAreNonStrict(t *testing.T) {
	w := buildRequest(basicRequest())
	if len(w.Tools) == 0 {
		t.Fatal("no tools")
	}
	for _, tool := range w.Tools {
		if tool.Strict {
			t.Fatalf("tool %q strict = true, want false", tool.Name)
		}
	}
}

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
		t.Fatalf("canonical unmarshal: %v", err)
	}
	out, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("canonical marshal: %v", err)
	}
	return string(out)
}

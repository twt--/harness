package openai

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"harness/internal/llm"
	"harness/internal/llm/llmtest"
)

func basicRequest() llm.Request { return llmtest.WeatherToolRequest("gpt-5.4", "call_", true) }

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

	if !llmtest.JSONEqual(t, got, want) {
		t.Errorf("request JSON mismatch.\n got: %s\nwant: %s", llmtest.CanonicalJSON(t, got), llmtest.CanonicalJSON(t, want))
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

	req.Temperature = llmtest.FloatPtr(0)
	b, err = json.Marshal(buildRequest(req))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"temperature":0`)) {
		t.Errorf("temperature 0 not sent though Temperature is non-nil: %s", b)
	}
}

func TestBuildRequestReasoningEffortOpenAI(t *testing.T) {
	req := basicRequest()
	req.Reasoning = llm.ReasoningConfig{Effort: "high"}
	w := buildRequest(req)
	if w.ReasoningEffort != "high" {
		t.Fatalf("reasoning_effort = %q, want high", w.ReasoningEffort)
	}
	if w.Reasoning != nil {
		t.Fatalf("reasoning object should be omitted for OpenAI mode: %+v", w.Reasoning)
	}

	b, err := json.Marshal(w)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"reasoning_effort":"high"`)) {
		t.Fatalf("reasoning_effort missing from JSON: %s", b)
	}
}

func TestBuildRequestReasoningEffortOpenRouter(t *testing.T) {
	req := basicRequest()
	req.Reasoning = llm.ReasoningConfig{Effort: "medium"}
	w := buildRequestForMode(req, "openrouter")
	if w.ReasoningEffort != "" {
		t.Fatalf("reasoning_effort = %q, want omitted for OpenRouter", w.ReasoningEffort)
	}
	if w.Reasoning == nil || w.Reasoning.Effort != "medium" {
		t.Fatalf("reasoning = %+v, want effort medium", w.Reasoning)
	}

	b, err := json.Marshal(w)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"reasoning":{"effort":"medium"}`)) {
		t.Fatalf("reasoning object missing from JSON: %s", b)
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

package responses

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

	req.Temperature = llmtest.FloatPtr(0)
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

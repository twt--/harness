package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"harness/internal/llm"
	"harness/internal/llm/factory"
	"harness/internal/llm/llmtest"
	"harness/internal/modelproxy/protocol"
)

func TestHandlerCatalogAndStreamResolveProviderConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "openrouter.json"), []byte(`{
  "name": "openrouter",
  "api_type": "openai",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key": "sk-file",
  "api_key_env": ["OPENROUTER_API_KEY"],
  "models": [
    {"name":"openai/gpt-5.5","context_window":1050000,"price":{"input":5,"output":30},"reasoning":true,"reasoning_options":[{"type":"effort","values":["low","medium","high"]}]}
  ]
}`), 0o600); err != nil {
		t.Fatalf("write provider config: %v", err)
	}

	var captured factory.Options
	fp := llmtest.New("fake", llmtest.Step{
		Events: []llm.StreamEvent{{Kind: llm.EventTextDelta, Text: "ok"}},
		Stop:   llm.StopEndTurn,
	})
	handler, err := NewHandler(Options{
		ConfigDir: dir,
		Config: Config{
			ProviderConfigs:      []string{"openrouter.json"},
			DefaultContextWindow: 512000,
		},
		Getenv: func(k string) string {
			if k == "OPENROUTER_API_KEY" {
				return "sk-env"
			}
			return ""
		},
		New: func(opts factory.Options) (llm.Provider, error) {
			captured = opts
			return fp, nil
		},
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	srv := httptest.NewServer(handler)
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatalf("GET models: %v", err)
	}
	var catalog protocol.Catalog
	if err := json.NewDecoder(resp.Body).Decode(&catalog); err != nil {
		t.Fatalf("decode catalog: %v", err)
	}
	resp.Body.Close()
	if len(catalog.Providers) != 1 || catalog.Providers[0].ID != "openrouter" {
		t.Fatalf("catalog providers = %+v", catalog.Providers)
	}

	body, _ := json.Marshal(protocol.StreamRequest{
		Provider: "openrouter",
		Request:  llm.Request{Model: "openai/gpt-5.5"},
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL+"/v1/stream", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err = srv.Client().Do(req)
	if err != nil {
		t.Fatalf("POST stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if captured.Provider != "openai" || captured.ProviderName != "openrouter" ||
		captured.BaseURL != "https://openrouter.ai/api/v1" || captured.APIKey != "sk-env" ||
		captured.ContextWindow != 1_050_000 {
		t.Fatalf("captured options = %+v", captured)
	}
	if len(fp.Requests) != 1 || fp.Requests[0].Model != "openai/gpt-5.5" {
		t.Fatalf("fake provider requests = %+v", fp.Requests)
	}
}

func TestHandlerRejectsUnknownModel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "openai.json"), []byte(`{
  "name": "openai",
  "api_type": "openai",
  "base_url": "http://localhost:11434/v1",
  "models": [{"name":"known","context_window":128000}]
}`), 0o600); err != nil {
		t.Fatalf("write provider config: %v", err)
	}
	handler, err := NewHandler(Options{
		ConfigDir: dir,
		Config:    Config{ProviderConfigs: []string{"openai.json"}},
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	srv := httptest.NewServer(handler)
	defer srv.Close()

	body, _ := json.Marshal(protocol.StreamRequest{
		Provider: "openai",
		Request:  llm.Request{Model: "missing"},
	})
	resp, err := srv.Client().Post(srv.URL+"/v1/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	var wireErr protocol.Error
	if err := json.NewDecoder(resp.Body).Decode(&wireErr); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if wireErr.Message == "" {
		t.Fatalf("expected error message")
	}
}

func TestHandlerRequiresExplicitProviderAndModel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "openai.json"), []byte(`{
  "name": "openai",
  "api_type": "openai",
  "base_url": "http://localhost:11434/v1",
  "models": [{"name":"known","context_window":128000}]
}`), 0o600); err != nil {
		t.Fatalf("write provider config: %v", err)
	}
	handler, err := NewHandler(Options{
		ConfigDir: dir,
		Config:    Config{ProviderConfigs: []string{"openai.json"}},
	})
	if err != nil {
		t.Fatalf("NewHandler: %v", err)
	}
	srv := httptest.NewServer(handler)
	defer srv.Close()

	body, _ := json.Marshal(protocol.StreamRequest{Request: llm.Request{Model: "known"}})
	resp, err := srv.Client().Post(srv.URL+"/v1/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("missing provider status = %d, want 400", resp.StatusCode)
	}

	body, _ = json.Marshal(protocol.StreamRequest{Provider: "openai"})
	resp, err = srv.Client().Post(srv.URL+"/v1/stream", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST stream: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("missing model status = %d, want 400", resp.StatusCode)
	}
}

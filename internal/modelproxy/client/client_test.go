package client

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"harness/internal/llm"
	"harness/internal/modelproxy/protocol"
)

func TestCatalogAndRegistry(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(protocol.Catalog{
			Providers: []protocol.Provider{{
				ID: "openrouter",
				Models: []protocol.Model{{
					ID:            "openai/gpt-5.5",
					ContextWindow: 1_050_000,
					Price:         llm.Price{Input: 5, Output: 30},
				}},
			}},
		})
	}))
	defer srv.Close()

	c, err := New(srv.URL, srv.Client())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	catalog, err := c.Catalog(context.Background())
	if err != nil {
		t.Fatalf("Catalog: %v", err)
	}
	if len(catalog.Providers) != 1 || catalog.Providers[0].ID != "openrouter" {
		t.Fatalf("catalog providers = %+v", catalog.Providers)
	}
	registry := Registry(catalog)
	if got := registry.ContextWindow("openrouter:openai/gpt-5.5"); got != 1_050_000 {
		t.Fatalf("qualified context window = %d, want 1050000", got)
	}
}

func TestProviderStreamEventsAndErrors(t *testing.T) {
	var sawProvider string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/stream" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		var req protocol.StreamRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		sawProvider = req.Provider
		w.Header().Set("content-type", protocol.ContentTypeNDJSON)
		enc := json.NewEncoder(w)
		text := llm.StreamEvent{Kind: llm.EventTextDelta, Text: "hello"}
		_ = enc.Encode(protocol.StreamEnvelope{Event: &text})
		_ = enc.Encode(protocol.StreamEnvelope{Error: &protocol.Error{
			StatusCode:   http.StatusTooManyRequests,
			Code:         "rate_limit",
			Message:      "slow down",
			Retryable:    true,
			RetryAfterMS: 250,
		}})
	}))
	defer srv.Close()

	c, err := New(srv.URL, srv.Client())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var texts []string
	var gotErr error
	for ev, err := range c.Provider("openai").Stream(context.Background(), llm.Request{Model: "gpt-5.5"}) {
		if err != nil {
			gotErr = err
			break
		}
		if ev.Kind == llm.EventTextDelta {
			texts = append(texts, ev.Text)
		}
	}
	if sawProvider != "openai" {
		t.Fatalf("provider sent to proxy = %q", sawProvider)
	}
	if len(texts) != 1 || texts[0] != "hello" {
		t.Fatalf("texts = %v", texts)
	}
	var apiErr *llm.APIError
	if !errors.As(gotErr, &apiErr) || apiErr.StatusCode != http.StatusTooManyRequests || !apiErr.Retryable {
		t.Fatalf("error = %v, want retryable APIError 429", gotErr)
	}
}

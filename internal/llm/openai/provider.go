// Package openai implements the llm.Provider contract against the OpenAI Chat
// Completions streaming API. The same code path serves OpenAI-compatible servers
// (vLLM, Ollama, llama.cpp, OpenRouter) via a configurable base URL. It covers
// tool-call assembly, usage normalization, and the retry-before-first-byte
// policy (design §5.3–§5.5).
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"harness/internal/llm"
	"harness/internal/retry"
	"harness/internal/sse"
)

const (
	defaultBaseURL      = "https://api.openai.com/v1"
	chatCompletionsPath = "/chat/completions"
	maxAttempts         = 5
)

// Config configures a Provider. A custom BaseURL supplies scheme/host/prefix
// only; the dialect appends its standard /chat/completions path, so
// -base-url http://localhost:11434/v1 works for Ollama (design §7).
type Config struct {
	APIKey        string
	BaseURL       string // default https://api.openai.com/v1
	ContextWindow int    // unused by OpenAI request building; kept for factory symmetry
	ReasoningMode string // "openai" or "openrouter"; empty defaults to "openai"
	HTTPClient    *http.Client
	Sleep         func(time.Duration) // nil = time.Sleep
}

// Provider is the OpenAI Chat Completions dialect.
type Provider struct {
	apiKey        string
	baseURL       string
	reasoningMode string
	client        *http.Client
	sleep         func(time.Duration)
}

// New constructs a Provider from cfg, applying defaults.
func New(cfg Config) *Provider {
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	base = strings.TrimSuffix(base, "/")

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	sleep := cfg.Sleep
	if sleep == nil {
		sleep = time.Sleep
	}
	return &Provider{
		apiKey:        cfg.APIKey,
		baseURL:       base,
		reasoningMode: cfg.ReasoningMode,
		client:        client,
		sleep:         sleep,
	}
}

func (p *Provider) Name() string { return "openai" }

// Stream runs one model call. Retries apply only before the first response byte;
// once tokens stream, any failure (invalid tool JSON, truncated body) is
// turn-fatal. ctx.Err() is checked before every attempt and sleep.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := json.Marshal(buildRequestForMode(req, p.reasoningMode))
		if err != nil {
			yield(llm.StreamEvent{}, &llm.APIError{Message: "marshal request: " + err.Error()})
			return
		}

		resp, err := p.connect(ctx, body, yield)
		if err != nil || resp == nil {
			return
		}
		defer resp.Body.Close()

		p.decode(ctx, resp.Body, yield)
	}
}

// connect performs the request with the retry-before-first-byte loop. It returns
// a live 200 response, or yields a terminal error and returns (nil, err). A nil
// response with nil error means a terminal error was already yielded.
func (p *Provider) connect(ctx context.Context, body []byte, yield func(llm.StreamEvent, error) bool) (*http.Response, error) {
	url := p.baseURL + chatCompletionsPath

	for attempt := 0; ; attempt++ {
		if err := ctx.Err(); err != nil {
			yield(llm.StreamEvent{}, err)
			return nil, err
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			yield(llm.StreamEvent{}, &llm.APIError{Message: "build request: " + err.Error()})
			return nil, err
		}
		httpReq.Header.Set("content-type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

		resp, err := p.client.Do(httpReq)
		if err != nil {
			// A cancelled context wins over transport-error classification.
			if ctxErr := ctx.Err(); ctxErr != nil {
				yield(llm.StreamEvent{}, ctxErr)
				return nil, ctxErr
			}
			apiErr := &llm.APIError{Message: err.Error(), Retryable: true}
			if !p.backoff(ctx, attempt, 0, apiErr, yield) {
				return nil, apiErr
			}
			continue
		}

		if resp.StatusCode == http.StatusOK {
			return resp, nil
		}

		apiErr := parseErrorResponse(resp)
		resp.Body.Close()
		if !apiErr.Retryable {
			yield(llm.StreamEvent{}, apiErr)
			return nil, apiErr
		}
		if !p.backoff(ctx, attempt, apiErr.RetryAfter, apiErr, yield) {
			return nil, apiErr
		}
	}
}

// backoff sleeps before the next attempt unless the budget is exhausted or ctx
// is cancelled. It returns true to continue retrying, false to stop (having
// yielded the terminal error in the stop case).
func (p *Provider) backoff(ctx context.Context, attempt int, retryAfter time.Duration, apiErr *llm.APIError, yield func(llm.StreamEvent, error) bool) bool {
	if attempt >= maxAttempts-1 {
		yield(llm.StreamEvent{}, apiErr)
		return false
	}
	if err := ctx.Err(); err != nil {
		yield(llm.StreamEvent{}, err)
		return false
	}
	p.sleep(retry.Next(attempt, retryAfter))
	return true
}

// decode reads the SSE stream, emits events, and accumulates usage. The literal
// data: [DONE] sentinel terminates the stream; a body EOF before it is a
// truncated stream wrapped in *llm.APIError (wrapping sse.ErrTruncatedStream).
// Buffered tool calls flush as Done when finish_reason "tool_calls" arrives.
func (p *Provider) decode(ctx context.Context, r io.Reader, yield func(llm.StreamEvent, error) bool) {
	asm := newToolAssembler()
	var usage llm.Usage
	var stop llm.StopReason = llm.StopEndTurn
	completed := false

	for ev, err := range sse.Read(ctx, r) {
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		data := strings.TrimSpace(ev.Data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			completed = true
			u := usage
			yield(llm.StreamEvent{Kind: llm.EventDone, Usage: &u, StopReason: stop}, nil)
			return
		}

		var chunk wireChunk
		if jsonErr := json.Unmarshal([]byte(data), &chunk); jsonErr != nil {
			yield(llm.StreamEvent{}, &llm.APIError{Message: "decode stream chunk: " + jsonErr.Error()})
			return
		}

		if chunk.Usage != nil {
			usage = normalizeUsage(chunk.Usage)
			u := usage
			if !yield(llm.StreamEvent{Kind: llm.EventUsage, Usage: &u}, nil) {
				return
			}
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				if !yield(llm.StreamEvent{Kind: llm.EventTextDelta, Text: choice.Delta.Content}, nil) {
					return
				}
			}
			for _, frag := range choice.Delta.ToolCalls {
				if !asm.observe(frag, yield) {
					return
				}
			}
			if choice.FinishReason == "" {
				continue
			}
			stop = normalizeStopReason(choice.FinishReason)
			if asm.has() {
				ok, fatal := asm.flush(yield)
				if fatal != nil {
					yield(llm.StreamEvent{}, fatal)
					return
				}
				if !ok {
					return
				}
			}
		}
	}

	if !completed {
		yield(llm.StreamEvent{}, fmt.Errorf("openai: stream ended before [DONE]: %w", sse.ErrTruncatedStream))
	}
}

// normalizeUsage maps the OpenAI usage object onto llm.Usage. prompt_tokens
// includes cached tokens, so cached_tokens is subtracted to recover the
// full-rate InputTokens; OpenAI has no separate cache-write charge (design §6).
func normalizeUsage(u *wireUsage) llm.Usage {
	cached := u.PromptTokensDetails.CachedTokens
	return llm.Usage{
		InputTokens:      u.PromptTokens - cached,
		OutputTokens:     u.CompletionTokens,
		CacheReadTokens:  cached,
		CacheWriteTokens: 0,
	}
}

// parseErrorResponse maps a non-2xx HTTP response onto an *llm.APIError,
// extracting the provider error type/message and the Retry-After floor, and
// classifying retryability by status (design §5.5).
func parseErrorResponse(resp *http.Response) *llm.APIError {
	apiErr := &llm.APIError{
		StatusCode: resp.StatusCode,
		Retryable:  retry.RetryableStatus(resp.StatusCode),
		RetryAfter: retry.ParseRetryAfter(resp.Header.Get("Retry-After")),
	}

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var env struct {
		Error *struct {
			Type    string `json:"type"`
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && env.Error != nil {
		apiErr.Code = env.Error.Type
		apiErr.Message = env.Error.Message
	}
	if apiErr.Message == "" {
		apiErr.Message = strings.TrimSpace(string(body))
	}
	return apiErr
}

// normalizeStopReason maps OpenAI finish_reason values onto the four normalized
// constants. Unknown or provider-specific reasons (content_filter, function_call,
// or compatible-server extensions) map to end_turn — the turn is over either way
// (design §5.1).
func normalizeStopReason(reason string) llm.StopReason {
	switch reason {
	case "stop":
		return llm.StopEndTurn
	case "length":
		return llm.StopMaxTokens
	case "tool_calls":
		return llm.StopToolUse
	default:
		return llm.StopEndTurn
	}
}

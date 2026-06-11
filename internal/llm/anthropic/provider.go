// Package anthropic implements the llm.Provider contract against the Anthropic
// Messages streaming API, including prompt caching, tool-call assembly, and the
// retry-before-first-byte policy (design §5.3–§5.5).
package anthropic

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
	defaultBaseURL = "https://api.anthropic.com"
	messagesPath   = "/v1/messages"
	apiVersion     = "2023-06-01"
	maxAttempts    = 5
)

// Config configures a Provider. A custom BaseURL supplies scheme/host/prefix
// only; the dialect appends its standard /v1/messages path (design §7).
type Config struct {
	APIKey        string
	BaseURL       string // default https://api.anthropic.com
	ContextWindow int    // resolved by main from provider config registry
	HTTPClient    *http.Client
	Sleep         func(time.Duration) // nil = time.Sleep
}

// Provider is the Anthropic Messages dialect.
type Provider struct {
	apiKey        string
	baseURL       string
	contextWindow int
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
		contextWindow: cfg.ContextWindow,
		client:        client,
		sleep:         sleep,
	}
}

func (p *Provider) Name() string { return "anthropic" }

// Stream runs one model call. Retries apply only before the first response byte;
// once tokens stream, any failure (mid-stream error frame, truncated body) is
// turn-fatal. ctx.Err() is checked before every attempt and sleep.
func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		window := p.contextWindow
		body, err := json.Marshal(buildRequest(req, window))
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
	url := p.baseURL + messagesPath

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
		httpReq.Header.Set("anthropic-version", apiVersion)
		httpReq.Header.Set("x-api-key", p.apiKey)

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

// decode reads the SSE stream, emits events, and accumulates usage. A body EOF
// before message_stop is a truncated stream; a mid-stream error frame is
// turn-fatal. Both are wrapped in *llm.APIError (truncation wraps
// sse.ErrTruncatedStream).
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

		var data wireEvent
		if ev.Data == "" {
			continue
		}
		if jsonErr := json.Unmarshal([]byte(ev.Data), &data); jsonErr != nil {
			yield(llm.StreamEvent{}, &llm.APIError{Message: "decode stream event: " + jsonErr.Error()})
			return
		}

		switch data.Type {
		case "message_start":
			if data.Message != nil {
				usage.InputTokens = data.Message.Usage.InputTokens
				usage.CacheWriteTokens = data.Message.Usage.CacheCreationInputTokens
				usage.CacheReadTokens = data.Message.Usage.CacheReadInputTokens
				usage.OutputTokens = data.Message.Usage.OutputTokens
				u := usage
				if !yield(llm.StreamEvent{Kind: llm.EventUsage, Usage: &u}, nil) {
					return
				}
			}

		case "content_block_start":
			if data.ContentBlock != nil && data.ContentBlock.Type == "tool_use" {
				if !yield(asm.start(data.Index, data.ContentBlock.ID, data.ContentBlock.Name), nil) {
					return
				}
			}

		case "content_block_delta":
			if data.Delta == nil {
				continue
			}
			switch data.Delta.Type {
			case "text_delta":
				if !yield(llm.StreamEvent{Kind: llm.EventTextDelta, Text: data.Delta.Text}, nil) {
					return
				}
			case "input_json_delta":
				if dev, ok := asm.delta(data.Index, data.Delta.PartialJSON); ok {
					if !yield(dev, nil) {
						return
					}
				}
			}

		case "content_block_stop":
			done, ferr, ok := asm.flush(data.Index)
			if ferr != nil {
				yield(llm.StreamEvent{}, ferr)
				return
			}
			if ok {
				if !yield(done, nil) {
					return
				}
			}

		case "message_delta":
			if data.Delta != nil && data.Delta.StopReason != "" {
				stop = normalizeStopReason(data.Delta.StopReason)
			}
			if data.Usage != nil {
				usage.OutputTokens = data.Usage.OutputTokens
				if data.Usage.InputTokens > 0 {
					usage.InputTokens = data.Usage.InputTokens
				}
				if data.Usage.CacheCreationInputTokens > 0 {
					usage.CacheWriteTokens = data.Usage.CacheCreationInputTokens
				}
				if data.Usage.CacheReadInputTokens > 0 {
					usage.CacheReadTokens = data.Usage.CacheReadInputTokens
				}
			}
			u := usage
			if !yield(llm.StreamEvent{Kind: llm.EventUsage, Usage: &u}, nil) {
				return
			}

		case "message_stop":
			completed = true
			u := usage
			yield(llm.StreamEvent{Kind: llm.EventDone, Usage: &u, StopReason: stop}, nil)
			return

		case "error":
			apiErr := &llm.APIError{Message: "stream error"}
			if data.Error != nil {
				apiErr.Code = data.Error.Type
				apiErr.Message = data.Error.Message
				apiErr.Retryable = retryableErrorType(data.Error.Type)
			}
			yield(llm.StreamEvent{}, apiErr)
			return

		case "ping":
			// ignored

		default:
			// Unknown event type: ignore per the versioning policy.
		}
	}

	if !completed {
		yield(llm.StreamEvent{}, fmt.Errorf("anthropic: stream ended before message_stop: %w", sse.ErrTruncatedStream))
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

// retryableErrorType classifies mid-stream error-frame types: transient server
// conditions are retryable by re-requesting the step; everything else
// (invalid_request_error, authentication_error, ...) is terminal.
func retryableErrorType(t string) bool {
	switch t {
	case "overloaded_error", "api_error", "rate_limit_error":
		return true
	}
	return false
}

// normalizeStopReason maps Anthropic stop reasons onto the four normalized
// constants. Unknown values map to end_turn — the turn is over either way.
func normalizeStopReason(reason string) llm.StopReason {
	switch reason {
	case "end_turn":
		return llm.StopEndTurn
	case "tool_use":
		return llm.StopToolUse
	case "max_tokens":
		return llm.StopMaxTokens
	case "stop_sequence":
		return llm.StopStop
	default:
		return llm.StopEndTurn
	}
}

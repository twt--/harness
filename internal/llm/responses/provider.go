// Package responses implements the llm.Provider contract against the OpenAI
// Responses streaming API.
package responses

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
	defaultBaseURL = "https://api.openai.com/v1"
	responsesPath  = "/responses"
	maxAttempts    = 5
)

type Config struct {
	APIKey        string
	BaseURL       string
	ContextWindow int
	HTTPClient    *http.Client
	Sleep         func(time.Duration)
}

type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
	sleep   func(time.Duration)
}

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
	return &Provider{apiKey: cfg.APIKey, baseURL: base, client: client, sleep: sleep}
}

func (p *Provider) Name() string { return "responses" }

func (p *Provider) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		body, err := json.Marshal(buildRequest(req))
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

func (p *Provider) connect(ctx context.Context, body []byte, yield func(llm.StreamEvent, error) bool) (*http.Response, error) {
	url := p.baseURL + responsesPath

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

func (p *Provider) decode(ctx context.Context, r io.Reader, yield func(llm.StreamEvent, error) bool) {
	asm := newToolAssembler()
	var usage llm.Usage
	completed := false

	for ev, err := range sse.Read(ctx, r) {
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		data := strings.TrimSpace(ev.Data)
		if data == "" || data == "[DONE]" {
			continue
		}

		var event wireEvent
		if jsonErr := json.Unmarshal([]byte(data), &event); jsonErr != nil {
			yield(llm.StreamEvent{}, &llm.APIError{Message: "decode stream event: " + jsonErr.Error()})
			return
		}

		switch event.Type {
		case "response.output_text.delta":
			if event.Delta != "" {
				if !yield(llm.StreamEvent{Kind: llm.EventTextDelta, Text: event.Delta}, nil) {
					return
				}
			}

		case "response.output_item.added":
			if !asm.outputItemAdded(event.OutputIndex, event.Item, yield) {
				return
			}

		case "response.function_call_arguments.delta":
			if !asm.argumentsDelta(event.OutputIndex, event.Delta, yield) {
				return
			}

		case "response.function_call_arguments.done":
			asm.argumentsDone(event.OutputIndex, event.ItemID, event.Name, event.Arguments)

		case "response.output_item.done":
			asm.outputItemDone(event.OutputIndex, event.Item)

		case "response.completed":
			completed = true
			if event.Response != nil {
				asm.responseOutput(event.Response.Output)
				if event.Response.Usage != nil {
					usage = normalizeUsage(event.Response.Usage)
					u := usage
					if !yield(llm.StreamEvent{Kind: llm.EventUsage, Usage: &u}, nil) {
						return
					}
				}
			}
			stop := llm.StopEndTurn
			if asm.has() {
				stop = llm.StopToolUse
				ok, fatal := asm.flush(yield)
				if fatal != nil {
					yield(llm.StreamEvent{}, fatal)
					return
				}
				if !ok {
					return
				}
			}
			u := usage
			yield(llm.StreamEvent{Kind: llm.EventDone, Usage: &u, StopReason: stop}, nil)
			return

		case "response.incomplete":
			completed = true
			stop := llm.StopEndTurn
			if event.Response != nil {
				asm.responseOutput(event.Response.Output)
				if event.Response.Usage != nil {
					usage = normalizeUsage(event.Response.Usage)
				}
				if event.Response.IncompleteDetails != nil && event.Response.IncompleteDetails.Reason == "max_output_tokens" {
					stop = llm.StopMaxTokens
				}
			}
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
			u := usage
			yield(llm.StreamEvent{Kind: llm.EventDone, Usage: &u, StopReason: stop}, nil)
			return

		case "response.failed":
			completed = true
			apiErr := &llm.APIError{Message: "response failed"}
			if event.Response != nil && event.Response.Error != nil {
				apiErr.Code = event.Response.Error.Code
				apiErr.Message = event.Response.Error.Message
				apiErr.Retryable = retryableErrorCode(apiErr.Code)
			}
			yield(llm.StreamEvent{}, apiErr)
			return

		case "error":
			completed = true
			apiErr := &llm.APIError{Code: event.Code, Message: event.Message, Retryable: retryableErrorCode(event.Code)}
			if apiErr.Message == "" {
				apiErr.Message = "stream error"
			}
			yield(llm.StreamEvent{}, apiErr)
			return

		default:
			// Lifecycle and unsupported tool events are ignored unless handled above.
		}
	}

	if !completed {
		yield(llm.StreamEvent{}, fmt.Errorf("responses: stream ended before terminal event: %w", sse.ErrTruncatedStream))
	}
}

func normalizeUsage(u *wireUsage) llm.Usage {
	cached := u.InputTokensDetails.CachedTokens
	return llm.Usage{
		InputTokens:      u.InputTokens - cached,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  cached,
		CacheWriteTokens: 0,
	}
}

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
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && env.Error != nil {
		if env.Error.Code != "" {
			apiErr.Code = env.Error.Code
		} else {
			apiErr.Code = env.Error.Type
		}
		apiErr.Message = env.Error.Message
	}
	if apiErr.Message == "" {
		apiErr.Message = strings.TrimSpace(string(body))
	}
	return apiErr
}

func retryableErrorCode(code string) bool {
	switch code {
	case "server_error", "rate_limit_exceeded", "rate_limit_error":
		return true
	}
	return false
}

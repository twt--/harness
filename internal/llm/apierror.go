package llm

import (
	"fmt"
	"time"
)

// APIError is the shared provider error surfaced to the agent loop (design §5.5).
// Both dialects construct it. Retryable marks the transport/status classes the
// provider retry loop may retry; RetryAfter carries a parsed Retry-After header
// (0 when absent) honored as a backoff floor.
type APIError struct {
	StatusCode int
	Code       string // provider error code/type if parseable
	Message    string
	Retryable  bool
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	switch {
	case e.Code != "" && e.Message != "":
		return fmt.Sprintf("api error %d (%s): %s", e.StatusCode, e.Code, e.Message)
	case e.Message != "":
		return fmt.Sprintf("api error %d: %s", e.StatusCode, e.Message)
	default:
		return fmt.Sprintf("api error %d", e.StatusCode)
	}
}

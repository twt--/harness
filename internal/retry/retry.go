// Package retry holds the backoff policy. Next is a pure function of the attempt
// number and a Retry-After floor; the retry loop (success/give-up/ctx handling)
// lives in each provider, which owns APIError.Retryable and injects a sleeper.
package retry

import (
	"crypto/rand"
	"math/big"
	"time"
)

const (
	baseDelay = 500 * time.Millisecond
	cap30s    = 30 * time.Second
)

// Next returns the backoff before the given attempt (0-based). It applies full
// jitter — a uniform draw from [0, min(30s, 500ms·2^attempt)] — and honors
// retryAfter as a floor, so the result is never below a server-supplied
// Retry-After even when jitter would pick a smaller value.
func Next(attempt int, retryAfter time.Duration) time.Duration {
	ceiling := cap30s
	if attempt < 60 {
		if scaled := baseDelay << uint(attempt); scaled > 0 && scaled < cap30s {
			ceiling = scaled
		}
	}

	d := time.Duration(randN(int64(ceiling) + 1))
	if d < retryAfter {
		d = retryAfter
	}
	return d
}

// randN returns a uniform draw from [0, n). n must be positive.
func randN(n int64) int64 {
	v, err := rand.Int(rand.Reader, big.NewInt(n))
	if err != nil {
		// crypto/rand.Reader does not fail in practice; degrade to no jitter.
		return 0
	}
	return v.Int64()
}

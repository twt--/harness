package retry

import (
	"testing"
	"time"
)

const base = 500 * time.Millisecond

// uncappedCeiling is base·2^attempt without the 30s clamp, used to bound jitter.
func uncappedCeiling(attempt int) time.Duration {
	return base * time.Duration(int64(1)<<uint(attempt))
}

func TestNextJitterWithinBounds(t *testing.T) {
	const draws = 10_000
	for attempt := 0; attempt < 6; attempt++ {
		ceil := uncappedCeiling(attempt)
		if ceil > cap30s {
			ceil = cap30s
		}
		for i := 0; i < draws; i++ {
			d := Next(attempt, 0)
			if d < 0 {
				t.Fatalf("attempt %d: Next = %v, want >= 0", attempt, d)
			}
			if d > ceil {
				t.Fatalf("attempt %d: Next = %v, want <= ceiling %v", attempt, d, ceil)
			}
		}
	}
}

func TestNextBoundedBy30sCap(t *testing.T) {
	// At a high attempt count base·2^attempt vastly exceeds 30s, so every draw
	// must still fall within [0, 30s].
	const draws = 10_000
	const attempt = 20
	for i := 0; i < draws; i++ {
		d := Next(attempt, 0)
		if d < 0 || d > cap30s {
			t.Fatalf("Next(%d) = %v, want within [0, %v]", attempt, d, cap30s)
		}
	}
}

func TestNextRetryAfterFloor(t *testing.T) {
	// Retry-After of 2s exceeds the attempt-0 ceiling (500ms), so the floor must
	// dominate every draw.
	const draws = 10_000
	retryAfter := 2 * time.Second
	for i := 0; i < draws; i++ {
		d := Next(0, retryAfter)
		if d < retryAfter {
			t.Fatalf("Next(0, %v) = %v, want >= floor %v", retryAfter, d, retryAfter)
		}
	}
}

func TestNextJitterIsRandom(t *testing.T) {
	// Full jitter must actually vary; many draws at a wide ceiling should not all
	// be identical.
	const attempt = 5
	first := Next(attempt, 0)
	varied := false
	for i := 0; i < 1000; i++ {
		if Next(attempt, 0) != first {
			varied = true
			break
		}
	}
	if !varied {
		t.Fatal("Next produced no jitter across 1000 draws")
	}
}

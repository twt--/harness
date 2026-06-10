package llm

import "testing"

func TestCostKnownModel(t *testing.T) {
	// A known model with non-zero usage must report a positive cost and known=true.
	u := Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}
	cost, known := Cost("claude-opus-4-8", u)
	if !known {
		t.Fatalf("Cost(known model) known = false, want true")
	}
	if cost <= 0 {
		t.Fatalf("Cost(known model) = %v, want > 0", cost)
	}
}

func TestCostUnknownModel(t *testing.T) {
	u := Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}
	cost, known := Cost("totally-made-up-model", u)
	if known {
		t.Fatalf("Cost(unknown model) known = true, want false")
	}
	if cost != 0 {
		t.Fatalf("Cost(unknown model) = %v, want 0", cost)
	}
}

func TestCostComponents(t *testing.T) {
	// All four token categories contribute to the cost. Compare a usage that
	// exercises cache fields against one that does not.
	base := Usage{InputTokens: 1_000_000}
	withCache := Usage{InputTokens: 1_000_000, CacheReadTokens: 1_000_000, CacheWriteTokens: 1_000_000}

	baseCost, ok1 := Cost("claude-opus-4-8", base)
	cacheCost, ok2 := Cost("claude-opus-4-8", withCache)
	if !ok1 || !ok2 {
		t.Fatalf("expected known model")
	}
	if cacheCost <= baseCost {
		t.Fatalf("cache usage cost %v should exceed base cost %v", cacheCost, baseCost)
	}
}

func TestContextWindowDefault(t *testing.T) {
	if got := ContextWindow("unknown-model"); got != 128000 {
		t.Fatalf("ContextWindow(unknown) = %d, want 128000", got)
	}
}

func TestContextWindowKnown(t *testing.T) {
	// A registry entry returns its own window. claude-opus-4-8 has a 1M window,
	// which is larger than the default, so this also proves the lookup happened.
	got := ContextWindow("claude-opus-4-8")
	if got == 128000 {
		t.Fatalf("ContextWindow(claude-opus-4-8) returned the default; expected the registered window")
	}
	if got != 1_000_000 {
		t.Fatalf("ContextWindow(claude-opus-4-8) = %d, want 1000000", got)
	}
}

func TestRegistryEntriesWellFormed(t *testing.T) {
	// Every registered model must have a positive context window and prices.
	for name, info := range modelRegistry {
		if info.ContextWindow <= 0 {
			t.Errorf("model %q has non-positive context window %d", name, info.ContextWindow)
		}
		if info.Price.Input <= 0 || info.Price.Output <= 0 {
			t.Errorf("model %q has non-positive input/output price %+v", name, info.Price)
		}
	}
}

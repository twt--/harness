package llm

// Price is the per-1M-token price in USD for each token category. CacheRead and
// CacheWrite are 0 when a provider has no separate cache pricing.
type Price struct {
	Input      float64 // uncached input, per 1M tokens
	Output     float64 // output, per 1M tokens
	CacheRead  float64 // cached-input read, per 1M tokens
	CacheWrite float64 // cache write, per 1M tokens
}

// ModelInfo is the registry entry for one model.
type ModelInfo struct {
	ContextWindow int
	Price         Price
}

// defaultContextWindow is used for any model not in the registry — arbitrary
// names on OpenAI-compatible servers. Conservative; overridable via
// -context-window.
const defaultContextWindow = 128_000

// modelRegistry is a hand-maintained, best-effort price/context table. Prices
// are USD per 1M tokens.
//
// Anthropic figures verified against the official pricing page (2026-06-09):
// cache-read = 0.1× input, 5-minute cache-write = 1.25× input. OpenAI GPT-5.x
// figures verified against developers.openai.com/api/docs/pricing (2026-06-09);
// OpenAI has no separate cache-write charge, and cached input is billed at a
// reduced rate (CacheRead), so CacheWrite is 0. The older OpenAI entries match
// their historical launch prices but are deprecated and no longer listed on the
// current pricing page (dated versions shut down 2026-10-23).
var modelRegistry = map[string]ModelInfo{
	// --- Anthropic ---
	"claude-fable-5": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 10.0, Output: 50.0, CacheRead: 1.0, CacheWrite: 12.5},
	},
	"claude-opus-4-8": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 5.0, Output: 25.0, CacheRead: 0.5, CacheWrite: 6.25},
	},
	"claude-opus-4-7": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 5.0, Output: 25.0, CacheRead: 0.5, CacheWrite: 6.25},
	},
	"claude-opus-4-6": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 5.0, Output: 25.0, CacheRead: 0.5, CacheWrite: 6.25},
	},
	"claude-opus-4-5": {
		ContextWindow: 200_000,
		Price:         Price{Input: 5.0, Output: 25.0, CacheRead: 0.5, CacheWrite: 6.25},
	},
	"claude-sonnet-4-6": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 3.0, Output: 15.0, CacheRead: 0.3, CacheWrite: 3.75},
	},
	"claude-sonnet-4-5": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 3.0, Output: 15.0, CacheRead: 0.3, CacheWrite: 3.75},
	},
	"claude-haiku-4-5": {
		ContextWindow: 200_000,
		Price:         Price{Input: 1.0, Output: 5.0, CacheRead: 0.1, CacheWrite: 1.25},
	},

	// --- OpenAI (Chat Completions) ---
	"gpt-5.5": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 5.0, Output: 30.0, CacheRead: 0.5},
	},
	"gpt-5.4": {
		ContextWindow: 1_000_000,
		Price:         Price{Input: 2.5, Output: 15.0, CacheRead: 0.25},
	},
	"gpt-5.4-mini": {
		ContextWindow: 400_000,
		Price:         Price{Input: 0.75, Output: 4.5, CacheRead: 0.075},
	},
	// nano's window is not stated on the official pricing/models pages; 400k is
	// inferred from the gpt-5.4 mini tier.
	"gpt-5.4-nano": {
		ContextWindow: 400_000,
		Price:         Price{Input: 0.2, Output: 1.25, CacheRead: 0.02},
	},
	"gpt-4.1": {
		ContextWindow: 1_047_576,
		Price:         Price{Input: 2.0, Output: 8.0, CacheRead: 0.5},
	},
	"gpt-4.1-mini": {
		ContextWindow: 1_047_576,
		Price:         Price{Input: 0.4, Output: 1.6, CacheRead: 0.1},
	},
	"gpt-4.1-nano": {
		ContextWindow: 1_047_576,
		Price:         Price{Input: 0.1, Output: 0.4, CacheRead: 0.025},
	},
	"gpt-4o": {
		ContextWindow: 128_000,
		Price:         Price{Input: 2.5, Output: 10.0, CacheRead: 1.25},
	},
	"gpt-4o-mini": {
		ContextWindow: 128_000,
		Price:         Price{Input: 0.15, Output: 0.6, CacheRead: 0.075},
	},
	"o3": {
		ContextWindow: 200_000,
		Price:         Price{Input: 2.0, Output: 8.0, CacheRead: 0.5},
	},
	"o4-mini": {
		ContextWindow: 200_000,
		Price:         Price{Input: 1.1, Output: 4.4, CacheRead: 0.275},
	},
}

// Cost returns the USD cost of the given usage for the named model, and whether
// the model was found in the registry. Unknown models report (0, false) so the
// UI can show token counts without a dollar figure.
func Cost(model string, u Usage) (usd float64, known bool) {
	info, ok := modelRegistry[model]
	if !ok {
		return 0, false
	}
	const perMillion = 1_000_000.0
	p := info.Price
	usd = float64(u.InputTokens)/perMillion*p.Input +
		float64(u.OutputTokens)/perMillion*p.Output +
		float64(u.CacheReadTokens)/perMillion*p.CacheRead +
		float64(u.CacheWriteTokens)/perMillion*p.CacheWrite
	return usd, true
}

// ContextWindow returns the model's context window from the registry, or the
// default (128k) for unknown models.
func ContextWindow(model string) int {
	if info, ok := modelRegistry[model]; ok {
		return info.ContextWindow
	}
	return defaultContextWindow
}

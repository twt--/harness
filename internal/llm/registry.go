package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// Price is the per-1M-token price in USD for each token category. CacheRead and
// CacheWrite are 0 when a provider has no separate cache pricing.
type Price struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
}

// ModelInfo is the registry entry for one model.
type ModelInfo struct {
	ContextWindow int            `json:"context_window"`
	Price         Price          `json:"price"`
	Reasoning     *ReasoningInfo `json:"reasoning,omitempty"`
}

// ProviderConfig is the on-disk schema for a provider JSON file.
type ProviderConfig struct {
	Name      string       `json:"name"`
	APIType   string       `json:"api_type"`
	BaseURL   string       `json:"base_url"`
	APIKey    string       `json:"api_key"`
	APIKeyEnv []string     `json:"api_key_env"`
	Models    []ModelEntry `json:"models"`
}

// ModelEntry is one model inside a ProviderConfig.
type ModelEntry struct {
	Name             string            `json:"name"`
	ContextWindow    int               `json:"context_window"`
	Price            Price             `json:"price"`
	Reasoning        *bool             `json:"reasoning,omitempty"`
	ReasoningOptions []ReasoningOption `json:"reasoning_options,omitempty"`
}

// DefaultContextWindow is used for any model not in the registry — arbitrary
// names on OpenAI-compatible servers. Conservative; configurable via
// -default-context-window and overridable per run via -context-window.
const DefaultContextWindow = 256_000

// Registry holds model info loaded from provider config files.
type Registry struct {
	models               map[string]ModelInfo
	defaultContextWindow int
}

// NewRegistry builds a Registry from a pre-built map. Tests use this to avoid
// file I/O.
func NewRegistry(models map[string]ModelInfo) *Registry {
	if models == nil {
		models = map[string]ModelInfo{}
	}
	return &Registry{
		models:               models,
		defaultContextWindow: DefaultContextWindow,
	}
}

// LoadProviderConfigs reads each provider config file, logs warnings for missing
// or malformed files, and returns a Registry containing all discovered models.
// Paths are resolved relative to configDir.
func LoadProviderConfigs(configDir string, files []string, warn func(string)) (*Registry, []ProviderConfig, error) {
	models := map[string]ModelInfo{}
	var providers []ProviderConfig
	for _, f := range files {
		path := f
		if !filepath.IsAbs(path) {
			path = filepath.Join(configDir, f)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			if warn != nil {
				warn(fmt.Sprintf("warning: skipping provider config %s: %v", f, err))
			}
			continue
		}
		pcs, err := decodeProviderConfigs(data)
		if err != nil {
			if warn != nil {
				warn(fmt.Sprintf("warning: skipping provider config %s: %v", f, err))
			}
			continue
		}
		for _, pc := range pcs {
			providers = append(providers, pc)
			for _, m := range pc.Models {
				models[m.Name] = ModelInfo{
					ContextWindow: m.ContextWindow,
					Price:         m.Price,
					Reasoning:     modelEntryReasoning(m),
				}
			}
		}
	}
	return NewRegistry(models), providers, nil
}

// Lookup returns the configured info for model, if any. The returned bool only
// says an entry exists; the entry may still omit context or price fields.
func (r *Registry) Lookup(model string) (ModelInfo, bool) {
	if r == nil {
		return ModelInfo{}, false
	}
	info, ok := r.models[model]
	return info, ok
}

// Models returns the configured model names in stable order.
func (r *Registry) Models() []string {
	if r == nil {
		return nil
	}
	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// HasPrice reports whether model has any non-zero configured price component.
func (r *Registry) HasPrice(model string) bool {
	info, ok := r.Lookup(model)
	return ok && !priceZero(info.Price)
}

// MergeModel fills missing registry metadata for model. Explicit provider
// config values win; discovered data is used only where the registry has no
// context window or no price at all.
func (r *Registry) MergeModel(model string, info ModelInfo) {
	if r == nil || model == "" {
		return
	}
	current := r.models[model]
	if current.ContextWindow <= 0 && info.ContextWindow > 0 {
		current.ContextWindow = info.ContextWindow
	}
	if priceZero(current.Price) && !priceZero(info.Price) {
		current.Price = info.Price
	}
	if current.Reasoning == nil && info.Reasoning != nil {
		current.Reasoning = info.Reasoning.Clone()
	} else if current.Reasoning != nil && len(current.Reasoning.Options) == 0 && info.Reasoning != nil && len(info.Reasoning.Options) > 0 {
		current.Reasoning.Options = info.Reasoning.Clone().Options
	}
	r.models[model] = current
}

// SetDefaultContextWindow sets the fallback window used when a model has no
// configured context window. Non-positive values reset to the built-in default.
func (r *Registry) SetDefaultContextWindow(window int) {
	if r == nil {
		return
	}
	if window <= 0 {
		window = DefaultContextWindow
	}
	r.defaultContextWindow = window
}

func decodeProviderConfigs(data []byte) ([]ProviderConfig, error) {
	var many []ProviderConfig
	if err := json.Unmarshal(data, &many); err == nil {
		return many, nil
	}

	var wrapper struct {
		Providers []ProviderConfig `json:"providers"`
	}
	if err := json.Unmarshal(data, &wrapper); err == nil && wrapper.Providers != nil {
		return wrapper.Providers, nil
	}

	var one ProviderConfig
	if err := json.Unmarshal(data, &one); err != nil {
		return nil, err
	}
	return []ProviderConfig{one}, nil
}

// Cost returns the USD cost of the given usage for the named model, and whether
// the model was found in the registry. Unknown models report (0, false) so the
// UI can show token counts without a dollar figure.
func (r *Registry) Cost(model string, u Usage) (usd float64, known bool) {
	if r == nil {
		return 0, false
	}
	info, ok := r.models[model]
	if !ok {
		return 0, false
	}
	const perMillion = 1_000_000.0
	p := info.Price
	if p.Input == 0 && p.Output == 0 && p.CacheRead == 0 && p.CacheWrite == 0 {
		return 0, false
	}
	usd = float64(u.InputTokens)/perMillion*p.Input +
		float64(u.OutputTokens)/perMillion*p.Output +
		float64(u.CacheReadTokens)/perMillion*p.CacheRead +
		float64(u.CacheWriteTokens)/perMillion*p.CacheWrite
	return usd, true
}

func priceZero(p Price) bool {
	return p.Input == 0 && p.Output == 0 && p.CacheRead == 0 && p.CacheWrite == 0
}

func modelEntryReasoning(m ModelEntry) *ReasoningInfo {
	if m.Reasoning == nil && len(m.ReasoningOptions) == 0 {
		return nil
	}
	supported := false
	if m.Reasoning != nil {
		supported = *m.Reasoning
	}
	return (&ReasoningInfo{
		Supported: supported,
		Options:   append([]ReasoningOption(nil), m.ReasoningOptions...),
	}).Clone()
}

// ContextWindow returns the model's context window from the registry, or the
// configured default for unknown models.
func (r *Registry) ContextWindow(model string) int {
	if r == nil {
		return DefaultContextWindow
	}
	if info, ok := r.models[model]; ok && info.ContextWindow > 0 {
		return info.ContextWindow
	}
	if r.defaultContextWindow > 0 {
		return r.defaultContextWindow
	}
	return DefaultContextWindow
}

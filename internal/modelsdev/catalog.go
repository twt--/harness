// Package modelsdev reads the public models.dev catalog and reduces it to the
// provider, endpoint, model context, and pricing fields harness needs.
package modelsdev

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"harness/internal/llm"
)

const DefaultURL = "https://models.dev/api.json"

// Catalog is the provider-keyed models.dev API response.
type Catalog struct {
	Providers map[string]Provider
}

// Provider is one provider entry in the models.dev API response.
type Provider struct {
	ID     string           `json:"id"`
	Name   string           `json:"name"`
	API    string           `json:"api"`
	Doc    string           `json:"doc"`
	NPM    string           `json:"npm"`
	Env    []string         `json:"env"`
	Models map[string]Model `json:"models"`
}

// Model is the subset of one models.dev model entry used by harness.
type Model struct {
	ID    string    `json:"id"`
	Name  string    `json:"name"`
	Limit Limit     `json:"limit"`
	Cost  llm.Price `json:"cost"`
}

// Limit carries token limits from models.dev.
type Limit struct {
	Context int `json:"context"`
}

// Fetch downloads and decodes a models.dev API catalog. A nil client uses the
// default HTTP client.
func Fetch(ctx context.Context, client *http.Client, url string) (*Catalog, error) {
	if url == "" {
		url = DefaultURL
	}
	if client == nil {
		client = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models.dev: GET %s: %s", url, resp.Status)
	}
	return Decode(resp.Body)
}

// Decode parses a models.dev API catalog.
func Decode(r io.Reader) (*Catalog, error) {
	var providers map[string]Provider
	if err := json.NewDecoder(r).Decode(&providers); err != nil {
		return nil, err
	}
	if providers == nil {
		providers = map[string]Provider{}
	}
	for key, p := range providers {
		if p.ID == "" {
			p.ID = key
		}
		if p.Models == nil {
			p.Models = map[string]Model{}
		}
		for modelKey, m := range p.Models {
			if m.ID == "" {
				m.ID = modelKey
			}
			p.Models[modelKey] = m
		}
		providers[key] = p
	}
	return &Catalog{Providers: providers}, nil
}

// Provider returns the provider with id, matching case-insensitively against the
// provider key and the provider's id field.
func (c *Catalog) Provider(id string) (Provider, bool) {
	if c == nil {
		return Provider{}, false
	}
	id = strings.ToLower(strings.TrimSpace(id))
	if id == "" {
		return Provider{}, false
	}
	if p, ok := c.Providers[id]; ok {
		return p, true
	}
	for key, p := range c.Providers {
		if strings.ToLower(key) == id || strings.ToLower(p.ID) == id {
			return p, true
		}
	}
	return Provider{}, false
}

// ProviderByAPI returns the provider whose models.dev api field matches baseURL
// after trimming trailing slashes.
func (c *Catalog) ProviderByAPI(baseURL string) (Provider, bool) {
	if c == nil {
		return Provider{}, false
	}
	baseURL = normalizeURL(baseURL)
	if baseURL == "" {
		return Provider{}, false
	}
	for _, p := range c.Providers {
		if normalizeURL(p.API) == baseURL {
			return p, true
		}
	}
	return Provider{}, false
}

// ResolveProvider resolves exact ids/names first, then unique prefixes over ids
// and display names. If the prefix is ambiguous, matches contains the candidates.
func (c *Catalog) ResolveProvider(input string) (Provider, []Provider, bool) {
	matches := c.MatchProviders(input)
	if len(matches) == 0 {
		return Provider{}, nil, false
	}
	input = strings.ToLower(strings.TrimSpace(input))
	for _, p := range matches {
		if strings.ToLower(p.ID) == input || strings.ToLower(p.Name) == input {
			return p, nil, true
		}
	}
	if len(matches) == 1 {
		return matches[0], nil, true
	}
	return Provider{}, matches, false
}

// MatchProviders returns provider candidates whose id or display name has input
// as a case-insensitive prefix.
func (c *Catalog) MatchProviders(input string) []Provider {
	if c == nil {
		return nil
	}
	input = strings.ToLower(strings.TrimSpace(input))
	if input == "" {
		return nil
	}
	var matches []Provider
	for _, p := range c.Providers {
		if strings.HasPrefix(strings.ToLower(p.ID), input) || strings.HasPrefix(strings.ToLower(p.Name), input) {
			matches = append(matches, p)
		}
	}
	sortProviders(matches)
	return matches
}

// BaseURL returns the provider API base URL known to models.dev, falling back to
// harness's built-in defaults for first-party providers.
func (p Provider) BaseURL() string {
	if p.API != "" {
		return p.API
	}
	switch p.ID {
	case "openai":
		return "https://api.openai.com/v1"
	case "anthropic":
		return "https://api.anthropic.com"
	default:
		return ""
	}
}

// APIType returns the harness dialect to use for this provider when it is known.
func (p Provider) APIType() string {
	if p.ID == "anthropic" || strings.Contains(strings.ToLower(p.NPM), "anthropic") || containsString(p.Env, "ANTHROPIC_API_KEY") {
		return "anthropic"
	}
	if p.ID == "openai" || p.API != "" || strings.Contains(strings.ToLower(p.NPM), "openai") {
		return "openai"
	}
	return ""
}

// ResolveModel resolves exact model ids/names first, then unique id prefixes.
// Ambiguous prefixes are returned in matches.
func (p Provider) ResolveModel(input string) (Model, []Model, bool) {
	matches := p.MatchModels(input)
	if len(matches) == 0 {
		return Model{}, nil, false
	}
	input = strings.ToLower(strings.TrimSpace(input))
	for _, m := range matches {
		if strings.ToLower(m.ID) == input || strings.ToLower(m.Name) == input {
			return m, nil, true
		}
	}
	if len(matches) == 1 {
		return matches[0], nil, true
	}
	return Model{}, matches, false
}

// MatchModels returns model candidates whose id or display name has input as a
// case-insensitive prefix.
func (p Provider) MatchModels(input string) []Model {
	input = strings.ToLower(strings.TrimSpace(input))
	if input == "" {
		return nil
	}
	var matches []Model
	for _, m := range p.Models {
		if strings.HasPrefix(strings.ToLower(m.ID), input) || strings.HasPrefix(strings.ToLower(m.Name), input) {
			matches = append(matches, m)
		}
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].ID < matches[j].ID })
	return matches
}

// ModelInfo returns harness registry metadata for modelID.
func (p Provider) ModelInfo(modelID string) (llm.ModelInfo, bool) {
	if p.Models == nil {
		return llm.ModelInfo{}, false
	}
	if m, ok := p.Models[modelID]; ok {
		return m.ModelInfo(), true
	}
	for _, m := range p.Models {
		if m.ID == modelID {
			return m.ModelInfo(), true
		}
	}
	return llm.ModelInfo{}, false
}

// ModelInfo converts one models.dev model into a harness registry entry.
func (m Model) ModelInfo() llm.ModelInfo {
	return llm.ModelInfo{
		ContextWindow: m.Limit.Context,
		Price:         m.Cost,
	}
}

func sortProviders(providers []Provider) {
	sort.Slice(providers, func(i, j int) bool {
		return providers[i].ID < providers[j].ID
	})
}

func normalizeURL(s string) string {
	return strings.TrimRight(strings.ToLower(strings.TrimSpace(s)), "/")
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

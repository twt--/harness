// Setup wizard and provider-config refresh for cmd harness: the `harness setup`
// interactive flow (models.dev-backed provider/model pickers) and the
// `-refresh-models` re-sync of provider config files. Split from main.go so the
// entrypoint stays the thin config -> factory -> tools -> agent -> ui wiring it
// documents.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"harness/internal/llm"
	"harness/internal/modelsdev"
)

type setupMainConfig struct {
	Provider             string   `json:"provider"`
	Model                string   `json:"model"`
	ProviderConfigs      []string `json:"provider_configs"`
	DefaultContextWindow int      `json:"default_context_window"`
}

type setupProviderConfig struct {
	Name      string             `json:"name"`
	APIType   string             `json:"api_type"`
	BaseURL   string             `json:"base_url"`
	APIKey    string             `json:"api_key"`
	APIKeyEnv []string           `json:"api_key_env,omitempty"`
	Models    []setupModelConfig `json:"models"`
}

type setupModelConfig struct {
	Name             string                `json:"name"`
	ContextWindow    int                   `json:"context_window,omitempty"`
	Price            *llm.Price            `json:"price,omitempty"`
	Reasoning        *bool                 `json:"reasoning,omitempty"`
	ReasoningOptions []llm.ReasoningOption `json:"reasoning_options,omitempty"`
}

func runSetup(env environment, force bool) error {
	dir := defaultConfigDir(env.getenv)
	configPath := filepath.Join(dir, "config.json")
	configExists, err := pathExists(configPath)
	if err != nil {
		return err
	}

	reader := bufio.NewReader(env.stdin)
	catalog, err := setupCatalog(env)
	if err != nil {
		return err
	}

	providerMeta, err := promptProviderSelection(reader, env.stdout, catalog, setupPageSize(env))
	if err != nil {
		return err
	}
	providerName := providerMeta.ID
	providerFile := providerConfigFilename(providerName)
	providerPath := filepath.Join(dir, providerFile)
	providerExists, err := pathExists(providerPath)
	if err != nil {
		return err
	}
	if providerExists && !force {
		return fmt.Errorf("%s already exists", providerPath)
	}
	if providerMeta.APIType() == "" || providerMeta.BaseURL() == "" {
		return fmt.Errorf("provider %q is not supported by harness", providerName)
	}
	apiKeyLabel := "API key (optional)"
	if len(providerMeta.Env) > 0 {
		apiKeyLabel = fmt.Sprintf("API key (optional; env %s also works)", strings.Join(providerMeta.Env, "/"))
	}
	apiKey, err := promptLine(reader, env.stdout, apiKeyLabel+": ")
	if err != nil {
		return err
	}
	model, err := promptModelSelection(reader, env.stdout, providerMeta, setupPageSize(env))
	if err != nil {
		return err
	}

	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}

	provider := setupProviderFromModelsDev(providerMeta, apiKey)

	mainConfig := setupMainConfig{
		Provider:             providerName,
		Model:                model.ID,
		ProviderConfigs:      []string{providerFile},
		DefaultContextWindow: llm.DefaultContextWindow,
	}

	var configBody any = mainConfig
	if configExists {
		updated, err := updatedSetupConfig(configPath, providerFile, providerName, model.ID, force)
		if err != nil {
			return err
		}
		configBody = updated
	}

	if err := writeSetupProviderConfig(providerPath, provider, force); err != nil {
		return err
	}

	writeConfig := writeJSONFileExclusive
	configVerb := "Wrote"
	if configExists {
		writeConfig = writeJSONFileAtomic
		configVerb = "Updated"
	}
	if err := writeConfig(configPath, configBody); err != nil {
		if !providerExists {
			_ = os.Remove(providerPath)
		}
		return err
	}

	providerVerb := "Wrote"
	if providerExists {
		providerVerb = "Overwrote"
	}
	fmt.Fprintf(env.stdout, "%s %s\n", configVerb, configPath)
	fmt.Fprintf(env.stdout, "%s %s\n", providerVerb, providerPath)
	return nil
}

func runRefreshModels(env environment, cfgPath string) error {
	if cfgPath == "" {
		return fmt.Errorf("no config file found")
	}
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	files, err := setupProviderConfigs(raw["provider_configs"])
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fmt.Errorf("%s has no provider_configs", cfgPath)
	}
	catalog, err := refreshCatalog(env)
	if err != nil {
		return err
	}

	dir := filepath.Dir(cfgPath)
	for _, file := range files {
		path := file
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, file)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		providers, err := llm.DecodeProviderConfigs(data)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if len(providers) == 0 {
			return fmt.Errorf("%s has no providers", path)
		}
		updated := make([]setupProviderConfig, 0, len(providers))
		for _, current := range providers {
			if current.Name == "" {
				return fmt.Errorf("%s has provider without name", path)
			}
			meta, ok := catalog.Provider(current.Name)
			if !ok {
				return fmt.Errorf("provider %q from %s was not found in models.dev", current.Name, path)
			}
			if meta.APIType() == "" || meta.BaseURL() == "" {
				return fmt.Errorf("provider %q from %s is not supported by harness", current.Name, path)
			}
			updated = append(updated, setupProviderFromModelsDev(meta, current.APIKey))
		}
		var body any = updated
		if len(updated) == 1 {
			body = updated[0]
		}
		if err := writeJSONFileAtomic(path, body); err != nil {
			return err
		}
		fmt.Fprintf(env.stdout, "Updated %s\n", path)
	}
	return nil
}

func refreshCatalog(env environment) (*modelsdev.Catalog, error) {
	if env.modelsDevCatalog != nil {
		return env.modelsDevCatalog(context.Background())
	}
	return defaultModelsDevCatalog(context.Background())
}

func updatedSetupConfig(path, providerFile, providerName, modelName string, force bool) (map[string]json.RawMessage, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg == nil {
		cfg = map[string]json.RawMessage{}
	}

	configs, err := setupProviderConfigs(cfg["provider_configs"])
	if err != nil {
		return nil, err
	}
	if slices.Contains(configs, providerFile) && !force {
		return nil, fmt.Errorf("%s already references provider config %s", path, providerFile)
	}
	if !slices.Contains(configs, providerFile) {
		configs = append(configs, providerFile)
	}
	if err := setJSONField(cfg, "provider_configs", configs); err != nil {
		return nil, err
	}

	if err := setSetupStringField(cfg, "provider", providerName, force); err != nil {
		return nil, err
	}
	if err := setSetupStringField(cfg, "model", modelName, force); err != nil {
		return nil, err
	}
	if _, ok := cfg["default_context_window"]; !ok || force {
		if err := setJSONField(cfg, "default_context_window", llm.DefaultContextWindow); err != nil {
			return nil, err
		}
	}
	return cfg, nil
}

func setupProviderConfigs(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var configs []string
	if err := json.Unmarshal(raw, &configs); err != nil {
		return nil, fmt.Errorf("provider_configs must be an array of strings: %w", err)
	}
	return configs, nil
}

func setupProviderFromModelsDev(provider modelsdev.Provider, apiKey string) setupProviderConfig {
	cfg := provider.ProviderConfig(apiKey)
	models := provider.ModelsByID()
	entries := make([]setupModelConfig, 0, len(models))
	for _, model := range models {
		entries = append(entries, setupModelFromModelsDev(model))
	}
	return setupProviderConfig{
		Name:      cfg.Name,
		APIType:   cfg.APIType,
		BaseURL:   cfg.BaseURL,
		APIKey:    cfg.APIKey,
		APIKeyEnv: cfg.APIKeyEnv,
		Models:    entries,
	}
}

func promptProviderSelection(r *bufio.Reader, w io.Writer, catalog *modelsdev.Catalog, pageSize int) (modelsdev.Provider, error) {
	providers := supportedSetupProviders(catalog)
	if len(providers) == 0 {
		return modelsdev.Provider{}, fmt.Errorf("models.dev catalog has no harness-supported providers")
	}
	filter := ""
	page := 0
	for {
		filtered := filterEntries(providers, filter)
		if len(filtered) == 0 {
			fmt.Fprintf(w, "No providers match %q\n", filter)
			filter = ""
			page = 0
			continue
		}
		page = clampPage(page, len(filtered), pageSize)
		printProviderPage(w, filtered, page, pageSize, filter)
		input, err := promptLine(r, w, "Provider (number/id, /search, n/p, q): ")
		if err != nil {
			return modelsdev.Provider{}, err
		}
		input = strings.TrimSpace(input)
		if input == "" || strings.EqualFold(input, "n") {
			if (page+1)*pageSize < len(filtered) {
				page++
			}
			continue
		}
		if strings.EqualFold(input, "p") {
			if page > 0 {
				page--
			}
			continue
		}
		if strings.EqualFold(input, "q") {
			return modelsdev.Provider{}, fmt.Errorf("setup cancelled")
		}
		if strings.HasPrefix(input, "/") {
			filter = strings.TrimSpace(strings.TrimPrefix(input, "/"))
			page = 0
			continue
		}
		if n, ok := parseSelectionNumber(input, len(filtered)); ok {
			return filtered[n-1], nil
		}
		if provider, matches, ok := resolveSelection(providers, input); ok {
			if provider.ID != input {
				fmt.Fprintf(w, "Using provider %s%s\n", provider.ID, displayNameSuffix(provider.Name, provider.ID))
			}
			return provider, nil
		} else if len(matches) > 1 {
			fmt.Fprintf(w, "Matches: %s\n", matchSummary(matches, 8))
			continue
		}
		filter = input
		page = 0
	}
}

func promptModelSelection(r *bufio.Reader, w io.Writer, provider modelsdev.Provider, pageSize int) (modelsdev.Model, error) {
	models := provider.ModelsByReleaseDate()
	if len(models) == 0 {
		return modelsdev.Model{}, fmt.Errorf("provider %q has no models", provider.ID)
	}
	filter := ""
	page := 0
	for {
		filtered := filterEntries(models, filter)
		if len(filtered) == 0 {
			fmt.Fprintf(w, "No models match %q\n", filter)
			filter = ""
			page = 0
			continue
		}
		page = clampPage(page, len(filtered), pageSize)
		printModelPage(w, provider, filtered, page, pageSize, filter)
		input, err := promptLine(r, w, "Default model (number/id, /search, n/p, q): ")
		if err != nil {
			return modelsdev.Model{}, err
		}
		input = strings.TrimSpace(input)
		if input == "" || strings.EqualFold(input, "n") {
			if (page+1)*pageSize < len(filtered) {
				page++
			}
			continue
		}
		if strings.EqualFold(input, "p") {
			if page > 0 {
				page--
			}
			continue
		}
		if strings.EqualFold(input, "q") {
			return modelsdev.Model{}, fmt.Errorf("setup cancelled")
		}
		if strings.HasPrefix(input, "/") {
			filter = strings.TrimSpace(strings.TrimPrefix(input, "/"))
			page = 0
			continue
		}
		if n, ok := parseSelectionNumber(input, len(filtered)); ok {
			return filtered[n-1], nil
		}
		if model, matches, ok := resolveSelection(models, input); ok {
			if model.ID != input {
				fmt.Fprintf(w, "Using model %s%s\n", model.ID, displayNameSuffix(model.Name, model.ID))
			}
			return model, nil
		} else if len(matches) > 1 {
			fmt.Fprintf(w, "Matches: %s\n", matchSummary(matches, 8))
			continue
		}
		filter = input
		page = 0
	}
}

func supportedSetupProviders(catalog *modelsdev.Catalog) []modelsdev.Provider {
	var providers []modelsdev.Provider
	for _, provider := range catalog.ProvidersList() {
		if provider.APIType() == "" || provider.BaseURL() == "" || len(provider.Models) == 0 {
			continue
		}
		providers = append(providers, provider)
	}
	sort.Slice(providers, func(i, j int) bool {
		if strings.EqualFold(providers[i].Name, providers[j].Name) {
			return providers[i].ID < providers[j].ID
		}
		return strings.ToLower(providers[i].Name) < strings.ToLower(providers[j].Name)
	})
	return providers
}

// pickEntry abstracts the two models.dev list element types the picker pages
// through; both expose an id and a display name, which entryIDName extracts.
type pickEntry interface {
	modelsdev.Provider | modelsdev.Model
}

func entryIDName[T pickEntry](v T) (id, name string) {
	switch e := any(v).(type) {
	case modelsdev.Provider:
		return e.ID, e.Name
	case modelsdev.Model:
		return e.ID, e.Name
	}
	return "", ""
}

// filterEntries keeps the entries whose id or display name contains filter,
// case-insensitively. An empty filter keeps everything.
func filterEntries[T pickEntry](items []T, filter string) []T {
	filter = strings.ToLower(strings.TrimSpace(filter))
	if filter == "" {
		return items
	}
	var out []T
	for _, item := range items {
		id, name := entryIDName(item)
		if strings.Contains(strings.ToLower(id), filter) || strings.Contains(strings.ToLower(name), filter) {
			out = append(out, item)
		}
	}
	return out
}

// resolveSelection resolves exact id/name matches first, then unique prefixes.
// An ambiguous prefix returns the candidates in matches.
func resolveSelection[T pickEntry](items []T, input string) (selected T, matches []T, ok bool) {
	input = strings.ToLower(strings.TrimSpace(input))
	var prefix []T
	for _, item := range items {
		id, name := entryIDName(item)
		id = strings.ToLower(id)
		name = strings.ToLower(name)
		if id == input || name == input {
			return item, nil, true
		}
		if strings.HasPrefix(id, input) || strings.HasPrefix(name, input) {
			prefix = append(prefix, item)
		}
	}
	if len(prefix) == 1 {
		return prefix[0], nil, true
	}
	var zero T
	return zero, prefix, false
}

func printProviderPage(w io.Writer, providers []modelsdev.Provider, page, pageSize int, filter string) {
	start, end := pageBounds(page, pageSize, len(providers))
	title := fmt.Sprintf("Providers %d-%d of %d", start+1, end, len(providers))
	if filter != "" {
		title += fmt.Sprintf(" matching %q", filter)
	}
	fmt.Fprintln(w, title)
	for i := start; i < end; i++ {
		provider := providers[i]
		fmt.Fprintf(w, "%4d. %-28s %5d models  %s\n", i+1, provider.ID, len(provider.Models), provider.Name)
	}
}

func printModelPage(w io.Writer, provider modelsdev.Provider, models []modelsdev.Model, page, pageSize int, filter string) {
	start, end := pageBounds(page, pageSize, len(models))
	title := fmt.Sprintf("Models for %s %d-%d of %d", provider.ID, start+1, end, len(models))
	if filter != "" {
		title += fmt.Sprintf(" matching %q", filter)
	}
	fmt.Fprintln(w, title)
	for i := start; i < end; i++ {
		model := models[i]
		release := model.ReleaseDate
		if release == "" {
			release = model.LastUpdated
		}
		if release == "" {
			release = "-"
		}
		fmt.Fprintf(w, "%4d. %-44s %10s  %s\n", i+1, clipSetup(model.ID, 44), release, model.Name)
	}
}

func parseSelectionNumber(input string, max int) (int, bool) {
	n, err := strconv.Atoi(input)
	if err != nil || n < 1 || n > max {
		return 0, false
	}
	return n, true
}

func clampPage(page, total, pageSize int) int {
	if pageSize <= 0 {
		pageSize = 20
	}
	maxPage := (total - 1) / pageSize
	if page < 0 {
		return 0
	}
	if page > maxPage {
		return maxPage
	}
	return page
}

func pageBounds(page, pageSize, total int) (start, end int) {
	if pageSize <= 0 {
		pageSize = 20
	}
	start = page * pageSize
	if start > total {
		start = total
	}
	end = start + pageSize
	if end > total {
		end = total
	}
	return start, end
}

func setupPageSize(env environment) int {
	rows := 0
	if env.terminalRows != nil {
		rows = env.terminalRows()
	}
	return pickerPageSize(rows)
}

func pickerPageSize(rows int) int {
	if rows <= 0 {
		return 20
	}
	size := rows - 6
	if size < 5 {
		return 5
	}
	return size
}

func setSetupStringField(cfg map[string]json.RawMessage, key, value string, force bool) error {
	if _, ok := cfg[key]; ok && !force {
		return nil
	}
	return setJSONField(cfg, key, value)
}

func setJSONField(cfg map[string]json.RawMessage, key string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	cfg[key] = data
	return nil
}

func writeSetupProviderConfig(path string, provider setupProviderConfig, force bool) error {
	if force {
		return writeJSONFileAtomic(path, provider)
	}
	return writeJSONFileExclusive(path, provider)
}

// marshalJSONLine renders v as indented JSON with a trailing newline, the
// on-disk form both config writers share.
func marshalJSONLine(v any) ([]byte, error) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

func writeJSONFileAtomic(path string, v any) error {
	data, err := marshalJSONLine(v)
	if err != nil {
		return err
	}

	dir := filepath.Dir(path)
	base := filepath.Base(path)
	tmp, err := os.CreateTemp(dir, "."+base+".*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	return nil
}

func setupCatalog(env environment) (*modelsdev.Catalog, error) {
	if env.modelsDevCatalog != nil {
		catalog, err := env.modelsDevCatalog(context.Background())
		if err == nil {
			return catalog, nil
		}
		fallback, fallbackErr := modelsdev.Fallback()
		if fallbackErr != nil {
			return nil, fmt.Errorf("models.dev lookup failed: %v; vendored fallback failed: %w", err, fallbackErr)
		}
		fmt.Fprintf(env.stderr, "harness: setup: warning: models.dev lookup failed: %v; using vendored fallback\n", err)
		return fallback, nil
	}
	return modelsdev.Fallback()
}

func setupModelFromModelsDev(model modelsdev.Model) setupModelConfig {
	cfg := setupModelConfig{
		Name:             model.ID,
		ContextWindow:    model.Limit.Context,
		ReasoningOptions: append([]llm.ReasoningOption(nil), model.ReasoningOptions...),
	}
	reasoning := model.Reasoning
	cfg.Reasoning = &reasoning
	if setupPriceKnown(model.Cost) {
		price := model.Cost
		cfg.Price = &price
	}
	return cfg
}

// matchSummary renders up to limit ambiguous-match candidates as "id (Name)"
// for the picker's "Matches: ..." hint line.
func matchSummary[T pickEntry](matches []T, limit int) string {
	if len(matches) > limit {
		matches = matches[:limit]
	}
	parts := make([]string, 0, len(matches))
	for _, m := range matches {
		id, name := entryIDName(m)
		parts = append(parts, id+displayNameSuffix(name, id))
	}
	return strings.Join(parts, ", ")
}

func displayNameSuffix(name, id string) string {
	if name == "" || name == id {
		return ""
	}
	return " (" + name + ")"
}

func clipSetup(s string, n int) string {
	if n <= 0 || len(s) <= n {
		return s
	}
	if n <= 3 {
		return s[:n]
	}
	return s[:n-3] + "..."
}

func setupPriceKnown(p llm.Price) bool {
	return p.Input != 0 || p.Output != 0 || p.CacheRead != 0 || p.CacheWrite != 0
}

func promptLine(r *bufio.Reader, w io.Writer, label string) (string, error) {
	if _, err := fmt.Fprint(w, label); err != nil {
		return "", err
	}
	line, err := r.ReadString('\n')
	if err != nil && !(errors.Is(err, io.EOF) && line != "") {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

func writeJSONFileExclusive(path string, v any) error {
	data, err := marshalJSONLine(v)
	if err != nil {
		return err
	}

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func pathExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, err
}

func providerConfigFilename(name string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.ToLower(name) {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '.'
		if ok {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	s := strings.Trim(b.String(), ".-")
	if s == "" {
		s = "provider"
	}
	return s + ".json"
}

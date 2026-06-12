package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"harness/internal/llm"
	"harness/internal/llm/factory"
	"harness/internal/modelproxy/protocol"
)

const maxStreamRequestBytes = 64 << 20

type Config struct {
	Provider             string   `json:"provider"`
	Model                string   `json:"model"`
	ProviderConfigs      []string `json:"provider_configs"`
	DefaultContextWindow int      `json:"default_context_window"`
}

type Options struct {
	ConfigDir string
	Config    Config
	Getenv    func(string) string
	Logger    *slog.Logger
	New       func(factory.Options) (llm.Provider, error)
	Warn      func(string)
}

type Handler struct {
	catalog              protocol.Catalog
	providers            []llm.ProviderConfig
	defaultContextWindow int
	getenv               func(string) string
	logger               *slog.Logger
	newProvider          func(factory.Options) (llm.Provider, error)
}

func NewHandler(opts Options) (*Handler, error) {
	getenv := opts.Getenv
	if getenv == nil {
		getenv = os.Getenv
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	newProvider := opts.New
	if newProvider == nil {
		newProvider = factory.New
	}
	warn := opts.Warn
	registry, providers, err := llm.LoadProviderConfigs(opts.ConfigDir, opts.Config.ProviderConfigs, warn)
	if err != nil {
		return nil, err
	}
	defaultWindow := opts.Config.DefaultContextWindow
	if defaultWindow <= 0 {
		defaultWindow = llm.DefaultContextWindow
	}
	registry.SetDefaultContextWindow(defaultWindow)
	if len(providers) == 0 {
		return nil, fmt.Errorf("model proxy: no provider configs are configured")
	}
	catalog := catalogFromProviderConfigs(providers, opts.Config.Provider, opts.Config.Model)
	if catalog.DefaultProvider == "" || catalog.DefaultModel == "" {
		provider, model, err := firstConfiguredModel(providers)
		if err != nil {
			return nil, err
		}
		if catalog.DefaultProvider == "" {
			catalog.DefaultProvider = provider
		}
		if catalog.DefaultModel == "" {
			catalog.DefaultModel = model
		}
	}
	return &Handler{
		catalog:              catalog,
		providers:            providers,
		defaultContextWindow: defaultWindow,
		getenv:               getenv,
		logger:               logger,
		newProvider:          newProvider,
	}, nil
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		h.handleModels(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/stream":
		h.handleStream(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (h *Handler) Catalog() protocol.Catalog {
	return h.catalog
}

func (h *Handler) handleModels(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("content-type", "application/json")
	if err := json.NewEncoder(w).Encode(h.catalog); err != nil {
		h.logger.Warn("write model catalog failed", "err", err)
	}
}

func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxStreamRequestBytes))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, &protocol.Error{StatusCode: http.StatusRequestEntityTooLarge, Message: "request body too large"})
		return
	}
	var req protocol.StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, &protocol.Error{StatusCode: http.StatusBadRequest, Message: "malformed stream request"})
		return
	}
	providerID := strings.TrimSpace(req.Provider)
	if providerID == "" {
		providerID = h.catalog.DefaultProvider
	}
	if req.Request.Model == "" {
		req.Request.Model = h.catalog.DefaultModel
	}

	opts, err := h.runtimeOptions(providerID, req.Request.Model)
	if err != nil {
		writeError(w, http.StatusBadRequest, &protocol.Error{StatusCode: http.StatusBadRequest, Message: err.Error()})
		return
	}
	provider, err := h.newProvider(opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, protocol.ErrorFrom(err))
		return
	}

	w.Header().Set("content-type", protocol.ContentTypeNDJSON)
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	enc := json.NewEncoder(w)
	flush := func() {
		if flusher != nil {
			flusher.Flush()
		}
	}

	for ev, err := range provider.Stream(r.Context(), req.Request) {
		if err != nil {
			_ = enc.Encode(protocol.StreamEnvelope{Error: protocol.ErrorFrom(err)})
			flush()
			return
		}
		event := ev
		if err := enc.Encode(protocol.StreamEnvelope{Event: &event}); err != nil {
			return
		}
		flush()
	}
}

func (h *Handler) runtimeOptions(providerID, model string) (factory.Options, error) {
	pc, ok := providerConfigByName(h.providers, providerID)
	if !ok {
		return factory.Options{}, fmt.Errorf("provider %q is not configured", providerID)
	}
	entry, ok := providerConfigModel(pc, model)
	if !ok {
		return factory.Options{}, fmt.Errorf("provider %q has no configured model %q", providerID, model)
	}
	apiType := pc.APIType
	if apiType == "" {
		apiType = pc.Name
	}
	apiKey := ""
	for _, name := range pc.APIKeyEnv {
		if value := h.getenv(name); value != "" {
			apiKey = value
			break
		}
	}
	if apiKey == "" {
		apiKey = providerAPIKeyEnv(apiType, h.getenv)
	}
	if apiKey == "" {
		apiKey = pc.APIKey
	}
	contextWindow := entry.ContextWindow
	if contextWindow <= 0 {
		contextWindow = h.defaultContextWindow
	}
	return factory.Options{
		Provider:      apiType,
		ProviderName:  pc.Name,
		Model:         model,
		BaseURL:       pc.BaseURL,
		APIKey:        apiKey,
		ContextWindow: contextWindow,
	}, nil
}

func writeError(w http.ResponseWriter, status int, e *protocol.Error) {
	if e == nil {
		e = &protocol.Error{StatusCode: status, Message: http.StatusText(status)}
	}
	if e.StatusCode == 0 {
		e.StatusCode = status
	}
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(e)
}

func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func ConfigPath(argsPath string, explicit bool, getenv func(string) string) string {
	if explicit {
		return argsPath
	}
	def := filepath.Join(DefaultConfigDir(getenv), "config.json")
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return ""
}

func DefaultConfigDir(getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if home := getenv("HOME"); home != "" {
		return filepath.Join(home, ".config", "harness-model-proxy")
	}
	return filepath.Join(os.TempDir(), "harness-model-proxy-config")
}

func catalogFromProviderConfigs(providers []llm.ProviderConfig, defaultProvider, defaultModel string) protocol.Catalog {
	out := protocol.Catalog{
		DefaultProvider: defaultProvider,
		DefaultModel:    defaultModel,
		Providers:       make([]protocol.Provider, 0, len(providers)),
	}
	for _, pc := range providers {
		if pc.Name == "" {
			continue
		}
		p := protocol.Provider{
			ID:     pc.Name,
			Name:   pc.Name,
			Models: make([]protocol.Model, 0, len(pc.Models)),
		}
		for _, entry := range pc.Models {
			if entry.Name == "" {
				continue
			}
			p.Models = append(p.Models, protocol.Model{
				ID:            entry.Name,
				Name:          entry.Name,
				ContextWindow: entry.ContextWindow,
				Price:         entry.Price,
				Reasoning:     modelEntryReasoning(entry),
			})
		}
		if len(p.Models) > 0 {
			out.Providers = append(out.Providers, p)
		}
	}
	return out
}

func firstConfiguredModel(providers []llm.ProviderConfig) (provider, model string, err error) {
	for _, pc := range providers {
		if pc.Name == "" {
			continue
		}
		for _, entry := range pc.Models {
			if entry.Name != "" {
				return pc.Name, entry.Name, nil
			}
		}
	}
	return "", "", fmt.Errorf("model proxy: no configured models")
}

func providerConfigByName(providers []llm.ProviderConfig, name string) (llm.ProviderConfig, bool) {
	for _, pc := range providers {
		if pc.Name == name {
			return pc, true
		}
	}
	return llm.ProviderConfig{}, false
}

func providerConfigModel(pc llm.ProviderConfig, model string) (llm.ModelEntry, bool) {
	for _, entry := range pc.Models {
		if entry.Name == model {
			return entry, true
		}
	}
	return llm.ModelEntry{}, false
}

func modelEntryReasoning(m llm.ModelEntry) *llm.ReasoningInfo {
	if m.Reasoning == nil && len(m.ReasoningOptions) == 0 {
		return nil
	}
	supported := false
	if m.Reasoning != nil {
		supported = *m.Reasoning
	}
	return (&llm.ReasoningInfo{
		Supported: supported,
		Options:   append([]llm.ReasoningOption(nil), m.ReasoningOptions...),
	}).Clone()
}

func providerAPIKeyEnv(provider string, getenv func(string) string) string {
	switch provider {
	case "anthropic":
		return getenv("ANTHROPIC_API_KEY")
	case "responses":
		if v := getenv("RESPONSES_API_KEY"); v != "" {
			return v
		}
		return getenv("OPENAI_API_KEY")
	default:
		return getenv("OPENAI_API_KEY")
	}
}

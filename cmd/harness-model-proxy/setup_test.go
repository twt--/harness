package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"harness/internal/modelsdev"
)

func TestRunSetupWritesOnlySelectedModelsAndNoProxyDefault(t *testing.T) {
	home := t.TempDir()
	var out, errw bytes.Buffer
	env := environment{
		stdin:  strings.NewReader("1\n\nnone\n2\ndone\n"),
		stdout: &out,
		stderr: &errw,
		getenv: func(k string) string {
			if k == "HOME" {
				return home
			}
			return ""
		},
		modelsDevCatalog: func(context.Context) (*modelsdev.Catalog, error) {
			return testSetupCatalog(), nil
		},
		terminalRows: func() int { return 12 },
	}

	if err := runSetup(env, false); err != nil {
		t.Fatalf("runSetup: %v; stderr=%q", err, errw.String())
	}
	if !strings.Contains(out.String(), "*") || !strings.Contains(out.String(), "\x1b[1m") {
		t.Fatalf("model selector should mark enabled rows with star and bold, output=%q", out.String())
	}

	dir := filepath.Join(home, ".config", "harness-model-proxy")
	configData, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatalf("read proxy config: %v", err)
	}
	var mainConfig map[string]json.RawMessage
	if err := json.Unmarshal(configData, &mainConfig); err != nil {
		t.Fatalf("decode proxy config: %v", err)
	}
	if _, ok := mainConfig["provider"]; ok {
		t.Fatalf("proxy config should not contain provider: %s", configData)
	}
	if _, ok := mainConfig["model"]; ok {
		t.Fatalf("proxy config should not contain model: %s", configData)
	}

	providerData, err := os.ReadFile(filepath.Join(dir, "testai.json"))
	if err != nil {
		t.Fatalf("read provider config: %v", err)
	}
	var provider setupProviderConfig
	if err := json.Unmarshal(providerData, &provider); err != nil {
		t.Fatalf("decode provider config: %v", err)
	}
	if len(provider.Models) != 1 || provider.Models[0].Name != "alpha" {
		t.Fatalf("provider models = %+v, want only alpha", provider.Models)
	}
}

func TestRunRefreshModelsPreservesConfiguredModelAllowlist(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfgPath, []byte(`{"provider_configs":["testai.json"]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "testai.json"), []byte(`{
  "name": "testai",
  "api_type": "openai",
  "base_url": "https://api.test/v1",
  "api_key": "sk-test",
  "models": [{"name":"alpha","context_window":1000}]
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	env := environment{
		stdout: &out,
		stderr: &bytes.Buffer{},
		modelsDevCatalog: func(context.Context) (*modelsdev.Catalog, error) {
			return testSetupCatalog(), nil
		},
	}

	if err := runRefreshModels(env, cfgPath); err != nil {
		t.Fatalf("runRefreshModels: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "testai.json"))
	if err != nil {
		t.Fatal(err)
	}
	var provider setupProviderConfig
	if err := json.Unmarshal(data, &provider); err != nil {
		t.Fatal(err)
	}
	if len(provider.Models) != 1 || provider.Models[0].Name != "alpha" || provider.Models[0].ContextWindow != 123000 {
		t.Fatalf("provider models after refresh = %+v, want refreshed alpha only", provider.Models)
	}
}

func testSetupCatalog() *modelsdev.Catalog {
	return &modelsdev.Catalog{Providers: map[string]modelsdev.Provider{
		"testai": {
			ID:   "testai",
			Name: "TestAI",
			API:  "https://api.test/v1",
			NPM:  "@ai-sdk/openai-compatible",
			Env:  []string{"TESTAI_API_KEY"},
			Models: map[string]modelsdev.Model{
				"alpha": {
					ID:          "alpha",
					Name:        "Alpha",
					ReleaseDate: "2025-01-01",
					Limit:       modelsdev.Limit{Context: 123000},
				},
				"beta": {
					ID:          "beta",
					Name:        "Beta",
					ReleaseDate: "2026-01-01",
					Limit:       modelsdev.Limit{Context: 456000},
				},
			},
		},
	}}
}

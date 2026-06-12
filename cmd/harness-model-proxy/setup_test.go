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
		stdin:  strings.NewReader("1\n\nsave\n2\nsave\n"),
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
	if !strings.Contains(out.String(), "(0 enabled)") {
		t.Fatalf("model selector should start with no enabled models, output=%q", out.String())
	}
	if !strings.Contains(out.String(), "Select at least one model before continuing.") {
		t.Fatalf("saving without a selected model should explain the required selection, output=%q", out.String())
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

func TestRunSetupModelSelectorCancelDoesNotWriteConfig(t *testing.T) {
	home := t.TempDir()
	var out, errw bytes.Buffer
	env := environment{
		stdin:  strings.NewReader("1\n\ncancel\n"),
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

	if err := runSetup(env, false); err == nil || err.Error() != "setup cancelled" {
		t.Fatalf("runSetup error = %v, want setup cancelled; stderr=%q", err, errw.String())
	}
	dir := filepath.Join(home, ".config", "harness-model-proxy")
	if _, err := os.Stat(filepath.Join(dir, "config.json")); !os.IsNotExist(err) {
		t.Fatalf("config.json stat error = %v, want not exist", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "testai.json")); !os.IsNotExist(err) {
		t.Fatalf("testai.json stat error = %v, want not exist", err)
	}
}

func TestRunSetupUpdatesExistingProviderConfig(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".config", "harness-model-proxy")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"provider_configs":["testai.json"],"default_context_window":256000}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "testai.json"), []byte(`{
  "name": "testai",
  "api_type": "openai",
  "base_url": "https://api.test/v1",
  "api_key": "sk-existing",
  "models": [{"name":"alpha","context_window":1000}]
}`), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errw bytes.Buffer
	env := environment{
		stdin:  strings.NewReader("1\n\n1\nsave\n"),
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
	output := out.String()
	if !strings.Contains(output, "*   1.") || !strings.Contains(output, "\x1b[1mtestai\x1b[0m") || !strings.Contains(output, "\x1b[1mTestAI\x1b[0m") {
		t.Fatalf("provider selector should mark existing provider with star and bold, output=%q", output)
	}
	if !strings.Contains(output, "(1 enabled)") {
		t.Fatalf("model selector should start from existing allowlist, output=%q", output)
	}
	if !strings.Contains(output, "Beta") {
		t.Fatalf("model selector should show disabled catalog models for existing providers, output=%q", output)
	}
	if !strings.Contains(output, "Updated "+filepath.Join(dir, "testai.json")) {
		t.Fatalf("setup should report provider update, output=%q", output)
	}

	providerData, err := os.ReadFile(filepath.Join(dir, "testai.json"))
	if err != nil {
		t.Fatalf("read provider config: %v", err)
	}
	var provider setupProviderConfig
	if err := json.Unmarshal(providerData, &provider); err != nil {
		t.Fatalf("decode provider config: %v", err)
	}
	if provider.APIKey != "sk-existing" {
		t.Fatalf("provider API key = %q, want preserved existing key", provider.APIKey)
	}
	if len(provider.Models) != 2 || provider.Models[0].Name != "alpha" || provider.Models[1].Name != "beta" {
		t.Fatalf("provider models = %+v, want alpha and beta", provider.Models)
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

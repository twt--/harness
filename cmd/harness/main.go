// Command harness is the entrypoint: it loads configuration, constructs the
// provider, tool registry, and agent, wires SIGINT handling, prints the session
// path, and dispatches to the interactive REPL or one-shot mode (design §10,
// §11). It is deliberately thin: config -> factory -> tools -> agent -> ui.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"harness/internal/agent"
	"harness/internal/config"
	"harness/internal/llm"
	"harness/internal/llm/factory"
	"harness/internal/logging"
	"harness/internal/mode"
	"harness/internal/modelsdev"
	"harness/internal/session"
	"harness/internal/skills"
	"harness/internal/sysprompt"
	"harness/internal/term"
	"harness/internal/tools"
	"harness/internal/ui"
)

func main() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)

	os.Exit(run(environment{
		args:             os.Args[1:],
		stdin:            os.Stdin,
		stdout:           os.Stdout,
		stderr:           os.Stderr,
		getenv:           os.Getenv,
		now:              time.Now,
		colorTTY:         isTTY(os.Stdout),
		stdinPiped:       pipedStdin(os.Stdin),
		sigCh:            sigCh,
		modelsDevCatalog: defaultModelsDevCatalog,
		terminalRows:     defaultTerminalRows,
	}))
}

// environment carries everything run depends on, so the wiring is testable with
// injected readers/writers, env, clock, TTY/pipe flags, and signal channel
// (design §13: no dependence on real time or terminals in tests). A nil sigCh
// disables SIGINT handling (tests).
type environment struct {
	args       []string
	stdin      io.Reader
	stdout     io.Writer
	stderr     io.Writer
	getenv     func(string) string
	now        func() time.Time
	colorTTY   bool // stdout is a terminal (gates color)
	stdinPiped bool // stdin is piped/redirected (gates one-shot stdin read)
	sigCh      chan os.Signal

	// newProvider builds the llm.Provider; nil uses factory.New. Tests inject a
	// scripted provider so run is exercised without real network calls.
	newProvider func(factory.Options) (llm.Provider, error)

	// modelsDevCatalog fetches optional model metadata. A nil fetcher disables
	// online enrichment, which keeps tests and offline runs deterministic.
	modelsDevCatalog func(context.Context) (*modelsdev.Catalog, error)
	terminalRows     func() int
}

// run wires everything together and returns the process exit code (design §10
// exit codes: 0 ok, 1 runtime, 2 usage, 130 interrupted).
func run(env environment) int {
	args := env.args
	stdout := env.stdout
	stderr := env.stderr
	getenv := env.getenv
	now := env.now
	if now == nil {
		now = time.Now
	}
	if len(args) > 0 && args[0] == "session" {
		return runSessionCommand(args[1:], stdout, stderr)
	}

	cfgPath := resolveConfigPath(args, getenv)

	cfg, err := config.Load(args, getenv, cfgPath)
	if err != nil {
		// -h/--help is a request, not a misuse: print the usage screen to stdout
		// and exit 0 (design §10).
		if errors.Is(err, config.ErrHelp) {
			config.Usage(stdout)
			return ui.ExitOK
		}
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}
	if cfg.Setup {
		if err := runSetup(env, cfg.SetupForce); err != nil {
			fmt.Fprintf(stderr, "harness: setup: %v\n", err)
			return ui.ExitUsage
		}
		return ui.ExitOK
	}
	if cfg.RefreshModels {
		if err := runRefreshModels(env, cfgPath); err != nil {
			fmt.Fprintf(stderr, "harness: refresh-models: %v\n", err)
			return ui.ExitUsage
		}
		return ui.ExitOK
	}
	if cfg.Model == "" {
		fmt.Fprintln(stderr, "harness: a model is required (-model or HARNESS_MODEL)")
		return ui.ExitUsage
	}
	logger, err := logging.NewLogger(stderr, cfg.LogLevel, cfg.Quiet)
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}

	modelRegistry, providerConfigs, err := llm.LoadProviderConfigs(configDir(cfgPath), cfg.ProviderConfigs, func(msg string) {
		fmt.Fprintf(stderr, "harness: %s\n", msg)
	})
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}
	modelRegistry.SetDefaultContextWindow(cfg.DefaultContextWindow)
	effectiveProvider, effectiveBaseURL, effectiveAPIKey := resolveProvider(cfg, providerConfigs, getenv)
	reasoning := llm.ReasoningConfig{Effort: cfg.ReasoningEffort}
	if apiType, ok := enrichRegistryFromModelsDev(modelRegistry, cfg.Model, cfg.Provider, effectiveProvider, effectiveBaseURL, env.modelsDevCatalog, !reasoning.Empty(), cfg.Verbose, stderr); ok {
		if useModelsDevAPIType(effectiveProvider, apiType, effectiveBaseURL) {
			effectiveProvider = apiType
		}
	}
	registryModel := registryModelKey(modelRegistry, cfg.Provider, cfg.Model)
	if err := validateReasoningEffort(modelRegistry, registryModel, reasoning); err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}
	effectiveContextWindow := cfg.ContextWindow
	if effectiveContextWindow <= 0 {
		effectiveContextWindow = modelRegistry.ContextWindow(registryModel)
	}

	// System prompt composition (design §8.5). -system and -system-override may
	// be @file references.
	appendText, err := resolveAtFile(cfg.System)
	if err != nil {
		fmt.Fprintf(stderr, "harness: -system: %v\n", err)
		return ui.ExitUsage
	}
	overrideText, err := resolveAtFile(cfg.SystemOverride)
	if err != nil {
		fmt.Fprintf(stderr, "harness: -system-override: %v\n", err)
		return ui.ExitUsage
	}
	// The env block must report the absolute working directory so the model can
	// reason about and resolve absolute file paths (design §8.5: `cwd:
	// /Users/twt/project`). Without an explicit Dir, EnvContext falls back to the
	// literal ".", which tells the agent its cwd is the string "." — useless for
	// path reasoning. An os.Getwd failure leaves Dir empty (the "." fallback), the
	// best we can do.
	wd, _ := os.Getwd()
	// AGENTS.md auto-discovery: if a file named AGENTS.md exists in the
	// directory harness was launched from, include its contents in the system
	// prompt so the model receives project-specific instructions without the
	// user needing to pass -system. A missing file is silently ignored; a read
	// error on an existing file is fatal so the user isn't silently surprised.
	agentsMD, err := loadAgentsMD(wd)
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitRuntime
	}
	if cfg.AgentsMDWarnBytes > 0 && len(agentsMD) > cfg.AgentsMDWarnBytes {
		fmt.Fprintf(stderr, "harness: warning: AGENTS.md is %d bytes, above agents_md_warn_bytes=%d; including it in full\n", len(agentsMD), cfg.AgentsMDWarnBytes)
	}
	// Skills discovery: scan project and user-level .agents/skills/ directories
	// for SKILL.md files, build a catalog for the system prompt, and surface
	// any warnings to stderr. Skills are disclosed via file-read activation so
	// the model uses its existing read_file tool to load them on demand.
	var skillWarnings skills.Warnings
	skillDirs := []skills.Dir{
		{Path: filepath.Join(wd, ".agents", "skills"), Scope: skills.ScopeProject},
		{Path: filepath.Join(homeDir(getenv), ".agents", "skills"), Scope: skills.ScopeUser},
	}
	discoveredSkills := skills.Discover(skillDirs, &skillWarnings)
	for _, w := range skillWarnings {
		fmt.Fprintf(stderr, "skills: %s\n", w)
	}
	catalog := skills.BuildCatalog(discoveredSkills)
	instructions := skills.Instructions(len(discoveredSkills))

	// buildSystem assembles the full system prompt for a given run-mode prompt,
	// reusing every other input. The skills instructions block is appended last,
	// exactly as at startup, so a /mode switch reproduces the same composition.
	buildSystem := func(modePrompt string) string {
		s := sysprompt.Build(sysprompt.Options{
			Append:        appendText,
			Override:      overrideText,
			NoEnv:         cfg.NoEnv,
			AgentsMD:      agentsMD,
			SkillsCatalog: catalog,
			ModePrompt:    modePrompt,
			Env:           sysprompt.EnvOptions{Dir: wd},
		})
		if instructions != "" {
			s += "\n\n" + instructions
		}
		return s
	}

	// Run modes (design: tool-gating layer). The tool catalog holds every
	// constructible tool; each mode selects a subset, realized by Subset so the
	// agent advertises and dispatches only the mode's tools. Built once and
	// shared with the /mode switch (write_tmp_file holds a per-run temp dir).
	toolCatalog, disabledTools := tools.CatalogWithOptions(tools.Options{
		MaxResultBytes:       cfg.ToolResultMaxBytes,
		MaxResultLines:       cfg.ToolResultMaxLines,
		ReadFileDefaultLimit: cfg.ReadFileDefaultLimit,
	})
	for _, disabled := range disabledTools {
		logger.Warn(disabled.Message(), logging.Category("cli_tools"))
	}
	fileModes := make(map[string]mode.FileMode, len(cfg.Modes))
	for name, fm := range cfg.Modes {
		fileModes[name] = mode.FileMode{AllowedTools: fm.AllowedTools, Prompt: fm.Prompt}
	}
	modes := mode.Resolve(fileModes)
	// Expand @file references in mode prompts once at startup: a bad reference
	// fails fast (rather than on a later /mode switch), and the cached text means
	// switching never touches the filesystem.
	for name, m := range modes {
		expanded, err := resolveAtFile(m.Prompt)
		if err != nil {
			fmt.Fprintf(stderr, "harness: mode %q prompt: %v\n", name, err)
			return ui.ExitUsage
		}
		m.Prompt = expanded
		modes[name] = m
	}

	// Load a resumed session up front: its saved mode selects the tool set when
	// no -mode flag overrides it (flags win, as with provider/model below).
	var resumed *session.Session
	if cfg.Resume != "" {
		s, err := session.Load(cfg.Resume)
		if err != nil {
			fmt.Fprintf(stderr, "harness: resume %s: %v\n", cfg.Resume, err)
			return ui.ExitRuntime
		}
		resumed = &s
	}

	modeName := cfg.Mode
	if resumed != nil && resumed.Mode != "" {
		if cfg.Mode == "" {
			modeName = resumed.Mode
		} else if cfg.Mode != resumed.Mode {
			fmt.Fprintf(stderr, "harness: session mode %q overridden by %q (flags win)\n", resumed.Mode, cfg.Mode)
		}
	}
	if modeName == "" {
		modeName = mode.Default
	}
	currentMode, ok := modes[modeName]
	if !ok {
		fmt.Fprintf(stderr, "harness: unknown mode %q (available: %s)\n", modeName, strings.Join(mode.Names(modes), ", "))
		return ui.ExitUsage
	}
	toolRegistry, err := toolCatalog.Subset(currentMode.AllowedTools)
	if err != nil {
		fmt.Fprintf(stderr, "harness: mode %q: %v\n", modeName, err)
		return ui.ExitUsage
	}
	systemPrompt := buildSystem(currentMode.Prompt)

	switchMode := func(name string) (ui.ModeSelection, error) {
		m, ok := modes[name]
		if !ok {
			return ui.ModeSelection{}, fmt.Errorf("unknown mode %q (available: %s)", name, strings.Join(mode.Names(modes), ", "))
		}
		reg, err := toolCatalog.Subset(m.AllowedTools)
		if err != nil {
			return ui.ModeSelection{}, err
		}
		return ui.ModeSelection{Name: m.Name, Tools: reg, System: buildSystem(m.Prompt)}, nil
	}

	newProvider := env.newProvider
	if newProvider == nil {
		newProvider = factory.New
	}
	provider, err := newProvider(factory.Options{
		Provider:      effectiveProvider,
		ProviderName:  cfg.Provider,
		Model:         cfg.Model,
		BaseURL:       effectiveBaseURL,
		APIKey:        effectiveAPIKey,
		ContextWindow: effectiveContextWindow,
		ReasoningMode: reasoningMode(cfg.Provider, effectiveProvider, effectiveBaseURL),
	})
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}

	switchModel := func(input string) (ui.ModelSelection, error) {
		model := strings.TrimSpace(input)
		if model == "" {
			return ui.ModelSelection{}, fmt.Errorf("model is required")
		}
		providerName, apiType, baseURL, apiKey, modelName, registryKey, err := resolveSwitchProvider(model, cfg, providerConfigs, getenv)
		if err != nil {
			return ui.ModelSelection{}, err
		}
		model = modelName
		if discoveredAPIType, ok := enrichRegistryFromModelsDev(modelRegistry, model, providerName, apiType, baseURL, env.modelsDevCatalog, !reasoning.Empty(), cfg.Verbose, stderr); ok {
			if useModelsDevAPIType(apiType, discoveredAPIType, baseURL) {
				apiType = discoveredAPIType
			}
		}
		registryKey = registryModelKey(modelRegistry, providerName, model)
		if err := validateReasoningEffort(modelRegistry, registryKey, reasoning); err != nil {
			return ui.ModelSelection{}, err
		}
		contextWindow := cfg.ContextWindow
		if contextWindow <= 0 {
			contextWindow = modelRegistry.ContextWindow(registryKey)
		}
		runtime, err := newProvider(factory.Options{
			Provider:      apiType,
			ProviderName:  providerName,
			Model:         model,
			BaseURL:       baseURL,
			APIKey:        apiKey,
			ContextWindow: contextWindow,
			ReasoningMode: reasoningMode(providerName, apiType, baseURL),
		})
		if err != nil {
			return ui.ModelSelection{}, err
		}
		return ui.ModelSelection{
			Provider:      providerName,
			Model:         model,
			RegistryModel: registryKey,
			BaseURL:       baseURL,
			Runtime:       runtime,
			ContextWindow: cfg.ContextWindow,
		}, nil
	}

	ag := agent.New(provider, toolRegistry, agent.Options{
		MaxSteps:                  cfg.MaxSteps,
		Model:                     cfg.Model,
		ContextWindow:             cfg.ContextWindow,
		Registry:                  modelRegistry,
		Reasoning:                 reasoning,
		AutoContinue:              cfg.OnMaxSteps == "continue",
		CompactKeepTurns:          cfg.CompactKeepTurns,
		CompactSummaryMaxTokens:   cfg.CompactSummaryMaxTokens,
		CompactToolResultMaxBytes: cfg.CompactToolResultMaxBytes,
	})

	created := now()
	var totals session.UsageTotals

	// Resume restores a prior transcript; flags win over the file's
	// provider/model with a warning (design §11). The mode was resolved above;
	// the tool registry already reflects it.
	if resumed != nil {
		s := *resumed
		if s.Provider != "" && s.Provider != cfg.Provider {
			fmt.Fprintf(stderr, "harness: session provider %q overridden by %q (flags win)\n", s.Provider, cfg.Provider)
		}
		if s.Model != "" && s.Model != cfg.Model {
			fmt.Fprintf(stderr, "harness: session model %q overridden by %q (flags win)\n", s.Model, cfg.Model)
		}
		ag.SetTranscript(s.Messages)
		if !s.Created.IsZero() {
			created = s.Created
		}
		totals = s.Usage
		// A resumed session keeps its system prompt unless overridden by flags.
		if cfg.System == "" && cfg.SystemOverride == "" && s.System != "" {
			systemPrompt = s.System
		}
	}
	ag.SetSystem(systemPrompt)

	sessionPath := cfg.Session
	if sessionPath == "" {
		if cfg.Resume != "" {
			sessionPath = cfg.Resume
		} else {
			sessionPath = session.DefaultPath(stateDir(getenv), created)
		}
	}

	color := !cfg.NoColor && env.colorTTY
	renderer := ui.NewRenderer(stdout, stderr, ui.RenderOptions{
		Color:      color,
		Verbose:    cfg.Verbose,
		ToolStream: cfg.ToolStream,
		Model:      registryModel,
		Registry:   modelRegistry,
		Now:        now,
	})

	app := &ui.App{
		Agent:           ag,
		Renderer:        renderer,
		Out:             stdout,
		Errw:            stderr,
		Provider:        cfg.Provider,
		Model:           cfg.Model,
		RegistryModel:   registryModel,
		BaseURL:         effectiveBaseURL,
		Registry:        modelRegistry,
		System:          systemPrompt,
		AvailableModels: modelRegistry.Models(),
		SwitchModel:     switchModel,
		PickModel:       configuredModelPicker(providerConfigs),
		PickerPageSize:  setupPageSize(env),
		Mode:            modeName,
		AvailableModes:  mode.Names(modes),
		SwitchMode:      switchMode,
		SessionPath:     sessionPath,
		StateDir:        stateDir(getenv),
		Created:         created,
		Now:             now,
		Prompt:          cfg.ReplPrompt,
		Skills:          discoveredSkills,
		SkillDirs:       skillDirs,
	}
	if resumed != nil {
		app.Turn = resumed.Turn
	}
	ag.SetCompactionArchiver(func(ctx context.Context, archive agent.CompactionArchive) (string, error) {
		return session.SaveCompaction(app.SessionPath, session.Compaction{
			Time:     now(),
			Summary:  archive.Summary,
			Usage:    archive.Usage,
			Messages: archive.Messages,
		})
	})
	app.SetUsage(totals)

	// SIGINT wiring (design §8.4): a single handler cancels the active turn or,
	// on a second press / at the idle prompt, requests exit.
	exitCh := make(chan struct{}, 1)
	if env.sigCh != nil {
		watcher := agent.NewInterruptWatcher(env.sigCh, now, func() {
			select {
			case exitCh <- struct{}{}:
			default:
			}
		})
		stop := watcher.Start()
		defer stop()
		app.Interrupt = watcher
	}

	// One-shot mode: a single turn, then exit (design §10).
	if cfg.PromptSet {
		prompt, err := ui.BuildPrompt(cfg.Prompt, env.stdin, env.stdinPiped)
		if err != nil {
			fmt.Fprintf(stderr, "harness: read prompt: %v\n", err)
			return ui.ExitRuntime
		}
		fmt.Fprintf(stderr, "session: %s\n", sessionPath)
		code := ui.OneShot(app, prompt)
		select {
		case <-exitCh:
			return ui.ExitInterrupt
		default:
		}
		return code
	}

	// Interactive REPL. ui.Run owns the session save in every exit path,
	// including SIGINT, so the exit-save never races an in-flight turn's own save
	// or usage update (design §8.4); main only forwards the exit request.
	fmt.Fprintf(stderr, "session: %s\n", sessionPath)
	return ui.Run(env.stdin, app, exitCh)
}

func configDir(path string) string {
	if path == "" {
		return "."
	}
	return filepath.Dir(path)
}

func runSessionCommand(args []string, stdout, stderr io.Writer) int {
	if len(args) < 1 {
		fmt.Fprintln(stderr, "usage: harness session replay <session-dir>")
		return ui.ExitUsage
	}
	switch args[0] {
	case "replay":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: harness session replay <session-dir>")
			return ui.ExitUsage
		}
		if err := session.Replay(args[1], stdout, session.ReplayOptions{}); err != nil {
			fmt.Fprintf(stderr, "harness: session replay: %v\n", err)
			return ui.ExitRuntime
		}
		return ui.ExitOK
	default:
		fmt.Fprintf(stderr, "harness: unknown session command %q\n", args[0])
		fmt.Fprintln(stderr, "usage: harness session replay <session-dir>")
		return ui.ExitUsage
	}
}

func defaultModelsDevCatalog(ctx context.Context) (*modelsdev.Catalog, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return modelsdev.Fetch(ctx, http.DefaultClient, modelsdev.DefaultURL)
}

func defaultTerminalRows() int {
	rows, _, ok := term.Size()
	if !ok {
		return 0
	}
	return rows
}

func enrichRegistryFromModelsDev(registry *llm.Registry, model, providerID, apiType, baseURL string, fetch func(context.Context) (*modelsdev.Catalog, error), needReasoning bool, verbose bool, errw io.Writer) (string, bool) {
	if registry == nil || fetch == nil || model == "" {
		return "", false
	}
	if isLocalBaseURL(baseURL) {
		return "", false
	}
	if info, ok := registry.Lookup(model); ok && info.ContextWindow > 0 && registry.HasPrice(model) && (!needReasoning || info.Reasoning != nil) {
		return "", false
	}

	catalog, err := fetch(context.Background())
	if err != nil {
		if verbose {
			fmt.Fprintf(errw, "harness: warning: models.dev lookup skipped: %v\n", err)
		}
		return "", false
	}
	provider, ok := catalog.Provider(providerID)
	if !ok {
		provider, ok = catalog.ProviderByAPI(baseURL)
	}
	if !ok {
		provider, ok = catalog.Provider(apiType)
	}
	if !ok && apiType == "responses" {
		provider, ok = catalog.Provider("openai")
	}
	if !ok {
		return "", false
	}
	info, ok := provider.ModelInfo(model)
	if !ok && apiType != "" && provider.ID != apiType {
		if fallback, found := catalog.Provider(apiType); found {
			info, ok = fallback.ModelInfo(model)
		}
	}
	if ok {
		registry.MergeModel(model, info)
		if discoveredAPIType := provider.APIType(); discoveredAPIType != "" {
			return discoveredAPIType, true
		}
	}
	return "", false
}

func useModelsDevAPIType(current, discovered, baseURL string) bool {
	if discovered == "" || discovered == current {
		return false
	}
	if discovered == "responses" {
		return isDefaultOpenAIBaseURL(baseURL)
	}
	return true
}

func validateReasoningEffort(registry *llm.Registry, model string, reasoning llm.ReasoningConfig) error {
	if reasoning.Empty() || registry == nil {
		return nil
	}
	info, ok := registry.Lookup(model)
	if !ok || info.Reasoning == nil {
		return nil
	}
	if info.Reasoning.SupportsEffort(reasoning.Effort) {
		return nil
	}
	if !info.Reasoning.Supported {
		return fmt.Errorf("model %q does not support reasoning effort", model)
	}
	if values, ok := info.Reasoning.EffortValues(); ok && len(values) > 0 {
		return fmt.Errorf("model %q does not support reasoning effort %q (supported: %s)", model, reasoning.Effort, strings.Join(values, ", "))
	}
	return fmt.Errorf("model %q does not support reasoning effort", model)
}

func reasoningMode(providerName, apiType, baseURL string) string {
	if apiType == "anthropic" {
		return "anthropic"
	}
	if strings.EqualFold(providerName, "openrouter") || strings.Contains(strings.ToLower(baseURL), "openrouter.ai") {
		return "openrouter"
	}
	return "openai"
}

func isLocalBaseURL(raw string) bool {
	if raw == "" {
		return false
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

func isDefaultOpenAIBaseURL(raw string) bool {
	raw = strings.TrimRight(strings.TrimSpace(raw), "/")
	return raw == "" || raw == "https://api.openai.com/v1"
}

func resolveProvider(cfg config.Config, providers []llm.ProviderConfig, getenv func(string) string) (provider, baseURL, apiKey string) {
	provider = cfg.Provider
	baseURL = cfg.BaseURL
	apiKey = cfg.APIKey
	pc, ok := providerConfigByName(providers, cfg.Provider)
	if !ok {
		return provider, baseURL, apiKey
	}
	if pc.APIType != "" {
		provider = pc.APIType
	}
	if baseURL == "" {
		baseURL = pc.BaseURL
	}
	for _, name := range pc.APIKeyEnv {
		if value := getenv(name); value != "" {
			apiKey = value
			break
		}
	}
	if apiKey == "" {
		apiKey = pc.APIKey
	}
	return provider, baseURL, apiKey
}

func resolveSwitchProvider(input string, cfg config.Config, providers []llm.ProviderConfig, getenv func(string) string) (provider, apiType, baseURL, apiKey, model, registryKey string, err error) {
	model = input
	if requestedProvider, requestedModel, ok := config.SplitProviderModel(input); ok {
		model = requestedModel
		pc, found := providerConfigByName(providers, requestedProvider)
		if !found {
			return "", "", "", "", "", "", fmt.Errorf("provider %q is not configured", requestedProvider)
		}
		if !providerConfigHasModel(pc, model) {
			return "", "", "", "", "", "", fmt.Errorf("provider %q has no configured model %q", requestedProvider, model)
		}
		provider, apiType, baseURL, apiKey = runtimeProviderConfig(pc, cfg, getenv)
		return provider, apiType, baseURL, apiKey, model, providerModelKey(provider, model), nil
	}
	if pc, ok := providerConfigForModel(providers, model, cfg.Provider); ok {
		provider, apiType, baseURL, apiKey = runtimeProviderConfig(pc, cfg, getenv)
		return provider, apiType, baseURL, apiKey, model, providerModelKey(provider, model), nil
	}
	if pc, ok := providerConfigByName(providers, cfg.Provider); ok {
		provider, apiType, baseURL, apiKey = runtimeProviderConfig(pc, cfg, getenv)
		return provider, apiType, baseURL, apiKey, model, providerModelKey(provider, model), nil
	}

	apiType = inferAPIType(model)
	provider = apiType
	baseURL = providerBaseURLEnv(apiType, getenv)
	apiKey = providerAPIKeyEnv(apiType, getenv)
	if cfg.Provider == provider {
		if cfg.BaseURL != "" {
			baseURL = cfg.BaseURL
		}
		if cfg.APIKey != "" {
			apiKey = cfg.APIKey
		}
	}
	return provider, apiType, baseURL, apiKey, model, providerModelKey(provider, model), nil
}

func configuredModelPicker(providers []llm.ProviderConfig) func(ui.PickerIO) (string, error) {
	providerEntries := configuredProviderPickerEntries(providers)
	if len(providerEntries) == 0 {
		return nil
	}
	return func(pio ui.PickerIO) (string, error) {
		w := pio.Writer
		if w == nil {
			w = io.Discard
		}
		provider, err := ui.Pick(pio.ReadLine, w, ui.PickerOptions[configuredProviderPick]{
			Items:       providerEntries,
			PageSize:    pio.PageSize,
			Prompt:      "Provider (number/id, /search, n/p, q): ",
			Kind:        "provider",
			CancelError: ui.ErrPickerCancelled,
			PrintPage:   ui.PrintProviderPickerPage[configuredProviderPick],
		})
		if err != nil {
			return "", err
		}
		models := configuredModelPickerEntries(provider.config.Models)
		model, err := ui.Pick(pio.ReadLine, w, ui.PickerOptions[configuredModelPick]{
			Items:       models,
			PageSize:    pio.PageSize,
			Prompt:      "Model (number/id, /search, n/p, q): ",
			Kind:        "model",
			CancelError: ui.ErrPickerCancelled,
			PrintPage: func(w io.Writer, models []configuredModelPick, page, pageSize int, filter string) {
				ui.PrintModelPickerPage(w, provider.config.Name, models, page, pageSize, filter)
			},
		})
		if err != nil {
			return "", err
		}
		return provider.config.Name + ":" + model.entry.Name, nil
	}
}

type configuredProviderPick struct {
	config llm.ProviderConfig
}

func configuredProviderPickerEntries(providers []llm.ProviderConfig) []configuredProviderPick {
	seen := make(map[string]bool, len(providers))
	entries := make([]configuredProviderPick, 0, len(providers))
	for _, provider := range providers {
		if provider.Name == "" || len(provider.Models) == 0 || seen[provider.Name] {
			continue
		}
		seen[provider.Name] = true
		entries = append(entries, configuredProviderPick{config: provider})
	}
	return entries
}

func (p configuredProviderPick) PickerID() string { return p.config.Name }

func (p configuredProviderPick) PickerName() string {
	if p.config.BaseURL != "" {
		return p.config.BaseURL
	}
	if p.config.APIType != "" && p.config.APIType != p.config.Name {
		return p.config.APIType
	}
	return p.config.Name
}

func (p configuredProviderPick) PickerModelCount() int {
	return len(p.config.Models)
}

type configuredModelPick struct {
	entry llm.ModelEntry
}

func configuredModelPickerEntries(models []llm.ModelEntry) []configuredModelPick {
	entries := make([]configuredModelPick, 0, len(models))
	for _, model := range models {
		if model.Name == "" {
			continue
		}
		entries = append(entries, configuredModelPick{entry: model})
	}
	return entries
}

func (m configuredModelPick) PickerID() string      { return m.entry.Name }
func (m configuredModelPick) PickerName() string    { return m.entry.Name }
func (m configuredModelPick) PickerPrice() string   { return formatPickerPrice(m.entry.Price) }
func (m configuredModelPick) PickerRelease() string { return "" }

// formatPickerPrice formats an llm.Price as "$in/$out" per 1M tokens,
// or "" when no price is configured.
func formatPickerPrice(p llm.Price) string {
	if p.Input == 0 && p.Output == 0 && p.CacheRead == 0 && p.CacheWrite == 0 {
		return ""
	}
	return fmt.Sprintf("$%s/$%s", formatPriceComponent(p.Input), formatPriceComponent(p.Output))
}

func formatPriceComponent(v float64) string {
	if v == float64(int64(v)) {
		return fmt.Sprintf("%.0f", v)
	}
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.2f", v), "0"), ".")
}

func providerModelKey(provider, model string) string {
	if provider == "" || model == "" {
		return model
	}
	return provider + ":" + model
}

func registryModelKey(registry *llm.Registry, provider, model string) string {
	key := providerModelKey(provider, model)
	if registry != nil && key != model {
		if _, ok := registry.Lookup(key); ok {
			return key
		}
	}
	return model
}

func runtimeProviderConfig(pc llm.ProviderConfig, cfg config.Config, getenv func(string) string) (provider, apiType, baseURL, apiKey string) {
	provider = pc.Name
	apiType = pc.APIType
	if apiType == "" {
		apiType = pc.Name
	}
	baseURL = pc.BaseURL
	if pc.Name == cfg.Provider && cfg.BaseURL != "" {
		baseURL = cfg.BaseURL
	}
	if pc.Name == cfg.Provider {
		apiKey = cfg.APIKey
	}
	for _, name := range pc.APIKeyEnv {
		if value := getenv(name); value != "" {
			apiKey = value
			break
		}
	}
	if apiKey == "" {
		apiKey = providerAPIKeyEnv(apiType, getenv)
	}
	if apiKey == "" {
		apiKey = pc.APIKey
	}
	return provider, apiType, baseURL, apiKey
}

func providerConfigForModel(providers []llm.ProviderConfig, model, preferred string) (llm.ProviderConfig, bool) {
	for _, pc := range providers {
		if pc.Name == preferred && providerConfigHasModel(pc, model) {
			return pc, true
		}
	}
	for _, pc := range providers {
		if providerConfigHasModel(pc, model) {
			return pc, true
		}
	}
	return llm.ProviderConfig{}, false
}

func providerConfigHasModel(pc llm.ProviderConfig, model string) bool {
	for _, entry := range pc.Models {
		if entry.Name == model {
			return true
		}
	}
	return false
}

func inferAPIType(model string) string {
	if strings.HasPrefix(model, "claude") {
		return "anthropic"
	}
	return "openai"
}

func providerBaseURLEnv(provider string, getenv func(string) string) string {
	switch provider {
	case "anthropic":
		return getenv("ANTHROPIC_BASE_URL")
	case "responses":
		if v := getenv("RESPONSES_BASE_URL"); v != "" {
			return v
		}
		return getenv("OPENAI_BASE_URL")
	default:
		return getenv("OPENAI_BASE_URL")
	}
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

func providerConfigByName(providers []llm.ProviderConfig, name string) (llm.ProviderConfig, bool) {
	for _, pc := range providers {
		if pc.Name == name {
			return pc, true
		}
	}
	return llm.ProviderConfig{}, false
}

// resolveConfigPath determines the config-file path config.Load should read: an
// explicit -config flag, or the implicit ~/.config/harness/config.json only when
// it exists (so an absent default is silently skipped, but a typo'd -config
// surfaces as an error in Load).
func resolveConfigPath(args []string, getenv func(string) string) string {
	if p := flagValue(args, "config"); p != "" {
		return p
	}
	def := filepath.Join(defaultConfigDir(getenv), "config.json")
	if _, err := os.Stat(def); err == nil {
		return def
	}
	return ""
}

func defaultConfigDir(getenv func(string) string) string {
	if home := getenv("HOME"); home != "" {
		return filepath.Join(home, ".config", "harness")
	}
	return filepath.Join(os.TempDir(), "harness-config")
}

// homeDir returns the user's home directory, or empty string if unavailable.
func homeDir(getenv func(string) string) string {
	return getenv("HOME")
}

// flagValue extracts a string flag's value from raw args, supporting both
// -flag=value and -flag value forms. It returns "" when absent.
func flagValue(args []string, name string) string {
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			break
		}
		for _, prefix := range []string{"-" + name, "--" + name} {
			if a == prefix {
				if i+1 < len(args) {
					return args[i+1]
				}
				return ""
			}
			if strings.HasPrefix(a, prefix+"=") {
				return a[len(prefix)+1:]
			}
		}
	}
	return ""
}

// resolveAtFile expands a @file reference to the file's contents; a plain string
// is returned unchanged. A literal leading @ can be escaped as @@.
func resolveAtFile(v string) (string, error) {
	if strings.HasPrefix(v, "@@") {
		return v[1:], nil
	}
	if strings.HasPrefix(v, "@") {
		data, err := os.ReadFile(v[1:])
		if err != nil {
			return "", err
		}
		return string(data), nil
	}
	return v, nil
}

// loadAgentsMD reads AGENTS.md from dir when present. A missing file returns
// an empty string with no error; other read failures (e.g. permissions) are
// returned so the user isn't silently surprised.
func loadAgentsMD(dir string) (string, error) {
	if dir == "" {
		return "", nil
	}
	path := filepath.Join(dir, "AGENTS.md")
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", fmt.Errorf("reading %s: %w", path, err)
	}
	return string(data), nil
}

// stateDir returns the base directory for auto-saved sessions: $XDG_STATE_HOME
// or ~/.local/state (design §11).
func stateDir(getenv func(string) string) string {
	if x := getenv("XDG_STATE_HOME"); x != "" {
		return x
	}
	if home := getenv("HOME"); home != "" {
		return filepath.Join(home, ".local", "state")
	}
	return filepath.Join(os.TempDir(), "harness-state")
}

// isTTY reports whether f is a terminal, gating dim color (design §2, §10).
func isTTY(f *os.File) bool {
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

// pipedStdin reports whether stdin is piped/redirected (not a terminal), so
// one-shot mode knows to read it (design §10).
func pipedStdin(f *os.File) bool {
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice == 0
}

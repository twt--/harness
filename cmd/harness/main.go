// Command harness is the entrypoint: it loads configuration, connects to
// harness-model-proxy, constructs the tool registry and agent, wires SIGINT
// handling, prints the session path, and dispatches to the interactive REPL or
// one-shot mode.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"harness/internal/agent"
	"harness/internal/config"
	"harness/internal/delegate"
	"harness/internal/llm"
	"harness/internal/logging"
	"harness/internal/mcptools"
	"harness/internal/mode"
	modelclient "harness/internal/modelproxy/client"
	"harness/internal/modelproxy/protocol"
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
		args:         os.Args[1:],
		stdin:        os.Stdin,
		stdout:       os.Stdout,
		stderr:       os.Stderr,
		getenv:       os.Getenv,
		now:          time.Now,
		colorTTY:     isTTY(os.Stdout),
		stdinPiped:   pipedStdin(os.Stdin),
		sigCh:        sigCh,
		terminalRows: defaultTerminalRows,
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

	terminalRows func() int
}

// run wires everything together and returns the process exit code (design §10
// exit codes: 0 ok, 1 runtime, 2 usage, 130 interrupted).
func run(env environment) int {
	args := env.args
	stdin := env.stdin
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
	logger, err := logging.NewLogger(stderr, cfg.LogLevel, cfg.Quiet)
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}

	proxyURL := cfg.ModelProxyURL
	if proxyURL == "" {
		proxyURL = protocol.DefaultURL
	}
	proxyClient, err := modelclient.New(proxyURL, nil)
	if err != nil {
		fmt.Fprintf(stderr, "harness: %v\n", err)
		return ui.ExitUsage
	}
	catalog, err := proxyClient.Catalog(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "harness: model proxy: %v\n", err)
		return ui.ExitRuntime
	}
	modelRegistry := modelclient.Registry(catalog)
	modelRegistry.SetDefaultContextWindow(cfg.DefaultContextWindow)
	reasoning := llm.ReasoningConfig{Effort: cfg.ReasoningEffort}
	selection, err := resolveCatalogSelection(catalog, cfg.Provider, cfg.Model, "")
	if err != nil {
		if cfg.Model != "" {
			fmt.Fprintf(stderr, "harness: %v\n", err)
			return ui.ExitUsage
		}
		reader := bufio.NewReader(stdin)
		stdin = reader
		selection, err = pickStartupModel(reader, stderr, catalog, pickerPageSize(env))
		if err != nil {
			if errors.Is(err, ui.ErrPickerCancelled) {
				fmt.Fprintln(stderr, "harness: model selection cancelled")
			} else {
				fmt.Fprintf(stderr, "harness: model selection: %v\n", err)
			}
			return ui.ExitUsage
		}
		cfg.Provider = selection.Provider
		cfg.Model = selection.Model
		if err := validateReasoningEffort(modelRegistry, selection.RegistryModel, reasoning); err != nil {
			fmt.Fprintf(stderr, "harness: %v\n", err)
			return ui.ExitUsage
		}
		configPath := writableConfigPath(args, getenv)
		if err := config.SaveSelectedModel(configPath, selection.Provider, selection.Model); err != nil {
			fmt.Fprintf(stderr, "harness: save selected model: %v\n", err)
			return ui.ExitRuntime
		}
		fmt.Fprintf(stderr, "harness: saved selected model to %s\n", configPath)
	}
	cfg.Provider = selection.Provider
	cfg.Model = selection.Model
	registryModel := selection.RegistryModel
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
	skillsCatalog := skills.BuildCatalog(discoveredSkills)
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
			SkillsCatalog: skillsCatalog,
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
	delegateTools, err := toolCatalog.Subset(delegateReadOnlyToolNames())
	if err != nil {
		fmt.Fprintf(stderr, "harness: delegate tools: %v\n", err)
		return ui.ExitUsage
	}
	delegateState := delegate.NewState(delegate.Runtime{
		Model:         cfg.Model,
		ContextWindow: cfg.ContextWindow,
		Registry:      modelRegistry,
		Reasoning:     reasoning,
	})
	toolCatalog.Register(delegate.New(delegateState.Snapshot, delegateTools, delegate.Options{
		MaxTurns:                  cfg.DelegateMaxTurns,
		CompactKeepTurns:          cfg.CompactKeepTurns,
		CompactSummaryMaxTokens:   cfg.CompactSummaryMaxTokens,
		CompactToolResultMaxBytes: cfg.CompactToolResultMaxBytes,
	}))
	// MCP (opt-in): connect to the proxy and register discovered tools into the
	// catalog before mode resolution, so any mode's subset can pick them up. It
	// never fails startup; on any error it warns and continues with no MCP tools.
	var mcpConn *mcptools.Conn
	var mcpSummary mcptools.Summary
	if cfg.MCP.Enable {
		conn, summary, cleanup, ok := setupMCP(context.Background(), cfg.MCP, toolCatalog, logger)
		defer cleanup()
		if ok {
			mcpConn, mcpSummary = conn, summary
		}
	}
	fileModes := make(map[string]mode.FileMode, len(cfg.Modes))
	for name, fm := range cfg.Modes {
		fileModes[name] = mode.FileMode{AllowedTools: fm.AllowedTools, Prompt: fm.Prompt}
	}
	modes := mode.Resolve(fileModes)
	// Default-inheriting modes (auto, independent, and config modes without an
	// explicit allowed_tools) expose the discovered MCP tools; explicit
	// whitelists opt out. Capture which modes inherit the default set BEFORE
	// augmenting them, so the refresh hook can re-derive their allowed lists.
	// mcpSummary.Names is empty when MCP is disabled, making both a no-op.
	mcpBases := defaultInheritingModeBases(modes)
	augmentModesWithMCP(modes, mcpSummary.Names)
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
		system := buildSystem(m.Prompt)
		snap := delegateState.Snapshot()
		snap.System = system
		delegateState.Set(snap)
		return ui.ModeSelection{Name: m.Name, Tools: reg, System: system}, nil
	}

	provider := proxyClient.Provider(cfg.Provider)

	switchModel := func(input string) (ui.ModelSelection, error) {
		input = strings.TrimSpace(input)
		if input == "" {
			return ui.ModelSelection{}, fmt.Errorf("model is required")
		}
		next, err := resolveCatalogSelection(catalog, "", input, cfg.Provider)
		if err != nil {
			return ui.ModelSelection{}, err
		}
		if err := validateReasoningEffort(modelRegistry, next.RegistryModel, reasoning); err != nil {
			return ui.ModelSelection{}, err
		}
		runtime := proxyClient.Provider(next.Provider)
		snap := delegateState.Snapshot()
		snap.Provider = runtime
		snap.Model = next.Model
		snap.ContextWindow = cfg.ContextWindow
		delegateState.Set(snap)
		return ui.ModelSelection{
			Provider:      next.Provider,
			Model:         next.Model,
			RegistryModel: next.RegistryModel,
			BaseURL:       proxyClient.URL(),
			Runtime:       runtime,
			ContextWindow: cfg.ContextWindow,
		}, nil
	}

	ag := agent.New(provider, toolRegistry, agent.Options{
		MaxTurns:                  cfg.MaxTurns,
		Model:                     cfg.Model,
		ContextWindow:             cfg.ContextWindow,
		Registry:                  modelRegistry,
		Reasoning:                 reasoning,
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
	delegateState.Set(delegate.Runtime{
		Provider:      provider,
		Model:         cfg.Model,
		ContextWindow: cfg.ContextWindow,
		Registry:      modelRegistry,
		Reasoning:     reasoning,
		System:        systemPrompt,
	})

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
		BaseURL:         proxyClient.URL(),
		Registry:        modelRegistry,
		System:          systemPrompt,
		AvailableModels: modelRegistry.Models(),
		SwitchModel:     switchModel,
		PickModel:       catalogModelPicker(catalog),
		PickerPageSize:  pickerPageSize(env),
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
		DisabledTools:   disabledTools,
	}
	if resumed != nil {
		app.Turn = resumed.Turn
	}
	// Wire the MCP tool-list refresh hook for the interactive REPL only: one-shot
	// runs a single turn with the tools fixed at startup, so it needs no hook.
	if mcpConn != nil && !cfg.PromptSet {
		app.RefreshMCP = newMCPRefresher(mcpConn, toolCatalog, modes, mcpBases, mcpSummary, logger)
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
		prompt, err := ui.BuildPrompt(cfg.Prompt, stdin, env.stdinPiped)
		if err != nil {
			fmt.Fprintf(stderr, "harness: read prompt: %v\n", err)
			return ui.ExitRuntime
		}
		fmt.Fprintf(stderr, "session: %s\n", sessionPath)
		fmt.Fprintf(stderr, "provider: %s  model: %s\n", cfg.Provider, cfg.Model)
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
	fmt.Fprintf(stderr, "provider: %s  model: %s\n", cfg.Provider, cfg.Model)
	return ui.Run(stdin, app, exitCh)
}

func delegateReadOnlyToolNames() []string {
	names := []string{"read_file", "list_dir", "grep"}
	if tools.RipgrepAvailable() {
		names = append(names, "rg")
	}
	names = append(names, "web_fetch")
	if tools.GitAvailable() {
		names = append(names, "git_readonly")
	}
	return names
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

func defaultTerminalRows() int {
	rows, _, ok := term.Size()
	if !ok {
		return 0
	}
	return rows
}

type catalogSelection struct {
	Provider      string
	Model         string
	RegistryModel string
}

func resolveCatalogSelection(catalog protocol.Catalog, provider, model, preferredProvider string) (catalogSelection, error) {
	provider = strings.TrimSpace(provider)
	model = strings.TrimSpace(model)
	if p, m, ok := config.SplitProviderModel(model); ok {
		provider = p
		model = m
	}
	if provider != "" && model == "" {
		if p, ok := catalogProvider(catalog, provider); ok && len(p.Models) == 1 {
			model = p.Models[0].ID
		}
	}
	if provider != "" && model != "" {
		p, ok := catalogProvider(catalog, provider)
		if !ok {
			return catalogSelection{}, fmt.Errorf("provider %q is not available from the model proxy", provider)
		}
		if !catalogProviderHasModel(p, model) {
			return catalogSelection{}, fmt.Errorf("provider %q has no model %q", provider, model)
		}
		return catalogSelection{Provider: provider, Model: model, RegistryModel: providerModelKey(provider, model)}, nil
	}
	if provider == "" && model != "" {
		if preferredProvider != "" {
			if p, ok := catalogProvider(catalog, preferredProvider); ok && catalogProviderHasModel(p, model) {
				return catalogSelection{Provider: preferredProvider, Model: model, RegistryModel: providerModelKey(preferredProvider, model)}, nil
			}
		}
		matches := catalogProvidersForModel(catalog, model)
		switch len(matches) {
		case 0:
			return catalogSelection{}, fmt.Errorf("model %q is not available from the model proxy", model)
		case 1:
			return catalogSelection{Provider: matches[0], Model: model, RegistryModel: providerModelKey(matches[0], model)}, nil
		default:
			return catalogSelection{}, fmt.Errorf("model %q is available from multiple providers (%s); use provider:%s", model, strings.Join(matches, ", "), model)
		}
	}
	return catalogSelection{}, fmt.Errorf("a model is required (-model or harness config model)")
}

func catalogProvider(catalog protocol.Catalog, id string) (protocol.Provider, bool) {
	for _, provider := range catalog.Providers {
		if provider.ID == id {
			return provider, true
		}
	}
	return protocol.Provider{}, false
}

func catalogProviderHasModel(provider protocol.Provider, model string) bool {
	for _, entry := range provider.Models {
		if entry.ID == model {
			return true
		}
	}
	return false
}

func catalogProvidersForModel(catalog protocol.Catalog, model string) []string {
	var providers []string
	for _, provider := range catalog.Providers {
		if catalogProviderHasModel(provider, model) {
			providers = append(providers, provider.ID)
		}
	}
	return providers
}

func catalogModelPicker(catalog protocol.Catalog) func(ui.PickerIO) (string, error) {
	providerEntries := catalogProviderPickerEntries(catalog)
	if len(providerEntries) == 0 {
		return nil
	}
	return func(pio ui.PickerIO) (string, error) {
		w := pio.Writer
		if w == nil {
			w = io.Discard
		}
		provider, err := ui.Pick(pio.ReadLine, w, ui.PickerOptions[catalogProviderPick]{
			Items:       providerEntries,
			PageSize:    pio.PageSize,
			Prompt:      "Provider (number/id, /search, n/p, q): ",
			Kind:        "provider",
			CancelError: ui.ErrPickerCancelled,
			PrintPage:   ui.PrintProviderPickerPage[catalogProviderPick],
		})
		if err != nil {
			return "", err
		}
		models := catalogModelPickerEntries(provider.provider.Models)
		model, err := ui.Pick(pio.ReadLine, w, ui.PickerOptions[catalogModelPick]{
			Items:       models,
			PageSize:    pio.PageSize,
			Prompt:      "Model (number/id, /search, n/p, q): ",
			Kind:        "model",
			CancelError: ui.ErrPickerCancelled,
			PrintPage: func(w io.Writer, models []catalogModelPick, page, pageSize int, filter string) {
				ui.PrintModelPickerPage(w, provider.provider.ID, models, page, pageSize, filter)
			},
		})
		if err != nil {
			return "", err
		}
		return provider.provider.ID + ":" + model.model.ID, nil
	}
}

func pickStartupModel(reader *bufio.Reader, w io.Writer, catalog protocol.Catalog, pageSize int) (catalogSelection, error) {
	picker := catalogModelPicker(catalog)
	if picker == nil {
		return catalogSelection{}, fmt.Errorf("model proxy catalog has no selectable models")
	}
	fmt.Fprintln(w, "Select a provider and model to use with harness.")
	input, err := picker(ui.PickerIO{
		ReadLine: func(prompt string) (string, error) {
			if _, err := fmt.Fprint(w, prompt); err != nil {
				return "", err
			}
			line, err := reader.ReadString('\n')
			if err != nil {
				if errors.Is(err, io.EOF) && line != "" {
					return strings.TrimSpace(line), nil
				}
				return "", err
			}
			return strings.TrimSpace(line), nil
		},
		Writer:   w,
		PageSize: pageSize,
	})
	if err != nil {
		return catalogSelection{}, err
	}
	return resolveCatalogSelection(catalog, "", input, "")
}

func writableConfigPath(args []string, getenv func(string) string) string {
	if p := flagValue(args, "config"); p != "" {
		return p
	}
	return filepath.Join(defaultConfigDir(getenv), "config.json")
}

type catalogProviderPick struct {
	provider protocol.Provider
}

func catalogProviderPickerEntries(catalog protocol.Catalog) []catalogProviderPick {
	seen := make(map[string]bool, len(catalog.Providers))
	entries := make([]catalogProviderPick, 0, len(catalog.Providers))
	for _, provider := range catalog.Providers {
		if provider.ID == "" || len(provider.Models) == 0 || seen[provider.ID] {
			continue
		}
		seen[provider.ID] = true
		entries = append(entries, catalogProviderPick{provider: provider})
	}
	return entries
}

func (p catalogProviderPick) PickerID() string { return p.provider.ID }

func (p catalogProviderPick) PickerName() string {
	if p.provider.Name != "" {
		return p.provider.Name
	}
	return p.provider.ID
}

func (p catalogProviderPick) PickerModelCount() int {
	return len(p.provider.Models)
}

type catalogModelPick struct {
	model protocol.Model
}

func catalogModelPickerEntries(models []protocol.Model) []catalogModelPick {
	entries := make([]catalogModelPick, 0, len(models))
	for _, model := range models {
		if model.ID == "" {
			continue
		}
		entries = append(entries, catalogModelPick{model: model})
	}
	return entries
}

func (m catalogModelPick) PickerID() string { return m.model.ID }
func (m catalogModelPick) PickerName() string {
	if m.model.Name != "" {
		return m.model.Name
	}
	return m.model.ID
}
func (m catalogModelPick) PickerPrice() string   { return formatPickerPrice(m.model.Price) }
func (m catalogModelPick) PickerRelease() string { return "" }

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

func pickerPageSize(env environment) int {
	rows := 0
	if env.terminalRows != nil {
		rows = env.terminalRows()
	}
	return ui.PickerPageSize(rows)
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

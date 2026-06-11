package ui

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/session"
	"harness/internal/skills"
	"harness/internal/term"
	"harness/internal/tools"
)

// ModelSelection is the runtime model/provider bundle returned by App.SwitchModel.
type ModelSelection struct {
	Provider      string
	Model         string
	RegistryModel string
	BaseURL       string
	Runtime       llm.Provider
	ContextWindow int // agent override; 0 means use the registry
}

// ModeSelection is the runtime run-mode bundle returned by App.SwitchMode: the
// new tool registry to advertise/dispatch from and the fully reassembled system
// prompt (with the mode's section) to send.
type ModeSelection struct {
	Name   string
	Tools  *tools.Registry
	System string
}

// App bundles the dependencies the REPL and one-shot driver need. main builds it
// from the resolved config, provider factory, tool registry, and renderer
// (design §10). The agent owns the running transcript; App tracks the cumulative
// session usage and the current save path (rotated by /clear).
type App struct {
	Agent    *agent.Agent
	Renderer *Renderer
	Out      io.Writer
	Errw     io.Writer

	Provider      string
	Model         string
	RegistryModel string
	BaseURL       string
	Registry      *llm.Registry
	System        string

	AvailableModels []string
	SwitchModel     func(model string) (ModelSelection, error)

	Mode           string   // current run mode name
	AvailableModes []string // sorted mode names for /mode listing
	SwitchMode     func(name string) (ModeSelection, error)

	SessionPath string    // current save path; /clear rotates it
	StateDir    string    // for rotating to a fresh auto-save path on /clear
	Created     time.Time // session creation time (preserved across saves)
	Now         func() time.Time

	// Interrupt is the optional SIGINT state machine. When set, the REPL marks
	// turn boundaries so ^C cancels a turn rather than the whole process
	// (design §8.4). Tests leave it nil.
	Interrupt *agent.InterruptWatcher

	// Prompt is the REPL input prompt string (default "> ").
	Prompt string

	// Skills is the discovered skills map for /skills listing and
	// $skillName invocation (design §10). nil disables both features.
	Skills map[string]skills.Skill

	// SkillDirs is the list of scanned skill directories with their scopes,
	// used by /skills to group output by source location.
	SkillDirs []skills.Dir

	usage session.UsageTotals // cumulative across the session
}

// helpText lists the meta-commands (design §10).
const helpText = `commands:
  /help            list commands
  /exit, /quit     save and exit
  /clear           reset conversation; rotate to a fresh session file
  /compact         force compaction now
  /usage           cumulative session tokens and cost
  /save [file]     force save (optionally elsewhere)
  /model [model]   list models, or switch to model
  /mode [name]     list run modes, or switch to mode
  /skills          list available skills
  $skillName       invoke a skill (reads SKILL.md and sends as prompt)
lines starting with / are commands; // sends a literal leading slash`

func (app *App) clock() func() time.Time {
	if app.Now != nil {
		return app.Now
	}
	return time.Now
}

// Run drives the interactive REPL: it reads lines from in, dispatches
// meta-commands, and runs one agent turn per prompt, saving the session after
// every turn (design §10, §11).
//
// exit carries SIGINT exit requests (design §8.4); a nil channel disables them.
// Run owns the final save in every exit path — /exit, EOF (^D), and SIGINT — so
// no second goroutine ever touches the transcript or session file concurrently
// with an in-flight turn. It returns 0 on /exit, /quit, or EOF, and
// ExitInterrupt (130) on a SIGINT exit request. Input is scanned in a helper
// goroutine so an exit request received while idle at the prompt is acted on
// immediately rather than blocking on the next line.
func Run(in io.Reader, app *App, exit <-chan struct{}) int {
	if app.Created.IsZero() {
		app.Created = app.clock()()
	}

	lines := make(chan string)
	scanDone := make(chan struct{})
	// scanErr holds a non-EOF read error from the input scanner. It is written
	// before lines is closed, so Run reads it only after observing the close
	// (the channel close establishes the happens-before, no race).
	var scanErr error
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(in)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			select {
			case lines <- sc.Text():
			case <-scanDone:
				return
			}
		}
		scanErr = sc.Err() // nil on clean EOF; a real error otherwise
	}()
	// Unblock the scanner goroutine's pending send on every return path.
	defer close(scanDone)

	prompt := app.Prompt
	if prompt == "" {
		prompt = "> "
	}

	// Restore a usable terminal before the first prompt (termios sane plus an
	// emulator soft reset), in case a prior process left it in raw, no-echo,
	// or mouse-reporting state. Targets /dev/tty directly; no-op without one.
	_ = term.Reset()
	fmt.Fprint(app.Errw, prompt)
	for {
		select {
		case <-exit:
			// SIGINT exit request (design §8.4). Any in-flight turn has already
			// returned (this loop runs turns synchronously), so the save here has
			// no concurrent writer.
			app.saveOrWarn(app.SessionPath)
			return ExitInterrupt
		case line, ok := <-lines:
			if !ok {
				// Input ended: clean EOF (^D) or a read error. Surface a read
				// error so it is not mistaken for a deliberate ^D, then save and
				// exit cleanly either way (design §8.4).
				if scanErr != nil {
					fmt.Fprintf(app.Errw, "[input error: %v]\n", scanErr)
				}
				app.saveOrWarn(app.SessionPath)
				return ExitOK
			}
			if line == "" {
				fmt.Fprint(app.Errw, prompt)
				continue
			}
			if strings.HasPrefix(line, "//") {
				app.runTurn(line[1:]) // // escapes one literal leading slash
				fmt.Fprint(app.Errw, prompt)
				continue
			}
			if strings.HasPrefix(line, "/") {
				if app.command(line) {
					return ExitOK
				}
				fmt.Fprint(app.Errw, prompt)
				continue
			}
			if strings.HasPrefix(line, "$$") && app.Skills != nil {
				app.runTurn(line[1:]) // $$ escapes one literal leading $
				fmt.Fprint(app.Errw, prompt)
				continue
			}
			if strings.HasPrefix(line, "$") && app.Skills != nil {
				if app.invokeSkill(line) {
					fmt.Fprint(app.Errw, prompt)
					continue
				}
			}
			app.runTurn(line)
			fmt.Fprint(app.Errw, prompt)
		}
	}
}

// command dispatches a meta-command line. It returns true when the REPL should
// exit (/exit, /quit).
func (app *App) command(line string) (exit bool) {
	cmd, arg, _ := strings.Cut(strings.TrimSpace(line), " ")
	arg = strings.TrimSpace(arg)

	switch cmd {
	case "/help":
		fmt.Fprintln(app.Errw, helpText)
	case "/exit", "/quit":
		app.saveOrWarn(app.SessionPath)
		return true
	case "/clear":
		app.clear()
	case "/compact":
		app.compact()
	case "/usage":
		fmt.Fprintln(app.Errw, app.usageSummary())
	case "/save":
		path := app.SessionPath
		if arg != "" {
			path = arg
		}
		if err := app.save(path); err != nil {
			fmt.Fprintf(app.Errw, "[save failed: %v]\n", err)
		} else {
			fmt.Fprintf(app.Errw, "[saved %s]\n", path)
		}
	case "/model":
		if arg == "" {
			fmt.Fprintln(app.Errw, app.modelSummary())
		} else {
			app.switchModel(arg)
		}
	case "/mode":
		if arg == "" {
			fmt.Fprintln(app.Errw, app.modeSummary())
		} else {
			app.switchMode(arg)
		}
	case "/skills":
		fmt.Fprintln(app.Errw, app.skillsSummary())
	default:
		fmt.Fprintf(app.Errw, "unknown command %q; type /help\n", cmd)
	}
	return false
}

// modelSummary renders the current model plus the configured models available
// for quick switching.
func (app *App) modelSummary() string {
	models := append([]string(nil), app.AvailableModels...)
	if app.Registry != nil {
		models = append(models, app.Registry.Models()...)
	}
	models = uniqueModels(models, app.Model)

	var b strings.Builder
	fmt.Fprintf(&b, "current: provider=%s model=%s base-url=%s\n", app.Provider, app.Model, app.BaseURL)
	b.WriteString("available models:")
	if len(models) == 0 {
		b.WriteString(" none configured")
		return b.String()
	}
	for _, model := range models {
		if model == app.Model {
			fmt.Fprintf(&b, "\n  %s (current)", model)
		} else {
			fmt.Fprintf(&b, "\n  %s", model)
		}
	}
	return b.String()
}

func uniqueModels(models []string, current string) []string {
	seen := make(map[string]bool, len(models)+1)
	var out []string
	for _, model := range models {
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		out = append(out, model)
	}
	if current != "" && !seen[current] {
		out = append(out, current)
	}
	sort.Strings(out)
	return out
}

func (app *App) switchModel(model string) {
	if app.SwitchModel == nil {
		fmt.Fprintln(app.Errw, "[model switch unavailable]")
		return
	}
	selection, err := app.SwitchModel(model)
	if err != nil {
		fmt.Fprintf(app.Errw, "[model switch failed: %v]\n", err)
		return
	}
	if selection.Runtime == nil {
		fmt.Fprintln(app.Errw, "[model switch failed: no provider was created]")
		return
	}
	if selection.Model == "" {
		selection.Model = model
	}
	if selection.Provider == "" {
		selection.Provider = app.Provider
	}
	app.Agent.SetProvider(selection.Runtime)
	app.Agent.SetModel(selection.Model, selection.ContextWindow)
	if selection.RegistryModel == "" {
		selection.RegistryModel = selection.Model
	}
	app.Renderer.SetModel(selection.RegistryModel)
	app.Provider = selection.Provider
	app.Model = selection.Model
	app.RegistryModel = selection.RegistryModel
	app.BaseURL = selection.BaseURL
	fmt.Fprintf(app.Errw, "[model switched: provider=%s model=%s base-url=%s]\n", app.Provider, app.Model, app.BaseURL)
}

// modeSummary renders the current run mode plus the modes available for
// switching, marking the current one.
func (app *App) modeSummary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "current mode: %s\n", app.Mode)
	b.WriteString("available modes:")
	if len(app.AvailableModes) == 0 {
		b.WriteString(" none configured")
		return b.String()
	}
	for _, name := range app.AvailableModes {
		if name == app.Mode {
			fmt.Fprintf(&b, "\n  %s (current)", name)
		} else {
			fmt.Fprintf(&b, "\n  %s", name)
		}
	}
	return b.String()
}

func (app *App) switchMode(name string) {
	if app.SwitchMode == nil {
		fmt.Fprintln(app.Errw, "[mode switch unavailable]")
		return
	}
	selection, err := app.SwitchMode(name)
	if err != nil {
		fmt.Fprintf(app.Errw, "[mode switch failed: %v]\n", err)
		return
	}
	app.Agent.SetTools(selection.Tools)
	app.Agent.SetSystem(selection.System)
	app.Mode = selection.Name
	app.System = selection.System // so saved sessions capture the mode's prompt
	fmt.Fprintf(app.Errw, "[mode switched: %s]\n", selection.Name)
}

// clear resets the conversation and rotates to a fresh auto-save file (design
// §10, §11). Cumulative usage resets with the conversation.
func (app *App) clear() {
	app.Agent.SetTranscript(nil)
	app.usage = session.UsageTotals{}
	app.Created = app.clock()()
	app.SessionPath = session.DefaultPath(app.StateDir, app.Created)
	fmt.Fprintf(app.Errw, "[cleared; new session %s]\n", app.SessionPath)
}

// invokeSkill handles $skillName invocations. It reads the SKILL.md file and
// sends it as a turn with the skill content embedded. Returns true when the
// skill was found and invoked; false when the skill name is unknown (caller
// falls through to a regular turn).
func (app *App) invokeSkill(line string) bool {
	words := strings.Fields(line)
	if len(words) == 0 {
		return false
	}
	skillName := strings.TrimPrefix(words[0], "$")
	skill, ok := app.Skills[skillName]
	if !ok {
		fmt.Fprintf(app.Errw, "unknown skill %q; type /skills\n", skillName)
		return true
	}
	body, err := skill.Read()
	if err != nil {
		fmt.Fprintf(app.Errw, "[skill %q read failed: %v]\n", skillName, err)
		return true
	}
	// Build the prompt: skill content + any additional text.
	var prompt strings.Builder
	fmt.Fprintf(&prompt, "%s\n\n", body)
	if len(words) > 1 {
		fmt.Fprintf(&prompt, "User: %s", strings.Join(words[1:], " "))
	} else {
		fmt.Fprintf(&prompt, "User: invoke skill %q", skillName)
	}
	app.runTurn(prompt.String())
	return true
}

// runTurn runs one user turn, accumulates usage, and saves the session. A turn
// error is reported but does not end the REPL (the next prompt may recover).
func (app *App) runTurn(prompt string) {
	ctx := context.Background()
	if app.Interrupt != nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		app.Interrupt.BeginTurn(cancel)
		defer func() {
			app.Interrupt.EndTurn()
			cancel()
		}()
	}

	app.Renderer.StartTurn()
	sink := &accumulatingSink{r: app.Renderer, app: app}
	err := app.Agent.RunTurn(ctx, prompt, sink)
	if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		fmt.Fprintf(app.Errw, "[error: %v]\n", err)
	}
	app.saveOrWarn(app.SessionPath)
}

// compact forces compaction now (/compact, design §12). The summary call's usage
// is folded into the cumulative session totals so /usage stays accurate, and the
// session is saved with the collapsed transcript. A summary-call error is already
// warned about via the sink by Compact; the transcript is left intact.
func (app *App) compact() {
	ctx := context.Background()
	sink := &accumulatingSink{r: app.Renderer, app: app}
	u, err := app.Agent.Compact(ctx, sink)
	if err != nil {
		return
	}
	app.addUsage(agent.TurnUsage{Usage: u})
	app.saveOrWarn(app.SessionPath)
}

// SetUsage seeds the cumulative session totals, used when resuming a session so
// /usage and saved totals continue from the prior run (design §11).
func (app *App) SetUsage(u session.UsageTotals) { app.usage = u }

// addUsage folds one turn's usage into the cumulative session totals.
func (app *App) addUsage(u agent.TurnUsage) {
	app.usage.InputTokens += u.Usage.InputTokens
	app.usage.OutputTokens += u.Usage.OutputTokens
	app.usage.CacheReadTokens += u.Usage.CacheReadTokens
	app.usage.CacheWriteTokens += u.Usage.CacheWriteTokens
	if app.Registry != nil {
		model := app.RegistryModel
		if model == "" {
			model = app.Model
		}
		if usd, known := app.Registry.Cost(model, u.Usage); known {
			app.usage.CostUSD += usd
		}
	}
}

// saveOrWarn is the automatic-save path used by every place that saves without a
// user explicitly asking (after-turn auto-save, exit saves, /compact). A failed
// save must never be silent: a visible warning beats silent data loss (design
// §11, §12), since a stale or missing on-disk transcript otherwise looks saved.
// The explicit /save command surfaces its own richer success/failure message and
// does not route through here.
func (app *App) saveOrWarn(path string) {
	if err := app.save(path); err != nil {
		fmt.Fprintf(app.Errw, "[save failed: %v]\n", err)
	}
}

// save writes the current transcript and usage totals to path (design §11).
func (app *App) save(path string) error {
	if path == "" {
		return nil
	}
	s := session.Session{
		Version:  session.Version,
		Provider: app.Provider,
		Model:    app.Model,
		Created:  app.Created,
		Updated:  app.clock()(),
		System:   app.System,
		Mode:     app.Mode,
		Messages: app.Agent.Transcript(),
		Usage:    app.usage,
	}
	return s.Save(path)
}

// usageSummary renders the cumulative session usage for /usage (design §10).
func (app *App) usageSummary() string {
	u := app.usage
	var b strings.Builder
	fmt.Fprintf(&b, "[session: %d in / %d out", u.InputTokens, u.OutputTokens)
	if u.CostUSD > 0 {
		fmt.Fprintf(&b, " · $%.4f", u.CostUSD)
	}
	b.WriteString("]")
	return b.String()
}

// skillsSummary renders the available skills for /skills (design §10), grouped
// by source directory (local vs user skills).
func (app *App) skillsSummary() string {
	if len(app.Skills) == 0 {
		return "[no skills available]"
	}

	// Group skills by scope
	byScope := make(map[skills.Scope][]string)
	for name, s := range app.Skills {
		byScope[s.Scope] = append(byScope[s.Scope], name)
	}

	// Find directory paths for each scope
	scopePath := make(map[skills.Scope]string)
	for _, d := range app.SkillDirs {
		scopePath[d.Scope] = d.Path
	}

	var b strings.Builder

	// Build directory label
	dirLabel := func(scope skills.Scope) string {
		if path, ok := scopePath[scope]; ok {
			return path
		}
		if scope == skills.ScopeProject {
			return "project"
		}
		return "user"
	}

	// Print local (project) skills first, then user skills
	for _, scope := range []skills.Scope{skills.ScopeProject, skills.ScopeUser} {
		names := byScope[scope]
		if len(names) == 0 {
			continue
		}
		sort.Strings(names)

		if scope == skills.ScopeProject {
			b.WriteString("local skills:\n")
		} else {
			fmt.Fprintf(&b, "user skills (%s):\n", dirLabel(scope))
		}

		for _, name := range names {
			s := app.Skills[name]
			fmt.Fprintf(&b, "  $%s - %s\n", name, s.Description)
		}
	}

	return b.String()
}

// accumulatingSink forwards events to the renderer while accumulating cumulative
// token totals and cost for the session (design §10 /usage, §11 saved totals).
type accumulatingSink struct {
	r   *Renderer
	app *App
}

func (s *accumulatingSink) TextDelta(text string)         { s.r.TextDelta(text) }
func (s *accumulatingSink) ToolStart(c llm.ToolCall)      { s.r.ToolStart(c) }
func (s *accumulatingSink) ToolResult(res llm.ToolResult) { s.r.ToolResult(res) }
func (s *accumulatingSink) Notice(msg string)             { s.r.Notice(msg) }

func (s *accumulatingSink) TurnComplete(u agent.TurnUsage) {
	s.app.addUsage(u)
	s.r.TurnComplete(u)
}

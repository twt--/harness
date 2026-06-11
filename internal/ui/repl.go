package ui

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/session"
	"harness/internal/skills"
	"harness/internal/term"
	"harness/internal/tools"
)

const (
	bracketedPasteStart = "\x1b[200~"
	bracketedPasteEnd   = "\x1b[201~"
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
	Turn        int       // last started user turn, persisted for replay numbering
	Now         func() time.Time

	// Interrupt is the optional SIGINT state machine. When set, the REPL marks
	// turn boundaries so ^C cancels a turn rather than the whole process
	// (design §8.4). Tests leave it nil.
	Interrupt *agent.InterruptWatcher

	// Prompt is the REPL input prompt string (default "> ").
	Prompt string

	// OpenEditor launches an editor for a temp prompt file. nil uses
	// $VISUAL, then $EDITOR, then vi. Tests inject this to edit deterministically.
	OpenEditor func(path string) error
	// BeforeEditor/AfterEditor temporarily hand the terminal back to the editor.
	// Run installs these hooks; tests and non-REPL callers can leave them nil.
	BeforeEditor func()
	AfterEditor  func()

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
  /edit [draft]    open $VISUAL/$EDITOR (or vi) for a multi-line prompt
  /save [file]     force save (optionally elsewhere)
  /model [model]   list models, or switch to model
  /mode [name]     list run modes, or switch to mode
  /skills          list available skills
  $skillName       invoke a skill (reads SKILL.md and sends as prompt)
Ctrl-G opens the editor from the prompt; lines starting with / are commands; // sends a literal leading slash`

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
// ExitInterrupt (130) on a SIGINT exit request. Input is scanned in an
// on-demand helper goroutine so an exit request received while idle at the
// prompt is acted on immediately rather than blocking on the next line. During
// an active turn the same helper also preserves typeahead and observes Esc-Esc
// without competing with an external editor launched from the idle prompt.
func Run(in io.Reader, app *App, exit <-chan struct{}) int {
	if app.Created.IsZero() {
		app.Created = app.clock()()
	}

	prompt := app.Prompt
	if prompt == "" {
		prompt = "> "
	}

	// Restore a usable terminal before the first prompt (termios sane plus an
	// emulator soft reset), in case a prior process left it in raw, no-echo,
	// or mouse-reporting state. Targets /dev/tty directly; no-op without one.
	var restoreCtrlG func() error
	disablePromptTerm := func() {
		_ = term.SetBracketedPaste(false)
		if restoreCtrlG != nil {
			_ = restoreCtrlG()
			restoreCtrlG = nil
		}
	}
	enablePromptTerm := func() {
		_ = term.Reset()
		if cleanup, err := term.EnableCtrlGLineEnd(); err == nil {
			restoreCtrlG = cleanup
		}
		_ = term.SetBracketedPaste(true)
	}
	enablePromptTerm()
	defer disablePromptTerm()

	prevBeforeEditor, prevAfterEditor := app.BeforeEditor, app.AfterEditor
	app.BeforeEditor = func() {
		disablePromptTerm()
		if prevBeforeEditor != nil {
			prevBeforeEditor()
		}
	}
	app.AfterEditor = func() {
		if prevAfterEditor != nil {
			prevAfterEditor()
		}
		enablePromptTerm()
	}
	defer func() {
		app.BeforeEditor = prevBeforeEditor
		app.AfterEditor = prevAfterEditor
	}()

	reader := newREPLReader(in)
	readReq := make(chan struct{})
	inputs := make(chan replReadResult, 1)
	go func() {
		for range readReq {
			input, ok, err := reader.read()
			inputs <- replReadResult{input: input, ok: ok, err: err}
			if !ok {
				return
			}
		}
	}()
	defer close(readReq)

	var (
		promptPrinted   bool
		readPending     bool
		inputEnded      bool
		inputErr        error
		active          bool
		activeReadPause bool
		exitAfterTurn   bool
		queued          []replInput
		turnDone        <-chan struct{}
		restoreEsc      func() error
		escPresses      escapePresses
	)

	requestRead := func() {
		if readPending || inputEnded {
			return
		}
		readPending = true
		readReq <- struct{}{}
	}
	setInputEnded := func(err error) {
		inputEnded = true
		inputErr = err
	}
	warnInputErr := func() {
		if inputErr != nil {
			fmt.Fprintf(app.Errw, "[input error: %v]\n", inputErr)
			inputErr = nil
		}
	}
	enableTurnTerm := func() {
		_ = term.SetBracketedPaste(false)
		if cleanup, err := term.EnableEscLineEnd(); err == nil {
			restoreEsc = cleanup
		}
		reader.setEscapeLineEnd(true)
	}
	disableTurnTerm := func() {
		reader.setEscapeLineEnd(false)
		if restoreEsc != nil {
			_ = restoreEsc()
			restoreEsc = nil
		}
		_ = term.SetBracketedPaste(true)
	}
	startTurn := func(prompt string) {
		run := app.prepareTurn(prompt)
		done := make(chan struct{}, 1)
		active = true
		activeReadPause = queuedContainsEditor(queued)
		exitAfterTurn = false
		promptPrinted = false
		escPresses.reset()
		enableTurnTerm()
		turnDone = done
		go func() {
			run()
			done <- struct{}{}
		}()
	}
	// applyAction dispatches one input at the idle prompt — both the queued-
	// typeahead drain and the fresh read use it — and reports whether the REPL
	// should exit.
	applyAction := func(input replInput) (exit bool) {
		action := app.handlePromptInput(input)
		promptPrinted = false
		if action.exit {
			return true
		}
		if action.run {
			if action.echoEditedPrompt {
				app.echoEditedPrompt(prompt, action.prompt)
			}
			startTurn(action.prompt)
		}
		return false
	}

	for {
		if active {
			if !activeReadPause {
				requestRead()
			}
			select {
			case <-exit:
				// SIGINT exit requests during a turn are honored only after the
				// turn goroutine finishes its own save and usage update.
				exitAfterTurn = true
			case <-turnDone:
				disableTurnTerm()
				active = false
				activeReadPause = false
				turnDone = nil
				escPresses.reset()
				if exitAfterTurn {
					app.saveOrWarn(app.SessionPath)
					return ExitInterrupt
				}
			case res := <-inputs:
				readPending = false
				if !res.ok {
					setInputEnded(res.err)
					continue
				}
				input := res.input
				if input.escape {
					if input.text != "" {
						queued = append(queued, replInput{text: input.text})
					}
					if escPresses.press(app.clock()()) && app.Interrupt != nil {
						app.Interrupt.CancelTurn()
					}
					continue
				}
				escPresses.reset()
				queued = append(queued, input)
				if inputMayOpenEditor(input) {
					activeReadPause = true
				}
			}
			continue
		}

		if len(queued) > 0 {
			input := queued[0]
			queued = queued[1:]
			if applyAction(input) {
				return ExitOK
			}
			continue
		}
		if inputEnded {
			warnInputErr()
			app.saveOrWarn(app.SessionPath)
			return ExitOK
		}
		if !promptPrinted {
			fmt.Fprint(app.Errw, prompt)
			promptPrinted = true
		}
		requestRead()
		select {
		case <-exit:
			// SIGINT exit request at the idle prompt (design §8.4).
			app.saveOrWarn(app.SessionPath)
			return ExitInterrupt
		case res := <-inputs:
			readPending = false
			if !res.ok {
				setInputEnded(res.err)
				continue
			}
			if applyAction(res.input) {
				return ExitOK
			}
		}
	}
}

type replInput struct {
	text   string
	pasted bool
	edit   bool
	escape bool
}

type replReadResult struct {
	input replInput
	ok    bool
	err   error
}

type replAction struct {
	prompt           string
	run              bool
	exit             bool
	echoEditedPrompt bool
}

type escapePresses struct {
	last time.Time
	seen bool
}

func (p *escapePresses) press(now time.Time) bool {
	if p.seen && now.Sub(p.last) <= time.Second {
		p.reset()
		return true
	}
	p.last = now
	p.seen = true
	return false
}

func (p *escapePresses) reset() {
	p.last = time.Time{}
	p.seen = false
}

func (app *App) handlePromptInput(input replInput) replAction {
	if input.escape {
		return replAction{}
	}
	line := input.text
	if line == "" && !input.edit {
		return replAction{}
	}
	if input.edit {
		if prompt, ok := app.editPrompt(line); ok {
			return replAction{prompt: prompt, run: true, echoEditedPrompt: true}
		}
		return replAction{}
	}
	if input.pasted {
		return replAction{prompt: line, run: true}
	}
	if strings.HasPrefix(line, "//") {
		return replAction{prompt: line[1:], run: true} // // escapes one literal leading slash
	}
	if strings.HasPrefix(line, "/") {
		cmd, arg := commandFields(line)
		if cmd == "/edit" {
			if prompt, ok := app.editPrompt(arg); ok {
				return replAction{prompt: prompt, run: true}
			}
			return replAction{}
		}
		if app.command(line) {
			return replAction{exit: true}
		}
		return replAction{}
	}
	if strings.HasPrefix(line, "$$") && app.Skills != nil {
		return replAction{prompt: line[1:], run: true} // $$ escapes one literal leading $
	}
	if strings.HasPrefix(line, "$") && app.Skills != nil {
		if prompt, handled, ok := app.skillPrompt(line); handled {
			if ok {
				return replAction{prompt: prompt, run: true}
			}
			return replAction{}
		}
	}
	return replAction{prompt: line, run: true}
}

func (app *App) echoEditedPrompt(replPrompt, submitted string) {
	if f, ok := app.Errw.(*os.File); ok && term.IsTerminal(f) {
		fmt.Fprintf(app.Errw, "\r\x1b[2K%s%s\n", replPrompt, submitted)
		return
	}
	fmt.Fprintln(app.Errw, submitted)
}

func commandFields(line string) (cmd, arg string) {
	cmd, arg, _ = strings.Cut(strings.TrimSpace(line), " ")
	return cmd, strings.TrimSpace(arg)
}

func inputMayOpenEditor(input replInput) bool {
	if input.edit {
		return true
	}
	if input.pasted {
		return false
	}
	cmd, _ := commandFields(input.text)
	return cmd == "/edit"
}

func queuedContainsEditor(inputs []replInput) bool {
	for _, input := range inputs {
		if inputMayOpenEditor(input) {
			return true
		}
	}
	return false
}

type replReader struct {
	r             *bufio.Reader
	paste         strings.Builder
	inPaste       bool
	escapeLineEnd atomic.Bool
}

func newREPLReader(in io.Reader) *replReader {
	return &replReader{r: bufio.NewReader(in)}
}

func (rr *replReader) setEscapeLineEnd(enabled bool) {
	rr.escapeLineEnd.Store(enabled)
}

func (rr *replReader) read() (replInput, bool, error) {
	for {
		line, terminator, err := readTerminalLine(rr.r, rr.escapeLineEnd.Load())
		if line != "" || terminator != lineTermNone {
			if input, emit := rr.handleLine(line, terminator); emit {
				return input, true, nil
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				if rr.inPaste && rr.paste.Len() > 0 {
					input := replInput{text: rr.paste.String(), pasted: true}
					rr.paste.Reset()
					rr.inPaste = false
					return input, true, nil
				}
				return replInput{}, false, nil
			}
			return replInput{}, false, err
		}
	}
}

type lineTerminator byte

const (
	lineTermNone    lineTerminator = 0
	lineTermNewline lineTerminator = '\n'
	lineTermEdit    lineTerminator = '\a'
	lineTermEscape  lineTerminator = '\x1b'
)

func readTerminalLine(r *bufio.Reader, escapeLineEnd bool) (line string, terminator lineTerminator, err error) {
	var b strings.Builder
	for {
		c, err := r.ReadByte()
		if err != nil {
			return b.String(), lineTermNone, err
		}
		switch c {
		case '\n':
			line := b.String()
			line = strings.TrimSuffix(line, "\r")
			return line, lineTermNewline, nil
		case byte(lineTermEdit):
			return b.String(), lineTermEdit, nil
		default:
			if escapeLineEnd && c == byte(lineTermEscape) {
				return b.String(), lineTermEscape, nil
			}
			b.WriteByte(c)
		}
	}
}

func (rr *replReader) handleLine(line string, terminator lineTerminator) (replInput, bool) {
	if !rr.inPaste {
		start := strings.Index(line, bracketedPasteStart)
		if start < 0 {
			return replInput{text: line, edit: terminator == lineTermEdit, escape: terminator == lineTermEscape}, true
		}
		rr.inPaste = true
		rr.paste.WriteString(line[:start])
		line = line[start+len(bracketedPasteStart):]
	}

	end := strings.Index(line, bracketedPasteEnd)
	if end >= 0 {
		rr.paste.WriteString(line[:end])
		text := rr.paste.String() + line[end+len(bracketedPasteEnd):]
		rr.paste.Reset()
		rr.inPaste = false
		return replInput{text: text, pasted: true}, true
	}

	rr.paste.WriteString(line)
	switch terminator {
	case lineTermNewline:
		rr.paste.WriteByte('\n')
	case lineTermEdit:
		rr.paste.WriteByte(byte(lineTermEdit))
	}
	return replInput{}, false
}

// command dispatches a meta-command line. It returns true when the REPL should
// exit (/exit, /quit).
func (app *App) command(line string) (exit bool) {
	cmd, arg := commandFields(line)

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
	case "/edit":
		if prompt, ok := app.editPrompt(arg); ok {
			app.runTurn(prompt)
		}
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
	app.Turn = 0
	app.SessionPath = session.DefaultPath(app.StateDir, app.Created)
	fmt.Fprintf(app.Errw, "[cleared; new session %s]\n", app.SessionPath)
}

func (app *App) skillPrompt(line string) (prompt string, handled bool, ok bool) {
	words := strings.Fields(line)
	if len(words) == 0 {
		return "", false, false
	}
	skillName := strings.TrimPrefix(words[0], "$")
	skill, ok := app.Skills[skillName]
	if !ok {
		fmt.Fprintf(app.Errw, "unknown skill %q; type /skills\n", skillName)
		return "", true, false
	}
	body, err := skill.Read()
	if err != nil {
		fmt.Fprintf(app.Errw, "[skill %q read failed: %v]\n", skillName, err)
		return "", true, false
	}
	// Build the prompt: skill content + any additional text.
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n\n", body)
	if len(words) > 1 {
		fmt.Fprintf(&b, "User: %s", strings.Join(words[1:], " "))
	} else {
		fmt.Fprintf(&b, "User: invoke skill %q", skillName)
	}
	return b.String(), true, true
}

// runTurn runs one user turn, accumulates usage, and saves the session. A turn
// error is reported but does not end the REPL (the next prompt may recover).
func (app *App) runTurn(prompt string) {
	app.prepareTurn(prompt)()
}

func (app *App) prepareTurn(prompt string) func() {
	turn := app.beginTurn(prompt)
	ctx := context.Background()
	var cancel context.CancelFunc
	if app.Interrupt != nil {
		ctx, cancel = context.WithCancel(ctx)
		app.Interrupt.BeginTurn(cancel)
	}

	app.Renderer.StartTurn()
	return func() {
		if app.Interrupt != nil {
			defer func() {
				app.Interrupt.EndTurn()
				cancel()
			}()
		}

		sink := newAccumulatingSink(app.Renderer, app, turn)
		err := app.Agent.RunTurn(ctx, prompt, sink)
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			fmt.Fprintf(app.Errw, "[error: %v]\n", err)
		}
		app.saveOrWarn(app.SessionPath)
	}
}

// compact forces compaction now (/compact, design §12). The summary call's usage
// is folded into the cumulative session totals so /usage stays accurate, and the
// session is saved with the collapsed transcript. A summary-call error is already
// warned about via the sink by Compact; the transcript is left intact.
func (app *App) compact() {
	ctx := context.Background()
	sink := newAccumulatingSink(app.Renderer, app, app.Turn)
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
		Turn:     app.Turn,
		Messages: app.Agent.Transcript(),
		Usage:    app.usage,
	}
	return s.Save(path)
}

func (app *App) beginTurn(prompt string) int {
	app.Turn++
	app.recordEvent(session.Event{
		Time: app.clock()(),
		Type: session.EventUser,
		Turn: app.Turn,
		Text: prompt,
	})
	return app.Turn
}

func (app *App) recordEvent(ev session.Event) {
	if app.SessionPath == "" {
		return
	}
	if ev.Time.IsZero() {
		ev.Time = app.clock()()
	}
	if err := session.AppendEvent(app.SessionPath, ev); err != nil {
		fmt.Fprintf(app.Errw, "[session event log failed: %v]\n", err)
	}
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

	// Build directory label (only user-scope sections render one)
	dirLabel := func(scope skills.Scope) string {
		if path, ok := scopePath[scope]; ok {
			return path
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
	r       *Renderer
	app     *App
	turn    int
	pending map[string]llm.ToolCall
}

func newAccumulatingSink(r *Renderer, app *App, turn int) *accumulatingSink {
	return &accumulatingSink{r: r, app: app, turn: turn, pending: make(map[string]llm.ToolCall)}
}

func (s *accumulatingSink) TextDelta(text string) {
	s.r.TextDelta(text)
	s.app.recordEvent(session.Event{Type: session.EventAssistantDelta, Turn: s.turn, Text: text})
}

func (s *accumulatingSink) ModelStepStart(step, attempt int, ctx agent.ContextEstimate) {
	s.r.ModelStepStart(step, attempt, ctx)
}

func (s *accumulatingSink) ToolUseStart(c llm.ToolCall) {
	s.r.ToolUseStart(c)
}

func (s *accumulatingSink) ToolUseDelta(index int, delta string) {
	s.r.ToolUseDelta(index, delta)
}

func (s *accumulatingSink) ToolStart(c llm.ToolCall) {
	s.pending[c.ID] = c
	s.r.ToolStart(c)
	s.app.recordEvent(session.Event{Type: session.EventToolStart, Turn: s.turn, ToolID: c.ID, Tool: c.Name, Input: c.Input})
}

func (s *accumulatingSink) ToolResult(res llm.ToolResult) {
	call := s.pending[res.ForID]
	delete(s.pending, res.ForID)
	line := ToolResultLine(call, res)
	s.r.ToolResult(res)
	s.app.recordEvent(session.Event{Type: session.EventToolResult, Turn: s.turn, ToolID: res.ForID, Tool: call.Name, Display: line})
	if res.Truncated {
		ref, err := session.SaveToolResultArtifact(s.app.SessionPath, s.turn, res)
		if err != nil {
			s.Notice(fmt.Sprintf("[tool result truncated; full output archive failed: %v]", err))
			return
		}
		msg := fmt.Sprintf("[tool result truncated: showing %s of %s", tools.HumanBytes(res.ShownBytes), tools.HumanBytes(res.OriginalBytes))
		if ref != "" {
			msg += "; full output: " + ref
		}
		msg += "]"
		s.Notice(msg)
	}
}

func (s *accumulatingSink) Notice(msg string) {
	s.r.Notice(msg)
	s.app.recordEvent(session.Event{Type: session.EventNotice, Turn: s.turn, Display: msg})
}

func (s *accumulatingSink) TurnComplete(u agent.TurnUsage) {
	s.app.addUsage(u)
	line := usageLine(s.r.registry, s.r.model, u, s.r.now().Sub(s.r.turnStart))
	s.r.TurnComplete(u)
	usage := u.Usage
	s.app.recordEvent(session.Event{
		Type:    session.EventTurnUsage,
		Turn:    s.turn,
		Display: line,
		Usage:   &usage,
		Steps:   u.Steps,
	})
}

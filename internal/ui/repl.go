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
	Reasoning     llm.ReasoningConfig
}

// AgentSummary is one configured agent row for /agent listing.
type AgentSummary struct {
	Name        string
	Description string
	Provider    string
	Model       string
}

// AgentSelection is the runtime agent bundle returned by App.SwitchAgent: the
// new tool registry, fully reassembled system prompt, and provider/model runtime
// for subsequent turns.
type AgentSelection struct {
	Name          string
	Tools         *tools.Registry
	System        string
	Provider      string
	Model         string
	RegistryModel string
	BaseURL       string
	Runtime       llm.Provider
	ContextWindow int
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
	Reasoning     llm.ReasoningConfig

	AvailableModels        []string
	SwitchModel            func(model string, reasoning llm.ReasoningConfig) (ModelSelection, error)
	PickModel              func(PickerIO) (string, error)
	PickerPageSize         int
	SetReasoning           func(model string, reasoning llm.ReasoningConfig) error
	SaveDefaultModel       func(provider, model string) error
	PromptDefaultModelSave bool

	AgentName       string         // current agent definition name
	AvailableAgents []AgentSummary // sorted agent names/descriptions for /agent listing
	SwitchAgent     func(name string) (AgentSelection, error)

	// RefreshMCP, when set, is consulted at the idle-prompt boundary (just
	// before a typed prompt starts a turn) to pick up proxy tool-list changes.
	// It is called with the current agent name; a non-nil registry replaces the
	// agent's tools and notice is rendered. A nil registry means "no change".
	// nil disables the hook (one-shot mode and tests leave it nil).
	RefreshMCP func(agentName string) (*tools.Registry, string)

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

	// DisabledTools lists optional built-in tools that could not be registered
	// (e.g., rg when ripgrep is not installed). Used by /tools.
	DisabledTools []tools.DisabledTool

	// SummaryWidth returns the terminal width for command summaries. nil or a
	// non-positive value disables forced wrapping.
	SummaryWidth func() int

	usage session.UsageTotals // cumulative across the session
}

// helpText lists the meta-commands (design §10).
const helpText = `commands:
  /help            list commands
  /exit, /quit     save and exit
  /clear           reset conversation; rotate to a fresh session file
  /compact         force compaction now
  /usage           cumulative session tokens and cost
  /tools           list available tools (built-in, MCP, and disabled)
  /edit [draft]    open $VISUAL/$EDITOR (or vi) for a multi-line prompt
  /save [file]     force save (optionally elsewhere)
  /model [model]   pick a configured provider/model, or switch directly
  /effort [level]  list or set reasoning effort for the current model
  /agent [name]    list agents, or switch to agent
  /mode [name]     alias for /agent
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
	return run(in, app, exit, promptLineEditorEnabled(in, app.Errw))
}

func run(in io.Reader, app *App, exit <-chan struct{}, usePromptEditor bool) int {
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
	var restorePromptTerm func() error
	disablePromptTerm := func() {
		_ = term.SetBracketedPaste(false)
		if restorePromptTerm != nil {
			_ = restorePromptTerm()
			restorePromptTerm = nil
		}
	}
	enablePromptTerm := func() {
		_ = term.Reset()
		if usePromptEditor {
			if cleanup, err := term.EnablePromptRawMode(); err == nil {
				restorePromptTerm = cleanup
			}
		} else if cleanup, err := term.EnableCtrlGLineEnd(); err == nil {
			restorePromptTerm = cleanup
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

	reader := newREPLReader(in, app.Errw, usePromptEditor)
	readReq := make(chan replReadRequest)
	inputs := make(chan replReadResult, 1)
	go func() {
		for req := range readReq {
			input, ok, err := reader.read(req)
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
		plainPromptRead bool
		queued          []replInput
		turnDone        <-chan struct{}
		restoreEsc      func() error
		escPresses      escapePresses
	)

	requestRead := func(req replReadRequest) {
		if readPending || inputEnded {
			return
		}
		readPending = true
		readReq <- req
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
	finish := func(code int) int {
		if app.Renderer != nil {
			app.Renderer.finishAssistantLine()
		}
		app.saveOrWarn(app.SessionPath)
		app.printExitUsageSummary()
		return code
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
		plainPromptRead = false
		promptPrinted = false
		escPresses.reset()
		disablePromptTerm()
		enableTurnTerm()
		turnDone = done
		go func() {
			run()
			done <- struct{}{}
		}()
	}
	readCommandLine := func(label string) (string, error) {
		if len(queued) > 0 {
			if _, err := fmt.Fprint(app.Errw, label); err != nil {
				return "", err
			}
			input := queued[0]
			queued = queued[1:]
			return strings.TrimSpace(input.text), nil
		}
		req := replReadRequest{}
		if usePromptEditor {
			req = replReadRequest{prompt: label, promptEditor: true}
		} else if _, err := fmt.Fprint(app.Errw, label); err != nil {
			return "", err
		}
		input, ok, err := reader.read(req)
		if !ok {
			if err != nil {
				return "", err
			}
			return "", io.EOF
		}
		return strings.TrimSpace(input.text), nil
	}
	// applyAction dispatches one input at the idle prompt — both the queued-
	// typeahead drain and the fresh read use it — and reports whether the REPL
	// should exit.
	applyAction := func(input replInput) (exit bool) {
		action := app.handlePromptInput(input, readCommandLine)
		promptPrinted = false
		if action.exit {
			return true
		}
		if action.run {
			if action.echoEditedPrompt {
				app.echoEditedPrompt(prompt, action.prompt)
			}
			app.refreshMCP()
			startTurn(action.prompt)
		}
		return false
	}

	for {
		if active {
			if !activeReadPause {
				requestRead(replReadRequest{})
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
					return finish(ExitInterrupt)
				}
				if usePromptEditor && readPending {
					// A plain read started during the model turn is still
					// blocked. Let it collect the next line in canonical mode;
					// starting the raw prompt editor now would leave no prompt
					// drawn and no terminal echo until that stale read finishes.
					plainPromptRead = true
				} else {
					enablePromptTerm()
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
				activeReadPause = true
			}
			continue
		}

		if len(queued) > 0 {
			input := queued[0]
			queued = queued[1:]
			if applyAction(input) {
				return finish(ExitOK)
			}
			continue
		}
		if inputEnded {
			warnInputErr()
			return finish(ExitOK)
		}
		if !promptPrinted {
			if !usePromptEditor || plainPromptRead {
				fmt.Fprint(app.Errw, prompt)
			}
			promptPrinted = true
		}
		if !plainPromptRead {
			requestRead(replReadRequest{prompt: prompt, promptEditor: usePromptEditor})
		}
		select {
		case <-exit:
			// SIGINT exit request at the idle prompt (design §8.4).
			return finish(ExitInterrupt)
		case res := <-inputs:
			readPending = false
			if plainPromptRead {
				plainPromptRead = false
				enablePromptTerm()
			}
			if !res.ok {
				setInputEnded(res.err)
				continue
			}
			if applyAction(res.input) {
				return finish(ExitOK)
			}
		}
	}
}

func promptLineEditorEnabled(in io.Reader, w io.Writer) bool {
	inf, ok := in.(*os.File)
	if !ok || !term.IsTerminal(inf) {
		return false
	}
	wf, ok := w.(*os.File)
	return ok && term.IsTerminal(wf)
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

type replReadRequest struct {
	prompt       string
	promptEditor bool
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

func (app *App) handlePromptInput(input replInput, readCommandLine func(string) (string, error)) replAction {
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
		if app.command(line, readCommandLine) {
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
	editor        *promptLineEditor
	paste         strings.Builder
	inPaste       bool
	escapeLineEnd atomic.Bool
}

func newREPLReader(in io.Reader, promptWriter io.Writer, promptEditor bool) *replReader {
	r := bufio.NewReader(in)
	rr := &replReader{r: r}
	if promptEditor {
		rr.editor = newPromptLineEditorWithReader(r, promptWriter)
	}
	return rr
}

func (rr *replReader) setEscapeLineEnd(enabled bool) {
	rr.escapeLineEnd.Store(enabled)
}

func (rr *replReader) read(req replReadRequest) (replInput, bool, error) {
	if req.promptEditor && rr.editor != nil {
		return rr.editor.read(req.prompt)
	}
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
func (app *App) command(line string, readCommandLine func(string) (string, error)) (exit bool) {
	cmd, arg := commandFields(line)

	switch cmd {
	case "/help":
		fmt.Fprintln(app.Errw, helpText)
	case "/exit", "/quit":
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
			app.pickModel(readCommandLine)
		} else {
			app.switchModelAndPromptDefault(arg, app.Reasoning, readCommandLine)
		}
	case "/effort":
		app.effort(arg)
	case "/agent", "/mode":
		if arg == "" {
			fmt.Fprintln(app.Errw, app.agentSummary())
		} else {
			app.switchAgent(arg)
		}
	case "/skills":
		fmt.Fprintln(app.Errw, app.skillsSummary())
	case "/tools":
		fmt.Fprintln(app.Errw, app.toolsSummary())
	default:
		fmt.Fprintf(app.Errw, "unknown command %q; type /help\n", cmd)
	}
	return false
}

func (app *App) pickModel(readLine func(string) (string, error)) {
	if app.PickModel == nil {
		fmt.Fprintln(app.Errw, app.modelSummary())
		return
	}
	fmt.Fprintf(app.Errw, "current: provider=%s model=%s\n", app.Provider, app.Model)
	model, err := app.PickModel(PickerIO{
		ReadLine: readLine,
		Writer:   app.Errw,
		PageSize: app.PickerPageSize,
	})
	if err != nil {
		if errors.Is(err, ErrPickerCancelled) {
			fmt.Fprintln(app.Errw, "[model selection cancelled]")
			return
		}
		fmt.Fprintf(app.Errw, "[model selection failed: %v]\n", err)
		return
	}
	reasoning, err := app.promptReasoningEffort(model, app.Reasoning, readLine)
	if err != nil {
		if errors.Is(err, ErrPickerCancelled) {
			fmt.Fprintln(app.Errw, "[model selection cancelled]")
			return
		}
		fmt.Fprintf(app.Errw, "[model selection failed: %v]\n", err)
		return
	}
	app.switchModelAndPromptDefault(model, reasoning, readLine)
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
	fmt.Fprintf(&b, "current: provider=%s model=%s proxy-url=%s\n", app.Provider, app.Model, app.BaseURL)
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

func (app *App) switchModel(model string, reasoning llm.ReasoningConfig) bool {
	if app.SwitchModel == nil {
		fmt.Fprintln(app.Errw, "[model switch unavailable]")
		return false
	}
	selection, err := app.SwitchModel(model, reasoning)
	if err != nil {
		fmt.Fprintf(app.Errw, "[model switch failed: %v]\n", err)
		return false
	}
	if selection.Runtime == nil {
		fmt.Fprintln(app.Errw, "[model switch failed: no provider was created]")
		return false
	}
	if selection.Model == "" {
		selection.Model = model
	}
	if selection.Provider == "" {
		selection.Provider = app.Provider
	}
	if selection.Reasoning.Empty() && !reasoning.Empty() {
		selection.Reasoning = reasoning
	}
	app.Agent.SetProvider(selection.Runtime)
	app.Agent.SetModel(selection.Model, selection.ContextWindow)
	app.Agent.SetReasoning(selection.Reasoning)
	if selection.RegistryModel == "" {
		selection.RegistryModel = selection.Model
	}
	app.Renderer.SetModel(selection.RegistryModel)
	app.Provider = selection.Provider
	app.Model = selection.Model
	app.RegistryModel = selection.RegistryModel
	app.BaseURL = selection.BaseURL
	app.Reasoning = selection.Reasoning
	fmt.Fprintf(app.Errw, "[model switched: provider=%s model=%s proxy-url=%s effort=%s]\n", app.Provider, app.Model, app.BaseURL, app.reasoningEffortLabel())
	return true
}

func (app *App) switchModelAndPromptDefault(model string, reasoning llm.ReasoningConfig, readLine func(string) (string, error)) {
	if !app.switchModel(model, reasoning) {
		return
	}
	app.promptSaveDefaultModel(readLine)
}

func (app *App) promptSaveDefaultModel(readLine func(string) (string, error)) {
	if app.SaveDefaultModel == nil || !app.PromptDefaultModelSave {
		return
	}
	save, err := PromptSaveDefaultModel(readLine, app.Errw, app.Provider, app.Model)
	if err != nil {
		if errors.Is(err, ErrPickerCancelled) {
			fmt.Fprintln(app.Errw, "[default model save cancelled]")
			return
		}
		fmt.Fprintf(app.Errw, "[default model save failed: %v]\n", err)
		return
	}
	if !save {
		return
	}
	if err := app.SaveDefaultModel(app.Provider, app.Model); err != nil {
		fmt.Fprintf(app.Errw, "[default model save failed: %v]\n", err)
		return
	}
	fmt.Fprintln(app.Errw, "[default model saved]")
}

func (app *App) effort(arg string) {
	if arg == "" {
		fmt.Fprintln(app.Errw, app.effortSummary())
		return
	}
	reasoning := app.Reasoning
	effort, ok := normalizeEffortInput(arg)
	if !ok {
		fmt.Fprintf(app.Errw, "[reasoning effort failed: invalid effort %q for model %q]\n", arg, app.currentRegistryModel())
		return
	}
	reasoning.Effort = effort
	if err := app.validateEffortForModel(app.currentRegistryModel(), effort); err != nil {
		fmt.Fprintf(app.Errw, "[reasoning effort failed: %v]\n", err)
		return
	}
	if err := app.setReasoning(reasoning); err != nil {
		fmt.Fprintf(app.Errw, "[reasoning effort failed: %v]\n", err)
		return
	}
	fmt.Fprintf(app.Errw, "[reasoning effort: %s]\n", app.reasoningEffortLabel())
}

func (app *App) setReasoning(reasoning llm.ReasoningConfig) error {
	if app.SetReasoning != nil {
		if err := app.SetReasoning(app.currentRegistryModel(), reasoning); err != nil {
			return err
		}
	}
	app.Reasoning = reasoning
	if app.Agent != nil {
		app.Agent.SetReasoning(reasoning)
	}
	return nil
}

func (app *App) effortSummary() string {
	model := app.currentRegistryModel()
	var b strings.Builder
	fmt.Fprintf(&b, "current reasoning effort: %s\n", app.reasoningEffortLabel())
	info, ok := app.reasoningInfoForModel(model)
	if !ok {
		fmt.Fprintf(&b, "available efforts for %s: unknown", model)
		return b.String()
	}
	if !info.Supported {
		fmt.Fprintf(&b, "available efforts for %s: none (model does not support reasoning)", model)
		return b.String()
	}
	values, hasEffort := info.EffortValues()
	if !hasEffort {
		fmt.Fprintf(&b, "available efforts for %s: none (catalog lists no effort levels)", model)
		return b.String()
	}
	if len(values) == 0 {
		fmt.Fprintf(&b, "available efforts for %s: provider-defined (catalog lists no fixed levels)", model)
		return b.String()
	}
	fmt.Fprintf(&b, "available efforts for %s:", model)
	app.writeEffortRows(&b, values)
	return b.String()
}

func (app *App) writeEffortRows(b *strings.Builder, values []string) {
	current := strings.ToLower(strings.TrimSpace(app.Reasoning.Effort))
	if current == "" {
		b.WriteString("\n  provider default (current)")
	} else {
		b.WriteString("\n  provider default")
	}
	for _, value := range values {
		fmt.Fprintf(b, "\n  %s", value)
		if strings.EqualFold(value, current) {
			b.WriteString(" (current)")
		}
	}
}

func (app *App) promptReasoningEffort(model string, reasoning llm.ReasoningConfig, readLine func(string) (string, error)) (llm.ReasoningConfig, error) {
	info, ok := app.reasoningInfoForModel(model)
	if !ok || !info.Supported {
		return reasoning, nil
	}
	values, hasEffort := info.EffortValues()
	if !hasEffort || len(values) == 0 {
		return reasoning, nil
	}
	current := strings.TrimSpace(reasoning.Effort)
	currentValid := current == "" || info.SupportsEffort(current)
	for {
		prompt := fmt.Sprintf("Reasoning effort (default/%s; current: %s): ", strings.Join(values, "/"), effortPromptCurrent(current, currentValid))
		input, err := readLine(prompt)
		if err != nil {
			return reasoning, err
		}
		input = strings.TrimSpace(input)
		if input == "" {
			if currentValid {
				return reasoning, nil
			}
			reasoning.Effort = ""
			return reasoning, nil
		}
		if strings.EqualFold(input, "q") {
			return reasoning, ErrPickerCancelled
		}
		effort, ok := normalizeEffortInput(input)
		if !ok || (effort != "" && !info.SupportsEffort(effort)) {
			fmt.Fprintf(app.Errw, "Invalid reasoning effort %q (supported: default, %s)\n", input, strings.Join(values, ", "))
			continue
		}
		reasoning.Effort = effort
		return reasoning, nil
	}
}

func PromptSaveDefaultModel(readLine func(string) (string, error), w io.Writer, provider, model string) (bool, error) {
	for {
		input, err := readLine(fmt.Sprintf("Save %s:%s as the default model? (y/N): ", provider, model))
		if err != nil {
			return false, err
		}
		switch strings.ToLower(strings.TrimSpace(input)) {
		case "", "n", "no":
			return false, nil
		case "y", "yes":
			return true, nil
		case "q":
			return false, ErrPickerCancelled
		default:
			fmt.Fprintln(w, `Please answer "yes" or "no".`)
		}
	}
}

func effortPromptCurrent(current string, valid bool) string {
	if strings.TrimSpace(current) == "" {
		return "provider default"
	}
	if valid {
		return current
	}
	return current + " (not valid for this model; Enter uses provider default)"
}

func normalizeEffortInput(input string) (string, bool) {
	effort := strings.ToLower(strings.TrimSpace(input))
	switch effort {
	case "":
		return "", false
	case "default", "none", "provider-default":
		return "", true
	default:
		return effort, true
	}
}

func (app *App) validateEffortForModel(model, effort string) error {
	if effort == "" {
		return nil
	}
	info, ok := app.reasoningInfoForModel(model)
	if !ok {
		return nil
	}
	if info.SupportsEffort(effort) {
		return nil
	}
	if !info.Supported {
		return fmt.Errorf("model %q does not support reasoning effort", model)
	}
	if values, ok := info.EffortValues(); ok && len(values) > 0 {
		return fmt.Errorf("model %q does not support reasoning effort %q (supported: %s)", model, effort, strings.Join(values, ", "))
	}
	return fmt.Errorf("model %q does not support reasoning effort", model)
}

func (app *App) reasoningInfoForModel(model string) (*llm.ReasoningInfo, bool) {
	if app.Registry == nil {
		return nil, false
	}
	for _, key := range app.reasoningLookupKeys(model) {
		info, ok := app.Registry.Lookup(key)
		if ok && info.Reasoning != nil {
			return info.Reasoning, true
		}
	}
	return nil, false
}

func (app *App) reasoningLookupKeys(model string) []string {
	var keys []string
	add := func(key string) {
		key = strings.TrimSpace(key)
		if key == "" {
			return
		}
		for _, existing := range keys {
			if existing == key {
				return
			}
		}
		keys = append(keys, key)
	}
	add(model)
	add(app.currentRegistryModel())
	if app.Provider != "" {
		add(app.Provider + ":" + model)
		add(app.Provider + ":" + app.Model)
	}
	return keys
}

func (app *App) currentRegistryModel() string {
	if app.RegistryModel != "" {
		return app.RegistryModel
	}
	if app.Provider != "" && app.Model != "" {
		if app.Registry != nil {
			if _, ok := app.Registry.Lookup(app.Provider + ":" + app.Model); ok {
				return app.Provider + ":" + app.Model
			}
		}
	}
	if app.Model != "" {
		return app.Model
	}
	return "unknown"
}

func (app *App) reasoningEffortLabel() string {
	if strings.TrimSpace(app.Reasoning.Effort) == "" {
		return "provider default"
	}
	return app.Reasoning.Effort
}

// agentSummary renders the current agent plus available agents and descriptions,
// marking the current one.
func (app *App) agentSummary() string {
	var b strings.Builder
	fmt.Fprintf(&b, "current agent: %s [%s]\n", app.AgentName, app.currentAgentModelSummary())
	b.WriteString("available agents:")
	if len(app.AvailableAgents) == 0 {
		b.WriteString(" none configured")
		return b.String()
	}
	labels := make([]string, len(app.AvailableAgents))
	for i, a := range app.AvailableAgents {
		label := a.Name
		if a.Name == app.AgentName {
			label += " (current)"
		}
		labels[i] = label
	}
	rows := make([]NameDescription, 0, len(app.AvailableAgents))
	for i, a := range app.AvailableAgents {
		modelInfo := app.agentModelSummary(a)
		description := "[" + modelInfo + "]"
		if strings.TrimSpace(a.Description) != "" {
			description += "  " + a.Description
		}
		rows = append(rows, NameDescription{
			Name:        labels[i],
			Description: description,
		})
	}
	b.WriteByte('\n')
	WriteNameDescriptionList(&b, rows, NameDescriptionListOptions{Indent: "  ", Width: app.summaryWidth()})
	return strings.TrimSuffix(b.String(), "\n")
}

func (app *App) currentAgentModelSummary() string {
	if app.Provider != "" || app.Model != "" {
		return fmt.Sprintf("%s/%s", app.Provider, app.Model)
	}
	return "unknown"
}

func (app *App) agentModelSummary(a AgentSummary) string {
	provider := strings.TrimSpace(a.Provider)
	model := strings.TrimSpace(a.Model)
	switch {
	case provider == "" && model == "":
		return "inherit current"
	case provider == "":
		return fmt.Sprintf("inherit provider/%s", model)
	case model == "":
		return fmt.Sprintf("%s/inherit current model", provider)
	default:
		return fmt.Sprintf("%s/%s", provider, model)
	}
}

func (app *App) switchAgent(name string) {
	if app.SwitchAgent == nil {
		fmt.Fprintln(app.Errw, "[agent switch unavailable]")
		return
	}
	oldProvider, oldModel := app.Provider, app.Model
	selection, err := app.SwitchAgent(name)
	if err != nil {
		fmt.Fprintf(app.Errw, "[agent switch failed: %v]\n", err)
		return
	}
	app.Agent.SetTools(selection.Tools)
	app.Agent.SetSystem(selection.System)
	if selection.Runtime != nil {
		app.Agent.SetProvider(selection.Runtime)
	}
	if selection.Model != "" {
		app.Agent.SetModel(selection.Model, selection.ContextWindow)
	}
	app.AgentName = selection.Name
	app.System = selection.System // so saved sessions capture the agent's prompt
	if selection.Provider != "" {
		app.Provider = selection.Provider
	}
	if selection.Model != "" {
		app.Model = selection.Model
	}
	if selection.RegistryModel == "" {
		selection.RegistryModel = app.Model
	}
	app.RegistryModel = selection.RegistryModel
	if app.Renderer != nil {
		app.Renderer.SetModel(selection.RegistryModel)
	}
	if selection.BaseURL != "" {
		app.BaseURL = selection.BaseURL
	}
	fmt.Fprintf(app.Errw, "[agent switched: %s]\n", selection.Name)
	fmt.Fprintf(app.Errw, "provider: %s  model: %s\n", app.Provider, app.Model)
	if oldProvider != app.Provider || oldModel != app.Model {
		fmt.Fprintln(app.Errw, "[warning: provider/model changed; the new model may start without prompt cache, increasing token usage or cost]")
	}
}

// refreshMCP applies any pending proxy tool-list change at the idle-prompt
// boundary, mirroring switchAgent's Agent.SetTools swap. It is a no-op when no
// hook is installed or the hook reports no change, so MCP-disabled runs (the
// default) and the one-shot path pay nothing.
func (app *App) refreshMCP() {
	if app.RefreshMCP == nil {
		return
	}
	sel, notice := app.RefreshMCP(app.AgentName)
	if sel == nil {
		return
	}
	app.Agent.SetTools(sel)
	if notice != "" {
		fmt.Fprintln(app.Errw, notice)
	}
}

// clear resets the conversation and rotates to a fresh auto-save file (design
// §10, §11). Cumulative usage resets with the conversation.
func (app *App) clear() {
	app.Agent.SetTranscript(nil)
	app.SetUsage(session.UsageTotals{})
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
func (app *App) SetUsage(u session.UsageTotals) {
	app.usage = u
	if app.Renderer != nil {
		app.Renderer.SetCumulativeUsage(u.InputTokens, u.OutputTokens, u.CostUSD)
	}
}

// addUsage folds one turn's usage into the cumulative session totals.
func (app *App) addUsage(u agent.TurnUsage) {
	app.usage.InputTokens += u.Usage.InputTokens
	app.usage.OutputTokens += u.Usage.OutputTokens
	app.usage.CacheReadTokens += u.Usage.CacheReadTokens
	app.usage.CacheWriteTokens += u.Usage.CacheWriteTokens
	app.usage.ReasoningTokens += u.Usage.ReasoningTokens
	if app.Registry != nil {
		model := app.RegistryModel
		if model == "" {
			model = app.Model
		}
		if usd, known := app.Registry.Cost(model, u.Usage); known {
			app.usage.CostUSD += usd
		}
	}
	if app.Renderer != nil {
		app.Renderer.SetCumulativeUsage(app.usage.InputTokens, app.usage.OutputTokens, app.usage.CostUSD)
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
		Agent:    app.AgentName,
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
	fmt.Fprintf(&b, "[session: %d input / %d cached input / %d output / %d reasoning",
		u.InputTokens, u.CacheReadTokens, u.OutputTokens, u.ReasoningTokens)
	if u.CacheWriteTokens > 0 {
		fmt.Fprintf(&b, " / %d cache write", u.CacheWriteTokens)
	}
	if u.CostUSD > 0 {
		fmt.Fprintf(&b, " · $%.4f", u.CostUSD)
	}
	b.WriteString("]")
	return b.String()
}

func (app *App) printExitUsageSummary() {
	fmt.Fprintf(app.Errw, "[session summary: %d input / %d cached input / %d output / %d reasoning",
		app.usage.InputTokens, app.usage.CacheReadTokens, app.usage.OutputTokens, app.usage.ReasoningTokens)
	if app.usage.CacheWriteTokens > 0 {
		fmt.Fprintf(app.Errw, " / %d cache write", app.usage.CacheWriteTokens)
	}
	if app.usage.CostUSD > 0 {
		fmt.Fprintf(app.Errw, " · $%.4f", app.usage.CostUSD)
	}
	fmt.Fprintln(app.Errw, "]")
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

		rows := make([]NameDescription, 0, len(names))
		for _, name := range names {
			s := app.Skills[name]
			rows = append(rows, NameDescription{Name: name, Description: s.Description})
		}
		WriteNameDescriptionList(&b, rows, NameDescriptionListOptions{
			Indent:     "  ",
			NamePrefix: "$",
			Width:      app.summaryWidth(),
		})
	}

	return b.String()
}

// toolsSummary renders the available tools for /tools: enabled built-in tools,
// enabled MCP tools (grouped by server), and disabled built-in tools with reasons.
func (app *App) toolsSummary() string {
	specs := app.Agent.ToolSpecs()

	var builtins, mcps []string
	descriptions := make(map[string]string, len(specs))
	for _, spec := range specs {
		descriptions[spec.Name] = spec.Description
		if isMCPToolName(spec.Name) {
			mcps = append(mcps, spec.Name)
		} else {
			builtins = append(builtins, spec.Name)
		}
	}

	var b strings.Builder

	// Enabled built-in tools
	if len(builtins) > 0 {
		b.WriteString("built-in tools:\n")
		rows := make([]NameDescription, 0, len(builtins))
		for _, name := range builtins {
			rows = append(rows, NameDescription{Name: name, Description: descriptions[name]})
		}
		WriteNameDescriptionList(&b, rows, NameDescriptionListOptions{Indent: "  ", Width: app.summaryWidth()})
	}

	// Enabled MCP tools, grouped by server
	if len(mcps) > 0 {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		byServer := make(map[string][]string)
		for _, name := range mcps {
			label := mcpServerLabel(name)
			byServer[label] = append(byServer[label], name)
		}
		// Sort server labels for stable output
		labels := make([]string, 0, len(byServer))
		for l := range byServer {
			labels = append(labels, l)
		}
		sort.Strings(labels)
		b.WriteString("mcp tools:\n")
		for _, label := range labels {
			fmt.Fprintf(&b, "  [%s]\n", label)
			rows := make([]NameDescription, 0, len(byServer[label]))
			for _, name := range byServer[label] {
				rows = append(rows, NameDescription{Name: name, Description: descriptions[name]})
			}
			WriteNameDescriptionList(&b, rows, NameDescriptionListOptions{Indent: "    ", Width: app.summaryWidth()})
		}
	}

	// Disabled tools
	if len(app.DisabledTools) > 0 {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("disabled tools:\n")
		for _, d := range app.DisabledTools {
			fmt.Fprintf(&b, "  %s  (%s)\n", d.Name, d.Reason)
		}
	}

	if b.Len() == 0 {
		return "[no tools available]"
	}
	return b.String()
}

// mcpServerLabel extracts a display-friendly server label from an MCP tool
// name of the form mcp__<server>__<tool>. It mirrors mcptools.serverLabel.
func mcpServerLabel(name string) string {
	const prefix = "mcp__"
	rest, _ := strings.CutPrefix(name, prefix)
	label, _, _ := strings.Cut(rest, "__")
	if label == "" {
		return "(unknown)"
	}
	return label
}

func isMCPToolName(name string) bool {
	return strings.HasPrefix(name, "mcp__")
}

func (app *App) summaryWidth() int {
	if app.SummaryWidth == nil {
		return 0
	}
	return app.SummaryWidth()
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

func (s *accumulatingSink) ModelTurnStart(modelTurn, attempt int, ctx agent.ContextEstimate) {
	s.r.ModelTurnStart(modelTurn, attempt, ctx)
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
	s.r.TurnComplete(u)
	s.app.addUsage(u)
	// Regenerate the line for the session event record after cumulative totals
	// have been updated by TurnComplete above.
	line := usageLine(s.r.registry, s.r.model, u, s.r.now().Sub(s.r.turnStart),
		s.r.cumInput, s.r.cumOutput, s.r.cumCost)
	usage := u.Usage
	s.app.recordEvent(session.Event{
		Type:       session.EventTurnUsage,
		Turn:       s.turn,
		Display:    line,
		Usage:      &usage,
		ModelTurns: u.ModelTurns,
	})
}

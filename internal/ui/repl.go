package ui

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"harness/internal/agent"
	"harness/internal/llm"
	"harness/internal/session"
)

// App bundles the dependencies the REPL and one-shot driver need. main builds it
// from the resolved config, provider factory, tool registry, and renderer
// (design §10). The agent owns the running transcript; App tracks the cumulative
// session usage and the current save path (rotated by /clear).
type App struct {
	Agent    *agent.Agent
	Renderer *Renderer
	Out      io.Writer
	Errw     io.Writer

	Provider string
	Model    string
	BaseURL  string
	System   string

	SessionPath string    // current save path; /clear rotates it
	StateDir    string    // for rotating to a fresh auto-save path on /clear
	Created     time.Time // session creation time (preserved across saves)
	Now         func() time.Time

	// Interrupt is the optional SIGINT state machine. When set, the REPL marks
	// turn boundaries so ^C cancels a turn rather than the whole process
	// (design §8.4). Tests leave it nil.
	Interrupt *agent.InterruptWatcher

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
  /model           print provider, model, and base URL
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

	for {
		select {
		case <-exit:
			// SIGINT exit request (design §8.4). Any in-flight turn has already
			// returned (this loop runs turns synchronously), so the save here has
			// no concurrent writer.
			app.save(app.SessionPath)
			return ExitInterrupt
		case line, ok := <-lines:
			if !ok {
				// Input ended: clean EOF (^D) or a read error. Surface a read
				// error so it is not mistaken for a deliberate ^D, then save and
				// exit cleanly either way (design §8.4).
				if scanErr != nil {
					fmt.Fprintf(app.Errw, "[input error: %v]\n", scanErr)
				}
				app.save(app.SessionPath)
				return ExitOK
			}
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "//") {
				app.runTurn(line[1:]) // // escapes one literal leading slash
				continue
			}
			if strings.HasPrefix(line, "/") {
				if app.command(line) {
					return ExitOK
				}
				continue
			}
			app.runTurn(line)
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
		app.save(app.SessionPath)
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
		fmt.Fprintf(app.Errw, "provider=%s model=%s base-url=%s\n", app.Provider, app.Model, app.BaseURL)
	default:
		fmt.Fprintf(app.Errw, "unknown command %q; type /help\n", cmd)
	}
	return false
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
	app.save(app.SessionPath)
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
	app.save(app.SessionPath)
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
	if usd, known := llm.Cost(app.Model, u.Usage); known {
		app.usage.CostUSD += usd
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

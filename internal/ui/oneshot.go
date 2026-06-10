package ui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"harness/internal/agent"
)

// Exit codes for one-shot mode (design §10).
const (
	ExitOK        = 0
	ExitRuntime   = 1
	ExitUsage     = 2
	ExitInterrupt = 130
)

// OneShot runs exactly one user turn, saves the session, and exits (design §10).
// Assistant text streams to app.Out; tool summaries, the usage line, notices,
// and errors go to app.Errw. The return value is the process exit code:
// 0 completed, 1 runtime error, 130 interrupted.
func OneShot(app *App, prompt string) int {
	if app.Created.IsZero() {
		app.Created = app.clock()()
	}

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

	// Save before deciding the exit code so a session is never lost (design §11).
	app.save(app.SessionPath)

	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return ExitInterrupt
		}
		fmt.Fprintf(app.Errw, "[error: %v]\n", err)
		return ExitRuntime
	}
	return ExitOK
}

// BuildPrompt assembles the one-shot prompt from the -p flag value and optional
// stdin (design §10). When flagText is "-", the whole prompt is read from stdin.
// Otherwise, when readStdin is set (piped input), stdin is appended after the
// flag text so `harness -p "summarize:" < notes.txt` works; with no stdin the
// flag text is used verbatim.
func BuildPrompt(flagText string, stdin io.Reader, readStdin bool) (string, error) {
	if flagText == "-" {
		data, err := readAll(stdin)
		if err != nil {
			return "", err
		}
		return strings.TrimRight(data, "\n"), nil
	}
	if !readStdin || stdin == nil {
		return flagText, nil
	}
	data, err := readAll(stdin)
	if err != nil {
		return "", err
	}
	piped := strings.TrimRight(data, "\n")
	if flagText == "" {
		return piped, nil
	}
	if piped == "" {
		return flagText, nil
	}
	return flagText + "\n" + piped, nil
}

func readAll(r io.Reader) (string, error) {
	if r == nil {
		return "", nil
	}
	b, err := io.ReadAll(r)
	return string(b), err
}

// ensure the accumulating sink stays an agent.EventSink (compile-time guard).
var _ agent.EventSink = (*accumulatingSink)(nil)

package tools

import (
	"context"
	"encoding/json"
	"os/exec"
)

type ripgrep struct {
	program string
}

func ripgrepProgram() (string, bool) {
	program, err := exec.LookPath("rg")
	if err != nil {
		return "", false
	}
	return program, true
}

func newRipgrep() (ripgrep, bool) {
	program, ok := ripgrepProgram()
	if !ok {
		return ripgrep{}, false
	}
	return ripgrep{program: program}, true
}

// RipgrepAvailable reports whether the optional rg tool can be registered from
// the current PATH.
func RipgrepAvailable() bool {
	_, ok := ripgrepProgram()
	return ok
}

func (ripgrep) Name() string { return "rg" }

func (ripgrep) Description() string {
	return `Run the host rg (ripgrep) command directly. Pass ripgrep options and operands as args, e.g. ["-n","TODO","."]. No shell; returns combined stdout+stderr and the exit code.`
}

func (ripgrep) Schema() json.RawMessage { return json.RawMessage(searchCommandSchema) }

func (ripgrep) ReadOnly() bool { return true }

func (r ripgrep) Run(ctx context.Context, input json.RawMessage) (string, error) {
	return runSearchCommand(ctx, input, "rg", r.program)
}

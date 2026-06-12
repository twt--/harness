package tools

import (
	"context"
	"encoding/json"
)

type grep struct{}

func (grep) Name() string { return "grep" }

func (grep) Description() string {
	return `Run the host grep command directly. Provide a JSON object with an args array, e.g. {"args":["-R","-n","TODO","."]}. No shell; returns combined stdout+stderr and the exit code.`
}

func (grep) Schema() json.RawMessage { return json.RawMessage(searchCommandSchema) }

func (grep) ReadOnly() bool { return true }

func (grep) Run(ctx context.Context, input json.RawMessage) (string, error) {
	return runSearchCommand(ctx, input, "grep", "grep")
}

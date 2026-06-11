// Package mode defines run modes: named bundles of an allowed-tool set and
// extra system-prompt instructions. Three built-in modes ship with the harness
// (auto, plan, independent); config-file entries field-level merge onto them
// (an omitted field keeps the built-in value) or define new modes. The mode
// prompt is appended to the system prompt as a final section; the tool list is
// realized by main via tools.Registry.Subset, which gates both what the model
// sees and what dispatches.
package mode

import (
	"maps"
	"slices"

	"harness/internal/tools"
)

// Default is the mode used when none is specified anywhere.
const Default = "auto"

// Mode is one resolved run mode. AllowedTools is always explicit after
// Builtins/Resolve (never empty), so callers need no nil special case. The
// struct is deliberately small; future per-mode knobs (e.g. max_steps) can be
// added alongside without changing the merge contract.
type Mode struct {
	Name         string
	AllowedTools []string
	Prompt       string
}

// FileMode mirrors one entry of the config file's "modes" object. Empty fields
// drive the field-level merge: they inherit from the same-named built-in, or
// for new modes from the defaults (default tool set, no prompt).
type FileMode struct {
	AllowedTools []string `json:"allowed_tools"`
	Prompt       string   `json:"prompt"`
}

const independentPrompt = `You are running in independent mode. Complete the requested task end to end without pausing to ask the user for input, confirmation, or clarification. Make reasonable assumptions, proceed, and note any assumptions in your final report. Only stop early for a hard blocker you cannot resolve with the available tools, or when you reach the step limit. Do the work, verify it, then report what you did.`

const planPrompt = `You are running in plan mode. Collaborate with the user to investigate the codebase and design an implementation plan; do not modify the project. You have read-only tools, git_readonly for history, and write_tmp_file for drafting notes to scratch files. Read the relevant code before proposing changes, ask clarifying questions when requirements are ambiguous, and present a concrete, ordered plan naming the exact files and changes involved. When the plan is ready, summarize it for the user rather than executing it.`

// Builtins returns fresh copies of the three built-in modes keyed by name.
func Builtins() map[string]Mode {
	return map[string]Mode{
		"auto": {
			Name:         "auto",
			AllowedTools: tools.DefaultNames(),
		},
		"independent": {
			Name:         "independent",
			AllowedTools: tools.DefaultNames(),
			Prompt:       independentPrompt,
		},
		"plan": {
			Name:         "plan",
			AllowedTools: []string{"read_file", "list_dir", "grep", "web_fetch", "git_readonly", "write_tmp_file"},
			Prompt:       planPrompt,
		},
	}
}

// Resolve merges config-file mode entries onto the built-ins and returns the
// full mode set. Merge is field-level: a non-empty field replaces, an empty
// field inherits (from the built-in of the same name, or from the defaults for
// a new mode).
func Resolve(file map[string]FileMode) map[string]Mode {
	modes := Builtins()
	for name, fm := range file {
		m, ok := modes[name]
		if !ok {
			m = Mode{Name: name, AllowedTools: tools.DefaultNames()}
		}
		if len(fm.AllowedTools) > 0 {
			m.AllowedTools = slices.Clone(fm.AllowedTools)
		}
		if fm.Prompt != "" {
			m.Prompt = fm.Prompt
		}
		modes[name] = m
	}
	return modes
}

// Names returns the mode names in sorted order, for listing and error text.
func Names(modes map[string]Mode) []string {
	return slices.Sorted(maps.Keys(modes))
}

// Package agentdef defines named agent definitions: bundles of an allowed-tool
// set, optional provider/model overrides, and extra system-prompt instructions.
// Three built-ins ship with the harness (auto, plan, independent); config-file
// entries field-level merge onto them (an omitted field keeps the built-in
// value) or define new agents. The agent prompt is appended to the system prompt
// as a final section; the tool list is realized by main via tools.Registry.Subset,
// which gates both what the model sees and what dispatches.
package agentdef

import (
	"maps"
	"slices"

	"harness/internal/tools"
	"harness/prompts"
)

// Default is the agent used when none is specified anywhere.
const Default = "auto"

// Definition is one resolved agent definition. AllowedTools is always explicit after
// Builtins/Resolve (never empty), so callers need no nil special case. The
// struct is deliberately small; future per-agent knobs (e.g. max_turns) can be
// added alongside without changing the merge contract.
type Definition struct {
	Name         string
	Description  string
	AllowedTools []string
	Prompt       string
	Provider     string
	Model        string
}

// FileDefinition mirrors one entry of the config file's "agents" object. Empty fields
// drive the field-level merge: they inherit from the same-named built-in, or
// for new agents from the defaults (default tool set, no prompt).
type FileDefinition struct {
	Description  string   `json:"description"`
	AllowedTools []string `json:"allowed_tools"`
	Prompt       string   `json:"prompt"`
	Provider     string   `json:"provider"`
	Model        string   `json:"model"`
}

// Builtins returns fresh copies of the three built-in agents keyed by name.
func Builtins() map[string]Definition {
	independentPrompt, _ := prompts.BuiltinAgentPrompt("independent")
	planPrompt, _ := prompts.BuiltinAgentPrompt("plan")
	return map[string]Definition{
		"auto": {
			Name:         "auto",
			Description:  "Default agent; the model decides what to do.",
			AllowedTools: defaultTools(),
		},
		"independent": {
			Name:         "independent",
			Description:  "Complete the task end to end without pausing for input.",
			AllowedTools: defaultTools(),
			Prompt:       independentPrompt,
		},
		"plan": {
			Name:         "plan",
			Description:  "Collaborate on an implementation plan without modifying the project.",
			AllowedTools: planTools(),
			Prompt:       planPrompt,
		},
	}
}

func planTools() []string {
	names := []string{"read_file", "list_dir", "grep"}
	if tools.RipgrepAvailable() {
		names = append(names, "rg")
	}
	names = append(names, "web_fetch")
	if tools.GitAvailable() {
		names = append(names, "git_readonly")
	}
	return append(names, "write_tmp_file", "delegate")
}

func defaultTools() []string {
	return append(tools.DefaultNames(), "delegate")
}

// DefaultTools returns the default allowed-tool set (the built-in tool names
// plus delegate) that auto/independent and any config agent without an explicit
// allowed_tools list inherit. main uses it to detect default-inheriting agents
// when extending them with discovered MCP tools.
func DefaultTools() []string { return defaultTools() }

// Resolve merges config-file agent entries onto the built-ins and returns the
// full agent set. Merge is field-level: a non-empty field replaces, an empty
// field inherits (from the built-in of the same name, or from the defaults for
// a new agent).
func Resolve(file map[string]FileDefinition) map[string]Definition {
	agents := Builtins()
	for name, fm := range file {
		a, ok := agents[name]
		if !ok {
			a = Definition{Name: name, AllowedTools: defaultTools()}
		}
		if fm.Description != "" {
			a.Description = fm.Description
		}
		if len(fm.AllowedTools) > 0 {
			a.AllowedTools = slices.Clone(fm.AllowedTools)
		}
		if fm.Prompt != "" {
			a.Prompt = fm.Prompt
		}
		if fm.Provider != "" {
			a.Provider = fm.Provider
		}
		if fm.Model != "" {
			a.Model = fm.Model
		}
		agents[name] = a
	}
	return agents
}

// Names returns the agent names in sorted order, for listing and error text.
func Names(agents map[string]Definition) []string {
	return slices.Sorted(maps.Keys(agents))
}

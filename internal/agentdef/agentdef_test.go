package agentdef

import (
	"slices"
	"testing"
)

func TestDefaultIsAuto(t *testing.T) {
	if Default != "auto" {
		t.Errorf("Default = %q, want \"auto\"", Default)
	}
}

func TestBuiltins(t *testing.T) {
	m := Builtins()
	if len(m) != 3 {
		t.Fatalf("want 3 builtin agents, got %d: %v", len(m), Names(m))
	}
	for name, a := range m {
		if a.Name != name {
			t.Errorf("agent %q has Name %q", name, a.Name)
		}
		if a.Description == "" {
			t.Errorf("agent %q has empty description", name)
		}
	}

	auto := m["auto"]
	if auto.Prompt != "" {
		t.Errorf("auto must have no extra prompt (current behavior), got %q", auto.Prompt)
	}
	if !slices.Equal(auto.AllowedTools, defaultTools()) {
		t.Errorf("auto tools = %v, want default set", auto.AllowedTools)
	}

	ind := m["independent"]
	if ind.Prompt == "" {
		t.Error("independent must carry a prompt")
	}
	if !slices.Equal(ind.AllowedTools, defaultTools()) {
		t.Errorf("independent tools = %v, want default set", ind.AllowedTools)
	}

	plan := m["plan"]
	if plan.Prompt == "" {
		t.Error("plan must carry a prompt")
	}
	wantPlan := planTools()
	if !slices.Equal(plan.AllowedTools, wantPlan) {
		t.Errorf("plan tools = %v, want %v", plan.AllowedTools, wantPlan)
	}
}

func TestResolveNilKeepsBuiltins(t *testing.T) {
	m := Resolve(nil)
	if !slices.Equal(Names(m), []string{"auto", "independent", "plan"}) {
		t.Errorf("Names = %v", Names(m))
	}
}

func TestPlanAgentOmitsGitReadonlyWhenGitMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	plan := Builtins()["plan"]
	if slices.Contains(plan.AllowedTools, "git_readonly") {
		t.Fatalf("plan agent includes unavailable git_readonly: %v", plan.AllowedTools)
	}
}

// Field-level merge: overriding only the prompt keeps the built-in tool list.
func TestResolvePromptOnlyOverrideKeepsTools(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"plan": {Prompt: "custom plan prompt"}})
	plan := m["plan"]
	if plan.Prompt != "custom plan prompt" {
		t.Errorf("Prompt = %q", plan.Prompt)
	}
	if !slices.Equal(plan.AllowedTools, Builtins()["plan"].AllowedTools) {
		t.Errorf("tool list not preserved: %v", plan.AllowedTools)
	}
}

// Field-level merge: overriding only the tools keeps the built-in prompt.
func TestResolveToolsOnlyOverrideKeepsPrompt(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"plan": {AllowedTools: []string{"read_file"}}})
	plan := m["plan"]
	if !slices.Equal(plan.AllowedTools, []string{"read_file"}) {
		t.Errorf("tools = %v", plan.AllowedTools)
	}
	if plan.Prompt != Builtins()["plan"].Prompt {
		t.Errorf("prompt not preserved: %q", plan.Prompt)
	}
}

func TestResolveMetadataOverrideKeepsOtherFields(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"plan": {
		Description: "custom desc",
		Provider:    "openai",
		Model:       "gpt-5.5",
	}})
	plan := m["plan"]
	if plan.Description != "custom desc" || plan.Provider != "openai" || plan.Model != "gpt-5.5" {
		t.Fatalf("metadata = description %q provider %q model %q", plan.Description, plan.Provider, plan.Model)
	}
	if plan.Prompt != Builtins()["plan"].Prompt {
		t.Errorf("prompt not preserved: %q", plan.Prompt)
	}
	if !slices.Equal(plan.AllowedTools, Builtins()["plan"].AllowedTools) {
		t.Errorf("tool list not preserved: %v", plan.AllowedTools)
	}
}

// A new agent with no allowed_tools inherits the default tool set.
func TestResolveNewAgentInheritsDefaultTools(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"review": {Prompt: "review the diff"}})
	rev, ok := m["review"]
	if !ok {
		t.Fatal("new agent not resolved")
	}
	if rev.Name != "review" || rev.Prompt != "review the diff" {
		t.Errorf("rev = %+v", rev)
	}
	if !slices.Equal(rev.AllowedTools, defaultTools()) {
		t.Errorf("tools = %v, want default set", rev.AllowedTools)
	}
}

func TestResolveNewAgentWithExplicitTools(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"ro": {AllowedTools: []string{"read_file", "grep"}}})
	ro := m["ro"]
	if !slices.Equal(ro.AllowedTools, []string{"read_file", "grep"}) {
		t.Errorf("tools = %v", ro.AllowedTools)
	}
	if ro.Prompt != "" {
		t.Errorf("prompt = %q, want empty", ro.Prompt)
	}
}

func TestNamesSorted(t *testing.T) {
	m := Resolve(map[string]FileDefinition{"zz": {}, "aa": {}})
	if got := Names(m); !slices.Equal(got, []string{"aa", "auto", "independent", "plan", "zz"}) {
		t.Errorf("Names = %v", got)
	}
}

package responses

import (
	"encoding/json"

	"harness/internal/llm"
)

// errorResultPrefix marks a failed tool result. Responses function_call_output
// items have no is_error field, so error results carry this prefix in output.
const errorResultPrefix = "ERROR: "

// emptyArgs is the canonical serialization for a tool call with no arguments.
const emptyArgs = "{}"

// wireRequest is the OpenAI Responses request body. Store is always sent false
// so harness remains stateless and resends its own transcript every step.
type wireRequest struct {
	Model           string          `json:"model"`
	Instructions    string          `json:"instructions,omitempty"`
	Input           []wireInputItem `json:"input"`
	Tools           []wireTool      `json:"tools,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	Reasoning       *wireReasoning  `json:"reasoning,omitempty"`
	Stream          bool            `json:"stream"`
	Store           bool            `json:"store"`
}

type wireReasoning struct {
	Effort string `json:"effort,omitempty"`
}

// wireInputItem covers the input item subset harness needs: messages, prior
// function calls, and function-call outputs.
type wireInputItem struct {
	Type string `json:"type"`

	// message
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`

	// function_call / function_call_output
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

type wireTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict"`
}

// --- streaming event wire structs ---

type wireEvent struct {
	Type string `json:"type"`

	// response.output_text.delta / response.function_call_arguments.delta
	Delta string `json:"delta"`

	// response.function_call_arguments.done
	Arguments string `json:"arguments"`

	// shared output item addressing
	ItemID      string `json:"item_id"`
	OutputIndex int    `json:"output_index"`
	Name        string `json:"name"`

	// response.output_item.added / response.output_item.done
	Item *wireOutputItem `json:"item"`

	// response.completed / response.failed / response.incomplete
	Response *wireResponse `json:"response"`

	// error
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param"`
}

type wireOutputItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}

type wireResponse struct {
	Status            string             `json:"status"`
	Error             *wireResponseError `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage  *wireUsage       `json:"usage"`
	Output []wireOutputItem `json:"output"`
}

type wireResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type wireUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

func buildRequest(req llm.Request) wireRequest {
	w := wireRequest{
		Model:        req.Model,
		Instructions: req.System,
		Input:        buildInput(req.Messages),
		Stream:       true,
		Store:        false,
		Temperature:  req.Temperature,
	}

	if req.MaxTokens > 0 {
		mt := req.MaxTokens
		w.MaxOutputTokens = &mt
	}
	if req.Reasoning.Effort != "" {
		w.Reasoning = &wireReasoning{Effort: req.Reasoning.Effort}
	}

	for _, t := range req.Tools {
		w.Tools = append(w.Tools, wireTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
			Strict:      false,
		})
	}

	return w
}

func buildInput(messages []llm.Message) []wireInputItem {
	var out []wireInputItem
	for _, m := range messages {
		var text string
		flushText := func() {
			if text == "" {
				return
			}
			out = append(out, wireInputItem{
				Type:    "message",
				Role:    string(m.Role),
				Content: text,
			})
			text = ""
		}

		for _, b := range m.Content {
			switch b.Kind {
			case llm.BlockText:
				text += b.Text
			case llm.BlockToolUse:
				flushText()
				args := string(b.ToolInput)
				if args == "" {
					args = emptyArgs
				}
				out = append(out, wireInputItem{
					Type:      "function_call",
					CallID:    b.ToolUseID,
					Name:      b.ToolName,
					Arguments: args,
				})
			case llm.BlockToolResult:
				flushText()
				output := b.ResultText
				if b.ResultError {
					output = errorResultPrefix + output
				}
				out = append(out, wireInputItem{
					Type:   "function_call_output",
					CallID: b.ResultForID,
					Output: output,
				})
			}
		}
		flushText()
	}
	return out
}

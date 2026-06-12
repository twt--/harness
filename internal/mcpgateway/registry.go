package mcpgateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"slices"
	"strconv"
	"sync"

	"harness/internal/logging"
	"harness/internal/mcp"
	"harness/internal/mcp/jsonrpc"
)

// pageSize is the tools/list page size. Most setups fit one page; pagination is
// implemented because the MCP spec requires cursor support and the aggregate can
// grow large.
const pageSize = 100

// qualifiedPrefix is the namespace prefix for every aggregated tool.
const qualifiedPrefix = "mcp__"

// route records where a qualified tool name dispatches.
type route struct {
	supervisor *Supervisor
	bareName   string
}

// Registry aggregates the tools of every supervised server into one namespaced
// surface and implements mcp.ToolProvider for the gateway's server sessions. It
// maintains a merged sorted tool list (names rewritten to mcp__<server>__<tool>)
// and a reverse route map, both rebuilt whenever any supervisor's tools change.
// It fans tools/list_changed out to subscribed sessions.
type Registry struct {
	supervisors []*Supervisor
	logger      *slog.Logger

	mu     sync.RWMutex
	tools  []mcp.Tool       // merged, namespaced, sorted by name
	routes map[string]route // qualified name -> dispatch target

	sessions map[*mcp.ServerSession]struct{}
}

// NewRegistry builds a registry over servers, injecting an onToolsChanged
// callback into each so a supervisor's tool change rebuilds the table and
// broadcasts list_changed. It builds the initial table immediately.
func NewRegistry(servers []*Supervisor, logger *slog.Logger) *Registry {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	r := &Registry{
		supervisors: servers,
		logger:      logger,
		routes:      map[string]route{},
		sessions:    map[*mcp.ServerSession]struct{}{},
	}
	for _, s := range servers {
		s.onToolsChanged = r.onSupervisorToolsChanged
	}
	r.rebuild()
	return r
}

// onSupervisorToolsChanged is the callback injected into each supervisor. It
// rebuilds the aggregated table then broadcasts list_changed to subscribers.
func (r *Registry) onSupervisorToolsChanged() {
	r.rebuild()
	r.BroadcastListChanged()
}

// rebuild recomputes the merged namespaced tool list and the reverse route map
// under the write lock. Tools whose qualified name is not provider-safe are
// dropped with a warning (never rewritten/truncated — a truncated name could
// collide and break routing).
func (r *Registry) rebuild() {
	var tools []mcp.Tool
	routes := map[string]route{}

	for _, sup := range r.supervisors {
		for _, t := range sup.Tools() {
			qualified := qualifiedPrefix + sup.Name() + "__" + t.Name
			if !serverNameRE.MatchString(qualified) {
				r.logger.Warn("tool omitted: qualified name not provider-safe",
					logging.Category(categoryGate), "server", sup.Name(), "tool", t.Name, "qualified", qualified)
				continue
			}
			if _, dup := routes[qualified]; dup {
				// Structurally impossible given distinct server names and unique
				// per-server tool names, but guard defensively.
				r.logger.Warn("tool omitted: duplicate qualified name",
					logging.Category(categoryGate), "qualified", qualified)
				continue
			}
			nt := t
			nt.Name = qualified
			tools = append(tools, nt)
			routes[qualified] = route{supervisor: sup, bareName: t.Name}
		}
	}

	slices.SortFunc(tools, func(a, b mcp.Tool) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		default:
			return 0
		}
	})

	r.mu.Lock()
	r.tools = tools
	r.routes = routes
	r.mu.Unlock()
}

// ListTools returns one page of the merged namespaced list, paginated by an
// opaque base64 cursor (the next index). An invalid cursor is a CodeInvalidParams
// jsonrpc error.
func (r *Registry) ListTools(ctx context.Context, cursor string) (mcp.ListToolsResult, error) {
	start, err := decodeCursor(cursor)
	if err != nil {
		return mcp.ListToolsResult{}, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "invalid cursor")
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if start > len(r.tools) {
		return mcp.ListToolsResult{}, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "invalid cursor")
	}

	end := min(start+pageSize, len(r.tools))
	page := slices.Clone(r.tools[start:end])
	result := mcp.ListToolsResult{Tools: page}
	if end < len(r.tools) {
		result.NextCursor = encodeCursor(end)
	}
	return result, nil
}

// CallTool routes a qualified call to the owning supervisor and returns its
// result verbatim. An unknown tool is a CodeInvalidParams jsonrpc error (the
// model should not have called it); a known tool's result (including IsError) is
// passed straight through.
func (r *Registry) CallTool(ctx context.Context, qualified string, args json.RawMessage) (*mcp.CallToolResult, error) {
	sup, bare, ok := r.route(qualified)
	if !ok {
		return nil, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "unknown tool: %s", qualified)
	}
	return sup.CallTool(ctx, bare, args)
}

// route resolves a qualified name to its supervisor and bare tool name via the
// reverse map (no string-splitting, so server names may contain underscores).
func (r *Registry) route(qualified string) (*Supervisor, string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	rt, ok := r.routes[qualified]
	if !ok {
		return nil, "", false
	}
	return rt.supervisor, rt.bareName, true
}

// Subscribe registers a session for tools/list_changed fan-out.
func (r *Registry) Subscribe(s *mcp.ServerSession) {
	r.mu.Lock()
	r.sessions[s] = struct{}{}
	r.mu.Unlock()
}

// Unsubscribe stops fan-out to a session (called when its Serve returns).
func (r *Registry) Unsubscribe(s *mcp.ServerSession) {
	r.mu.Lock()
	delete(r.sessions, s)
	r.mu.Unlock()
}

// BroadcastListChanged notifies every subscribed session that the tool list
// changed. Dead sessions (whose Done channel is closed) are dropped. Notify is
// peer-buffered, so a slow session does not block the fan-out.
func (r *Registry) BroadcastListChanged() {
	r.mu.RLock()
	sessions := make([]*mcp.ServerSession, 0, len(r.sessions))
	for s := range r.sessions {
		sessions = append(sessions, s)
	}
	r.mu.RUnlock()

	var dead []*mcp.ServerSession
	for _, s := range sessions {
		select {
		case <-s.Done():
			dead = append(dead, s)
			continue
		default:
		}
		if err := s.NotifyToolsListChanged(); err != nil {
			dead = append(dead, s)
		}
	}

	if len(dead) > 0 {
		r.mu.Lock()
		for _, s := range dead {
			delete(r.sessions, s)
		}
		r.mu.Unlock()
	}
}

// encodeCursor encodes a next-index as an opaque base64 string.
func encodeCursor(idx int) string {
	return base64.StdEncoding.EncodeToString([]byte(strconv.Itoa(idx)))
}

// decodeCursor decodes a cursor to its index. An empty cursor is index 0. A
// non-base64 or non-numeric/negative cursor is an error.
func decodeCursor(cursor string) (int, error) {
	if cursor == "" {
		return 0, nil
	}
	raw, err := base64.StdEncoding.DecodeString(cursor)
	if err != nil {
		return 0, err
	}
	idx, err := strconv.Atoi(string(raw))
	if err != nil {
		return 0, err
	}
	if idx < 0 {
		return 0, strconv.ErrRange
	}
	return idx, nil
}

package mcp

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"harness/internal/mcp/jsonrpc"
)

// httpServerMaxBodyBytes bounds a POST body. It matches the client's 4 MB cap
// (httpMaxBodyBytes) so a request the client is willing to send is one the
// server is willing to read; a larger body is rejected rather than read.
const httpServerMaxBodyBytes = 4 << 20

// sessionIDBytes is the entropy of a generated session id: 16 bytes = 128 bits,
// hex-encoded to 32 visible-ASCII characters (0x21–0x7E per spec).
const sessionIDBytes = 16

// sessionIdleTTL bounds how long a session may sit idle before it is purged. It
// is enforced lazily on each request (no background goroutine): a request that
// finds its session older than this is treated as expired (404), and any other
// request opportunistically sweeps all stale sessions. 30 minutes mirrors a
// generous interactive-client idle window.
const sessionIdleTTL = 30 * time.Minute

// mcpSessionHeader and mcpProtocolHeader are the streamable-HTTP control
// headers. Go canonicalizes header keys, so these match the client's casing.
const (
	mcpSessionHeader  = "Mcp-Session-Id"
	mcpProtocolHeader = "MCP-Protocol-Version"
)

// HTTPHandlerOptions configures an HTTP MCP handler.
type HTTPHandlerOptions struct {
	Info     Implementation
	Provider ToolProvider
	Logger   *slog.Logger

	// now is the injectable clock for idle-expiry tests; nil → time.Now.
	now func() time.Time
}

// NewHTTPHandler returns a spec-conforming streamable-HTTP MCP server handler
// (spec revision 2025-06-18) for a tools-only provider. It interoperates with
// the HTTPTransport client in this package.
//
// Responses are ALWAYS application/json: this server never replies to a POST
// with text/event-stream and never offers a GET stream. There is therefore no
// server-push channel, so tools/list_changed is not advertised (ListChanged is
// reported false in initialize) and a client that cares about tool changes must
// re-list. The client accepts JSON responses (it sends the dual Accept header
// but handles application/json), so this is fully interoperable.
//
// Sessions are created on initialize, carried via the Mcp-Session-Id header,
// and purged lazily after sessionIdleTTL of inactivity.
func NewHTTPHandler(opts HTTPHandlerOptions) http.Handler {
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	now := opts.now
	if now == nil {
		now = time.Now
	}
	return &httpHandler{
		info:     opts.Info,
		provider: opts.Provider,
		logger:   logger,
		now:      now,
		sessions: make(map[string]*httpSession),
	}
}

// httpSession is the per-session server state: when it was last touched (for
// idle expiry) and the in-flight tools/call cancellations keyed by canonical id
// (mirroring server.go), so a notifications/cancelled in a later POST can cancel
// a call still running from an earlier POST.
type httpSession struct {
	mu       sync.Mutex
	lastSeen time.Time
	inflight map[string]context.CancelFunc
}

// httpHandler implements http.Handler for the streamable-HTTP server side.
type httpHandler struct {
	info     Implementation
	provider ToolProvider
	logger   *slog.Logger
	now      func() time.Time

	mu       sync.Mutex
	sessions map[string]*httpSession
}

func (h *httpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		h.handlePost(w, r)
	case http.MethodDelete:
		h.handleDelete(w, r)
	case http.MethodGet:
		// No server-initiated stream in v1: a tools-only client treats 405 as
		// "no server-push offered" and proceeds over POST alone.
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handlePost decodes the single JSON-RPC message in the body and routes it. A
// malformed body or a top-level array (batching removed) is a 400 with a
// JSON-RPC parse-error envelope.
func (h *httpHandler) handlePost(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, httpServerMaxBodyBytes+1))
	if err != nil {
		h.writeParseError(w, "read request body")
		return
	}
	if len(body) > httpServerMaxBodyBytes {
		h.writeParseError(w, "request body exceeds size limit")
		return
	}

	// Reject a top-level array up front: batching was removed from MCP, and the
	// jsonrpc.Message decode below would otherwise fail with a less specific
	// error. Skip leading JSON whitespace before sniffing the first token.
	if trimmed := trimJSONLeadingSpace(body); len(trimmed) > 0 && trimmed[0] == '[' {
		h.writeParseError(w, "batch arrays are not supported")
		return
	}

	var msg jsonrpc.Message
	if err := json.Unmarshal(body, &msg); err != nil {
		h.writeParseError(w, "malformed JSON")
		return
	}

	switch msg.Kind() {
	case jsonrpc.KindRequest:
		h.handleRequest(w, r, msg)
	case jsonrpc.KindNotification:
		h.handleNotification(w, r, msg)
	default:
		// A response or an invalid envelope is not something a tools-only server
		// expects from a client. Per spec a non-request body gets 202 with no
		// body; we treat an invalid envelope the same way rather than erroring.
		w.WriteHeader(http.StatusAccepted)
	}
}

// handleRequest dispatches a single JSON-RPC request. initialize creates a
// session; every other method requires a live session.
func (h *httpHandler) handleRequest(w http.ResponseWriter, r *http.Request, msg jsonrpc.Message) {
	id := *msg.ID

	if msg.Method == MethodInitialize {
		h.handleInitialize(w, id, msg.Params)
		return
	}

	sess, status := h.authorize(r)
	if status != 0 {
		http.Error(w, http.StatusText(status), status)
		return
	}

	switch msg.Method {
	case MethodPing:
		h.writeResult(w, id, json.RawMessage(`{}`))
	case MethodListTools:
		h.handleListTools(w, r, id, msg.Params)
	case MethodCallTool:
		h.handleCallTool(w, r, sess, id, msg.Params)
	default:
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeMethodNotFound, "method not found: %s", msg.Method))
	}
}

// handleNotification acknowledges any notification with 202 and an empty body.
// notifications/cancelled cancels a matching in-flight call in the session;
// every other notification (initialized included) is a no-op ack.
func (h *httpHandler) handleNotification(w http.ResponseWriter, r *http.Request, msg jsonrpc.Message) {
	// A cancellation must target a live session to find its in-flight map; an
	// unknown/expired session is a 404 so the client re-initializes. Any other
	// authorize failure on a notification (missing session header, or an
	// unsupported protocol-version header) is tolerated as a 202: a notification
	// is fire-and-forget, so a best-effort cancel that cannot resolve a session
	// is simply a no-op rather than an error the client must handle.
	if msg.Method == NotifCancelled {
		if sess, status := h.authorize(r); status == 0 {
			h.cancelInflight(sess, msg.Params)
		} else if status == http.StatusNotFound {
			http.Error(w, http.StatusText(status), status)
			return
		}
	}
	w.WriteHeader(http.StatusAccepted)
}

// handleInitialize negotiates the protocol version, creates a session, and
// returns InitializeResult with the session id in the Mcp-Session-Id header.
// ListChanged is always false: there is no server-push channel over HTTP.
func (h *httpHandler) handleInitialize(w http.ResponseWriter, id jsonrpc.ID, params json.RawMessage) {
	var p InitializeParams
	if err := json.Unmarshal(params, &p); err != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "invalid initialize params: %v", err))
		return
	}

	// Deliberate: no cap on session count. An unauthenticated initialize can mint
	// sessions; we rely on the local-trust/front-proxy boundary plus the 30min
	// idle TTL (sessionIdleTTL) to bound accumulation rather than a hard limit.

	// Negotiate: echo the client's version if we support it, else offer ours.
	// The client surfaces a server-selected version it cannot speak as a
	// VersionError, so offering ours is the correct downgrade signal.
	version := ProtocolVersion
	if Supports(p.ProtocolVersion) {
		version = p.ProtocolVersion
	}

	sessionID, err := newSessionID()
	if err != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInternal, "create session: %v", err))
		return
	}
	h.mu.Lock()
	h.sessions[sessionID] = &httpSession{
		lastSeen: h.now(),
		inflight: make(map[string]context.CancelFunc),
	}
	h.mu.Unlock()

	result := InitializeResult{
		ProtocolVersion: version,
		Capabilities: ServerCapabilities{
			Tools: &ToolsCapability{ListChanged: false},
		},
		ServerInfo: h.info,
	}
	raw, err := json.Marshal(result)
	if err != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInternal, "marshal initialize result: %v", err))
		return
	}
	w.Header().Set(mcpSessionHeader, sessionID)
	h.writeResult(w, id, raw)
}

func (h *httpHandler) handleListTools(w http.ResponseWriter, r *http.Request, id jsonrpc.ID, params json.RawMessage) {
	var p ListToolsParams
	if len(params) > 0 {
		if err := json.Unmarshal(params, &p); err != nil {
			h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "invalid tools/list params: %v", err))
			return
		}
	}
	result, err := h.provider.ListTools(r.Context(), p.Cursor)
	if err != nil {
		h.writeProviderError(w, id, err, "list tools")
		return
	}
	raw, err := json.Marshal(result)
	if err != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInternal, "marshal tools/list result: %v", err))
		return
	}
	h.writeResult(w, id, raw)
}

// handleCallTool runs one tools/call. The provider call's context derives from
// the request context AND a per-session cancellation entry keyed by the call id,
// so a notifications/cancelled in a concurrent POST can cancel it (mirroring
// server.go).
func (h *httpHandler) handleCallTool(w http.ResponseWriter, r *http.Request, sess *httpSession, id jsonrpc.ID, params json.RawMessage) {
	var p CallToolParams
	if err := json.Unmarshal(params, &p); err != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInvalidParams, "invalid tools/call params: %v", err))
		return
	}

	callCtx, cancel := context.WithCancel(r.Context())
	key := id.String()
	sess.mu.Lock()
	sess.inflight[key] = cancel
	sess.mu.Unlock()
	defer func() {
		sess.mu.Lock()
		delete(sess.inflight, key)
		sess.mu.Unlock()
		cancel()
	}()

	result, err := h.provider.CallTool(callCtx, p.Name, p.Arguments)
	if err != nil {
		h.writeProviderError(w, id, err, "call tool")
		return
	}
	raw, mErr := json.Marshal(result)
	if mErr != nil {
		h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInternal, "marshal tools/call result: %v", mErr))
		return
	}
	h.writeResult(w, id, raw)
}

// cancelInflight cancels the in-flight call matching a notifications/cancelled
// payload's requestId, if one is registered. Best-effort: a cancel that races
// ahead of the call registering its entry finds no match and is dropped, exactly
// as in server.go.
func (h *httpHandler) cancelInflight(sess *httpSession, params json.RawMessage) {
	var p CancelledParams
	if err := json.Unmarshal(params, &p); err != nil {
		return
	}
	key, ok := canonicalIDKey(p.RequestID)
	if !ok {
		return
	}
	sess.mu.Lock()
	cancel := sess.inflight[key]
	sess.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// authorize validates the session and protocol-version headers for a
// non-initialize request. It returns the live session and status 0 on success,
// or 0-session with an HTTP status to send: 400 for a missing session header or
// an unsupported protocol version, 404 for an unknown/expired session. It also
// lazily purges all stale sessions. A side effect of success is bumping the
// session's lastSeen.
func (h *httpHandler) authorize(r *http.Request) (*httpSession, int) {
	// An MCP-Protocol-Version header, when present, must be one we support;
	// absent is tolerated (the spec lets the server assume a default).
	if v := r.Header.Get(mcpProtocolHeader); v != "" && !Supports(v) {
		return nil, http.StatusBadRequest
	}

	id := r.Header.Get(mcpSessionHeader)
	if id == "" {
		return nil, http.StatusBadRequest
	}

	now := h.now()
	h.mu.Lock()
	h.purgeStaleLocked(now)
	sess, ok := h.sessions[id]
	if ok {
		sess.lastSeen = now
	}
	h.mu.Unlock()
	if !ok {
		return nil, http.StatusNotFound
	}
	return sess, 0
}

// purgeStaleLocked drops every session idle longer than sessionIdleTTL. The
// caller holds h.mu. Reading lastSeen here without the per-session lock is safe:
// lastSeen is only written under h.mu (in authorize and initialize).
func (h *httpHandler) purgeStaleLocked(now time.Time) {
	for id, sess := range h.sessions {
		if now.Sub(sess.lastSeen) > sessionIdleTTL {
			delete(h.sessions, id)
		}
	}
}

// handleDelete terminates the session named by the Mcp-Session-Id header. A
// missing header is 400; an unknown/expired session is 404; a live session is
// removed and answered 204. Removing the session does NOT cancel its in-flight
// calls: each is still bound by its own POST request context, and explicit
// cancellation is the notifications/cancelled path's job.
func (h *httpHandler) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(mcpSessionHeader)
	if id == "" {
		http.Error(w, http.StatusText(http.StatusBadRequest), http.StatusBadRequest)
		return
	}
	h.mu.Lock()
	h.purgeStaleLocked(h.now())
	_, ok := h.sessions[id]
	if ok {
		delete(h.sessions, id)
	}
	h.mu.Unlock()
	if !ok {
		http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// writeProviderError maps a provider error onto a JSON-RPC error response with
// the same semantics as server.go: a *jsonrpc.Error becomes the error response
// verbatim; any other error is CodeInternal with the given context prefix.
func (h *httpHandler) writeProviderError(w http.ResponseWriter, id jsonrpc.ID, err error, what string) {
	var je *jsonrpc.Error
	if errors.As(err, &je) {
		h.writeError(w, id, je)
		return
	}
	h.writeError(w, id, jsonrpc.Errorf(jsonrpc.CodeInternal, "%s: %v", what, err))
}

// writeResult writes a JSON-RPC success response (200, application/json).
func (h *httpHandler) writeResult(w http.ResponseWriter, id jsonrpc.ID, result json.RawMessage) {
	h.writeMessage(w, http.StatusOK, jsonrpc.NewResponse(id, result))
}

// writeError writes a JSON-RPC error response carrying err. The HTTP status is
// 200: the error is in the JSON-RPC envelope, not the HTTP layer, matching the
// client's readJSONResult which decodes the body on a 2xx. (Header/transport
// failures, by contrast, use real HTTP status codes.)
func (h *httpHandler) writeError(w http.ResponseWriter, id jsonrpc.ID, err *jsonrpc.Error) {
	h.writeMessage(w, http.StatusOK, jsonrpc.NewErrorResponse(id, err))
}

// writeMessage marshals m and writes it as the application/json body with the
// given status. A marshal failure (not expected for these envelopes) degrades to
// a bare 500 so the client sees a transport error rather than a truncated body.
func (h *httpHandler) writeMessage(w http.ResponseWriter, status int, m jsonrpc.Message) {
	body, err := json.Marshal(m)
	if err != nil {
		h.logger.Error("mcp http: marshal response", "err", err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := w.Write(body); err != nil {
		h.logger.Debug("mcp http: write response", "err", err)
	}
}

// writeParseError writes a JSON-RPC parse-error envelope with a 400 status and a
// null id. JSON-RPC convention is that a parse error (the server could not even
// read the id) carries "id":null. The jsonrpc package's ID type rejects null on
// unmarshal and refuses to marshal an unset id, so the null-id envelope cannot
// be expressed through jsonrpc.Message; it is marshalled literally here. The
// detail is logged, not sent, to avoid leaking parser internals to the client.
func (h *httpHandler) writeParseError(w http.ResponseWriter, detail string) {
	h.logger.Debug("mcp http: rejecting POST body", "reason", detail)
	const body = `{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}`
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)
	_, _ = io.WriteString(w, body)
}

// newSessionID returns a fresh hex-encoded 128-bit session id (visible ASCII per
// spec).
func newSessionID() (string, error) {
	buf := make([]byte, sessionIDBytes)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}

// trimJSONLeadingSpace returns b without leading JSON whitespace (space, tab,
// newline, carriage return), so the array sniff sees the first real token.
func trimJSONLeadingSpace(b []byte) []byte {
	for len(b) > 0 {
		switch b[0] {
		case ' ', '\t', '\n', '\r':
			b = b[1:]
		default:
			return b
		}
	}
	return b
}

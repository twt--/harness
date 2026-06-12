package mcp

import (
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"strconv"
)

// maxUnixSocketPath bounds the usable bytes of a unix-domain socket path. The
// kernel sun_path array is 104 bytes on macOS (108 on Linux) including the
// trailing NUL; 103 is the safe usable ceiling on the tightest platform.
const maxUnixSocketPath = 103

// DefaultSocketPath resolves the cross-binary default unix socket path for the
// MCP gateway. It is a pure function of getenv and process identity (os.TempDir,
// os.Getuid, os.UserHomeDir); it performs no filesystem writes — directory
// creation and permission checks are the daemon's job. getenv injects the
// environment (use os.Getenv in production) so the resolution is testable.
//
// Resolution, in order:
//  1. $XDG_RUNTIME_DIR/harness-mcp-gateway/gateway.sock when XDG_RUNTIME_DIR is
//     set (the Linux happy path: a short, per-user, 0700 tmpfs runtime dir).
//  2. Otherwise <tmp>/harness-mcp-gateway-<uid>/gateway.sock, where <tmp> is
//     os.TempDir() and <uid> is os.Getuid(). The per-uid suffix keeps it unique
//     per user on a shared /tmp.
//  3. If the chosen path exceeds the sun_path ceiling (long $TMPDIR or
//     XDG_RUNTIME_DIR), fall back to a short hashed name in <tmp>:
//     hmg-<8 hex of an FNV hash of uid+home>.sock, which is bounded in length.
func DefaultSocketPath(getenv func(string) string) string {
	tmp := os.TempDir()
	uid := os.Getuid()
	home, _ := os.UserHomeDir()
	return defaultSocketPath(getenv, tmp, uid, home)
}

// defaultSocketPath is the injectable core of DefaultSocketPath: tmp and uid are
// parameters so tests can exercise the length-fallback without mutating process
// state (os.TempDir caches $TMPDIR at first use, so it cannot be reliably
// overridden via t.Setenv).
func defaultSocketPath(getenv func(string) string, tmp string, uid int, home string) string {
	var path string
	if rt := getenv("XDG_RUNTIME_DIR"); rt != "" {
		path = filepath.Join(rt, "harness-mcp-gateway", "gateway.sock")
	} else {
		path = filepath.Join(tmp, "harness-mcp-gateway-"+strconv.Itoa(uid), "gateway.sock")
	}
	if len(path) <= maxUnixSocketPath {
		return path
	}
	// Fall back to a short hashed name in tmp to stay under the sun_path limit.
	h := fnv.New32a()
	_, _ = h.Write([]byte(strconv.Itoa(uid) + "\x00" + home))
	return filepath.Join(tmp, fmt.Sprintf("hmg-%08x.sock", h.Sum32()))
}

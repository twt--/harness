package mcp

import (
	"strings"
	"testing"
)

func TestDefaultSocketPathXDGSet(t *testing.T) {
	getenv := func(k string) string {
		if k == "XDG_RUNTIME_DIR" {
			return "/run/user/1000"
		}
		return ""
	}
	got := defaultSocketPath(getenv, "/tmp", 1000, "/home/u")
	want := "/run/user/1000/harness-mcp-gateway/gateway.sock"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestDefaultSocketPathXDGUnset(t *testing.T) {
	getenv := func(string) string { return "" }
	got := defaultSocketPath(getenv, "/tmp", 501, "/Users/u")
	want := "/tmp/harness-mcp-gateway-501/gateway.sock"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestDefaultSocketPathLongTmpFallback(t *testing.T) {
	getenv := func(string) string { return "" }
	// A moderately long tmp dir pushes the per-uid path
	// (<tmp>/harness-mcp-gateway-<uid>/gateway.sock) past the sun_path ceiling
	// while the short hashed fallback (<tmp>/hmg-XXXXXXXX.sock) still fits.
	longTmp := "/" + strings.Repeat("d", 70)
	got := defaultSocketPath(getenv, longTmp, 501, "/Users/u")
	if len(got) > maxUnixSocketPath {
		t.Fatalf("fallback path still too long (%d): %q", len(got), got)
	}
	if !strings.HasPrefix(got, longTmp+"/hmg-") {
		t.Fatalf("fallback should live in tmp with hmg- prefix, got %q", got)
	}
	if !strings.HasSuffix(got, ".sock") {
		t.Fatalf("fallback should end in .sock, got %q", got)
	}
	// 8 hex chars between "hmg-" and ".sock".
	base := got[len(longTmp)+1:]
	if len(base) != len("hmg-")+8+len(".sock") {
		t.Fatalf("hashed name has unexpected length: %q", base)
	}
}

func TestDefaultSocketPathLongXDGFallback(t *testing.T) {
	longRT := "/" + strings.Repeat("r", 120) + "/run"
	getenv := func(k string) string {
		if k == "XDG_RUNTIME_DIR" {
			return longRT
		}
		return ""
	}
	got := defaultSocketPath(getenv, "/tmp", 1000, "/home/u")
	if len(got) > maxUnixSocketPath {
		t.Fatalf("fallback path still too long (%d): %q", len(got), got)
	}
	if !strings.HasPrefix(got, "/tmp/hmg-") {
		t.Fatalf("fallback should live in tmp, got %q", got)
	}
}

func TestDefaultSocketPathHashStable(t *testing.T) {
	getenv := func(string) string { return "" }
	longTmp := "/" + strings.Repeat("d", 70)
	a := defaultSocketPath(getenv, longTmp, 501, "/Users/u")
	b := defaultSocketPath(getenv, longTmp, 501, "/Users/u")
	if a != b {
		t.Fatalf("hash not stable: %q vs %q", a, b)
	}
	// Different identity -> different hash (very high probability).
	c := defaultSocketPath(getenv, longTmp, 502, "/Users/u")
	if a == c {
		t.Fatalf("different uid produced same hash: %q", a)
	}
}

// TestDefaultSocketPathPublic exercises the exported wrapper to confirm it does
// not panic and returns a non-empty path.
func TestDefaultSocketPathPublic(t *testing.T) {
	got := DefaultSocketPath(func(string) string { return "" })
	if got == "" {
		t.Fatal("empty path")
	}
}

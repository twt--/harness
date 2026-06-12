package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestOpenLogSinkPrecedence covers the resolution order flag > config >
// tty-stderr > default file, with the isTTY decision injected via stderrIsTTY.
func TestOpenLogSinkPrecedence(t *testing.T) {
	dir := t.TempDir()
	flagPath := filepath.Join(dir, "flag.log")
	configPath := filepath.Join(dir, "config.log")
	socket := filepath.Join(dir, "sub", "g.sock")
	stderr := &bytes.Buffer{}

	t.Run("flag wins over config and tty", func(t *testing.T) {
		sink, closeFn, err := openLogSink(logSinkParams{
			flagPath: flagPath, configPath: configPath,
			socket: socket, stderr: stderr, stderrIsTTY: true,
		})
		if err != nil {
			t.Fatalf("openLogSink: %v", err)
		}
		defer closeFn()
		mustBeFile(t, sink, flagPath)
	})

	t.Run("config wins over tty when no flag", func(t *testing.T) {
		sink, closeFn, err := openLogSink(logSinkParams{
			configPath: configPath, socket: socket, stderr: stderr, stderrIsTTY: true,
		})
		if err != nil {
			t.Fatalf("openLogSink: %v", err)
		}
		defer closeFn()
		mustBeFile(t, sink, configPath)
	})

	t.Run("stderr when a tty and no flag or config", func(t *testing.T) {
		sink, closeFn, err := openLogSink(logSinkParams{
			socket: socket, stderr: stderr, stderrIsTTY: true,
		})
		if err != nil {
			t.Fatalf("openLogSink: %v", err)
		}
		defer closeFn()
		if sink != stderr {
			t.Fatalf("tty stderr sink = %T, want the injected stderr buffer", sink)
		}
	})

	t.Run("default file when not a tty (detached spawn)", func(t *testing.T) {
		sink, closeFn, err := openLogSink(logSinkParams{
			socket: socket, stderr: stderr, stderrIsTTY: false,
		})
		if err != nil {
			t.Fatalf("openLogSink: %v", err)
		}
		defer closeFn()
		wantPath := filepath.Join(filepath.Dir(socket), gatewayLogName)
		mustBeFile(t, sink, wantPath)
		// The default-file path must be created next to the socket, so a detached
		// gateway never loses logs.
		if _, err := os.Stat(wantPath); err != nil {
			t.Fatalf("default log file not created: %v", err)
		}
	})
}

// TestOpenLogFileIsAppend0600 verifies the file sink is append-mode and 0600 so
// repeated serve runs accumulate rather than truncate, and logs are not world-
// readable.
func TestOpenLogFileIsAppend0600(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.log")
	if err := os.WriteFile(path, []byte("existing\n"), 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}
	sink, closeFn, err := openLogFile(path)
	if err != nil {
		t.Fatalf("openLogFile: %v", err)
	}
	if _, err := sink.Write([]byte("appended\n")); err != nil {
		t.Fatalf("write: %v", err)
	}
	closeFn()

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(got) != "existing\nappended\n" {
		t.Errorf("log file = %q, want append (not truncate)", string(got))
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("log file perm = %o, want 0600", perm)
	}
}

// mustBeFile asserts the sink is an *os.File pointing at wantPath.
func mustBeFile(t *testing.T, sink any, wantPath string) {
	t.Helper()
	f, ok := sink.(*os.File)
	if !ok {
		t.Fatalf("sink type = %T, want *os.File", sink)
	}
	if f.Name() != wantPath {
		t.Fatalf("sink path = %q, want %q", f.Name(), wantPath)
	}
}

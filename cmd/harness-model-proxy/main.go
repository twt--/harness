// Command harness-model-proxy owns provider configuration, API keys, model
// catalog metadata, and concrete provider calls for harness.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"harness/internal/httpserve"
	"harness/internal/logging"
	"harness/internal/modelproxy/server"
	"harness/internal/modelsdev"
	"harness/internal/term"
)

const (
	exitOK        = 0
	exitRuntime   = 1
	exitUsage     = 2
	defaultListen = "127.0.0.1:8765"
)

type environment struct {
	args             []string
	stdin            io.Reader
	stdout           io.Writer
	stderr           io.Writer
	getenv           func(string) string
	sigCh            chan os.Signal
	modelsDevCatalog func(context.Context) (*modelsdev.Catalog, error)
	terminalRows     func() int
}

func main() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	os.Exit(run(environment{
		args:             os.Args[1:],
		stdin:            os.Stdin,
		stdout:           os.Stdout,
		stderr:           os.Stderr,
		getenv:           os.Getenv,
		sigCh:            sigCh,
		modelsDevCatalog: defaultModelsDevCatalog,
		terminalRows:     defaultTerminalRows,
	}))
}

func run(env environment) int {
	fs := flag.NewFlagSet("harness-model-proxy", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	configPath := fs.String("config", "", "config file path")
	listen := fs.String("listen", "", "HTTP listen address")
	setup := fs.Bool("setup", false, "create or update proxy config")
	force := fs.Bool("force", false, "with --setup, overwrite existing provider files and defaults")
	refreshModels := fs.Bool("refresh-models", false, "fetch models.dev and update configured provider model metadata")
	logLevel := fs.String("log-level", logging.LevelInfo, "log level: debug, info, warn, error")
	if err := fs.Parse(env.args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			usage(env.stdout)
			return exitOK
		}
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitUsage
	}
	if *setup {
		if err := runSetup(env, *force); err != nil {
			fmt.Fprintf(env.stderr, "harness-model-proxy: setup: %v\n", err)
			return exitUsage
		}
		return exitOK
	}

	path := server.ConfigPath(*configPath, flagWasSet(fs, "config"), env.getenv)
	if *refreshModels {
		if err := runRefreshModels(env, path); err != nil {
			fmt.Fprintf(env.stderr, "harness-model-proxy: refresh-models: %v\n", err)
			return exitUsage
		}
		return exitOK
	}
	if path == "" {
		fmt.Fprintln(env.stderr, "harness-model-proxy: no config file found; run harness-model-proxy --setup")
		return exitUsage
	}
	cfg, err := server.LoadConfig(path)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitRuntime
	}

	level, err := logging.CanonicalLevel(*logLevel)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitUsage
	}
	logger, err := logging.NewLogger(env.stderr, level, false)
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitUsage
	}

	handler, err := server.NewHandler(server.Options{
		ConfigDir: filepath.Dir(path),
		Config:    cfg,
		Getenv:    env.getenv,
		Logger:    logger,
		Warn: func(msg string) {
			logger.Warn(msg)
		},
	})
	if err != nil {
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitRuntime
	}
	addr := defaultListen
	if *listen != "" {
		addr = *listen
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if env.sigCh != nil {
		go func() {
			select {
			case <-env.sigCh:
				cancel()
			case <-ctx.Done():
			}
		}()
	}
	srv := httpserve.New(addr, handler)
	logger.Info("model proxy listening", "addr", addr)
	if err := httpserve.Run(ctx, srv); err != nil {
		fmt.Fprintf(env.stderr, "harness-model-proxy: %v\n", err)
		return exitRuntime
	}
	return exitOK
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "harness-model-proxy — provider and model proxy for harness.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  harness-model-proxy [flags]           serve HTTP")
	fmt.Fprintln(w, "  harness-model-proxy --setup [--force] configure providers")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fs := flag.NewFlagSet("harness-model-proxy", flag.ContinueOnError)
	fs.SetOutput(w)
	fs.String("config", "", "config file path")
	fs.String("listen", "", "HTTP listen address")
	fs.Bool("setup", false, "create or update proxy config")
	fs.Bool("force", false, "with --setup, overwrite existing provider files and defaults")
	fs.Bool("refresh-models", false, "fetch models.dev and update configured provider model metadata")
	fs.String("log-level", logging.LevelInfo, "log level: debug, info, warn, error")
	fs.PrintDefaults()
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

func defaultConfigDir(getenv func(string) string) string {
	return server.DefaultConfigDir(getenv)
}

func defaultModelsDevCatalog(ctx context.Context) (*modelsdev.Catalog, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return modelsdev.Fetch(ctx, http.DefaultClient, modelsdev.DefaultURL)
}

func defaultTerminalRows() int {
	rows, _, ok := term.Size()
	if !ok {
		return 0
	}
	return rows
}

.PHONY: build test

build:
	go build ./cmd/harness

test:
	go test ./...

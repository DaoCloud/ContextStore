# ContextStore unified build entry point.
#
# The KVService is self-contained under kv-service/, but project-level builds
# should always be invoked from the repository root with `make build`.

.PHONY: all build server server-debug run-server client-rs rdma-ffi proto proto-rust \
	proto-python test test-server test-integration bench fmt lint docker docker-push clean help

KV_SERVICE_DIR := kv-service

all: build

build:
	$(MAKE) -C $(KV_SERVICE_DIR) build

server:
	$(MAKE) -C $(KV_SERVICE_DIR) server

server-debug:
	$(MAKE) -C $(KV_SERVICE_DIR) server-debug

run-server:
	$(MAKE) -C $(KV_SERVICE_DIR) run-server

client-rs:
	$(MAKE) -C $(KV_SERVICE_DIR) client-rs

rdma-ffi:
	$(MAKE) -C $(KV_SERVICE_DIR) rdma-ffi

proto:
	$(MAKE) -C $(KV_SERVICE_DIR) proto

proto-rust:
	$(MAKE) -C $(KV_SERVICE_DIR) proto-rust

proto-python:
	$(MAKE) -C $(KV_SERVICE_DIR) proto-python

test: test-server test-integration

test-server:
	$(MAKE) -C $(KV_SERVICE_DIR) test-server

test-integration:
	$(MAKE) -C $(KV_SERVICE_DIR) test-integration

fmt:
	$(MAKE) -C $(KV_SERVICE_DIR) fmt

lint:
	$(MAKE) -C $(KV_SERVICE_DIR) lint

bench:
	$(MAKE) -C $(KV_SERVICE_DIR) bench

docker:
	$(MAKE) -C $(KV_SERVICE_DIR) docker

docker-push:
	$(MAKE) -C $(KV_SERVICE_DIR) docker-push

clean:
	$(MAKE) -C $(KV_SERVICE_DIR) clean

help:
	@echo "ContextStore build targets:"
	@echo ""
	@echo "  make build            Build KVService server, Rust SDK, and RDMA C ABI"
	@echo "  make server           Build only the KVService server"
	@echo "  make client-rs        Build only the Rust client SDK"
	@echo "  make rdma-ffi         Build only the RDMA C ABI library"
	@echo "  make proto            Regenerate Rust and Python protobuf code"
	@echo "  make test             Run KVService server and integration tests"
	@echo "  make bench            Run KVService benchmarks"
	@echo "  make docker           Build the KVService Docker image"
	@echo "  make fmt / lint       Format or statically check KVService code"
	@echo "  make clean            Remove KVService build artifacts"

# ContextStore unified build entry point.
#
# The KVService is self-contained under kv-service/, but project-level builds
# should always be invoked from the repository root with `make build`.

.PHONY: all build deploy check-deploy-config server server-debug run-server client-rs rdma-ffi meta proto proto-rust \
	proto-python test test-server test-integration bench fmt lint docker docker-push clean help

KV_SERVICE_DIR := kv-service
CONFIG ?= kv-service/configs/server.toml
LOG_LEVEL ?= info

all: build

build:
	$(MAKE) -C $(KV_SERVICE_DIR) build

# Start the freshly built server in the foreground so its logs remain attached
# to the invoking terminal. Environment variables such as CS_RDMA_DEVICES are
# inherited unchanged for RDMA test deployments.
deploy: check-deploy-config build
	@echo "==> Starting ContextStore KVService"
	@echo "    config: $(CONFIG)"
	@echo "    log level: $(LOG_LEVEL)"
	@echo "    RDMA devices: $${CS_RDMA_DEVICES:-not configured}"
	@echo "    force disk read: $${CS_FORCE_DISK_READ:-0}"
	@exec ./target/release/contextstore-server --config "$(CONFIG)" --log-level "$(LOG_LEVEL)"

check-deploy-config:
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "error: config file not found: $(CONFIG)"; \
		exit 1; \
	fi

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

meta:
	$(MAKE) -C $(KV_SERVICE_DIR) meta

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
	@echo "  make build            Build server, clients, RDMA FFI, and metadata CLI"
	@echo "  make deploy           Build and start the server in the foreground"
	@echo "  make server           Build only the KVService server"
	@echo "  make client-rs        Build only the Rust client SDK"
	@echo "  make rdma-ffi         Build only the RDMA C ABI library"
	@echo "  make meta             Build the Redis metadata inspection CLI"
	@echo "  make proto            Regenerate Rust and Python protobuf code"
	@echo "  make test             Run KVService server and integration tests"
	@echo "  make bench            Run KVService benchmarks"
	@echo "  make docker           Build the KVService Docker image"
	@echo "  make fmt / lint       Format or statically check KVService code"
	@echo "  make clean            Remove KVService build artifacts"

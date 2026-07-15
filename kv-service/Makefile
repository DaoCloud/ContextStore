# ContextStore KV Service — Rust services + protobuf build
#
# The Python client SDK has been merged into the top-level package
# (src/contextstore/kvservice_client/) and is shipped by the root
# pyproject.toml; it is no longer built as a separate wheel here.

.PHONY: all build server server-debug run-server client-rs rdma-ffi proto proto-rust proto-python \
        test test-server test-integration bench fmt lint docker docker-push clean help

# ===== Config =====
CARGO  ?= cargo
PYTHON ?= python3
DOCKER ?= docker

SERVER_DIR   := server
CLIENT_RS    := client-rs
RDMA_FFI_DIR := rdma-ffi
PROTO_DIR    := proto
BUILD_TYPE   ?= release
SERVER_FEATURES ?= io-uring,rdma,metrics
CLIENT_RS_FEATURES ?= rdma

# Path to the Python client package inside the root repo (post-merge)
PY_PB_OUT := ../src/contextstore/kvservice_client/_pb

# ===== Top-level targets =====
all: build

build: server client-rs rdma-ffi

# ===== Server (Rust) =====
server:
	@echo "==> Building contextstore-server ($(BUILD_TYPE), features: $(SERVER_FEATURES))"
	cd $(SERVER_DIR) && $(CARGO) build --$(BUILD_TYPE) --features $(SERVER_FEATURES)

server-debug:
	cd $(SERVER_DIR) && $(CARGO) build

run-server: server
	../target/$(BUILD_TYPE)/contextstore-server \
		--config configs/server.toml

# ===== Rust client + bench =====
client-rs:
	@echo "==> Building contextstore-client-rs ($(BUILD_TYPE), features: $(CLIENT_RS_FEATURES))"
	cd $(CLIENT_RS) && $(CARGO) build --$(BUILD_TYPE) --features $(CLIENT_RS_FEATURES)

# ===== RDMA FFI cdylib (consumed by Python via ctypes) =====
rdma-ffi:
	cd $(RDMA_FFI_DIR) && $(CARGO) build --$(BUILD_TYPE)

# ===== Protobuf =====
proto: proto-rust proto-python

proto-rust:
	@echo "==> Generating Rust protobuf (triggered by build.rs)"
	cd $(SERVER_DIR) && $(CARGO) build --$(BUILD_TYPE)

proto-python:
	@echo "==> Generating Python protobuf -> $(PY_PB_OUT)"
	mkdir -p $(PY_PB_OUT)
	$(PYTHON) -m grpc_tools.protoc \
		-I $(PROTO_DIR) \
		--python_out=$(PY_PB_OUT) \
		--grpc_python_out=$(PY_PB_OUT) \
		$(PROTO_DIR)/kv_service.proto

# ===== Tests =====
test: test-server test-integration

test-server:
	cd $(SERVER_DIR) && $(CARGO) test --$(BUILD_TYPE)

# Python integration tests (require a running server)
test-integration:
	$(PYTHON) -m pytest tests/ -v

bench:
	cd benchmarks && $(PYTHON) run_benchmark.py

# ===== Code quality =====
fmt:
	cd $(SERVER_DIR)   && $(CARGO) fmt
	cd $(CLIENT_RS)    && $(CARGO) fmt
	cd $(RDMA_FFI_DIR) && $(CARGO) fmt

lint:
	cd $(SERVER_DIR)   && $(CARGO) clippy -- -D warnings
	cd $(CLIENT_RS)    && $(CARGO) clippy -- -D warnings
	cd $(RDMA_FFI_DIR) && $(CARGO) clippy -- -D warnings

# ===== Packaging =====
docker:
	$(DOCKER) build -t contextstore/kv-service:latest \
		-f deploy/docker/Dockerfile ..

docker-push:
	$(DOCKER) push contextstore/kv-service:latest

# ===== Cleanup =====
clean:
	cd $(SERVER_DIR)   && $(CARGO) clean
	cd $(CLIENT_RS)    && $(CARGO) clean
	cd $(RDMA_FFI_DIR) && $(CARGO) clean
	rm -rf $(PY_PB_OUT)/*_pb2.py $(PY_PB_OUT)/*_pb2_grpc.py $(PY_PB_OUT)/*.pyi
	find . -type d -name __pycache__ -exec rm -rf {} +

help:
	@echo "ContextStore KV Service build targets:"
	@echo ""
	@echo "  make build         Build server + client-rs + rdma-ffi with deployment features"
	@echo "  make server        Build only the Rust server"
	@echo "  make client-rs     Build only the Rust client SDK + bench"
	@echo "  make rdma-ffi      Build only the RDMA C ABI cdylib"
	@echo "  make proto         Regenerate protobuf code (Rust + Python)"
	@echo "  make test          Rust unit tests + Python integration tests"
	@echo "  make bench         Run the perf benchmark (requires a running server)"
	@echo "  make docker        Build the Docker image"
	@echo "  make fmt / lint    Format / static analysis"
	@echo "  make clean         Clean build artifacts"

# ContextStore KV Service — Rust services + protobuf build
#
# The Python client SDK has been merged into the top-level package
# (src/contextstore/kvservice_client/) and is shipped by the root
# pyproject.toml; it is no longer built as a separate wheel here.

.PHONY: all build build-info check-build-tools check-e2e-tools server e2e-server server-debug run-server client-rs rdma-ffi meta proto proto-rust proto-python \
        test test-server test-integration e2e bench fmt lint docker docker-push clean help

# ===== Config =====
CARGO  ?= cargo
PYTHON ?= python3
DOCKER ?= docker

# Prefer an explicitly supplied protoc. Otherwise discover a bundled compiler
# before falling back to the system installation.
PROTOC_ORIGIN := explicit
ifeq ($(strip $(PROTOC)),)
PROTOC_ORIGIN := auto-discovered
PROTOC_CANDIDATES := \
	$(CURDIR)/../tools/protoc/protoc \
	/opt/contextstore/tools/protoc/protoc \
	$(shell command -v protoc 2>/dev/null)
PROTOC := $(firstword $(foreach candidate,$(PROTOC_CANDIDATES),$(wildcard $(candidate))))
endif

PROTOC_BIN_DIR := $(patsubst %/,%,$(dir $(PROTOC)))
PROTOC_LIBRARY_CANDIDATES := $(PROTOC_BIN_DIR)/lib $(PROTOC_BIN_DIR)
PROTOC_LIBRARY_PATH := $(firstword $(foreach candidate,$(PROTOC_LIBRARY_CANDIDATES),$(if $(wildcard $(candidate)/libprotoc.so*),$(candidate))))
ifneq ($(strip $(PROTOC_LIBRARY_PATH)),)
export LD_LIBRARY_PATH := $(PROTOC_LIBRARY_PATH)$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))
else
PROTOC_LIBRARY_PATH := system loader path
endif
export PROTOC

SERVER_DIR   := server
CLIENT_RS    := client-rs
RDMA_FFI_DIR := rdma-ffi
PROTO_DIR    := proto
BUILD_TYPE   ?= release
SERVER_FEATURES ?= io-uring,rdma,metrics
CLIENT_RS_FEATURES ?= rdma
CARGO_RELEASE_FLAG := $(if $(filter release,$(BUILD_TYPE)),--release,)

# Path to the Python client package inside the root repo (post-merge)
PY_PB_OUT := ../src/contextstore/kvservice_client/_pb

# ===== Top-level targets =====
all: build

build: build-info server client-rs rdma-ffi meta

build-info: check-build-tools
	@echo "==> ContextStore KVService build configuration"
	@echo "    profile: $(BUILD_TYPE)"
	@echo "    server features: $(SERVER_FEATURES)"
	@echo "    Rust client features: $(CLIENT_RS_FEATURES)"
	@echo "    protoc ($(PROTOC_ORIGIN)): $(PROTOC)"
	@echo "    protoc library path: $(PROTOC_LIBRARY_PATH)"

check-build-tools:
	@if [ -z "$(PROTOC)" ] || { ! command -v "$(PROTOC)" >/dev/null 2>&1 && [ ! -x "$(PROTOC)" ]; }; then \
		echo "error: protoc was not found."; \
		echo "Set PROTOC=/path/to/protoc or install protoc on PATH."; \
		exit 1; \
	fi

# ===== Server (Rust) =====
server: check-build-tools
	@echo "==> Building contextstore-server ($(BUILD_TYPE), features: $(SERVER_FEATURES))"
	cd $(SERVER_DIR) && $(CARGO) build $(CARGO_RELEASE_FLAG) --features $(SERVER_FEATURES)

# E2E is intentionally hardware-independent: use the default server build instead
# of the deployment build, which enables the optional RDMA feature.
e2e-server: check-build-tools
	@echo "==> Building contextstore-server for hardware-independent E2E ($(BUILD_TYPE))"
	cd $(SERVER_DIR) && $(CARGO) build $(CARGO_RELEASE_FLAG) --bin contextstore-server

server-debug:
	cd $(SERVER_DIR) && $(CARGO) build

run-server: server
	../target/$(BUILD_TYPE)/contextstore-server \
		--config configs/server.toml

# ===== Rust client + bench =====
client-rs: check-build-tools
	@echo "==> Building contextstore-client-rs ($(BUILD_TYPE), features: $(CLIENT_RS_FEATURES))"
	cd $(CLIENT_RS) && $(CARGO) build $(CARGO_RELEASE_FLAG) --features $(CLIENT_RS_FEATURES)

# ===== RDMA FFI cdylib (consumed by Python via ctypes) =====
rdma-ffi:
	cd $(RDMA_FFI_DIR) && $(CARGO) build $(CARGO_RELEASE_FLAG)

# ===== Metadata inspection =====
meta: check-build-tools
	@echo "==> Building cs-meta ($(BUILD_TYPE), features: $(SERVER_FEATURES))"
	cd $(SERVER_DIR) && $(CARGO) build $(CARGO_RELEASE_FLAG) --features $(SERVER_FEATURES) --bin cs-meta

# ===== Protobuf =====
proto: proto-rust proto-python

proto-rust: check-build-tools
	@echo "==> Generating Rust protobuf (triggered by build.rs)"
	cd $(SERVER_DIR) && $(CARGO) build $(CARGO_RELEASE_FLAG)

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
	cd $(SERVER_DIR) && $(CARGO) test $(CARGO_RELEASE_FLAG)

# Python integration tests (require a running server)
test-integration:
	$(PYTHON) -m pytest tests/ -v

# Starts an isolated Redis plus two local KVService processes. The test owns all
# ports, metadata keys, and data directories and requires no pre-existing service.
e2e: check-e2e-tools e2e-server
	cd $(CLIENT_RS) && CS_E2E_SERVER_BIN="$(CURDIR)/../target/$(BUILD_TYPE)/contextstore-server" \
		$(CARGO) test $(CARGO_RELEASE_FLAG) --test local_cluster_e2e -- --nocapture

check-e2e-tools:
	@command -v redis-server >/dev/null 2>&1 || { echo "error: redis-server is required for make e2e"; exit 1; }

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
	@echo "  make build         Build server, clients, RDMA FFI, and metadata CLI"
	@echo "  make server        Build only the Rust server"
	@echo "  make e2e-server    Build the default-feature server used by make e2e"
	@echo "  make client-rs     Build only the Rust client SDK + bench"
	@echo "  make rdma-ffi      Build only the RDMA C ABI cdylib"
	@echo "  make meta          Build the Redis metadata inspection CLI"
	@echo "  make proto         Regenerate protobuf code (Rust + Python)"
	@echo "  make test          Rust unit tests + Python integration tests"
	@echo "  make e2e           Run isolated Redis + two-node local-cluster E2E tests"
	@echo "  make bench         Run the perf benchmark (requires a running server)"
	@echo "  make docker        Build the Docker image"
	@echo "  make fmt / lint    Format / static analysis"
	@echo "  make clean         Clean build artifacts"

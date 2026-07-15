# Multi-stage build: Rust server only. The Python client SDK is shipped by
# the root pip package, so nothing Python-specific is built here.
# NOTE: the build context must be the repository root:
#   docker build -t contextstore/kv-service:latest -f kv-service/deploy/docker/Dockerfile .

# ===== Stage 1: Build Rust server =====
FROM rust:1.75-bookworm AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libclang-dev \
    libibverbs-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY kv-service/server/ ./server/
COPY kv-service/proto/  ./proto/

WORKDIR /build/server
RUN cargo build --release --features io-uring,rdma,metrics

# ===== Stage 2: Runtime =====
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libibverbs1 \
    libssl3 \
    nvme-cli \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /build/server/target/release/contextstore-server /usr/local/bin/
COPY kv-service/configs/ /etc/contextstore/

RUN mkdir -p /var/lib/contextstore /var/log/contextstore

EXPOSE 50051

ENTRYPOINT ["/usr/local/bin/contextstore-server"]
CMD ["--config", "/etc/contextstore/server.toml"]

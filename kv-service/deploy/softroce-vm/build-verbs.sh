#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
cargo build --locked --release -p contextstore-server --features rdma \
  --bin contextstore-server --bin cs-meta
cargo build --locked --release -p contextstore-client-rs --features rdma \
  --bin cs-bench --bin cs-rail-read-bench
cargo test --locked --release -p contextstore-client-rs --features rdma \
  --test rail_read_e2e --no-run

# KVService configuration

`contextstore-server` reads one TOML configuration file at startup. The file is
selected with `--config` / `-c`; when omitted, the server reads
`configs/server.toml` relative to its current working directory.

Redis is an external dependency, not a second KVService config file. The server
connects to Redis through `[metadata].redis_url` and fails startup if the Redis
metadata store is unavailable.

## Example files

| File | Purpose |
|------|---------|
| `server.toml` | Default local development config using `./data/nvme*` paths. |
| `server-test.toml` | Test environment config using `/mnt/cs/nvme2..9`. |
| `server-nvmeof.toml` | NVMe-oF / JBOF config using mounted remote namespaces. |

Use these files as deployment templates and keep environment-specific values
such as disk mount paths, advertised endpoints, and Redis URLs outside source
control when they contain private infrastructure details.

## Start commands

From the repository root:

```bash
make build
./target/release/contextstore-server \
    --config kv-service/configs/server.toml
```

From the `kv-service/` directory:

```bash
make build
../target/release/contextstore-server --config configs/server.toml
```

The log level can be set with `--log-level`:

```bash
../target/release/contextstore-server \
    --config configs/server.toml \
    --log-level info
```

## Minimal config

Every section has Rust-side defaults, but production configs should be explicit
about storage paths, Redis metadata, and public endpoints.

```toml
[api]
listen = "0.0.0.0:50051"
max_connections = 1000

[storage]
devices = ["./data/nvme0"]
data_subdir = "contextstore"
striping_threshold = 268435456
striping_chunk_size = 67108864

[memory_tier]
capacity_mb = 4096
slab_size_mb = 64
use_pinned_memory = false

[io_executor]
kind = "tier_a"
thread_pool_size = 32
io_uring_depth = 256

[router]
strategy = "object_hash"

[metadata]
redis_url = "redis://127.0.0.1:6379/"
redis_key_prefix = "contextstore:metadata:"
redis_connect_timeout_ms = 1000
redis_command_timeout_ms = 1000

[metrics]
enabled = false
listen = "0.0.0.0:9090"
```

## Sections

### `[api]`

| Field | Default | Description |
|-------|---------|-------------|
| `listen` | `"0.0.0.0:50051"` | gRPC listen address. |
| `max_connections` | `1000` | Intended connection limit for the service. |

### `[storage]`

| Field | Default | Description |
|-------|---------|-------------|
| `devices` | `["./data/nvme0"]` | Mount directories or directories backing local/NVMe-oF devices. At least one device is required. |
| `data_subdir` | `"contextstore"` | Subdirectory created below each configured device. |
| `striping_threshold` | `268435456` | Object size threshold in bytes for internal striping. `0` disables striping. |
| `striping_chunk_size` | `67108864` | Chunk size in bytes when striping is enabled. |

KVService stores object data under each device's `data_subdir`. It creates
subdirectories and object files; it does not format raw devices.

### `[memory_tier]`

| Field | Default | Description |
|-------|---------|-------------|
| `capacity_mb` | `4096` | L1 memory cache capacity. |
| `slab_size_mb` | `64` | Slab allocation size for the memory tier. |
| `use_pinned_memory` | `false` | Enables pinned memory when the runtime environment supports it. |

### `[io_executor]`

| Field | Default | Description |
|-------|---------|-------------|
| `kind` | `"tier_a"` | I/O backend: `tier_a`, `tier_b`, or `tier_c`. |
| `thread_pool_size` | `32` | Worker count for the thread-pool path. |
| `io_uring_depth` | `256` | Ring depth for the `tier_b` io_uring path. |

`tier_a` works with the default build. `tier_b` requires a Linux build with the
`io-uring` Cargo feature. `tier_c` is reserved for the SPDK path.

### `[router]`

| Field | Default | Description |
|-------|---------|-------------|
| `strategy` | `"object_hash"` | Object routing strategy. `object_hash` is currently the supported value. |

### `[metadata]`

| Field | Default | Description |
|-------|---------|-------------|
| `redis_url` | `"redis://127.0.0.1:6379/"` | Shared Redis metadata endpoint. Must not be empty. |
| `redis_key_prefix` | `"contextstore:metadata:"` | Prefix for all KVService metadata keys in Redis. Must not be empty. |
| `redis_connect_timeout_ms` | `1000` | Redis connection timeout in milliseconds. Must be greater than `0`. |
| `redis_command_timeout_ms` | `1000` | Redis read/write command timeout in milliseconds. Must be greater than `0`. |

All KVService nodes that should share object metadata must use the same
`redis_url` and `redis_key_prefix`. Use different prefixes to isolate
environments that share one Redis instance.

### `[cluster]`

This section is optional. When `cluster.data_nodes` is empty, the server runs in
single-node placement mode. Cross-node placement is enabled only when more than
one data node is configured and the object is large enough to be striped.

```toml
[cluster]
node_id = "node-a"
grpc_advertise = "10.0.0.11:50051"
rdma_advertise = "10.0.0.11:50053"

[[cluster.data_nodes]]
node_id = "node-a"
grpc_endpoint = "10.0.0.11:50051"
rdma_endpoint = "10.0.0.11:50053"

[[cluster.data_nodes]]
node_id = "node-b"
grpc_endpoint = "10.0.0.12:50051"
rdma_endpoint = "10.0.0.12:50053"
```

| Field | Default | Description |
|-------|---------|-------------|
| `node_id` | `""` | This node's stable ID. If empty, the server reads `CS_NODE_ID`, then falls back to `local`. |
| `grpc_advertise` | `""` | Public gRPC endpoint for this node. If empty, the server reads `CS_GRPC_ADVERTISE`, then falls back to `[api].listen`. |
| `rdma_advertise` | `""` | Public RDMA endpoint for this node. If empty, the server reads `CS_RDMA_ADVERTISE`, then falls back to empty. |
| `data_nodes` | `[]` | Data nodes eligible for object stripe placement. Each entry requires `grpc_endpoint`; `node_id` and `rdma_endpoint` are optional. |

### `[metrics]`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enables the Prometheus exporter when the binary is built with the `metrics` feature. |
| `listen` | `"0.0.0.0:9090"` | Prometheus HTTP listen address. |

If `metrics.enabled = true` but the binary was not built with `--features
metrics`, the server still starts and logs a warning.

### `[gds]`

This section is optional and only has an effect when the server is built with
the `gds` feature.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Attempts to initialize GPUDirect Storage support. |
| `device` | `-1` | CUDA device ordinal. `-1` means do not force a device. |

GDS initialization failures are logged as warnings and normal I/O paths remain
available.

## Deployment notes

- Create all configured storage device directories before startup, or run the
  server with permissions that can create the needed subdirectories.
- Start Redis before KVService. For local testing, `redis-server --port 6379`
  is sufficient.
- In Kubernetes, provide Redis separately and set `[metadata].redis_url` to its
  service DNS name, for example `redis://contextstore-redis:6379/`.
- For multi-node deployments, every node must use the same Redis metadata store
  and a consistent `cluster.data_nodes` list.

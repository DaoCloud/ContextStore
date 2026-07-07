# ContextStore NIXL Plugin

This directory contains a standalone NIXL backend plugin named `CONTEXTSTORE`.
It is designed to be loaded by NIXL without modifying Dynamo, KVBM, or vLLM.

The plugin follows the same object-style contract as NIXL's `OBJ` backend:

- local descriptors are `DRAM_SEG`
- remote descriptors are `OBJ_SEG`
- the object key is taken from the `OBJ_SEG` descriptor `metaInfo`
- `NIXL_WRITE` stores local DRAM into ContextStore
- `NIXL_READ` loads ContextStore data into local DRAM

## Current Scope

The first implementation has two client modes:

| Mode | Use |
|------|-----|
| `client_library` | Production path. The plugin `dlopen`s a ContextStore client C ABI library. |
| `file_root` | Local smoke-test fallback. It stores objects as files and does not use ContextStore. |

The `file_root` mode is intentionally only for validating NIXL plugin loading and descriptor semantics before wiring the KVService/RDMA client library.

## C ABI

A production client library must export the functions declared in
`contextstore_nixl_client.h`:

```c
int cs_nixl_client_open(const struct cs_nixl_client_config *config, void **out_client);
void cs_nixl_client_close(void *client);
int cs_nixl_client_put(void *client, const char *key, const void *data, size_t len, uint64_t offset);
int cs_nixl_client_get(void *client, const char *key, void *data, size_t len, uint64_t offset);
int cs_nixl_client_exists(void *client, const char *key, uint64_t *size, int *found);
```

`cs_nixl_client_get` and `cs_nixl_client_put` return `0` on success.

## Build

Build against an installed NIXL prefix:

```bash
cmake -S nixl-plugin/contextstore -B /tmp/contextstore-nixl-build \
  -DNIXL_PREFIX=/opt/nvidia/nvda_nixl
cmake --build /tmp/contextstore-nixl-build -j
```

Or build against a NIXL source checkout:

```bash
cmake -S nixl-plugin/contextstore -B /tmp/contextstore-nixl-build \
  -DNIXL_SOURCE_DIR=/path/to/nixl
cmake --build /tmp/contextstore-nixl-build -j
```

When building against the `nixl-cu12` wheel, the public backend headers are not
installed with the wheel. Use the matching NIXL source tag for headers and link
against the wheel shared libraries:

```bash
NIXL_LIB_DIR=/data/dynamo-vllm-venv/lib/python3.12/site-packages/.nixl_cu12.mesonpy.libs
cmake -S nixl-plugin/contextstore -B /tmp/contextstore-nixl-build \
  -DNIXL_SOURCE_DIR=/path/to/nixl-0.10.1 \
  -DNIXL_LIBRARY_DIR=${NIXL_LIB_DIR}
cmake --build /tmp/contextstore-nixl-build -j
```

For NIXL `0.10.1`, linking `libnixl_build.so` is required because descriptor
list methods used by backend plugins are not header-only in that release.

The output is `libplugin_CONTEXTSTORE.so`.

## Run With NIXL

Place the built library in a directory listed by `NIXL_PLUGIN_DIR`, then create
the backend from NIXL with plugin name `CONTEXTSTORE`.

Example smoke-test backend parameters:

```text
file_root=/tmp/contextstore-nixl-objects
```

Example production backend parameters:

```text
client_library=/data/ContextStore/build/libcontextstore_nixl_client.so
endpoint=127.0.0.1:50051
namespace=dynamo-kvbm
rdma_server_addr=127.0.0.1:50053
rdma_enabled=true
```

The plugin maps each NIXL OBJ key to KVService as:

```text
KVService ObjectKey {
  namespace: <backend parameter "namespace">
  object_key: <NIXL OBJ key>
}
```

## Build Client Library

The production `client_library` is built from `client-ffi/`.

gRPC-only build:

```bash
cargo build --manifest-path nixl-plugin/contextstore/client-ffi/Cargo.toml --release
```

RDMA data path build:

```bash
cargo build --manifest-path nixl-plugin/contextstore/client-ffi/Cargo.toml \
  --release --features rdma
```

Output:

```text
nixl-plugin/contextstore/client-ffi/target/release/libcontextstore_nixl_client.so
```

RDMA runtime environment:

| Variable | Default | Use |
|----------|---------|-----|
| `CS_NIXL_RDMA_DEVICE` | `mlx5_0` | RDMA HCA name |
| `CS_NIXL_RDMA_PORT` | `1` | HCA port |
| `CS_NIXL_RDMA_GID_INDEX` | `3` | RoCE GID index |
| `CS_NIXL_RDMA_INTERNAL_BUFFER_MB` | `64` | Legacy internal receive buffer size |
| `CS_NIXL_RDMA_FALLBACK_GRPC` | `false` | Fallback to gRPC if RDMA open fails |

With `rdma_enabled=true`, `put/get` use the existing ContextStore RDMA wire
protocol. The first version supports full-object transfers only
(`OBJ_SEG.addr == 0`); non-zero object offsets return an error instead of
silently writing the wrong range.

The RDMA client caches external memory registrations by `(ptr, len)` for the
lifetime of the client handle. This matches the expected vLLM/NIXL pinned-buffer
pool behavior and avoids an `ibv_reg_mr`/`ibv_dereg_mr` pair on every transfer.
The underlying RDMA client deregisters all cached regions during
`cs_nixl_client_close`.

`queryMem`/`exists` uses KVService's metadata-light `Exists` RPC in gRPC mode and
does not pull the object payload. KVService does not currently expose a
metadata-only size/HEAD API, so `queryMem` reports presence without a size when
the size is unknown. In RDMA mode the current wire protocol does not have a
metadata-only exists request, so the client reports "unknown/not found" and
callers should issue a READ when they need the object.

## Smoke Test And Bench

Use the Python roundtrip script to validate plugin loading, descriptor semantics,
and gRPC/RDMA data paths.

File smoke test:

```bash
export NIXL_PLUGIN_DIR=/tmp/contextstore-nixl-build
python nixl-plugin/contextstore/scripts/nixl_contextstore_roundtrip.py \
  --mode file \
  --size 1m \
  --iters 3
```

gRPC client-library test:

```bash
export NIXL_PLUGIN_DIR=/tmp/contextstore-nixl-build
python nixl-plugin/contextstore/scripts/nixl_contextstore_roundtrip.py \
  --mode grpc \
  --client-library nixl-plugin/contextstore/client-ffi/target/release/libcontextstore_nixl_client.so \
  --endpoint 127.0.0.1:50051 \
  --namespace nixl-grpc-test \
  --size 1m \
  --size 16m
```

RDMA client-library test:

```bash
export NIXL_PLUGIN_DIR=/tmp/contextstore-nixl-build
export CS_NIXL_RDMA_DEVICE=mlx5_0
export CS_NIXL_RDMA_PORT=1
export CS_NIXL_RDMA_GID_INDEX=3
python nixl-plugin/contextstore/scripts/nixl_contextstore_roundtrip.py \
  --mode rdma \
  --client-library nixl-plugin/contextstore/client-ffi/target/release/libcontextstore_nixl_client.so \
  --endpoint 127.0.0.1:50051 \
  --rdma-server-addr 127.0.0.1:50053 \
  --namespace nixl-rdma-test \
  --size 1m \
  --size 16m \
  --size 64m
```

512MiB eviction / multi-NVMe benchmark:

```bash
export NIXL_PLUGIN_DIR=/tmp/contextstore-nixl-build
export CS_NIXL_RDMA_DEVICE=mlx5_0
export CS_NIXL_RDMA_PORT=1
export CS_NIXL_RDMA_GID_INDEX=3
python nixl-plugin/contextstore/scripts/nixl_contextstore_eviction.py \
  --mode rdma \
  --client-library nixl-plugin/contextstore/client-ffi/target/release/libcontextstore_nixl_client.so \
  --endpoint 127.0.0.1:50051 \
  --rdma-server-addr 127.0.0.1:50053 \
  --namespace nixl-eviction-test \
  --object-size 512m \
  --evict-count 3 \
  --timeout-s 300
```

Use this case when validating the disk reload path rather than just NIXL plugin
loading. `512m` is larger than the default 256MiB striping threshold and should
produce 8 stripes with the default 64MiB stripe size. The script writes one base
object, writes eviction objects to push the base object out of the slab cache,
then reads the base object back and verifies the payload.

Validation checklist:

- The script exits with status 0.
- `NIXL_SUMMARY` contains `verified=1`.
- KVService logs contain `n_stripes=8` for the reload object.
- KVService logs contain eight `TIERB_READ_PTR` lines, each with
  `planned_bytes=67108864`, when the reload comes from disk.

KVBM-side usage should eventually look like a regular NIXL backend enablement:

```bash
export NIXL_PLUGIN_DIR=/data/ContextStore/build/nixl-plugins:${NIXL_PLUGIN_DIR:-}
export DYN_KVBM_NIXL_BACKEND_CONTEXTSTORE=true
```

The exact KVBM descriptor creation still needs integration work: KVBM must create
`OBJ_SEG` descriptors whose `metaInfo` is the ContextStore key.

# KVService — Deployment guide

Four supported deployment shapes for the Rust `contextstore-server`. Pick one; the layouts are independent.

> **Production bare-metal deployments use the versioned toolkit at
> [`deploy/baremetal/`](../../deploy/baremetal/README.md)** (immutable release
> directories, atomic `current` symlink switching, hardened systemd unit,
> post-upgrade acceptance checklist). The shapes below cover local dev,
> containers, and the JBOF/NVMe-oF topology. Where they overlap — Shape 4's
> systemd step — the baremetal toolkit is authoritative.

The server reads one TOML config file at startup. See
[`../configs/README.md`](../configs/README.md) for the complete config reference
and the Redis metadata requirements.

```
deploy/
├── docker/
│   ├── Dockerfile             # multi-stage build (Rust server only; Python client
│   │                            ships as part of the main pip package)
│   └── docker-compose.yml     # single-host compose
├── k8s/
│   ├── statefulset.yaml       # StatefulSet + Service + ConfigMap (gRPC path only)
│   ├── statefulset-rdma.yaml  # RDMA variant A: hostNetwork + hugepages + IPC_LOCK
│   └── statefulset-rdma-spiderpool.yaml
│                              # RDMA variant B: Multus/spiderpool secondary NIC,
│                              # fixed-IP pools, dynamic GID probing, no privileged.
│                              # Use this when client pods reach storage over the
│                              # same secondary network (validated at 23 GB/s
│                              # cold-read aggregate; see manifest comments for
│                              # the ten field lessons)
└── jbof/
    ├── spdk_target.sh         # JBOF side: start SPDK NVMe-oF target
    ├── initiator_connect.sh   # Storage host: attach + mount remote namespaces
    └── initiator_disconnect.sh
```

All commands below assume the working directory is the repository root.

---

## Shape 1 — Local dev (single host)

```bash
make build
./target/release/contextstore-server \
    --config kv-service/configs/server-test.toml
```

---

## Shape 2 — Docker Compose

```bash
docker compose -f kv-service/deploy/docker/docker-compose.yml up -d
```

The Dockerfile expects the repository root as its build context (it copies both `kv-service/server/` and `kv-service/proto/`).

---

## Shape 3 — Kubernetes

Two manifests, by data path:

| manifest | data path | typical read BW | host prerequisites |
|---|---|---|---|
| `k8s/statefulset.yaml` | gRPC/TCP only | ~0.5 GB/s | none beyond NVMe mounts |
| `k8s/statefulset-rdma.yaml` | gRPC + RDMA | disk-bound (25 GiB/s-class on 2 nodes x 2 NVMe) | RDMA NIC, hugepages, labeled nodes |

**gRPC-only:**

```bash
kubectl apply -f kv-service/deploy/k8s/statefulset.yaml
kubectl get pods -l app=contextstore-kv
```

**RDMA-enabled** — per storage node, once:

```bash
# 1. Reserve 2Mi hugepages for the 8 GB pre-registered slab (+ headroom)
echo 5632 > /proc/sys/vm/nr_hugepages        # 11 GiB; persist via sysctl.d

# 2. Verify the RDMA stack: device present and RoCE v2 GID for the host IP
ibdev2netdev                                  # note the device name (e.g. mlx5_1)
show_gids | grep v2                           # gid-index 3 = RoCE v2 IPv4

# 3. Label the node so the StatefulSet schedules onto it
kubectl label node <node> contextstore.io/rdma=true
```

Then edit `statefulset-rdma.yaml` (HCA device name in `CS_RDMA_DEVICES`,
NVMe hostPath mounts, Redis URL in the ConfigMap) and apply.

`CS_RDMA_DEVICES` accepts `device:ip:port[:gid_index]`. The fourth segment
matters on secondary-NIC networks (macvlan/SR-IOV), where the RoCE v2 GID
index is derived from the pod IP and drifts on every pod recreation — probe
it at startup instead of hard-coding (the spiderpool manifest does this
automatically; hostNetwork deployments can usually keep the static index).

```bash
kubectl apply -f kv-service/deploy/k8s/statefulset-rdma.yaml
kubectl get pods -l app=contextstore-kv-rdma
# readiness gates on the RDMA control port: Ready means the slab is registered
```

Validated end to end on a 2-worker cluster (K8s 1.32, containerd 2.x,
ConnectX HCAs in ETH/RoCE v2 mode): both pods Ready with the 8 GB hugepage
slab registered, gRPC PUT through the pod, then single-endpoint RDMA GET
from a host client — same-node 40.7 GB/s, cross-node 26.8 GB/s (slab-served;
identical to the bare-metal numbers on the same fabric).

Design notes (also inlined as comments in the manifest):

- **hostNetwork** so the RDMA listener binds the host IP and RoCE v2 GIDs
  match the advertised address; clients keep using `--gid-index 3`.
- **privileged** (validated on containerd 2.x / K8s 1.32): a hostPath mount
  of `/dev/infiniband` plus IPC_LOCK is not sufficient — the device cgroup
  still blocks the verbs char devices and `ibv_open_device` fails. Deploying
  `k8s-rdma-shared-dev-plugin` and requesting `rdma/hca` lets you drop
  privileged back to IPC_LOCK. **hugepages-2Mi** resource requests back the
  slab's `MAP_HUGETLB` allocation (falls back to normal pages if unavailable).
- **seccomp Unconfined** because default profiles on some runtimes block
  io_uring; harden with a custom profile allowing
  `io_uring_setup/enter/register`, or set `io_executor.kind = "tier_a"`.
- On clusters with Multus + `k8s-rdma-shared-dev-plugin`, swap hostNetwork,
  the hostPath, and privileged for an `rdma/hca` resource request and a
  secondary RoCE NIC (see the comment block at the end of the manifest).
- Multi-node placement (`[cluster]`, coordinator-routed multi-endpoint
  reads) needs per-node `node_id`/advertise values: render per-node
  ConfigMaps with kustomize/helm, or generate the `[cluster]` section in an
  init container from `HOST_IP`. Without `[cluster]` each pod serves
  standalone.

The manifest expects Redis to be reachable at the URL configured in the
`[metadata]` section; deploy Redis separately or point the ConfigMap at an
existing Redis service.

---

## Shape 4 — Bare metal, JBOF over NVMe-oF (production)

Three hosts: a **JBOF** exporting NVMe namespaces, a **storage host** running `contextstore-server` on top of remote namespaces, and one or more **compute hosts** running vLLM / Dynamo.

**Step 1 — start the SPDK target on the JBOF**

```bash
ssh jbof-host
sudo kv-service/deploy/jbof/spdk_target.sh RDMA 0.0.0.0 4420
```

The script auto-discovers local PCIe NVMe controllers and exports each as a namespace under
`nqn.2024-01.contextstore:jbof0`. Requires SPDK installed at `/usr/local/spdk` (override with `SPDK_DIR`).

**Step 2 — attach and mount on the storage host**

```bash
ssh storage-host
sudo kv-service/deploy/jbof/initiator_connect.sh <jbof_ip> 4420 rdma /mnt/cs
```

Remote namespaces appear as `/dev/nvmeXn1` and are formatted (xfs) + mounted under `/mnt/cs/nvmeX`. The script prints a ready-to-paste `[storage].devices = [...]` block for `configs/server-nvmeof.toml`.

To reverse this later:

```bash
sudo kv-service/deploy/jbof/initiator_disconnect.sh /mnt/cs
```

**Step 3 — start `contextstore-server` under systemd**

Use the versioned bare-metal toolkit — it installs immutable releases under
`/opt/contextstore/releases/<id>/` with atomic `current` switching and a
hardened unit:

```bash
cd deploy/baremetal
sudo ./bootstrap-node.sh
./build-release.sh --release-id <version-sha>          # on the Linux builder
sudo ./install-release.sh --release-id <version-sha> --source <release-dir>
sudo ./configure-node.sh ...                            # see deploy/baremetal/README.md
sudo ./install-service.sh
sudo ./activate-release.sh --release-id <version-sha> --restart
```

(The legacy single-file unit previously shipped at
`kv-service/deploy/systemd/contextstore-server.service` is deprecated; it
predates the release-directory layout and lacks the sandbox hardening.)

For the RDMA data path, build the server with `--features rdma` and open the RDMA TCP control port
(default `50053`) from compute hosts to the storage host.

**Step 4 — install the Connector on each compute host**

```bash
pip install -e '/path/to/ContextStore'   # installs contextstore + contextstore.kvservice_client
```

Then point vLLM / Dynamo at the storage host — see the top-level [`README.md`](../../README.md) for the `--kv-transfer-config` payload.

## Failure handling FAQ

KVService keeps object metadata in Redis, object bytes in the configured
storage tier, and optional hot data in its memory tier. These layers have
different recovery behavior.

| Failure | Service behavior | Client-visible result | Recovery |
|---------|------------------|-----------------------|----------|
| KVService restart | Drops process memory and RDMA connections; keeps committed Redis metadata and storage files. | Requests in flight fail. | Restart the service and retry. The first read may come from storage. |
| Redis unavailable | Applies the configured timeout, reconnects, and retries the metadata command once. | The operation succeeds after a short interruption or returns an error. | Restore Redis, then retry. There is no in-process metadata fallback. |
| Object or stripe missing | Rejects the incomplete object and invalidates stale metadata where the object read path owns it. | `NOT_FOUND` or an empty optional result. | Recompute or restore the complete object, then write it again. |
| Stripe checksum mismatch | Rejects the object instead of returning partial or corrupt data. | `NOT_FOUND` on the normal object read path. | Investigate the device, then rewrite the object. There is no stripe reconstruction. |
| TTL expired | Treats the object as absent and attempts to remove its metadata and files. | `NOT_FOUND` or an empty optional result. | No client recovery is required unless the object is still needed; then write it again. |

Checksum and TTL behavior require a server and client release that supports
the corresponding configuration and protocol fields. Do not mix client and
server artifacts generated from incompatible protocol revisions.

### What survives a KVService restart?

An acknowledged PUT has written the object to the storage tier and committed
its metadata. Those objects survive a service restart when Redis and the
storage mounts remain available. The memory tier, registered RDMA buffers, and
connections are process state and must be rebuilt.

A process termination during PUT is different from an acknowledged PUT. The
request fails and may leave an unreferenced file if the process stops after a
physical write but before metadata commit. KVService does not replay in-flight
requests after restart.

For striped objects, every referenced data node and stripe must be available.
KVService does not return a partial object and does not reconstruct a missing
stripe from replicas.

### What happens when Redis is temporarily unavailable?

Each metadata operation uses `redis_connect_timeout_ms` and
`redis_command_timeout_ms`. After a command failure, KVService opens a new
connection and retries that command once. If Redis recovers within this
window, the operation can complete. Otherwise the request returns an error;
it is not converted into a cache miss.

KVService requires Redis during startup and has no process-local metadata
fallback. A write whose data reached storage but whose metadata commit failed
is not readable and may leave an unreferenced file.

### How are missing objects and stripes handled?

Missing metadata is a normal cache miss. If metadata exists but the referenced
file or stripe is absent, KVService does not return the remaining bytes. The
normal object read path removes matching stale metadata and returns a miss.
Direct placement APIs return `NOT_FOUND`; their caller must discard the stale
descriptor and perform a fresh lookup before deciding whether to recompute.

A data-node connection failure is reported as `UNAVAILABLE`, not
`NOT_FOUND`. Clients must not delete metadata merely because a node is
temporarily unreachable.

### What happens when checksum verification fails?

When `storage.verify_stripe_checksums` is enabled, KVService stores an
xxh3-64 checksum for each physical stripe and verifies it before serving a
disk read. A missing checksum, short read, or checksum mismatch makes the
whole object unavailable. KVService logs the integrity failure and never
returns a partially verified object.

Checksum verification is disabled by default because it adds a memory scan to
the storage read path. Objects written before verification was enabled have no
stored checksums and must be rewritten before they can pass verification.
KVService detects corruption but does not repair or reconstruct the stripe.

### How does TTL cleanup work?

A positive `ttl_seconds` expires an object at `created_at + ttl_seconds`; zero
or a negative value disables expiry. Expiry is checked by read, existence,
descriptor, RDMA, and conditional-write paths. The cleanup operation verifies
the current object identity before deleting metadata, so an expired generation
cannot delete a newer value written under the same key.

TTL cleanup is lazy. KVService removes an expired object when an operation
touches it; it does not periodically scan the complete Redis namespace. An
expired object that is never accessed again can therefore continue to occupy
storage. File deletion is best effort, so failed deletion can also leave an
unreferenced file for an external maintenance job to reclaim.

### Which errors should a client retry?

| Status | Client action |
|--------|---------------|
| `NOT_FOUND` | Treat as a cache miss and recompute or fetch from another authoritative source. |
| `UNAVAILABLE` | Retry with bounded exponential backoff; do not invalidate metadata immediately. |
| `FAILED_PRECONDITION` | Refresh the descriptor or correct the client/server configuration before retrying. |
| `INTERNAL` | Preserve the error, inspect service and Redis health, and retry only when the failing dependency is healthy. |
| `ALREADY_EXISTS` or `committed=false` | Treat an immutable or conditional PUT as already committed according to the calling API contract. |

Clients should bound retries and keep transport failures distinct from cache
misses. KVService is not a write-ahead log and does not provide automatic
application-request replay.

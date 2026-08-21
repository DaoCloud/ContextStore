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
│   └── statefulset-rdma.yaml  # RDMA-enabled variant (hostNetwork + hugepages + IPC_LOCK)
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
NVMe hostPath mounts, Redis URL in the ConfigMap) and apply:

```bash
kubectl apply -f kv-service/deploy/k8s/statefulset-rdma.yaml
kubectl get pods -l app=contextstore-kv-rdma
# readiness gates on the RDMA control port: Ready means the slab is registered
```

Design notes (also inlined as comments in the manifest):

- **hostNetwork** so the RDMA listener binds the host IP and RoCE v2 GIDs
  match the advertised address; clients keep using `--gid-index 3`.
- **IPC_LOCK** capability for `ibv_reg_mr` of the slab; **hugepages-2Mi**
  resource requests back its `MAP_HUGETLB` allocation (falls back to normal
  pages with a warning if unavailable).
- **seccomp Unconfined** because default profiles on some runtimes block
  io_uring; harden with a custom profile allowing
  `io_uring_setup/enter/register`, or set `io_executor.kind = "tier_a"`.
- `/dev/infiniband` is mounted via hostPath; no `privileged` needed. On
  clusters with Multus + `k8s-rdma-shared-dev-plugin`, swap hostNetwork and
  the hostPath for an `rdma/hca` resource request and a secondary RoCE NIC
  (see the comment block at the end of the manifest).
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

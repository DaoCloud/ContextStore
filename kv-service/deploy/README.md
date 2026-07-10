# KVService — Deployment guide

Four supported deployment shapes for the Rust `contextstore-server`. Pick one; the layouts are independent.

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
│   └── statefulset.yaml       # StatefulSet + Service + ConfigMap
├── systemd/
│   └── contextstore-server.service
└── jbof/
    ├── spdk_target.sh         # JBOF side: start SPDK NVMe-oF target
    ├── initiator_connect.sh   # Storage host: attach + mount remote namespaces
    └── initiator_disconnect.sh
```

All commands below assume the working directory is the repository root.

---

## Shape 1 — Local dev (single host)

```bash
make -C kv-service build
./kv-service/server/target/release/contextstore-server \
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

```bash
kubectl apply -f kv-service/deploy/k8s/statefulset.yaml
kubectl get pods -l app=contextstore-kv
```

Edit the ConfigMap section of `statefulset.yaml` (or replace it with a mounted `configs/server.toml`) before applying to a real cluster.
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

```bash
sudo install -m 644 kv-service/deploy/systemd/contextstore-server.service /etc/systemd/system/
sudo install -m 644 kv-service/configs/server-nvmeof.toml /etc/contextstore/server.toml
sudo systemctl daemon-reload
sudo systemctl enable --now contextstore-server
sudo systemctl status contextstore-server
```

For the RDMA data path, build the server with `--features rdma` and open the RDMA TCP control port
(default `50053`) from compute hosts to the storage host.

**Step 4 — install the Connector on each compute host**

```bash
pip install -e '/path/to/ContextStore'   # installs contextstore + contextstore.kvservice_client
```

Then point vLLM / Dynamo at the storage host — see the top-level [`README.md`](../../README.md) for the `--kv-transfer-config` payload.

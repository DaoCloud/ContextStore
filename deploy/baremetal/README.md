# ContextStore bare-metal installation

This toolkit installs KVService on storage nodes without building on those
nodes. Build and verify artifacts on a Linux builder, then copy the release
directory and versioned wheels to every target host.

**This toolkit is versioned at `deploy/baremetal/` in the repository.** The
copy under `/opt/contextstore/deploy/` on any node is a deployment artifact:
refresh it from the repository when upgrading, never edit it in place. Keep
the toolkit identical on every node — divergent per-node copies of these
scripts have caused installs to skip steps (missing `activate-release.sh`).

## Standard layout

```text
/opt/contextstore/artifacts/wheels/   immutable ContextStore and Redhare wheels
/opt/contextstore/releases/<id>/      immutable Rust server release
/opt/contextstore/current             active release symlink (relative target)
/opt/contextstore/deploy/             copy of this toolkit, refreshed per release
/opt/contextstore/logs/               manual diagnostic output only
/etc/contextstore/server.toml         node configuration
/etc/contextstore/contextstore.env    diagnostic environment overrides
/mnt/contextstore/nvme{0,1}/data/     KVService object data
```

The systemd unit writes normal logs to journald. Do not place object data,
Redis data, or large benchmark output under `/opt/contextstore`.

Run the deployment scripts below from this directory. `build-release.sh`
resolves the repository root as two levels above itself; when running a
copied toolkit outside the repository, pass `--repo <checkout>` explicitly.
It refuses to build from a dirty tree.

## 1. Bootstrap each storage node

Run once after the data mounts are ready:

```bash
sudo ./bootstrap-node.sh
```

This creates the `contextstore` system account and gives it access only to the
`data` subdirectory beneath every configured mount.

## 2. Build and transfer the release

Build on the Linux builder. The release identifier must contain the intended
semantic version and source SHA. The script invokes the single supported build
entry point, `make build`, with the `io-uring`, RDMA, and metrics feature set.

For a Debian or Ubuntu host intentionally designated as the builder, install
the build prerequisites once:

```bash
sudo ./install-builder-prereqs.sh
```

```bash
./build-release.sh \
  --release-id 0.4.0-d1c2a77112c7 \
  --output /tmp/contextstore-release-0.4.0-d1c2a77112c7
```

When a Linux target has Podman but no Rust toolchain, use the companion
container builder instead. It mounts only the selected source tree and leaves
the resulting release under its `artifacts/releases/` directory.

```bash
./build-release-container.sh --release-id 0.4.0-d1c2a77112c7
```

Transfer that immutable directory to a staging path on each storage node. The
transfer method is deliberately outside the scripts so an environment can use
its approved SSH, artifact repository, or configuration-management channel.

```bash
rsync -a --checksum /tmp/contextstore-release-0.4.0-d1c2a77112c7/ \
  worker01:/tmp/contextstore-release-0.4.0-d1c2a77112c7/
```

## 3. Install and activate the immutable server release

Run on each storage node:

```bash
sudo ./install-release.sh \
  --release-id 0.4.0-d1c2a77112c7 \
  --source /tmp/contextstore-release-0.4.0-d1c2a77112c7

sudo ./activate-release.sh --release-id 0.4.0-d1c2a77112c7
```

`activate-release.sh` changes only the `current` symlink. Use `--restart` only
after the configuration and systemd unit have been installed.

## 4. Configure both nodes

Every node uses the same Redis URL, key prefix, and complete peer list. Only
the local node identity and advertise addresses differ. Replace the Redis URL
with the approved shared test Redis endpoint.

```bash
# worker01: 10.80.2.89
sudo ./configure-node.sh \
  --node-id worker01 \
  --grpc-advertise 10.80.2.89:50051 \
  --rdma-advertise 10.80.2.89:50053 \
  --redis-url 'redis://<shared-redis>:6379/' \
  --redis-prefix 'contextstore:worker-test:' \
  --peer 'worker01,10.80.2.89:50051,10.80.2.89:50053' \
  --peer 'worker02,10.80.2.49:50051,10.80.2.49:50053'

# worker02: 10.80.2.49
sudo ./configure-node.sh \
  --node-id worker02 \
  --grpc-advertise 10.80.2.49:50051 \
  --rdma-advertise 10.80.2.49:50053 \
  --redis-url 'redis://<shared-redis>:6379/' \
  --redis-prefix 'contextstore:worker-test:' \
  --peer 'worker01,10.80.2.89:50051,10.80.2.89:50053' \
  --peer 'worker02,10.80.2.49:50051,10.80.2.49:50053'
```

The generated configuration uses `tier_b`, two local device directories,
64 MiB striping chunks, and the Prometheus exporter on port 9090.

## 5. Install and start the service

```bash
sudo ./install-service.sh
sudo ./service.sh validate
sudo systemctl enable --now contextstore-kvservice
sudo ./service.sh status
sudo ./service.sh logs
```

The service will not start successfully until the configured shared Redis is
reachable. The supplied environment file sets `CS_RDMA_DISABLED=1`: this makes
the first deployment gRPC/TCP only. After the NIC, GID, route, and peer
reachability have been verified, set `CS_RDMA_DISABLED=0`, configure
`CS_RDMA_DEVICES`, then restart the service. Do not persist diagnostic toggles
such as `CS_FORCE_DISK_READ` in normal operation.

## 6. Install pinned Python wheels on compute nodes

Use the exact Python interpreter that runs vLLM or Redhare. The script uses
`--no-deps` to avoid changing the existing Torch and vLLM dependency graph.

```bash
sudo ./install-wheels.sh \
  --python /path/to/vllm-venv/bin/python \
  --contextstore-wheel contextstore-0.4.0-cp312-cp312-linux_x86_64.whl \
  --redhare-wheel redhare-0.4.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

## Upgrade and rollback

Install every new release into a fresh version directory. Verify the manifest,
then atomically switch `current` and restart. Rollback is the same operation
with the previous release identifier.

```bash
sha256sum -c /opt/contextstore/releases/<release-id>/manifest.sha256
sudo ./activate-release.sh --release-id <release-id> --restart
```

## Acceptance after upgrade

Run these after every release activation, before declaring the upgrade done:

1. `./service.sh validate` and `systemctl is-active contextstore-kvservice`
   on every node.
2. `sha256sum -c /opt/contextstore/releases/<release-id>/manifest.sha256` and
   confirm `release.env` `SOURCE_COMMIT` matches the tip of `main` that was
   intended to ship. A release whose SOURCE_COMMIT is not on `main` means a
   pull request was merged out of order — stop and reconcile before serving
   traffic.
3. Multi-endpoint RDMA read benchmark against a striped test object with a
   clean page cache. Record bandwidth, latency, and `data_ok` count; compare
   against the previous release's accepted numbers.
4. Confirm no diagnostic toggles are active:
   `tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value contextstore-kvservice)/environ | grep -E 'CS_FORCE|CS_SYNC'`
   must print nothing.

## Merge-order discipline for stacked pull requests

When PR B is based on PR A's branch, merge B into A's branch **before** A is
merged to `main`; a squash-merge of A snapshots the branch at merge time and
silently drops anything merged into it afterwards. After every merge to
`main`, verify the expected symbols actually landed (for example
`git grep <new-function> origin/main`).

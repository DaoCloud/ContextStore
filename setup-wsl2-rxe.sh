#!/usr/bin/env bash
# =============================================================================
# setup-wsl2-rxe.sh — bring up two independent Soft-RoCE paths on WSL2
#                     (rxe0 -> veth0, rxe1 -> veth1)
#
# Purpose: construct two independent network paths for the multi-rail parallel
#          read proposal (Issue #33), using the Soft-RoCE approach that the
#          contest constraints explicitly allow on machines without RDMA NICs.
# Risk:    network/RDMA soft-device configuration only. Does NOT build kernels,
#          does NOT touch .wslconfig, does NOT restart WSL. Idempotent (existing
#          interfaces/devices are skipped). Requires root.
#
# Prerequisites (one-time, see wsl2-softroce-setup.md):
#   1. Custom WSL2 kernel built with CONFIG_RDMA_RXE=y
#   2. rdma-core / infiniband-diags / ibverbs-utils installed (rdma, ibv_*)
#
# Usage:  sudo bash setup-wsl2-rxe.sh
# =============================================================================
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "error: run as root -> sudo bash $0" >&2
    exit 1
fi

echo "== [1/4] load the Soft-RoCE kernel module =="
if ! modprobe rdma_rxe; then
    echo "modprobe rdma_rxe failed: your WSL2 kernel lacks CONFIG_RDMA_RXE." >&2
    echo "Build and switch kernels per wsl2-softroce-setup.md first." >&2
    exit 1
fi
echo "    rdma_rxe loaded"

echo "== [2/4] create the veth pair (two independent L3 paths) =="
# veth pair: veth0 <--> veth1, distinct addresses on a dedicated subnet,
# giving two genuinely independent paths.
ip link add veth0 type veth peer name veth1 2>/dev/null || echo "    veth0/veth1 exist, skip"
ip addr add 192.168.96.110/24 dev veth0 2>/dev/null || true
ip addr add 192.168.96.111/24 dev veth1 2>/dev/null || true
ip link set veth0 up
ip link set veth1 up
echo "    veth0=192.168.96.110  veth1=192.168.96.111  up"

echo "== [3/4] allow local delivery between the two veth ends =="
sysctl -w net.ipv4.conf.veth0.accept_local=1 >/dev/null
sysctl -w net.ipv4.conf.veth1.accept_local=1 >/dev/null

echo "== [4/4] bind one rxe soft-RDMA device to each veth =="
rdma link add rxe0 type rxe netdev veth0 2>/dev/null || echo "    rxe0 exists, skip"
rdma link add rxe1 type rxe netdev veth1 2>/dev/null || echo "    rxe1 exists, skip"

echo ""
echo "==================== verification ===================="
echo "--- rdma link ---"; rdma link
echo "--- ibv_devices ---"; ibv_devices
echo "--- cross-rail ping (veth0 -> veth1) ---"
if ping -I veth0 -c1 -W2 192.168.96.111 >/dev/null 2>&1; then
    echo "OK: two independent Soft-RoCE paths (rxe0->veth0, rxe1->veth1) are ready."
    echo "next steps:"
    echo "  - real-verbs dual-rail bandwidth:  ib_send_bw -d rxe0 192.168.96.111 & ib_send_bw -d rxe1 192.168.96.110 &"
    echo "  - multi-rail e2e demo:             cargo run --features rdma --release --example softroce_dual_rail"
else
    echo "WARN: cross-veth ping failed; check accept_local and link states." >&2
fi

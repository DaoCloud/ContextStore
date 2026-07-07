#!/usr/bin/env bash
# JBOF side — start an SPDK NVMe-oF target that exports every local PCIe NVMe.
#
# Usage: ./deploy/jbof/spdk_target.sh [TRANSPORT] [LISTEN_IP] [LISTEN_PORT]
#   TRANSPORT:   RDMA | TCP   (default: RDMA)
#   LISTEN_IP:   listen IP    (default: 0.0.0.0)
#   LISTEN_PORT: listen port  (default: 4420)
#
# Environment:
#   SPDK_DIR     SPDK install prefix       (default: /usr/local/spdk)
#   HUGEMEM_MB   HugePages in MiB          (default: 8192)
#   SUBSYS_NQN   subsystem NQN             (default: nqn.2024-01.contextstore:jbof0)

set -euo pipefail

TRANSPORT="${1:-RDMA}"
LISTEN_IP="${2:-0.0.0.0}"
LISTEN_PORT="${3:-4420}"

SPDK_DIR="${SPDK_DIR:-/usr/local/spdk}"
HUGEMEM_MB="${HUGEMEM_MB:-8192}"
SUBSYS_NQN="${SUBSYS_NQN:-nqn.2024-01.contextstore:jbof0}"

RPC="${SPDK_DIR}/scripts/rpc.py"
TGT_BIN="${SPDK_DIR}/build/bin/nvmf_tgt"

if [[ ! -x "$TGT_BIN" ]]; then
    echo "[ERROR] SPDK target binary not found: $TGT_BIN" >&2
    echo "Install SPDK first: https://spdk.io/doc/getting_started.html" >&2
    exit 1
fi

echo "==> Configuring HugePages (${HUGEMEM_MB} MB)..."
HUGEMEM=$HUGEMEM_MB "${SPDK_DIR}/scripts/setup.sh"

echo "==> Starting nvmf_tgt..."
"$TGT_BIN" -m 0x0F &  # pin to 4 CPU cores
TGT_PID=$!
sleep 2

echo "==> Creating transport: $TRANSPORT"
"$RPC" nvmf_create_transport -t "$TRANSPORT" -u 8192 -m 4 -c 0

echo "==> Creating subsystem: $SUBSYS_NQN"
"$RPC" nvmf_create_subsystem "$SUBSYS_NQN" -a -s "ContextStore-JBOF-0"

echo "==> Attaching local NVMe devices..."
# Auto-discover PCIe NVMe controllers
i=0
for pci in $(lspci -D | awk '/Non-Volatile memory controller/ {print $1}'); do
    echo "    nvme${i} <- ${pci}"
    "$RPC" bdev_nvme_attach_controller -b "nvme${i}" -t pcie -a "$pci" || true
    "$RPC" nvmf_subsystem_add_ns "$SUBSYS_NQN" "nvme${i}n1" || true
    i=$((i + 1))
done

echo "==> Adding listener: ${TRANSPORT}://${LISTEN_IP}:${LISTEN_PORT}"
"$RPC" nvmf_subsystem_add_listener "$SUBSYS_NQN" \
    -t "$TRANSPORT" -a "$LISTEN_IP" -s "$LISTEN_PORT"

echo ""
echo "==> SPDK NVMe-oF target running (PID: $TGT_PID)"
echo "    Subsystem: $SUBSYS_NQN"
echo "    Listener:  ${TRANSPORT}://${LISTEN_IP}:${LISTEN_PORT}"
echo "    Namespaces: $i"
echo ""
echo "On the initiator, run: ./initiator_connect.sh $LISTEN_IP $LISTEN_PORT $TRANSPORT"

wait $TGT_PID

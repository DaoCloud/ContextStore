#!/usr/bin/env bash
# Storage host side — attach a remote JBOF NVMe-oF target, format, and mount.
#
# Usage: ./deploy/jbof/initiator_connect.sh <JBOF_IP> [PORT] [TRANSPORT] [MOUNT_ROOT]
#   JBOF_IP:    JBOF listen address       (required)
#   PORT:       target port               (default: 4420)
#   TRANSPORT:  rdma | tcp                (default: rdma)
#   MOUNT_ROOT: root mountpoint           (default: /mnt/cs)
#
# After running:
#   - remote namespaces appear as /dev/nvmeXn1
#   - each is mkfs.xfs'd (if not already formatted) and mounted at ${MOUNT_ROOT}/nvmeX

set -euo pipefail

JBOF_IP="${1:?Usage: $0 <JBOF_IP> [PORT] [TRANSPORT] [MOUNT_ROOT]}"
PORT="${2:-4420}"
TRANSPORT="${3:-rdma}"
MOUNT_ROOT="${4:-/mnt/cs}"

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] root required (needed for nvme connect / mkfs / mount)" >&2
    exit 1
fi

echo "==> Loading kernel modules..."
case "$TRANSPORT" in
    rdma) modprobe nvme-rdma ;;
    tcp)  modprobe nvme-tcp ;;
    *)    echo "[ERROR] unknown transport: $TRANSPORT"; exit 1 ;;
esac

echo "==> Discovering subsystems..."
nvme discover -t "$TRANSPORT" -a "$JBOF_IP" -s "$PORT" || true

echo "==> Connecting to all subsystems..."
nvme connect-all -t "$TRANSPORT" -a "$JBOF_IP" -s "$PORT"

sleep 2  # wait for udev to create device nodes

echo "==> Current NVMe devices:"
nvme list

mkdir -p "$MOUNT_ROOT"
echo "==> Mounting remote namespaces under ${MOUNT_ROOT}/nvmeX ..."
# Only process fabric-attached remote controllers
for dev in $(nvme list -o json | python3 -c '
import json, sys
data = json.load(sys.stdin)
for d in data.get("Devices", []):
    # NVMe-oF devices typically have ModelNumber starting with "SPDK" or ProductName="Linux"
    if d.get("ModelNumber", "").startswith("SPDK") or d.get("ProductName") == "Linux":
        print(d["DevicePath"])
'); do
    name=$(basename "$dev")             # nvmeXn1
    mount_point="${MOUNT_ROOT}/${name}"
    mkdir -p "$mount_point"
    # Skip mkfs if a filesystem already exists
    if ! blkid "$dev" >/dev/null 2>&1; then
        echo "    Formatting $dev (xfs)..."
        mkfs.xfs -f "$dev"
    fi
    if ! mountpoint -q "$mount_point"; then
        echo "    mount $dev -> $mount_point"
        mount -o noatime,nodiratime "$dev" "$mount_point"
    fi
done

echo ""
echo "==> Done. Use these devices in your contextstore-server config:"
ls -d "${MOUNT_ROOT}"/nvme* 2>/dev/null || echo "    (no mountpoints found)"
echo ""
echo "Snippet for configs/server-nvmeof.toml:"
echo "  [storage]"
echo "  devices = ["
for d in "${MOUNT_ROOT}"/nvme*; do
    [[ -d "$d" ]] && echo "      \"$d\","
done
echo "  ]"

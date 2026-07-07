#!/usr/bin/env bash
# Storage host side — unmount and disconnect all NVMe-oF sessions.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] root required" >&2; exit 1
fi

MOUNT_ROOT="${1:-/mnt/cs}"

echo "==> Unmounting ${MOUNT_ROOT}/nvme* ..."
for d in "${MOUNT_ROOT}"/nvme*; do
    [[ -d "$d" ]] || continue
    mountpoint -q "$d" && umount "$d" || true
done

echo "==> Disconnecting all NVMe-oF sessions..."
nvme disconnect-all

echo "Done."

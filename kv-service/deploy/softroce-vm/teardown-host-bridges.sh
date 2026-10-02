#!/usr/bin/env bash
set -euo pipefail

for name in csrs0 csrc0 csrs1 csrc1; do
  if ip link show dev "$name" >/dev/null 2>&1; then
    sudo ip link delete "$name"
  fi
done
for name in csrb0 csrb1; do
  if ip link show dev "$name" >/dev/null 2>&1; then
    sudo ip link delete "$name"
  fi
done

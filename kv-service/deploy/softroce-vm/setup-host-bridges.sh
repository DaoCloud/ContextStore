#!/usr/bin/env bash
set -euo pipefail

tap_user=${CS_VM_TAP_USER:-$(id -un)}

for name in csrb0 csrb1 csrs0 csrc0 csrs1 csrc1; do
  if ip link show dev "$name" >/dev/null 2>&1; then
    echo "Refusing to replace existing interface: $name" >&2
    exit 1
  fi
done

sudo ip link add name csrb0 type bridge
sudo ip link add name csrb1 type bridge
for name in csrs0 csrc0 csrs1 csrc1; do
  sudo ip tuntap add dev "$name" mode tap user "$tap_user"
done
sudo ip link set csrs0 master csrb0
sudo ip link set csrc0 master csrb0
sudo ip link set csrs1 master csrb1
sudo ip link set csrc1 master csrb1
for name in csrb0 csrb1 csrs0 csrc0 csrs1 csrc1; do
  sudo ip link set "$name" up
done
ip -br link show csrb0
ip -br link show csrb1
bridge link show | grep -E 'csrs[01]|csrc[01]'

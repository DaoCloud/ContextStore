#!/usr/bin/env bash
set -euo pipefail

# Prepare two separate Ubuntu cloud-image overlays and NoCloud seeds.
# This reads one public SSH key; private keys never enter the repository.
vm_dir=${CS_VM_DIR:?set CS_VM_DIR to an empty test-only directory}
public_key=${CS_VM_SSH_PUBLIC_KEY:?set CS_VM_SSH_PUBLIC_KEY to a public key path}
guest_user=${CS_VM_GUEST_USER:-railtest}
if [[ ! -r "$public_key" ]]; then
  echo "Public SSH key is not readable: $public_key" >&2
  exit 1
fi
mkdir -p "$vm_dir"
cd "$vm_dir"

image=jammy-server-cloudimg-amd64.img
base_url=https://cloud-images.ubuntu.com/jammy/current
if [[ ! -f "$image" ]]; then
  curl -fsSL --retry 3 "$base_url/$image" -o "$image"
fi
curl -fsSL --retry 3 "$base_url/SHA256SUMS" -o SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing | grep -qx "$image: OK"

for role in server client; do
  disk="disk-${role}.qcow2"
  if [[ ! -f "$disk" ]]; then
    qemu-img create -q -f qcow2 -F qcow2 -b "$image" "$disk" 12G
  fi
  seed_dir="seed-${role}"
  mkdir -p "$seed_dir"
  cat > "$seed_dir/user-data" <<EOF
#cloud-config
hostname: cs-railvm-${role}
manage_etc_hosts: true
ssh_pwauth: false
package_update: false
users:
  - name: ${guest_user}
    groups: [sudo]
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    lock_passwd: true
    ssh_authorized_keys:
      - $(cat "$public_key")
EOF
  cat > "$seed_dir/meta-data" <<EOF
instance-id: cs-railvm-${role}-local
local-hostname: cs-railvm-${role}
EOF
  (
    cd "$seed_dir"
    genisoimage -quiet -output "../seed-${role}.iso" -volid cidata -joliet -rock \
      user-data meta-data
  )
  if [[ ! -f "OVMF_VARS-${role}.fd" ]]; then
    cp /usr/share/OVMF/OVMF_VARS.fd "OVMF_VARS-${role}.fd"
  fi
done

ls -lh "$image" disk-*.qcow2 seed-*.iso

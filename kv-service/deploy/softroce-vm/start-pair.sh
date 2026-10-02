#!/usr/bin/env bash
set -euo pipefail

vm_dir=${CS_VM_DIR:?set CS_VM_DIR to the prepared VM directory}
server_session=cs-railvm-server-20261002
client_session=cs-railvm-client-20261002
cd "$vm_dir"
if tmux has-session -t "$server_session" 2>/dev/null; then
  echo "Server VM already running" >&2
  exit 1
fi
if tmux has-session -t "$client_session" 2>/dev/null; then
  echo "Client VM already running" >&2
  exit 1
fi

tmux new-session -d -s "$server_session" \
  "cd '$vm_dir' && exec qemu-system-x86_64 -enable-kvm -cpu host -smp 4 -m 4096 -machine q35 -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd -drive if=pflash,format=raw,file=OVMF_VARS-server.fd -drive file=disk-server.qcow2,if=virtio,format=qcow2 -drive file=seed-server.iso,media=cdrom,if=ide -netdev user,id=mgmt,hostfwd=tcp:127.0.0.1:22222-:22 -device virtio-net-pci,netdev=mgmt,mac=52:54:00:12:34:56 -netdev tap,id=rail0,ifname=csrs0,script=no,downscript=no -device virtio-net-pci,netdev=rail0,mac=52:54:00:31:00:02 -netdev tap,id=rail1,ifname=csrs1,script=no,downscript=no -device virtio-net-pci,netdev=rail1,mac=52:54:00:32:00:02 -nographic -serial mon:stdio > qemu-server.log 2>&1"
sleep 2
tmux has-session -t "$server_session"
ss -ltn | grep ':22222\b'

tmux new-session -d -s "$client_session" \
  "cd '$vm_dir' && exec qemu-system-x86_64 -enable-kvm -cpu host -smp 4 -m 4096 -machine q35 -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd -drive if=pflash,format=raw,file=OVMF_VARS-client.fd -drive file=disk-client.qcow2,if=virtio,format=qcow2 -drive file=seed-client.iso,media=cdrom,if=ide -netdev user,id=mgmt,hostfwd=tcp:127.0.0.1:22223-:22 -device virtio-net-pci,netdev=mgmt,mac=52:54:00:12:34:57 -netdev tap,id=rail0,ifname=csrc0,script=no,downscript=no -device virtio-net-pci,netdev=rail0,mac=52:54:00:31:00:01 -netdev tap,id=rail1,ifname=csrc1,script=no,downscript=no -device virtio-net-pci,netdev=rail1,mac=52:54:00:32:00:01 -nographic -serial mon:stdio > qemu-client.log 2>&1"
sleep 2
tmux has-session -t "$client_session"
ss -ltn | grep -E ':(22222|22223)\b'

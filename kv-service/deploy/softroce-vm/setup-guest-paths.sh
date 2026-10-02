#!/usr/bin/env bash
set -euo pipefail

role=${1:?usage: configure_railvm_paths.sh server|client}
case "$role" in
  server)
    ip0=10.31.0.2/24
    ip1=10.32.0.2/24
    device0=rxe_s0
    device1=rxe_s1
    ;;
  client)
    ip0=10.31.0.1/24
    ip1=10.32.0.1/24
    device0=rxe_c0
    device1=rxe_c1
    ;;
  *)
    echo "role must be server or client" >&2
    exit 2
    ;;
esac

sudo ip addr add "$ip0" dev enp0s3
sudo ip addr add "$ip1" dev enp0s4
sudo ip link set enp0s3 up
sudo ip link set enp0s4 up
sudo modprobe rdma_rxe
sudo rdma link add "$device0" type rxe netdev enp0s3
sudo rdma link add "$device1" type rxe netdev enp0s4
rdma link show
ibv_devices

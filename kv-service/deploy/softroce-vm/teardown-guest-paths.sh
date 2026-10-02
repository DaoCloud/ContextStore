#!/usr/bin/env bash
set -euo pipefail

role=${1:?usage: teardown_railvm_paths.sh server|client}
case "$role" in
  server) device0=rxe_s0; device1=rxe_s1; ip0=10.31.0.2/24; ip1=10.32.0.2/24 ;;
  client) device0=rxe_c0; device1=rxe_c1; ip0=10.31.0.1/24; ip1=10.32.0.1/24 ;;
  *) exit 2 ;;
esac
sudo rdma link delete "$device0"
sudo rdma link delete "$device1"
sudo ip addr del "$ip0" dev enp0s3
sudo ip addr del "$ip1" dev enp0s4
sudo ip link set enp0s3 down
sudo ip link set enp0s4 down

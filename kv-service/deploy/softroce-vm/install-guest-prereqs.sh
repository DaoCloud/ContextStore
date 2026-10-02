#!/usr/bin/env bash
set -euo pipefail

# Run only inside a fresh Ubuntu 22.04 test guest, before setup-guest-paths.sh.
sudo env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  --no-install-recommends \
  "linux-modules-extra-$(uname -r)" rdma-core libibverbs1 \
  ibverbs-providers ibverbs-utils redis-server
modinfo -n rdma_rxe

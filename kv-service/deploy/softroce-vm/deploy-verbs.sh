#!/usr/bin/env bash
set -euo pipefail

# Copy Linux x86_64 binaries to two isolated test guests via an SSH config.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
bin_dir=${CS_BIN_DIR:-$repo_root/target/release}
ssh_config=${CS_VM_SSH_CONFIG:?set CS_VM_SSH_CONFIG to SSH aliases for both guests}
server_alias=${CS_VM_SERVER_ALIAS:-cs-railvm-server}
client_alias=${CS_VM_CLIENT_ALIAS:-cs-railvm-client}
guest_root=${CS_GUEST_ROOT:-/home/railtest}

for guest in "$server_alias" "$client_alias"; do
  ssh -F "$ssh_config" "$guest" "mkdir -p '$guest_root/bin' '$guest_root/evidence' '$guest_root/data/rail0' '$guest_root/data/rail1'"
done
scp -F "$ssh_config" "$bin_dir/contextstore-server" "$bin_dir/cs-meta" \
  "$server_alias:$guest_root/bin/"
scp -F "$ssh_config" "$bin_dir/cs-bench" "$bin_dir/cs-rail-read-bench" \
  "$client_alias:$guest_root/bin/"
scp -F "$ssh_config" \
  "$repo_root/kv-service/configs/server-softroce-vm.toml" \
  "$server_alias:$guest_root/evidence/server.toml"
scp -F "$ssh_config" \
  "$repo_root/kv-service/deploy/softroce-vm/start-guest-server.sh" \
  "$repo_root/kv-service/deploy/softroce-vm/stop-guest-server.sh" \
  "$server_alias:$guest_root/"
scp -F "$ssh_config" \
  "$repo_root/kv-service/benchmarks/collect_rail_verbs.py" \
  "$repo_root/kv-service/benchmarks/collect_rail_concurrent.py" \
  "$client_alias:$guest_root/"
if [[ -n "${CS_E2E_BIN:-}" ]]; then
  scp -F "$ssh_config" "$CS_E2E_BIN" \
    "$client_alias:$guest_root/bin/rail_read_e2e"
fi

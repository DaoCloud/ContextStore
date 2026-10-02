#!/usr/bin/env bash
set -euo pipefail

root=${CS_GUEST_ROOT:-/home/railtest}
redis_session=cs-softroce-redis
server_session=cs-softroce-server
slab_mb=${CS_RDMA_SLAB_MB:-128}
write_delay_ms=${CS_RDMA_TEST_PRE_WRITE_DELAY_MS:-0}
write_delay_nic=${CS_RDMA_TEST_PRE_WRITE_NIC_IDX:-0}
cq_timeout_ms=${CS_RDMA_CQ_TIMEOUT_MS:-30000}
mkdir -p "$root/data/redis" "$root/data/rail0" "$root/data/rail1" "$root/evidence"

if ! tmux has-session -t "$redis_session" 2>/dev/null; then
  tmux new-session -d -s "$redis_session" \
    "exec redis-server --bind 127.0.0.1 --port 6388 --save '' --appendonly no --dir '$root/data/redis' >> '$root/evidence/redis.log' 2>&1"
fi
for _ in $(seq 1 30); do
  if redis-cli -h 127.0.0.1 -p 6388 PING 2>/dev/null | grep -qx PONG; then
    break
  fi
  sleep 0.2
done
redis-cli -h 127.0.0.1 -p 6388 PING

if tmux has-session -t "$server_session" 2>/dev/null; then
  echo "Server session already exists: $server_session" >&2
  exit 1
fi
tmux new-session -d -s "$server_session" \
  "CS_RDMA_DEVICES='rxe_s0:10.31.0.2:55153:1,rxe_s1:10.32.0.2:55154:1' CS_RDMA_SLAB_MB='$slab_mb' CS_RDMA_CQ_TIMEOUT_MS='$cq_timeout_ms' CS_RDMA_TEST_PRE_WRITE_DELAY_MS='$write_delay_ms' CS_RDMA_TEST_PRE_WRITE_NIC_IDX='$write_delay_nic' exec '$root/bin/contextstore-server' --config '$root/evidence/server.toml' >> '$root/evidence/server.log' 2>&1"
sleep 2
tmux has-session -t "$server_session"
ss -ltn | grep -E ':(55151|55153|55154)\b'

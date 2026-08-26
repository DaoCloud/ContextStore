#!/usr/bin/env bash
# Generate the complete KVService TOML configuration for one storage node.
set -euo pipefail

CONFIG_ROOT=/etc/contextstore
CONFIG_FILE="${CONFIG_ROOT}/server.toml"
NODE_ID=
GRPC_ADVERTISE=
RDMA_ADVERTISE=
REDIS_URL=
REDIS_PREFIX=
METRICS_LISTEN=0.0.0.0:9090
MEMORY_MB=4096
IO_KIND=tier_b
THREAD_POOL_SIZE=32
IO_URING_DEPTH=256
DATA_SUBDIR=data
DEVICES=(/mnt/contextstore/nvme0 /mnt/contextstore/nvme1)
DEVICES_EXPLICIT=false
PEERS=()

usage() {
    cat <<'EOF'
Usage: configure-node.sh --node-id <id> --grpc-advertise <host:port> \
  --redis-url <url> --redis-prefix <prefix> \
  --peer <node-id,grpc-endpoint,rdma-endpoint> [options]

Required: --node-id, --grpc-advertise, --redis-url, --redis-prefix, and at
least one --peer. Supply every cluster node as --peer, including this node.

Options:
  --rdma-advertise <host:port>       Optional RDMA endpoint for this node
  --device <mount-path>              Repeatable; defaults to nvme0 and nvme1
  --metrics-listen <host:port>       Default: 0.0.0.0:9090
  --memory-mb <count>                Default: 4096
  --io-kind <tier_a|tier_b>          Default: tier_b
  --thread-pool-size <count>         Default: 32
  --io-uring-depth <count>           Default: 256
  --data-subdir <name>               Default: data
EOF
}

require_root() {
    if [[ ${EUID} -ne 0 ]]; then
        echo "ERROR: run as root" >&2
        exit 1
    fi
}

require_toml_safe() {
    local value=$1 field=$2
    if [[ -z "${value}" || "${value}" == *$'\n'* || "${value}" == *'"'* || "${value}" == *'\\'* ]]; then
        echo "ERROR: invalid TOML value for ${field}" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --node-id) NODE_ID="${2:?missing node id}"; shift 2 ;;
        --grpc-advertise) GRPC_ADVERTISE="${2:?missing gRPC endpoint}"; shift 2 ;;
        --rdma-advertise) RDMA_ADVERTISE="${2:?missing RDMA endpoint}"; shift 2 ;;
        --redis-url) REDIS_URL="${2:?missing Redis URL}"; shift 2 ;;
        --redis-prefix) REDIS_PREFIX="${2:?missing Redis prefix}"; shift 2 ;;
        --peer) PEERS+=("${2:?missing peer}"); shift 2 ;;
        --device)
            if [[ "${DEVICES_EXPLICIT}" == false ]]; then
                DEVICES=()
                DEVICES_EXPLICIT=true
            fi
            DEVICES+=("${2:?missing device path}")
            shift 2
            ;;
        --metrics-listen) METRICS_LISTEN="${2:?missing metrics endpoint}"; shift 2 ;;
        --memory-mb) MEMORY_MB="${2:?missing memory size}"; shift 2 ;;
        --io-kind) IO_KIND="${2:?missing I/O executor}"; shift 2 ;;
        --thread-pool-size) THREAD_POOL_SIZE="${2:?missing worker count}"; shift 2 ;;
        --io-uring-depth) IO_URING_DEPTH="${2:?missing queue depth}"; shift 2 ;;
        --data-subdir) DATA_SUBDIR="${2:?missing data subdirectory}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

require_root
for required in NODE_ID GRPC_ADVERTISE REDIS_URL REDIS_PREFIX; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: --${required,,} is required" >&2
        usage >&2
        exit 2
    fi
done
if ((${#PEERS[@]} == 0)); then
    echo "ERROR: provide the complete cluster with at least one --peer" >&2
    exit 2
fi
if [[ ! "${NODE_ID}" =~ ^[A-Za-z0-9._-]+$ || ! "${DATA_SUBDIR}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: node id and data subdirectory may contain only letters, numbers, dot, dash, and underscore" >&2
    exit 2
fi
require_toml_safe "${NODE_ID}" "node id"
require_toml_safe "${GRPC_ADVERTISE}" "gRPC advertise endpoint"
require_toml_safe "${REDIS_URL}" "Redis URL"
require_toml_safe "${REDIS_PREFIX}" "Redis key prefix"
require_toml_safe "${METRICS_LISTEN}" "metrics listen endpoint"
if [[ -n "${RDMA_ADVERTISE}" ]]; then
    require_toml_safe "${RDMA_ADVERTISE}" "RDMA advertise endpoint"
fi
if [[ "${IO_KIND}" != tier_a && "${IO_KIND}" != tier_b ]]; then
    echo "ERROR: --io-kind must be tier_a or tier_b" >&2
    exit 2
fi
for number in "${MEMORY_MB}" "${THREAD_POOL_SIZE}" "${IO_URING_DEPTH}"; do
    [[ "${number}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: numeric settings must be positive integers" >&2; exit 2; }
done
for device in "${DEVICES[@]}"; do
    [[ -d "${device}" ]] || { echo "ERROR: storage device directory does not exist: ${device}" >&2; exit 1; }
    findmnt -rn --target "${device}" >/dev/null || { echo "ERROR: storage device is not mounted: ${device}" >&2; exit 1; }
done

install -d -o root -g contextstore -m 0750 "${CONFIG_ROOT}"
temp_file=$(mktemp "${CONFIG_ROOT}/server.toml.XXXXXX")
trap 'rm -f "${temp_file}"' EXIT

{
    cat <<EOF
[api]
listen = "0.0.0.0:50051"
max_connections = 2000

[storage]
devices = [
EOF
    for device in "${DEVICES[@]}"; do
        require_toml_safe "${device}" "storage device"
        printf '    "%s",\n' "${device}"
    done
    cat <<EOF
]
data_subdir = "${DATA_SUBDIR}"
striping_threshold = 268435456
striping_chunk_size = 67108864

[memory_tier]
capacity_mb = ${MEMORY_MB}
slab_size_mb = 64
use_pinned_memory = false

[io_executor]
kind = "${IO_KIND}"
thread_pool_size = ${THREAD_POOL_SIZE}
io_uring_depth = ${IO_URING_DEPTH}

[router]
strategy = "object_hash"

[metadata]
redis_url = "${REDIS_URL}"
redis_key_prefix = "${REDIS_PREFIX}"
redis_connect_timeout_ms = 1000
redis_command_timeout_ms = 1000

[metrics]
enabled = true
listen = "${METRICS_LISTEN}"

[cluster]
node_id = "${NODE_ID}"
grpc_advertise = "${GRPC_ADVERTISE}"
rdma_advertise = "${RDMA_ADVERTISE}"
EOF
    for peer in "${PEERS[@]}"; do
        IFS=, read -r peer_id peer_grpc peer_rdma extra <<<"${peer}"
        if [[ -n "${extra:-}" || -z "${peer_id}" || -z "${peer_grpc}" ]]; then
            echo "ERROR: invalid peer '${peer}', expected node-id,grpc-endpoint,rdma-endpoint" >&2
            exit 2
        fi
        if [[ ! "${peer_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
            echo "ERROR: invalid peer node id: ${peer_id}" >&2
            exit 2
        fi
        require_toml_safe "${peer_id}" "peer id"
        require_toml_safe "${peer_grpc}" "peer gRPC endpoint"
        if [[ -n "${peer_rdma}" ]]; then
            require_toml_safe "${peer_rdma}" "peer RDMA endpoint"
        fi
        cat <<EOF

[[cluster.data_nodes]]
node_id = "${peer_id}"
grpc_endpoint = "${peer_grpc}"
rdma_endpoint = "${peer_rdma}"
EOF
    done
} >"${temp_file}"

chown root:contextstore "${temp_file}"
chmod 0640 "${temp_file}"
mv -f "${temp_file}" "${CONFIG_FILE}"
trap - EXIT

echo "Wrote node configuration: ${CONFIG_FILE}"

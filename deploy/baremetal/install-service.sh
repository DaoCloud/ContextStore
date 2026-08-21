#!/usr/bin/env bash
# Install the standardized ContextStore systemd service and environment file.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UNIT_SOURCE="${SCRIPT_DIR}/contextstore-kvservice.service"
UNIT_TARGET=/etc/systemd/system/contextstore-kvservice.service
ENV_FILE=/etc/contextstore/contextstore.env

usage() {
    cat <<'EOF'
Usage: install-service.sh

Installs the ContextStore systemd unit and creates the default environment file
when it does not already exist.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (($# != 0)); then
    echo "ERROR: install-service.sh does not accept arguments" >&2
    usage >&2
    exit 2
fi
if [[ ${EUID} -ne 0 ]]; then
    echo "ERROR: run as root" >&2
    exit 1
fi
if [[ ! -f "${UNIT_SOURCE}" ]]; then
    echo "ERROR: service template not found: ${UNIT_SOURCE}" >&2
    exit 1
fi

install -d -o root -g contextstore -m 0750 /etc/contextstore
install -m 0644 "${UNIT_SOURCE}" "${UNIT_TARGET}"
# Stale drop-in overrides (e.g. an emergency ExecStart patch from a previous
# incident) silently shadow the unit shipped here. Surface them loudly.
OVERRIDE_DIR=/etc/systemd/system/contextstore-kvservice.service.d
if [[ -d "${OVERRIDE_DIR}" ]] && compgen -G "${OVERRIDE_DIR}/*.conf" >/dev/null; then
    echo "WARNING: drop-in overrides exist and take precedence over the installed unit:" >&2
    ls -l "${OVERRIDE_DIR}"/*.conf >&2
    echo "         review and remove them unless intentionally kept:" >&2
    echo "         rm ${OVERRIDE_DIR}/*.conf && systemctl daemon-reload" >&2
fi
if [[ ! -e "${ENV_FILE}" ]]; then
    cat >"${ENV_FILE}" <<'EOF'
# Stable production default. Adjust RUST_LOG only during diagnosis.
RUST_LOG=info
# RDMA is explicitly disabled until the NIC, GID, route, and peers are validated.
# Change this to 0 and set CS_RDMA_DEVICES before enabling the RDMA data path.
CS_RDMA_DISABLED=1
# Diagnostic toggles are intentionally disabled by default.
# CS_FORCE_DISK_READ=1
# CS_SYNC_WRITES=1
# CS_RDMA_DEVICES=mlx5_0:0.0.0.0:50053
# CS_RDMA_SLAB_MB=0
EOF
    chown root:contextstore "${ENV_FILE}"
    chmod 0640 "${ENV_FILE}"
fi

systemctl daemon-reload
systemd-analyze verify "${UNIT_TARGET}"
echo "Installed ${UNIT_TARGET}"
echo "Start after configuring Redis and activating a release:"
echo "  systemctl enable --now contextstore-kvservice"

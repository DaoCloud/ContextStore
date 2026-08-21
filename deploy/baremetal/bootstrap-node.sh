#!/usr/bin/env bash
# Prepare one storage node for a ContextStore systemd deployment.
set -euo pipefail

ROOT=/opt/contextstore
CONFIG_ROOT=/etc/contextstore
SERVICE_USER=contextstore
SERVICE_GROUP=contextstore
DATA_SUBDIR=data
DEVICES=(/mnt/contextstore/nvme0 /mnt/contextstore/nvme1)
DEVICES_EXPLICIT=false

usage() {
    cat <<'EOF'
Usage: bootstrap-node.sh [--device <mount-path>]...

Creates the ContextStore service account and standardized runtime directories.
When --device is omitted, /mnt/contextstore/nvme0 and nvme1 are prepared.
EOF
}

require_root() {
    if [[ ${EUID} -ne 0 ]]; then
        echo "ERROR: run as root" >&2
        exit 1
    fi
}

while (($#)); do
    case "$1" in
        --device)
            if [[ "${DEVICES_EXPLICIT}" == false ]]; then
                DEVICES=()
                DEVICES_EXPLICIT=true
            fi
            DEVICES+=("${2:?missing mount path}")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_root

if ! getent group "${SERVICE_GROUP}" >/dev/null; then
    groupadd --system "${SERVICE_GROUP}"
fi
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    useradd --system --gid "${SERVICE_GROUP}" --home-dir /nonexistent \
        --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

install -d -m 0755 "${ROOT}/artifacts/wheels" "${ROOT}/releases"
install -d -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" -m 0750 "${ROOT}/logs"
install -d -o root -g "${SERVICE_GROUP}" -m 0750 "${CONFIG_ROOT}"

for device in "${DEVICES[@]}"; do
    if [[ ! -d "${device}" ]]; then
        echo "ERROR: storage mount path does not exist: ${device}" >&2
        exit 1
    fi
    if ! findmnt -rn --target "${device}" >/dev/null; then
        echo "ERROR: storage path is not mounted: ${device}" >&2
        exit 1
    fi
    install -d -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" -m 0750 \
        "${device}/${DATA_SUBDIR}"
done

echo "ContextStore node bootstrap complete"
echo "  artifacts: ${ROOT}/artifacts/wheels"
echo "  releases:  ${ROOT}/releases"
echo "  config:    ${CONFIG_ROOT}"
printf '  data:      %s\n' "${DEVICES[@]/%//${DATA_SUBDIR}}"

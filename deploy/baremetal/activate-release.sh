#!/usr/bin/env bash
# Atomically select an installed ContextStore release for the systemd service.
set -euo pipefail

ROOT=/opt/contextstore
RELEASE_ID=
RESTART=false

usage() {
    cat <<'EOF'
Usage: activate-release.sh --release-id <id> [--restart]

Updates /opt/contextstore/current atomically. --restart restarts the
contextstore-kvservice systemd unit after the switch.
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
        --release-id)
            RELEASE_ID="${2:?missing release id}"
            shift 2
            ;;
        --restart)
            RESTART=true
            shift
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
if [[ -z "${RELEASE_ID}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "ERROR: invalid release id: ${RELEASE_ID}" >&2
    exit 2
fi

RELEASE_DIR="${ROOT}/releases/${RELEASE_ID}"
if [[ ! -x "${RELEASE_DIR}/bin/contextstore-server" ]]; then
    echo "ERROR: invalid release: ${RELEASE_DIR}" >&2
    exit 1
fi

TEMP_LINK="${ROOT}/.current.${RELEASE_ID}.$$"
ln -s "releases/${RELEASE_ID}" "${TEMP_LINK}"
mv -Tf "${TEMP_LINK}" "${ROOT}/current"
echo "Activated release: ${RELEASE_ID}"

if [[ "${RESTART}" == true ]]; then
    systemctl restart contextstore-kvservice
    systemctl status contextstore-kvservice --no-pager
fi

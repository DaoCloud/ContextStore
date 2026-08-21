#!/usr/bin/env bash
# Install an immutable ContextStore server release on a storage node.
set -euo pipefail

ROOT=/opt/contextstore
RELEASE_ID=
SOURCE_DIR=

usage() {
    cat <<'EOF'
Usage: install-release.sh --release-id <id> --source <directory>

The source directory must contain bin/contextstore-server. The destination is
/opt/contextstore/releases/<id> and is never overwritten.
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
        --source)
            SOURCE_DIR="${2:?missing source directory}"
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
if [[ -z "${RELEASE_ID}" || -z "${SOURCE_DIR}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "ERROR: invalid release id: ${RELEASE_ID}" >&2
    exit 2
fi
if [[ ! -x "${SOURCE_DIR}/bin/contextstore-server" ]]; then
    echo "ERROR: missing executable: ${SOURCE_DIR}/bin/contextstore-server" >&2
    exit 1
fi

DESTINATION="${ROOT}/releases/${RELEASE_ID}"
if [[ -e "${DESTINATION}" ]]; then
    echo "ERROR: release already exists and is immutable: ${DESTINATION}" >&2
    exit 1
fi

install -d -m 0755 "${ROOT}/releases" "${DESTINATION}"
cp -a "${SOURCE_DIR}/." "${DESTINATION}/"
# cp -a preserves the source release root mode. Builder staging directories are
# commonly created by mktemp as 0700, so restore traversal access for the
# non-root service account after the copy.
find "${DESTINATION}" -type d -exec chmod 0755 {} +
chmod 0755 "${DESTINATION}/bin/contextstore-server"

cat >"${DESTINATION}/install.env" <<EOF
RELEASE_ID=${RELEASE_ID}
INSTALLED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

(
    cd "${DESTINATION}"
    find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"${DESTINATION}/manifest.sha256"
chmod 0644 "${DESTINATION}/manifest.sha256"

echo "Installed immutable release: ${DESTINATION}"
echo "Verify with: sha256sum -c ${DESTINATION}/manifest.sha256"

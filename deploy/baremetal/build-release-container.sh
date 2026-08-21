#!/usr/bin/env bash
# Build an immutable ContextStore server release in a disposable Linux container.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
RELEASE_ID=
IMAGE=docker.io/library/rust:1.81-bookworm

usage() {
    cat <<'EOF'
Usage: build-release-container.sh --release-id <id> [options]

Builds the standard deployment release in a disposable Podman container. The
source tree is mounted at /workspace and the release is written to
<source>/artifacts/releases/<id>.

Options:
  --source <directory>  ContextStore source tree; default is this script's repository
  --image <image>       Builder image; default: docker.io/library/rust:1.81-bookworm
EOF
}

while (($#)); do
    case "$1" in
        --release-id) RELEASE_ID="${2:?missing release id}"; shift 2 ;;
        --source) SOURCE_DIR=$(cd -- "${2:?missing source directory}" && pwd); shift 2 ;;
        --image) IMAGE="${2:?missing image}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "${RELEASE_ID}" ]] || { usage >&2; exit 2; }
[[ "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || {
    echo "ERROR: invalid release id: ${RELEASE_ID}" >&2
    exit 2
}
[[ -f "${SOURCE_DIR}/Makefile" && -x "${SOURCE_DIR}/kv-service/deploy/baremetal/build-release.sh" ]] || {
    echo "ERROR: not a ContextStore source tree: ${SOURCE_DIR}" >&2
    exit 1
}
command -v podman >/dev/null || { echo "ERROR: Podman is required" >&2; exit 1; }

echo "==> Building release in Podman"
echo "    source:  ${SOURCE_DIR}"
echo "    release: ${RELEASE_ID}"
echo "    image:   ${IMAGE}"
podman run --rm \
    --volume "${SOURCE_DIR}:/workspace:Z" \
    --workdir /workspace \
    "${IMAGE}" \
    bash -ec '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install --no-install-recommends -y \
            build-essential clang libibverbs-dev pkg-config protobuf-compiler
        kv-service/deploy/baremetal/build-release.sh \
            --release-id "$1" \
            --output "/workspace/artifacts/releases/$1"
    ' -- "${RELEASE_ID}"

#!/usr/bin/env bash
# Build a reproducible, immutable ContextStore server release on a Linux builder.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# The toolkit lives at <repo>/deploy/baremetal/, so the repository root is two
# levels up. Resolve through git when available so the script also works from
# a copied toolkit directory pointed at a checkout via --repo.
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
RELEASE_ID=
OUTPUT_DIR=

usage() {
    cat <<'EOF'
Usage: build-release.sh --release-id <id> [--output <directory>] [--repo <checkout>]

Builds the standard deployment feature set and creates an immutable release
directory containing bin/contextstore-server, release.env, and manifest.sha256.
Run this on a Linux builder with the required Rust, io_uring, and RDMA build
dependencies. The output directory is never overwritten.

--repo overrides source-tree autodetection (needed when this script is run
from a copied toolkit directory instead of <repo>/deploy/baremetal/).
EOF
}

while (($#)); do
    case "$1" in
        --release-id) RELEASE_ID="${2:?missing release id}"; shift 2 ;;
        --output) OUTPUT_DIR="${2:?missing output directory}"; shift 2 ;;
        --repo) REPO_ROOT=$(cd -- "${2:?missing repo directory}" && pwd); shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${RELEASE_ID}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${RELEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    echo "ERROR: invalid release id: ${RELEASE_ID}" >&2
    exit 2
fi
# Fail loudly when the resolved root is not actually a ContextStore checkout —
# a mis-resolved root previously made `make build` fail with a confusing
# "No rule to make target" from an unrelated directory.
if [[ ! -f "${REPO_ROOT}/Makefile" || ! -d "${REPO_ROOT}/kv-service" ]]; then
    echo "ERROR: ${REPO_ROOT} is not a ContextStore repository root" >&2
    echo "       (run from <repo>/deploy/baremetal/ or pass --repo <checkout>)" >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "ERROR: releases with io_uring and RDMA must be built on Linux" >&2
    exit 1
fi
# Immutable releases must come from clean, committed source: a dirty tree
# makes SOURCE_COMMIT a lie and the build unreproducible.
if [[ -n "$(git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    echo "ERROR: source tree has uncommitted changes; releases must be built from clean source" >&2
    git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" status --short >&2 || true
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/artifacts/releases/${RELEASE_ID}"
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "ERROR: release output already exists and is immutable: ${OUTPUT_DIR}" >&2
    exit 1
fi

# The standard builder setup installs Rust under this isolated prefix, keeping
# the storage-node runtime package set independent from the compiler version.
if [[ -x /opt/contextstore/toolchains/cargo/bin/cargo ]]; then
    export CARGO_HOME=/opt/contextstore/toolchains/cargo
    export RUSTUP_HOME=/opt/contextstore/toolchains/rustup
    export PATH="/opt/contextstore/toolchains/cargo/bin:${PATH}"
fi

echo "==> Building ContextStore deployment release"
echo "    release: ${RELEASE_ID}"
echo "    source:  ${REPO_ROOT}"
echo "    output:  ${OUTPUT_DIR}"
echo "    rustc:   $(rustc --version)"
echo "    cargo:   $(cargo --version)"
make -C "${REPO_ROOT}" build

SERVER_BIN="${REPO_ROOT}/target/release/contextstore-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "ERROR: expected server binary was not built: ${SERVER_BIN}" >&2
    exit 1
fi

GIT_COMMIT=$(git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" rev-parse HEAD)
GIT_DESCRIBE=$(git -c safe.directory="${REPO_ROOT}" -C "${REPO_ROOT}" describe --always --dirty --tags 2>/dev/null || true)
OUTPUT_PARENT=$(dirname -- "${OUTPUT_DIR}")
install -d -m 0755 "${OUTPUT_PARENT}"
TMP_OUTPUT_DIR=$(mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX")
trap 'rm -rf "${TMP_OUTPUT_DIR}"' EXIT

install -d -m 0755 "${TMP_OUTPUT_DIR}/bin"
install -m 0755 "${SERVER_BIN}" "${TMP_OUTPUT_DIR}/bin/contextstore-server"

cat >"${TMP_OUTPUT_DIR}/release.env" <<EOF
RELEASE_ID=${RELEASE_ID}
SOURCE_COMMIT=${GIT_COMMIT}
SOURCE_DESCRIBE=${GIT_DESCRIBE}
SERVER_FEATURES=io-uring,rdma,metrics
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUILT_BY=$(id -un)@$(hostname -f 2>/dev/null || hostname)
EOF

(
    cd "${TMP_OUTPUT_DIR}"
    find . -type f ! -name manifest.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"${TMP_OUTPUT_DIR}/manifest.sha256"

mv "${TMP_OUTPUT_DIR}" "${OUTPUT_DIR}"
trap - EXIT

echo "==> Release ready: ${OUTPUT_DIR}"
echo "    server:   ${OUTPUT_DIR}/bin/contextstore-server"
echo "    manifest: ${OUTPUT_DIR}/manifest.sha256"

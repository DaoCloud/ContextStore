#!/usr/bin/env bash
# Install the Linux-only build dependencies for a ContextStore release builder.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: install-builder-prereqs.sh [--skip-update]

Installs C/C++ build tools, Protobuf, Clang, RDMA development headers, and an
isolated current Rust stable toolchain under /opt/contextstore/toolchains. Do
not run this on a minimal runtime-only storage node unless that node is
intentionally designated as the release builder.

Optional environment variables:
  RUSTUP_INIT_URL     Rustup bootstrap binary URL (defaults to the official URL)
  RUSTUP_DIST_SERVER  Rust toolchain distribution server (defaults to official)
  RUSTUP_UPDATE_ROOT  Rustup update server (defaults to official)
EOF
}

SKIP_UPDATE=false
while (($#)); do
    case "$1" in
        --skip-update) SKIP_UPDATE=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ${EUID} -ne 0 ]]; then
    echo "ERROR: run as root" >&2
    exit 1
fi
if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: cannot identify the operating system" >&2
    exit 1
fi
. /etc/os-release
if [[ "${ID}" != debian && "${ID_LIKE:-}" != *debian* ]]; then
    echo "ERROR: this script supports Debian and Ubuntu only; detected ${PRETTY_NAME}" >&2
    exit 1
fi
command -v apt-get >/dev/null || { echo "ERROR: apt-get is required" >&2; exit 1; }

if [[ "${SKIP_UPDATE}" == false ]]; then
    apt-get update
fi
apt-get install --no-install-recommends -y \
    build-essential clang curl git libibverbs-dev librdmacm-dev pkg-config protobuf-compiler

TOOLCHAIN_ROOT=/opt/contextstore/toolchains
export CARGO_HOME="${TOOLCHAIN_ROOT}/cargo"
export RUSTUP_HOME="${TOOLCHAIN_ROOT}/rustup"
export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://static.rust-lang.org}"
export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://static.rust-lang.org/rustup}"
RUSTUP_INIT_URL="${RUSTUP_INIT_URL:-https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init}"
install -d -m 0755 "${TOOLCHAIN_ROOT}"

if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
    installer=$(mktemp)
    trap 'rm -f "${installer}"' EXIT
    curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
        "${RUSTUP_INIT_URL}" --output "${installer}"
    chmod 0755 "${installer}"
    RUSTUP_INIT_SKIP_PATH_CHECK=yes "${installer}" \
        -y --profile minimal --default-toolchain stable --no-modify-path
    rm -f "${installer}"
    trap - EXIT
else
    "${CARGO_HOME}/bin/rustup" toolchain install stable --profile minimal
    "${CARGO_HOME}/bin/rustup" default stable
fi

export PATH="${CARGO_HOME}/bin:${PATH}"

echo "==> ContextStore builder prerequisites installed"
rustc --version
cargo --version
clang --version | head -1
pkg-config --modversion libibverbs
protoc --version

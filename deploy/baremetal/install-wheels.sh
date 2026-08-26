#!/usr/bin/env bash
# Install pinned ContextStore and Redhare wheels into an existing application Python environment.
set -euo pipefail

ARTIFACT_DIR=/opt/contextstore/artifacts/wheels
PYTHON=
CONTEXTSTORE_WHEEL=
REDHARE_WHEEL=

usage() {
    cat <<'EOF'
Usage: install-wheels.sh --python <python-path> --contextstore-wheel <file> [options]

Options:
  --redhare-wheel <file>       Install this pinned Redhare wheel too
  --artifact-dir <directory>   Default: /opt/contextstore/artifacts/wheels

Wheel names are resolved relative to --artifact-dir unless an absolute path is supplied.
Dependencies are intentionally not resolved, so this does not mutate the vLLM runtime stack.
EOF
}

resolve_wheel() {
    local value=$1 path
    if [[ "${value}" = /* ]]; then
        path=${value}
    else
        path="${ARTIFACT_DIR}/${value}"
    fi
    [[ -f "${path}" ]] || { echo "ERROR: wheel not found: ${path}" >&2; exit 1; }
    printf '%s\n' "${path}"
}

while (($#)); do
    case "$1" in
        --python) PYTHON="${2:?missing python path}"; shift 2 ;;
        --contextstore-wheel) CONTEXTSTORE_WHEEL="${2:?missing wheel}"; shift 2 ;;
        --redhare-wheel) REDHARE_WHEEL="${2:?missing wheel}"; shift 2 ;;
        --artifact-dir) ARTIFACT_DIR="${2:?missing artifact directory}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${PYTHON}" || -z "${CONTEXTSTORE_WHEEL}" ]]; then
    usage >&2
    exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "ERROR: Python interpreter is not executable: ${PYTHON}" >&2; exit 1; }

contextstore_path=$(resolve_wheel "${CONTEXTSTORE_WHEEL}")
packages=("${contextstore_path}")
if [[ -n "${REDHARE_WHEEL}" ]]; then
    packages+=("$(resolve_wheel "${REDHARE_WHEEL}")")
fi

"${PYTHON}" -m pip install --force-reinstall --no-deps "${packages[@]}"
"${PYTHON}" - <<'PY'
import contextstore
print(f"ContextStore installed: {contextstore.__version__}")
try:
    import redhare
    print(f"Redhare installed: {getattr(redhare, '__version__', 'unknown')}")
except ModuleNotFoundError:
    pass
PY

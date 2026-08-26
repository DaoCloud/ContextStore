#!/usr/bin/env bash
# Standard service lifecycle and log access wrapper.
set -euo pipefail

SERVICE=contextstore-kvservice
ACTION=${1:-status}

case "${ACTION}" in
    start|stop|restart)
        systemctl "${ACTION}" "${SERVICE}"
        ;;
    status)
        systemctl status "${SERVICE}" --no-pager
        ;;
    validate)
        # The server has no --check-config flag; validate by parsing the TOML
        # and confirming the binary starts far enough to read it. A config
        # error surfaces as a non-zero exit with the parse error on stderr.
        if /opt/contextstore/current/bin/contextstore-server --help 2>/dev/null | grep -q -- --check-config; then
            exec /opt/contextstore/current/bin/contextstore-server \
                --config /etc/contextstore/server.toml --check-config
        fi
        echo "NOTE: server binary has no --check-config; running TOML syntax check only" >&2
        python3 - /etc/contextstore/server.toml <<'PYEOF'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    tomllib.load(f)
print("config parse OK:", sys.argv[1])
PYEOF
        ;;
    logs)
        journalctl -u "${SERVICE}" -f
        ;;
    recent-logs)
        journalctl -u "${SERVICE}" -n "${2:-200}" --no-pager
        ;;
    *)
        echo "Usage: service.sh {start|stop|restart|status|validate|logs|recent-logs [count]}" >&2
        exit 2
        ;;
esac

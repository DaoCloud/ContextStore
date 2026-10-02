#!/usr/bin/env bash
set -euo pipefail

# Prefer guest `sudo poweroff` over this forced stop to preserve qcow2 state.

for session in cs-railvm-client-20261002 cs-railvm-server-20261002; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
  fi
done

#!/usr/bin/env bash
set -euo pipefail

for session in cs-softroce-server cs-softroce-redis; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
  fi
done

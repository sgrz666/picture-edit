#!/usr/bin/env bash
# ==============================================================================
# stop.sh — Gracefully Terminate Dashboard Daemon
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$BASE_DIR/logs/dashboard.pid"

# 1. Check if PID file exists
if [[ ! -f "$PID_FILE" ]]; then
  # Fallback check scoped to this dashboard directory's server process
  EXISTING_PID=$(pgrep -f "$BASE_DIR/src/server.py" 2>/dev/null || true)
  if [[ -z "$EXISTING_PID" ]]; then
    echo "[INFO] Dashboard is not running (no PID file found)."
    exit 0
  fi
else
  EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
fi

# 2. Check if process is actually alive
if [[ -z "$EXISTING_PID" ]] || ! kill -0 "$EXISTING_PID" 2>/dev/null; then
  echo "[INFO] Dashboard is not running (stale PID removed)."
  rm -f "$PID_FILE"
  exit 0
fi

# 3. Send SIGTERM first for graceful cleanup
echo "[INFO] Stopping Dashboard (PID: $EXISTING_PID) with SIGTERM..."
kill -15 "$EXISTING_PID" 2>/dev/null || true

# Wait up to 10 seconds for graceful shutdown
KILLED=0
for i in {1..20}; do
  sleep 0.5
  if ! kill -0 "$EXISTING_PID" 2>/dev/null; then
    KILLED=1
    break
  fi
done

if [[ "$KILLED" -eq 1 ]]; then
  echo "[SUCCESS] Dashboard stopped gracefully."
  rm -f "$PID_FILE"
  exit 0
fi

# 4. If still alive after 10s, escalate to SIGKILL
echo "[WARN] Dashboard did not terminate within 10s. Escalating to SIGKILL..."
kill -9 "$EXISTING_PID" 2>/dev/null || true
for i in {1..10}; do
  sleep 0.2
  if ! kill -0 "$EXISTING_PID" 2>/dev/null; then
    break
  fi
done
rm -f "$PID_FILE"
echo "[SUCCESS] Dashboard terminated with SIGKILL."
exit 0

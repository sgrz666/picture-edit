#!/usr/bin/env bash
# ==============================================================================
# status.sh — Query Status and Health of Dashboard Daemon
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$BASE_DIR/logs/dashboard.pid"
PORT="${PORT:-9527}"

# Parse optional command-line flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# 1. Verify PID file presence
if [[ ! -f "$PID_FILE" ]]; then
  echo "[STATUS] STOPPED (No PID file)"
  exit 3
fi

PID=$(cat "$PID_FILE" 2>/dev/null || true)

# 2. Verify process liveness
if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  echo "[STATUS] STOPPED (Stale PID: ${PID:-none})"
  rm -f "$PID_FILE"
  exit 3
fi

# 3. Report process runtime details
echo "=========================================================================="
echo " [STATUS] Dashboard is RUNNING"
echo " PID:          $PID"
PROC_INFO=$(ps -p "$PID" -o pid,user,%cpu,%mem,etime,cmd --no-headers 2>/dev/null | awk '{print "User: "$2" | CPU: "$3"% | MEM: "$4"% | Uptime: "$5}' || echo "N/A")
echo " Process Info: $PROC_INFO"

# 4. Probe API endpoint latency
START_TIME=$(date +%s%N 2>/dev/null || echo 0)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://127.0.0.1:$PORT/api/health" 2>/dev/null || echo "FAIL")
if [[ "$START_TIME" != "0" ]]; then
  END_TIME=$(date +%s%N 2>/dev/null || echo 0)
  LATENCY_MS=$(( (END_TIME - START_TIME) / 1000000 ))
else
  LATENCY_MS="<5"
fi

if [[ "$HTTP_CODE" == "200" ]]; then
  echo " Health Probe: 200 OK (${LATENCY_MS} ms)"
else
  echo " Health Probe: FAILED (HTTP Status: $HTTP_CODE)"
fi
echo " Dashboard URL: http://172.24.247.154:$PORT"
echo " Log Tail:     $(tail -n 1 "$BASE_DIR/logs/dashboard.log" 2>/dev/null || echo "No logs")"
echo "=========================================================================="
exit 0

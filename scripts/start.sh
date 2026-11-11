#!/usr/bin/env bash
# ==============================================================================
# start.sh — Launch GPU & RAM Monitoring Dashboard Daemon
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

PORT="${PORT:-9527}"
HOST="${HOST:-0.0.0.0}"
PID_FILE="$BASE_DIR/logs/dashboard.pid"
LOG_FILE="$BASE_DIR/logs/dashboard.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024)) # 10MB

# Parse optional command-line flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$BASE_DIR/logs"

# 1. Check if process is already running (idempotency check)
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "[WARN] Dashboard is ALREADY running (PID: $EXISTING_PID)."
    echo "[INFO] Check status via: ./scripts/status.sh --port $PORT"
    exit 0
  else
    echo "[INFO] Removing stale PID file: $PID_FILE"
    rm -f "$PID_FILE"
  fi
fi

# 2. Pre-flight Port Conflict Check
PORT_OCCUPIED=0
if ss -tuln 2>/dev/null | grep -qE "(:|\])$PORT\b"; then
  PORT_OCCUPIED=1
elif python3 -c "import socket, sys; s = socket.socket(); s.settimeout(0.3); res = s.connect_ex(('127.0.0.1', int('$PORT'))); s.close(); sys.exit(0 if res == 0 else 1)" 2>/dev/null; then
  PORT_OCCUPIED=1
fi

if [[ "$PORT_OCCUPIED" -eq 1 ]]; then
  echo "[ERROR] Port $PORT is already in use by another service!"
  echo "[DIAGNOSTIC]:"
  ss -tulpn 2>/dev/null | grep -E "(:|\])$PORT\b" || true
  echo "[SUGGESTION] Start with an alternate port: ./scripts/start.sh --port <port>"
  exit 1
fi

# 3. Log Rotation if > 10MB
if [[ -f "$LOG_FILE" ]]; then
  LOG_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || wc -c < "$LOG_FILE" || echo 0)
  if [[ "$LOG_SIZE" -gt "$MAX_LOG_SIZE" ]]; then
    mv "$LOG_FILE" "${LOG_FILE}.1"
    echo "[INFO] Rotated log file ($LOG_SIZE bytes) to ${LOG_FILE}.1"
  fi
fi

# 4. Set Compat Library Path for NVML Driver Mismatch Workaround
if [[ -d "$BASE_DIR/compat" ]]; then
  export LD_LIBRARY_PATH="$BASE_DIR/compat:${LD_LIBRARY_PATH:-}"
elif [[ -d "/data/shangguanrz/仪表盘/compat" ]]; then
  export LD_LIBRARY_PATH="/data/shangguanrz/仪表盘/compat:${LD_LIBRARY_PATH:-}"
fi

# 5. Determine Python 3 binary
if command -v /usr/bin/python3 >/dev/null 2>&1; then
  PYTHON_BIN="/usr/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

# 6. Launch Daemon in background
echo "[INFO] Starting GPU & RAM Monitoring Dashboard daemon on $HOST:$PORT..."
nohup "$PYTHON_BIN" "$BASE_DIR/src/server.py" --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# 7. Post-launch Healthcheck Verification (up to 6 seconds)
HEALTHY=0
for i in {1..12}; do
  sleep 0.5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] Dashboard process exited immediately. Last 15 lines of log:"
    tail -n 15 "$LOG_FILE" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/api/health" > /dev/null 2>&1; then
    HEALTHY=1
    break
  fi
done

if [[ "$HEALTHY" -eq 1 ]]; then
  echo "=========================================================================="
  echo " [SUCCESS] GPU & RAM Monitoring Dashboard is RUNNING!"
  echo " PID:         $SERVER_PID"
  echo " Port:        $PORT"
  echo " Local URL:   http://127.0.0.1:$PORT"
  echo " LAN URL:     http://172.24.247.154:$PORT"
  echo " SSH Tunnel:  ssh -L $PORT:127.0.0.1:$PORT sg -N"
  echo " Log File:    $LOG_FILE"
  echo "=========================================================================="
  exit 0
else
  echo "[WARN] Process running (PID: $SERVER_PID) but HTTP health probe did not respond within 6s."
  echo "[INFO] Check logs: tail -f $LOG_FILE"
  exit 0
fi

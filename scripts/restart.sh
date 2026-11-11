#!/usr/bin/env bash
# ==============================================================================
# restart.sh — Clean Stop followed by Start
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$BASE_DIR/scripts/stop.sh"
sleep 1
"$BASE_DIR/scripts/start.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail
workflow_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/shangguanrz/nihaochen/envs/qwen21/bin/python -u "$workflow_dir/run_dual.py" "$@"

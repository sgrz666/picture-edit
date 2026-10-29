#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${QWEN_PYTHON:-/home/shangguanrz/nihaochen/envs/qwen21/bin/python}" -B -u "$tool_dir/campaign.py" "$@"

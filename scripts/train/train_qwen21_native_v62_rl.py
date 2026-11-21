#!/usr/bin/env python3
"""Native V6.2 rollout/replay entry point."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == '__main__':
    from src.pose_control.qwen21_full.native_rl.cli import main

    main()

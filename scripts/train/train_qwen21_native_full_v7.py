#!/usr/bin/env python3
"""Native Full V7; default prepare resolves configuration without training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    from src.pose_control.qwen21_full.native_v7.training import main as shared_main
    return shared_main(argv)


if __name__ == '__main__':
    main()

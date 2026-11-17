#!/usr/bin/env python3
"""Explicit V6.2 to V7 initialization; reuse immutable migration and readback."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    from scripts.data.migrate_qwen21_native_full_v62 import main as shared_main
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    return shared_main(argv, source_version='v62', target_version='v7',
                       runtime_type=NativeFullRuntimeV7, pose_config_type=PoseLinkConfigV7)


if __name__ == '__main__':
    main()

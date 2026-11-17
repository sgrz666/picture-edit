#!/usr/bin/env python3
"""V7 inference reuses native V6.2 conditioning, sampling and output safeguards."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SWITCHES=('mask_pose_fusion','spatial_rope','geometry_contact','identity_modulation')


def add_arguments(parser):
    for name in SWITCHES:
        parser.add_argument('--'+name.replace('_','-'),action=argparse.BooleanOptionalAction,default=None,
                            help='Explicit inference ablation; checkpoint configuration is used when omitted')


def configure_runtime(runtime,args):
    original=runtime.pose_config.to_dict()
    overrides={name:getattr(args,name) for name in SWITCHES if getattr(args,name) is not None}
    runtime.set_ablation(**overrides)
    return {'checkpoint_pose_config':original,'paper_switch_overrides':overrides}


def main(argv=None):
    from scripts.run.run_qwen21_native_full_v62 import main as shared_main
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    from src.pose_control.qwen21_full.native_v7.integration import install_native_full_v7
    return shared_main(argv, runtime_type=NativeFullRuntimeV7, installer=install_native_full_v7,
                       architecture='native_full_v7',add_arguments=add_arguments,configure_runtime=configure_runtime)


if __name__ == '__main__':
    main()

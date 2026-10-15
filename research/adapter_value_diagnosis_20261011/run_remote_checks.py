"""Run existing remote checks with temporary aliases for relocated scripts.

Usage from the local PowerShell workspace:
  Get-Content -Raw <this-file> | ssh sg <qwen21-python> - --mode tests
  Get-Content -Raw <this-file> | ssh sg <qwen21-python> - --mode parity

Aliases exist only in this Python process. Source files and weights are not
modified. Parity writes diagnostic outputs to a separate directory.
"""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('tests', 'parity'), required=True)
args = parser.parse_args()
root = Path('/home/shangguanrz/project/pic-edit')
os.chdir(root)
sys.path.insert(0, str(root))
aliases = {
    'scripts.precompute_iper_face_features_v65':
        'scripts.data.precompute_iper_face_features_v65',
    'scripts.precompute_iper_hand_features_v66':
        'scripts.data.precompute_iper_hand_features_v66',
}
for old, new in aliases.items():
    sys.modules[old] = importlib.import_module(new)
print('Temporary import aliases:', aliases, flush=True)

if args.mode == 'tests':
    import pytest
    raise SystemExit(pytest.main([
        '-q', 'tests/test_qwen21_full_preservation.py',
        'tests/test_qwen21_full_bridge.py', 'tests/test_qwen21_joint_training.py',
    ]))

sys.argv = [
    'check_qwen21_native_value_prediction',
    '--pair-ids', 'iper_00', 'tiktok_00', '--resolution', '512',
    '--output', '/home/shangguanrz/nihaochen/qwen21_full/'
                'adapter_value_diagnosis_20261011/parity_512',
]
runpy.run_module('scripts.audit.check_qwen21_native_value_prediction',
                run_name='__main__')

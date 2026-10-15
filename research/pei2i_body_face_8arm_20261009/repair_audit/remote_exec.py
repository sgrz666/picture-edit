"""Run a UTF-8 audit script on sg without cross-shell interpolation."""
import argparse
import base64
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
parser.add_argument('--python', default='/home/shangguanrz/nihaochen/envs/qwen21/bin/python')
args = parser.parse_args()
if not args.python.startswith('/') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/-.' for c in args.python):
    raise ValueError('Remote Python must be a plain absolute path')
encoded = base64.b64encode(args.script.read_bytes()).decode('ascii')
command = args.python + " -c 'import base64;exec(base64.b64decode(\"" + encoded + "\"))'"
raise SystemExit(subprocess.run(['ssh', 'sg', command], check=False).returncode)

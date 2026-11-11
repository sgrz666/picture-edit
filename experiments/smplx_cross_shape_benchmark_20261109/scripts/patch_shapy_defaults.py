"""Patch the remaining multi-line Metrics default in defaults.py."""
from __future__ import annotations

import re

import config

p = config.SHAPY_REPO / "regressor/human_shape/config/defaults.py"
s = p.read_text()
pattern = re.compile(r"    body: Metrics = Metrics\(.*?\n    \)", re.S)
replacement = """    body: Metrics = field(default_factory=lambda: Metrics(
        mpjpe=MPJPE(root_joints=('left_hip', 'right_hip')),
        fscores_thresh=(10.0 / 1000, 20.0 / 1000, 50.0 / 1000,
                        75.0 / 1000, 100.0 / 1000),
    ))"""
s2, n = pattern.subn(replacement, s, count=1)
if n != 1:
    raise SystemExit("could not patch Evaluation.body")
p.write_text(s2)
print("[defaults] body patched")

"""Py3.12 dataclass defaults fix for SHAPY's attributes/config.py."""
from __future__ import annotations

import re
from pathlib import Path

import config

p = config.SHAPY_REPO / "attributes/attributes/utils/config.py"
s = p.read_text()
# All dataclass fields of the form ``name: Type = Type()``.
s2, n = re.subn(
    r"(\b\w+): (\w+) = \2\(\)",
    r"\1: \2 = field(default_factory=\2)",
    s,
)
p.write_text(s2)
print("[attrs] changed", n)

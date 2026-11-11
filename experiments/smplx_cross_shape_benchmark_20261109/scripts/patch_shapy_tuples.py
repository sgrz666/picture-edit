"""Modern OmegaConf needs Tuple[T, ...] for variable-length tuple defaults."""
from __future__ import annotations

import re
from pathlib import Path

import config

root = config.SHAPY_REPO / "regressor/human_shape/config"
pattern = re.compile(r"Tuple\[(\w+)\]")
total = 0
for p in sorted(root.rglob("*.py")):
    s = p.read_text()
    s2, n = pattern.subn(r"Tuple[\1, ...]", s)
    if n:
        p.write_text(s2)
        total += n
        print(f"[tuple] {p.name}: {n}")
print("[tuple] total", total)

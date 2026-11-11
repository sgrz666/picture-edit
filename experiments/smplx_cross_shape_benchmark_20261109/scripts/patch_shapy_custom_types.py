"""Expand SHAPY's custom NewType annotations for modern OmegaConf."""
from __future__ import annotations

import re
from pathlib import Path

import config

root = config.SHAPY_REPO / "regressor/human_shape/config"
total = 0
for p in sorted(root.rglob("*.py")):
    s = p.read_text()
    s2 = re.sub(r":\s*StringTuple\b", ": Tuple[str, ...]", s)
    s2 = re.sub(r":\s*FloatTuple\b", ": Tuple[float, ...]", s2)
    s2 = s2.replace("Optional[StringTuple]", "Optional[Tuple[str, ...]]")
    if s2 != s:
        if "Tuple" not in s2.split("from typing import", 1)[-1].split("\n", 1)[0]:
            s2 = s2.replace("from typing import", "from typing import Tuple,", 1)
        p.write_text(s2)
        n = len(re.findall(r"Tuple\[str, \.\.\.\]", s2)) - len(re.findall(r"Tuple\[str, \.\.\.\]", s))
        total += max(n, 1)
        print(f"[types] {p.name}")
print("[types] files", total)

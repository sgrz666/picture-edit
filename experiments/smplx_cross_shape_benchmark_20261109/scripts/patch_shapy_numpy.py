"""Replace numpy scalar aliases removed in numpy >= 1.24."""
from __future__ import annotations

import re
from pathlib import Path

import config

roots = [config.SHAPY_REPO / "regressor", config.SHAPY_REPO / "attributes"]
pattern = re.compile(r"\bnp\.(int|float|bool|object|str|long)\b")
repl = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "object": "object",
    "str": "str",
    "long": "int",
}
total = 0
for root in roots:
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        s = p.read_text()
        def sub(m):
            return repl[m.group(1)]
        s2, n = pattern.subn(sub, s)
        if n:
            p.write_text(s2)
            total += n
            print(f"[numpy] {p.relative_to(config.SHAPY_REPO)}: {n}")
print("[numpy] total", total)

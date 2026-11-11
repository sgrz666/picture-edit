"""Py3.12 mutable dataclass defaults across SHAPY's config modules."""
from __future__ import annotations

import re
from pathlib import Path

import config

root = config.SHAPY_REPO / "regressor/human_shape/config"
# name: Type = Type(...), allowing one level of nested parentheses and
# multi-line arguments (e.g. Stage(num_blocks=(4, 4), ...)).
pattern = re.compile(r"(\b\w+): (\w+) = \2\((?:[^()]|\([^()]*\))*\)", re.S)


def repl(m):
    name, typ = m.group(1), m.group(2)
    expr = m.group(0).split("=", 1)[1].strip()
    return f"{name}: {typ} = field(default_factory=lambda: {expr})"


total = 0
for p in sorted(root.rglob("*.py")):
    s = p.read_text()
    s2, n = pattern.subn(repl, s)
    if n:
        head = "\n".join(s2.splitlines()[:8])
        if "field" not in head:
            if "from dataclasses import dataclass, make_dataclass" in s2:
                s2 = s2.replace(
                    "from dataclasses import dataclass, make_dataclass",
                    "from dataclasses import dataclass, make_dataclass, field", 1)
            else:
                s2 = s2.replace(
                    "from dataclasses import dataclass\n",
                    "from dataclasses import dataclass, field\n", 1)
        p.write_text(s2)
        total += n
        print(f"[dataclass] {p.name}: {n}")
print("[dataclass] total", total)

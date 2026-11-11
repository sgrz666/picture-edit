"""Py3.12: make_dataclass cannot use numpy arrays as mutable defaults."""
from __future__ import annotations

from pathlib import Path

import config

p = config.SHAPY_REPO / "regressor/human_shape/utils/data_structs.py"
s = p.read_text()
s = s.replace(
    """                    key, type(val), field(
                        default_factory=dict if isinstance(val, dict) else None,
                    )]""",
    """                    key, type(val), field(
                        default_factory=lambda v=val: v,
                    )]""",
)
s = s.replace(
    """                    [key, type(val), field(
                        default=val,
                        #  default_factory=dict if isinstance(val, dict) else None,
                    )]""",
    """                    [key, type(val), field(
                        default_factory=lambda v=val: v,
                    )]""",
)
if "default=val" in s:
    raise SystemExit("Struct patch did not apply")
p.write_text(s)
print("[struct] patched")

"""Qualify the nested HRNet.Stage default factories."""
from __future__ import annotations

import config

p = config.SHAPY_REPO / "regressor/human_shape/config/network_defaults.py"
s = p.read_text()
a = s.count("field(default_factory=Stage)")
b = s.count("field(default_factory=lambda: Stage(")
s = s.replace("field(default_factory=Stage)", "field(default_factory=lambda: HRNet.Stage())")
s = s.replace("field(default_factory=lambda: Stage(", "field(default_factory=lambda: HRNet.Stage(")
p.write_text(s)
print("[stage] simple", a, "lambda", b)

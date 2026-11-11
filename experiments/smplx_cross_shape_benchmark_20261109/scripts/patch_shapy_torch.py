"""torch 2.8: Checkpointer must load the 2022 checkpoint with weights_only=False."""
from __future__ import annotations

import config

p = config.SHAPY_REPO / "regressor/human_shape/utils/checkpointer.py"
s = p.read_text()
old = "torch.load(latest_ckpt_fn, map_location=map_location)"
new = "torch.load(latest_ckpt_fn, map_location=map_location, weights_only=False)"
n = s.count(old)
s = s.replace(old, new)
p.write_text(s)
print("[torch] patched", n, "checkpoint loads")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run the existing SMPLest-X inference pipeline without pyrender overlay.

Important:
- Does NOT modify main/inference.py.
- Does NOT modify model inference.
- Does NOT modify detector.
- Does NOT modify SMPL-X outputs.
- Only replaces utils.visualization_utils.render_mesh() with
  an identity function, because TikTok asset construction only
  needs exported SMPL-X parameters / mesh_cam / camera data.
"""

import runpy

# ------------------------------------------------------------
# Ensure the local SMPLest-X packages take precedence over
# site-packages (notably HuggingFace's `datasets` package).
# ------------------------------------------------------------
import sys
from pathlib import Path

SMPL_ROOT = Path(__file__).resolve().parents[1]

smpl_root_str = str(SMPL_ROOT)

if smpl_root_str in sys.path:
    sys.path.remove(smpl_root_str)

sys.path.insert(
    0,
    smpl_root_str,
)

# If an external `datasets` package was already imported,
# remove it so `datasets.dataset` resolves to SMPLest-X local code.
for _name in list(sys.modules):
    if (
        _name == "datasets"
        or _name.startswith("datasets.")
    ):
        del sys.modules[_name]

import sys
import types

# Make sure the real utils package can still be used for:
#   utils.data_utils
#   utils.inference_utils
import utils  # noqa: F401


# ------------------------------------------------------------
# Stub ONLY utils.visualization_utils.
# This prevents its top-level `import pyrender`.
# ------------------------------------------------------------

stub = types.ModuleType("utils.visualization_utils")


def render_mesh(img, *args, **kwargs):
    # Keep the visualization frame unchanged.
    # This has no effect on SMPL-X estimation or exported params.
    return img


stub.render_mesh = render_mesh

sys.modules["utils.visualization_utils"] = stub


# ------------------------------------------------------------
# Run the existing local inference.py with the original CLI args.
# sys.argv is preserved automatically.
# ------------------------------------------------------------

runpy.run_module(
    "main.inference",
    run_name="__main__",
)

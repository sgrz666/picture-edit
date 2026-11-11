"""Local compatibility patches for the 2022 SHAPY code on torch 2.8.

Idempotent; only touches the experiment's shallow clone.  Run with the
experiment venv or deepgen python (no torch needed).
"""
from __future__ import annotations

import re
from pathlib import Path

import config

REPO = config.SHAPY_REPO


def patch_resnet():
    p = REPO / "regressor/human_shape/models/backbone/resnet.py"
    s = p.read_text()
    correct = """    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }"""
    pattern = re.compile(r"    model_urls = \{\n(?:.*\n)*?    \}", re.M)
    if "resnet18: https://" in s:
        s2, n = pattern.subn(correct, s, count=1)
        if n != 1:
            raise RuntimeError("could not repair resnet model_urls block")
        s = s2
    if "from torchvision.models.resnet import model_urls" not in s:
        # Already replaced by an older patch; ensure the fallback exists.
        if "model_urls = {" not in s:
            raise RuntimeError("resnet patch state unknown")
    p.write_text(s)
    print("[patch] resnet ok")


def patch_img_utils():
    p = REPO / "regressor/human_shape/utils/img_utils.py"
    s = p.read_text()
    if "import jpeg4py as jpeg\n" in s:
        s = s.replace(
            "import jpeg4py as jpeg\n",
            "try:\n    import jpeg4py as jpeg\nexcept Exception:\n    jpeg = None\n",
            1,
        )
        s = s.replace("except jpeg.JPEGRuntimeError:", "except Exception:", 1)
        p.write_text(s)
        print("[patch] img_utils jpeg4py optional")
    else:
        print("[patch] img_utils already patched")


def patch_open3d():
    p = REPO / "regressor/human_shape/utils/np_utils.py"
    s = p.read_text()
    if "import open3d as o3d\n" in s:
        s = s.replace("import open3d as o3d\n", "", 1)
        s = s.replace(
            "def np2o3d_pcl(x: np.ndarray) -> o3d.geometry.PointCloud:\n",
            "def np2o3d_pcl(x: np.ndarray):\n    import open3d as o3d\n",
            1,
        )
        p.write_text(s)
        print("[patch] np_utils open3d lazy")
    else:
        print("[patch] np_utils already patched")
    p = REPO / "regressor/human_shape/utils/metrics.py"
    s = p.read_text()
    if "import open3d as o3d\n" in s:
        s = s.replace("import open3d as o3d\n", "", 1)
        p.write_text(s)
        print("[patch] metrics open3d removed")
    else:
        print("[patch] metrics already patched")


def main():
    patch_resnet()
    patch_img_utils()
    patch_open3d()


if __name__ == "__main__":
    main()

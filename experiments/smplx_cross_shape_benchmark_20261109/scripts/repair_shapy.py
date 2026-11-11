"""One-off repair of the two SHAPY files damaged by a non-idempotent patch."""
from __future__ import annotations

import re
from pathlib import Path

import config

REPO = config.SHAPY_REPO


def repair_img_utils():
    p = REPO / "regressor/human_shape/utils/img_utils.py"
    s = p.read_text()
    bad = (
        "import numpy as np\n"
        "try:\n"
        "    try:\n"
        "    import jpeg4py as jpeg\n"
        "except Exception:\n"
        "    jpeg = None\n"
        "except Exception:\n"
        "    jpeg = None\n"
    )
    good = (
        "import numpy as np\n"
        "try:\n"
        "    import jpeg4py as jpeg\n"
        "except Exception:\n"
        "    jpeg = None\n"
    )
    if bad in s:
        s = s.replace(bad, good, 1)
    elif "try:\n    try:" in s:
        raise RuntimeError("unknown img_utils corruption")
    if "except jpeg.JPEGRuntimeError:" in s:
        s = s.replace("except jpeg.JPEGRuntimeError:", "except Exception:", 1)
    p.write_text(s)
    print("[repair] img_utils", s.splitlines()[:6])


def repair_np_utils():
    p = REPO / "regressor/human_shape/utils/np_utils.py"
    s = p.read_text()
    pattern = re.compile(r"def np2o3d_pcl\(x: np\.ndarray\):(?:.*\n)*?    return pcl")
    good = (
        "def np2o3d_pcl(x: np.ndarray):\n"
        "    import open3d as o3d\n"
        "\n"
        "    pcl = o3d.geometry.PointCloud()\n"
        "    pcl.points = o3d.utility.Vector3dVector(x)\n"
        "\n"
        "    return pcl"
    )
    s2, n = pattern.subn(good, s, count=1)
    if n != 1:
        raise RuntimeError("could not repair np2o3d_pcl")
    s2 = s2.replace("import open3d as o3d\n", "", 1)
    p.write_text(s2)
    print("[repair] np_utils ok")


def main():
    repair_img_utils()
    repair_np_utils()


if __name__ == "__main__":
    main()

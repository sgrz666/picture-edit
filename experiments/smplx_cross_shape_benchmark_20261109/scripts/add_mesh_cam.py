"""One-off: add the exported camera-space mesh (mesh_cam) to the merged params.

Kept separate from prepare_hbw.py so the 705-image SHA pass does not need to be
repeated.  Future runs of prepare_hbw.py include mesh_cam directly.
"""
from __future__ import annotations

import glob

import numpy as np
import torch

import config


def main():
    z = dict(np.load(config.SMPLESTX_PARAMS, allow_pickle=True))
    z.pop("mesh_cam", None)
    paths = sorted(glob.glob(str(config.SMPLESTX_PARAMS_GLOB)))
    meshes = []
    for p in paths:
        params = torch.load(p, map_location="cpu", weights_only=False)
        meshes.append(params["mesh_cam"].detach().cpu().numpy())
    mesh = np.concatenate(meshes, axis=0).astype(np.float32)
    if len(mesh) != len(z["betas"]):
        raise RuntimeError(f"mesh_cam rows {len(mesh)} != betas {len(z['betas'])}")
    np.savez_compressed(config.SMPLESTX_PARAMS, **z, mesh_cam=mesh)
    print("[OK] mesh_cam", mesh.shape, "->", config.SMPLESTX_PARAMS)


if __name__ == "__main__":
    main()

"""Thin SMPL-X forward wrapper for P0 (betas + body_pose -> posed vertices)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SMPLXWrapper:
    def __init__(self, model_dir: str | Path, num_betas: int = 10, device: str = "cpu"):
        import smplx

        self.device = torch.device(device)
        self.num_betas = num_betas
        # smplx.create appends "smplx/SMPLX_NEUTRAL.npz"; accept either the
        # human_model_files root or the smplx subdirectory.
        model_dir = Path(model_dir)
        if model_dir.name == "smplx":
            model_dir = model_dir.parent
        self.model = smplx.create(
            str(model_dir), model_type="smplx", gender="neutral",
            num_betas=num_betas, use_pca=False, batch_size=1,
        ).to(self.device).eval()

    @torch.no_grad()
    def vertices(self, betas, body_pose=None, global_orient=None, transl=None):
        b = np.asarray(betas, dtype=np.float32)
        if b.ndim == 1:
            b = b[None]
        n = b.shape[0]
        if body_pose is None:
            body_pose = np.zeros((n, 63), dtype=np.float32)
        if global_orient is None:
            global_orient = np.zeros((n, 3), dtype=np.float32)
        if transl is None:
            transl = np.zeros((n, 3), dtype=np.float32)
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        out = self.model(
            betas=t(b),
            body_pose=t(body_pose),
            global_orient=t(global_orient),
            transl=t(transl),
            return_verts=True,
        )
        return out.vertices.detach().cpu().numpy()


def apply_pose_batch(wrapper: SMPLXWrapper, betas, body_pose, global_orient=None, transl=None):
    """Convenience: one beta repeated for each pose or vice versa.

    ``betas`` may be (K,10) and ``body_pose`` (P,63); they are broadcast only
    when one of them has length 1.  Returns (N,10475,3).
    """
    b = np.asarray(betas, dtype=np.float32)
    p = np.asarray(body_pose, dtype=np.float32)
    if b.ndim == 1:
        b = b[None]
    if p.ndim == 1:
        p = p[None]
    if len(b) == 1 and len(p) > 1:
        b = np.repeat(b, len(p), axis=0)
    if len(p) == 1 and len(b) > 1:
        p = np.repeat(p, len(b), axis=0)
    if len(b) != len(p):
        raise ValueError(f"beta rows {len(b)} != pose rows {len(p)}")
    g = None if global_orient is None else np.repeat(np.asarray(global_orient, np.float32).reshape(1, 3), len(b), axis=0)
    t = None if transl is None else np.repeat(np.asarray(transl, np.float32).reshape(1, 3), len(b), axis=0)
    return wrapper.vertices(b, p, g, t)

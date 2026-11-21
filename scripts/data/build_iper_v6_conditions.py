#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build DeepGen V6.3 Native Adapter conditions for iPER dataset.

Generates:
1. 14-Class Part Map: rendered via nvdiffrast using SMPL-X skinning weights on mesh_cam.
   Saved as 512x512 uint8 PNG in iper_assets_512_nvdiffrast/{app}/part_512/{role}/{stem}.png
   (0 = background, 1..14 = body parts matching conditions.PART_NAMES)
2. SMPL-X Global 26D parameters + 25-keypoint coordinates:
   Saved in iper_assets_512_nvdiffrast/{app}/v6_conditions.pt containing:
   - smplx_global: [70, 26] (betas[10] + rot6d[6] + transl[3] + cam[7])
   - kps_25_coords: [70, 25, 3] (x_norm, y_norm, score)
   - stem_to_idx: dict mapping stem to frame index 0..69
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Try importing nvdiffrast
try:
    import nvdiffrast.torch as dr
except ImportError:
    dr = None


# 14 Part Names from src.pose_control.v6.conditions:
# ("head", "torso", "left_upper_arm", "left_lower_arm", "left_hand",
#  "right_upper_arm", "right_lower_arm", "right_hand", "left_upper_leg",
#  "left_lower_leg", "left_foot", "right_upper_leg", "right_lower_leg", "right_foot")

# 55 SMPL-X parts to 14 classes (0..13)
PART_MAPPING_55_TO_14 = {
    # 0: head
    15: 0, 12: 0, 22: 0, 23: 0, 24: 0,
    # 1: torso
    0: 1, 3: 1, 6: 1, 9: 1, 13: 1, 14: 1,
    # 2: left_upper_arm
    16: 2,
    # 3: left_lower_arm
    18: 3,
    # 4: left_hand (Hand 20 + fingers 25..39)
    20: 4,
    # 5: right_upper_arm
    17: 5,
    # 6: right_lower_arm
    19: 6,
    # 7: right_hand (Hand 21 + fingers 40..54)
    21: 7,
    # 8: left_upper_leg
    1: 8,
    # 9: left_lower_leg
    4: 9,
    # 10: left_foot (Foot 7 + Toes 10)
    7: 10, 10: 10,
    # 11: right_upper_leg
    2: 11,
    # 12: right_lower_leg
    5: 12,
    # 13: right_foot (Foot 8 + Toes 11)
    8: 13, 11: 13,
}
for _i in range(25, 40):
    PART_MAPPING_55_TO_14[_i] = 4
for _i in range(40, 55):
    PART_MAPPING_55_TO_14[_i] = 7


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation vectors [..., 3] to continuous 6D rotation [..., 6]."""
    shape = axis_angle.shape
    axis_angle = axis_angle.reshape(-1, 3)
    
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    eps = 1e-7
    axis = axis_angle / (angle + eps)
    
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c = torch.cos(angle).squeeze(-1)
    s = torch.sin(angle).squeeze(-1)
    c1 = 1.0 - c
    
    # Rotation matrix
    r00 = c + x * x * c1
    r01 = x * y * c1 - z * s
    r10 = y * x * c1 + z * s
    r11 = c + y * y * c1
    r20 = z * x * c1 - y * s
    r21 = z * y * c1 + x * s
    
    rot6d = torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)
    return rot6d.reshape(*shape[:-1], 6)


class SMPLXPartRenderer:
    """Renders 14-class body part maps using nvdiffrast."""

    def __init__(self, smplx_model_path: str, size: int = 512, device: str = "cuda:0"):
        if dr is None:
            raise RuntimeError("nvdiffrast is required for SMPLXPartRenderer")
            
        self.size = size
        self.device = torch.device(device)
        
        # Load SMPL-X model data
        model_data = np.load(smplx_model_path, allow_pickle=True)
        self.weights = model_data["weights"]  # (10475, 55)
        self.faces_np = model_data["f"].astype(np.int32)  # (20908, 3)
        
        # Compute vertex part class (1..14, where 0 is reserved for background)
        vert_part_55 = self.weights.argmax(axis=1)  # (10475,)
        vert_part_14 = np.array([PART_MAPPING_55_TO_14[p] + 1 for p in vert_part_55], dtype=np.uint8)
        
        # Compute face part class via majority vote among 3 vertices
        tri_parts = vert_part_14[self.faces_np]  # (20908, 3)
        face_part = np.zeros(len(self.faces_np), dtype=np.uint8)
        for i in range(len(self.faces_np)):
            counts = np.bincount(tri_parts[i], minlength=15)
            face_part[i] = counts.argmax()
            
        self.face_part_cuda = torch.from_numpy(face_part).to(self.device)
        self.faces_cuda = torch.from_numpy(self.faces_np).to(device=self.device, dtype=torch.int32).contiguous()
        self.glctx = dr.RasterizeCudaContext(device=self.device)

    def _opencv_camera_to_clip(
        self, vertices_cv: torch.Tensor, fx: float, fy: float, cx: float, cy: float
    ) -> torch.Tensor:
        """Pinhole camera coordinates to OpenGL clip coordinates."""
        x = vertices_cv[:, 0]
        y = vertices_cv[:, 1]
        z = vertices_cv[:, 2]

        valid_z = z[torch.isfinite(z) & (z > 1e-6)]
        if valid_z.numel() == 0:
            raise RuntimeError("mesh has no positive camera Z")

        zmin = float(valid_z.min().item())
        zmax = float(valid_z.max().item())
        near = max(1e-4, zmin * 0.50)
        far = max(near + 1.0, zmax * 1.50)

        w = float(self.size)
        h = float(self.size)

        x_clip = (2.0 * float(fx) / w) * x + (2.0 * float(cx) / w - 1.0) * z
        y_clip = (-2.0 * float(fy) / h) * y + (1.0 - 2.0 * float(cy) / h) * z
        A = (far + near) / (far - near)
        B = (-2.0 * far * near) / (far - near)
        z_clip = A * z + B
        w_clip = z

        return torch.stack([x_clip, y_clip, z_clip, w_clip], dim=-1)

    @torch.no_grad()
    def render_part_map(
        self, vertices_cv_np: np.ndarray, fx: float, fy: float, cx: float, cy: float
    ) -> np.ndarray:
        """Render 512x512 part map (uint8, values 0..14)."""
        verts = torch.from_numpy(vertices_cv_np).to(device=self.device, dtype=torch.float32).contiguous()
        clip = self._opencv_camera_to_clip(verts, fx=fx, fy=fy, cx=cx, cy=cy).unsqueeze(0).contiguous()

        rast, _ = dr.rasterize(
            self.glctx, clip, self.faces_cuda, resolution=[self.size, self.size], grad_db=False
        )

        valid = rast[0, ..., 3] > 0
        tri_id = rast[0, ..., 3].long() - 1
        part_map = torch.where(valid, self.face_part_cuda[tri_id], torch.zeros_like(tri_id, dtype=torch.uint8))
        # Flip bottom-up to top-down
        part_map = torch.flip(part_map, dims=[0])
        return part_map.cpu().numpy()


def extract_25_keypoints(
    kp_data: Dict[str, np.ndarray], width: float, height: float
) -> np.ndarray:
    """Extract 25 body keypoints normalized to [0, 1] with scores.
    
    Returns: [25, 3] with (x_norm, y_norm, score)
    """
    kp134 = kp_data.get("keypoints_134_px")
    sc134 = kp_data.get("scores_134")
    
    if kp134 is None or sc134 is None:
        raise ValueError("Missing keypoints_134_px or scores_134")
        
    out = np.zeros((25, 3), dtype=np.float32)
    
    # 0..7: Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist
    out[0:8, :2] = kp134[0:8]
    out[0:8, 2] = sc134[0:8]
    
    # 8: MidHip = (RHip + LHip) / 2
    r_hip = kp134[8]
    l_hip = kp134[11]
    out[8, :2] = (r_hip + l_hip) * 0.5
    out[8, 2] = min(float(sc134[8]), float(sc134[11]))
    
    # 9..14: RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
    out[9, :2] = kp134[8]
    out[9, 2] = sc134[8]
    out[10, :2] = kp134[9]
    out[10, 2] = sc134[9]
    out[11, :2] = kp134[10]
    out[11, 2] = sc134[10]
    out[12, :2] = kp134[11]
    out[12, 2] = sc134[11]
    out[13, :2] = kp134[12]
    out[13, 2] = sc134[12]
    out[14, :2] = kp134[13]
    out[14, 2] = sc134[13]
    
    # 15..18: REye, LEye, REar, LEar
    out[15:19, :2] = kp134[14:18]
    out[15:19, 2] = sc134[14:18]
    
    # 19..24: LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel
    out[19:25, :2] = kp134[18:24]
    out[19:25, 2] = sc134[18:24]
    
    # Normalize coordinates to [0, 1]
    out[:, 0] /= max(float(width), 1.0)
    out[:, 1] /= max(float(height), 1.0)
    
    # Filter low confidence / NaNs
    invalid = ~np.isfinite(out) | (out[:, 2:3] < 0.05)
    out[invalid[:, 0], 0] = 0.0
    out[invalid[:, 1], 1] = 0.0
    out[invalid[:, 2], 2] = 0.0
    
    return out


def process_appearance(
    app: str,
    sampled_root: Path,
    assets_root: Path,
    renderer: SMPLXPartRenderer,
    overwrite: bool = False,
) -> bool:
    """Process all 70 frames of one appearance."""
    app_sampled = sampled_root / app
    app_assets = assets_root / app
    
    fmap_path = app_sampled / "smplestx" / "frame_map.json"
    params_path = app_sampled / "smplestx" / "params.pt"
    
    if not fmap_path.exists() or not params_path.exists():
        print(f"[{app}] Missing frame_map or params.pt, skipping.")
        return False
        
    with open(fmap_path, "r", encoding="utf-8") as f:
        fmap = json.load(f)
        
    frames = fmap.get("frames", [])
    if len(frames) != 70:
        print(f"[{app}] Frames count {len(frames)} != 70, skipping.")
        return False
        
    params = torch.load(params_path, map_location="cpu", weights_only=False)
    
    # Verify required tensors
    required = ("mesh_cam", "betas", "global_orient", "transl", "focal_length_xy_raw", "princpt_raw")
    for k in required:
        if k not in params:
            print(f"[{app}] Missing param {k}, skipping.")
            return False
            
    mesh_cam_all = params["mesh_cam"].float()  # [70, 10475, 3]
    betas_all = params["betas"].float()        # [70, 10]
    orient_all = params["global_orient"].float() # [70, 3]
    transl_all = params["transl"].float()      # [70, 3]
    focal_all = params["focal_length_xy_raw"].float() # [70, 2]
    princpt_all = params["princpt_raw"].float()       # [70, 2]
    
    if "width" in params and "height" in params:
        w_all = params["width"].float().reshape(-1)
        h_all = params["height"].float().reshape(-1)
    else:
        w_all = torch.full((70,), 1024.0)
        h_all = torch.full((70,), 1024.0)
        
    # Compute 6D continuous rotation
    rot6d_all = axis_angle_to_rotation_6d(orient_all)  # [70, 6]
    
    # Compute camera 7D: [fx/W, fy/H, cx/W, cy/H, tx, ty, tz]
    sx = 512.0 / w_all
    sy = 512.0 / h_all
    fx_512 = focal_all[:, 0] * sx
    fy_512 = focal_all[:, 1] * sy
    cx_512 = princpt_all[:, 0] * sx
    cy_512 = princpt_all[:, 1] * sy
    
    cam_7d = torch.stack([
        focal_all[:, 0] / w_all,
        focal_all[:, 1] / h_all,
        princpt_all[:, 0] / w_all,
        princpt_all[:, 1] / h_all,
        transl_all[:, 0],
        transl_all[:, 1],
        transl_all[:, 2],
    ], dim=-1)  # [70, 7]
    
    # Concatenate 26D Global Parameters: betas[10] + rot6d[6] + transl[3] + cam[7]
    smplx_global_26d = torch.cat([betas_all[:, :10], rot6d_all, transl_all, cam_7d], dim=-1)  # [70, 26]
    
    # Process keypoints and render Part Maps
    stems = []
    roles = []
    kps_25_list = []
    
    part_out_dir = app_assets / "part_512"
    
    for i, frame in enumerate(frames):
        stem = frame["stem"]
        role = frame.get("role", "target")
        stems.append(stem)
        roles.append(role)
        
        # 1. Part Map rendering
        part_file = part_out_dir / role / f"{stem}.png"
        if overwrite or not part_file.exists():
            part_file.parent.mkdir(parents=True, exist_ok=True)
            verts_np = mesh_cam_all[i].numpy()
            part_map = renderer.render_part_map(
                verts_np,
                fx=float(fx_512[i]),
                fy=float(fy_512[i]),
                cx=float(cx_512[i]),
                cy=float(cy_512[i]),
            )
            cv2.imwrite(str(part_file), part_map)
            
        # 2. Keypoints 25D extraction
        kp_file = app_sampled / "keypoints" / role / f"{stem}.npz"
        if kp_file.exists():
            kp_data = np.load(kp_file)
            kps_25 = extract_25_keypoints(kp_data, float(w_all[i]), float(h_all[i]))
        else:
            kps_25 = np.zeros((25, 3), dtype=np.float32)
        kps_25_list.append(kps_25)
        
    kps_25_coords = torch.from_numpy(np.stack(kps_25_list, axis=0)).float()  # [70, 25, 3]
    
    # Save structured conditions
    v6_bundle = {
        "smplx_global": smplx_global_26d,     # [70, 26] float32
        "kps_25_coords": kps_25_coords,       # [70, 25, 3] float32
        "stems": stems,
        "roles": roles,
        "stem_to_idx": {stem: idx for idx, stem in enumerate(stems)},
    }
    
    bundle_out_path = app_assets / "v6_conditions.pt"
    torch.save(v6_bundle, bundle_out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess iPER dataset for V6.3 Native Adapter")
    parser.add_argument(
        "--sampled_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument("--single", type=str, default=None, help="Process single appearance ID, e.g. 001_1")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing assets")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    sampled_root = Path(args.sampled_root)
    assets_root = Path(args.assets_root)
    
    print("=" * 60)
    print("DeepGen V6.3 Condition Preprocessor")
    print(f"Sampled Root: {sampled_root}")
    print(f"Assets Root : {assets_root}")
    print(f"SMPL-X Path : {args.smplx_path}")
    print("=" * 60)
    
    # Initialize renderer
    t0 = time.time()
    print("Initializing SMPLXPartRenderer with nvdiffrast...")
    renderer = SMPLXPartRenderer(args.smplx_path, size=512, device=args.device)
    print(f"Renderer ready in {time.time() - t0:.2f}s")
    
    if args.single:
        apps = [args.single]
    else:
        apps = sorted([d.name for d in sampled_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
        
    print(f"Total appearances to process: {len(apps)}")
    
    success_count = 0
    total_frames = 0
    t_start = time.time()
    
    for idx, app in enumerate(apps):
        app_t0 = time.time()
        ok = process_appearance(
            app=app,
            sampled_root=sampled_root,
            assets_root=assets_root,
            renderer=renderer,
            overwrite=args.overwrite,
        )
        if ok:
            success_count += 1
            total_frames += 70
            elapsed = time.time() - app_t0
            print(f"[{idx + 1:02d}/{len(apps):02d}] {app}: PASS (70 frames in {elapsed:.2f}s, {70 / elapsed:.1f} fps)")
        else:
            print(f"[{idx + 1:02d}/{len(apps):02d}] {app}: SKIPPED/FAILED")
            
    total_elapsed = time.time() - t_start
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Successful Appearances: {success_count} / {len(apps)}")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Total Wall Time       : {total_elapsed:.2f}s ({total_frames / max(total_elapsed, 1e-3):.1f} fps overall)")
    print("=" * 60)


if __name__ == "__main__":
    main()

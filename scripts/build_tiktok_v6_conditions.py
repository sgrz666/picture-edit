#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build DeepGen V6.3 Native Adapter conditions for the TikTok Dataset.

Generates:
1. 14-Class Part Map: rendered via nvdiffrast using SMPL-X skinning weights on mesh_cam.
   Saved as 512x512 uint8 PNG in TikTok_3d_assets_native/{seq}/part_512/{stem}.png
   (0 = background, 1..14 = body parts matching conditions.PART_NAMES).
2. SMPL-X Global 26D parameters + 25-keypoint coordinates:
   Saved in TikTok_3d_assets_native/{seq}/v6_conditions.pt containing:
   - smplx_global: [N, 26] (betas[10] + rot6d[6] + transl[3] + cam[7])
   - kps_25_coords: [N, 25, 3] (x_norm, y_norm, score)
   - stems: list of frame stems
   - stem_to_idx: dict mapping stem to frame index 0..N-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

        part_cuda = torch.zeros((self.size, self.size), dtype=torch.uint8, device=self.device)
        part_cuda[valid] = self.face_part_cuda[tri_id[valid]]

        # Invert Y to match OpenCV image coordinate convention
        part_map = part_cuda.flip(0).cpu().numpy()
        return part_map


def extract_25_keypoints_from_smplx_joints(
    joints_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> np.ndarray:
    """Project SMPL-X 3D joints to OpenPose 25 keypoint layout [25, 3] normalized in [0, 1]."""
    # OpenPose 25 layout from SMPL-X 55 joints:
    # 0: Nose (Head 15)
    # 1: Neck (Neck 12)
    # 2: RShoulder (17), 3: RElbow (19), 4: RWrist (21)
    # 5: LShoulder (16), 6: LElbow (18), 7: LWrist (20)
    # 8: MidHip (Pelvis 0)
    # 9: RHip (2), 10: RKnee (5), 11: RAnkle (8)
    # 12: LHip (1), 13: LKnee (4), 14: LAnkle (7)
    # 15: REye (24), 16: LEye (23), 17: REar (15), 18: LEar (15)
    # 19: LBigToe (10), 20: LSmallToe (10), 21: LHeel (10)
    # 22: RBigToe (11), 23: RSmallToe (11), 24: RHeel (11)
    mapping = [
        15, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7,
        24, 23, 15, 15, 10, 10, 10, 11, 11, 11,
    ]
    out = np.zeros((25, 3), dtype=np.float32)
    for i, j_idx in enumerate(mapping):
        if j_idx < len(joints_3d):
            pt = joints_3d[j_idx]
            z = max(float(pt[2]), 1e-4)
            u = (fx * pt[0] / z) + cx
            v = (fy * pt[1] / z) + cy
            out[i, 0] = float(u / max(width, 1.0))
            out[i, 1] = float(v / max(height, 1.0))
            out[i, 2] = 1.0 if 0.0 <= out[i, 0] <= 1.0 and 0.0 <= out[i, 1] <= 1.0 else 0.5
    return out


# SMPLest-X's first 25 projected joints are in its named body order, not
# OpenPose-25 order. See human_models/human_models.py: joints_name.
SMPLestX_TO_OPENPOSE25 = (
    24, 7, 9, 11, 13, 8, 10, 12, 0, 2, 4, 6, 1, 3, 5,
    23, 22, 21, 20, 14, 15, 16, 17, 18, 19,
)


def project_smplestx_body25_to_openpose(
    params: Mapping[str, Any], frame_index: int
) -> np.ndarray:
    """Invert SMPLest-X's 12x16 input-crop projection for one RGB frame.

    Returns OpenPose-25 normalized xy/confidence. Out-of-frame joints have
    confidence zero; projected mesh vertices are never accepted as joints.
    """
    if "joint_proj_model" not in params:
        raise ValueError("SMPLest-X joint_proj_model is required for body pose")
    joints = torch.as_tensor(params["joint_proj_model"], dtype=torch.float32)
    if joints.ndim != 3 or joints.shape[1:] != (137, 2):
        raise ValueError("joint_proj_model must have shape [N,137,2]")
    if not 0 <= frame_index < joints.shape[0]:
        raise IndexError("frame_index is outside joint_proj_model")
    focal_key = "focal_length_xy_raw" if "focal_length_xy_raw" in params else "focal_length_xy"
    principal_key = "princpt_raw" if "princpt_raw" in params else "princpt"
    if focal_key not in params or principal_key not in params:
        raise ValueError("SMPLest-X focal length and principal point are required")
    focal = torch.as_tensor(params[focal_key], dtype=torch.float32)[frame_index]
    principal = torch.as_tensor(params[principal_key], dtype=torch.float32)[frame_index]
    if focal.shape != (2,) or principal.shape != (2,):
        raise ValueError("camera focal/principal arrays must have shape [N,2]")

    def frame_dimension(name: str) -> float:
        values = torch.as_tensor(params[name], dtype=torch.float32).reshape(-1)
        value = float(values[0 if values.numel() == 1 else frame_index])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    width, height = frame_dimension("width"), frame_dimension("height")
    bbox_width = focal[0] * (192.0 / 5000.0)
    bbox_height = focal[1] * (256.0 / 5000.0)
    selected = joints[frame_index, list(SMPLestX_TO_OPENPOSE25)]
    x = principal[0] - bbox_width * 0.5 + selected[:, 0] / 12.0 * bbox_width
    y = principal[1] - bbox_height * 0.5 + selected[:, 1] / 16.0 * bbox_height
    xy = torch.stack((x / width, y / height), dim=-1)
    valid = torch.isfinite(xy).all(dim=-1) & ((xy >= 0) & (xy <= 1)).all(dim=-1)
    result = torch.cat((torch.where(valid[:, None], xy, torch.zeros_like(xy)), valid[:, None].float()), dim=-1)
    return result.cpu().numpy().astype(np.float32)


def process_tiktok_sequence(
    seq_id: str,
    raw_root: Path,
    assets_root: Path,
    renderer: Optional[SMPLXPartRenderer] = None,
    overwrite: bool = False,
) -> bool:
    """Process one sequence in TikTok dataset, writing part_512/ and v6_conditions.pt."""
    seq_raw = raw_root / seq_id
    seq_assets = assets_root / seq_id
    smplestx_dir = seq_assets / "smplestx"
    fmap_path = smplestx_dir / "frame_map.json"
    params_path = smplestx_dir / "params.pt"

    if not fmap_path.exists() or not params_path.exists():
        print(f"[{seq_id}] Missing frame_map or params.pt, skipping.")
        return False

    with open(fmap_path, "r", encoding="utf-8") as f:
        fmap = json.load(f)
    frames = fmap.get("frames", [])
    frame_count = len(frames)
    if frame_count == 0:
        print(f"[{seq_id}] Empty frames in frame_map, skipping.")
        return False

    params = torch.load(params_path, map_location="cpu", weights_only=False)
    mesh_cam_all = params["mesh_cam"].float()  # [N, 10475, 3]
    betas_all = params["betas"].float()        # [N, 10]
    orient_all = params["global_orient"].float() # [N, 3]
    transl_all = params["transl"].float()      # [N, 3]

    focal_all = params.get("focal_length_xy_raw", params.get("focal_length_xy")).float()
    princpt_all = params.get("princpt_raw", params.get("princpt")).float()

    w_val = float(params.get("width", torch.tensor([604.0]))[0])
    h_val = float(params.get("height", torch.tensor([1080.0]))[0])
    w_all = torch.full((frame_count,), w_val)
    h_all = torch.full((frame_count,), h_val)

    # Compute 6D continuous rotation
    rot6d_all = axis_angle_to_rotation_6d(orient_all)  # [N, 6]

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
    ], dim=-1)  # [N, 7]

    # Concatenate 26D Global Parameters: betas[10] + rot6d[6] + transl[3] + cam[7]
    smplx_global_26d = torch.cat([betas_all[:, :10], rot6d_all, transl_all, cam_7d], dim=-1)  # [N, 26]

    stems = []
    kps_25_list = []
    part_out_dir = seq_assets / "part_512"

    for i, frame in enumerate(frames):
        stem = frame.get("stem", f"{frame['index']+1:04d}")
        stems.append(stem)

        # 1. Part Map rendering
        if renderer is not None:
            part_file = part_out_dir / f"{stem}.png"
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

        # 2. Keypoints: optional DWPose, otherwise the named SMPLest-X joints.
        kp_dwpose = seq_raw / "dwpose" / f"{stem}.npz"
        if kp_dwpose.exists():
            data = np.load(kp_dwpose)
            kps_25 = data.get("kps_25", np.zeros((25, 3), dtype=np.float32))
        else:
            kps_25 = project_smplestx_body25_to_openpose(params, i)
        kps_25_list.append(kps_25)

    kps_25_coords = torch.from_numpy(np.stack(kps_25_list, axis=0)).float()  # [N, 25, 3]

    v6_bundle = {
        "smplx_global": smplx_global_26d,     # [N, 26] float32
        "kps_25_coords": kps_25_coords,       # [N, 25, 3] float32
        "stems": stems,
        "stem_to_idx": {stem: idx for idx, stem in enumerate(stems)},
        "width": w_val,
        "height": h_val,
    }

    bundle_out_path = seq_assets / "v6_conditions.pt"
    torch.save(v6_bundle, bundle_out_path)
    return True


def write_single_frame_override(
    sequence_assets: Path,
    *,
    sequence: str,
    target_stem: str,
    output_path: Path,
    renderer: Optional[SMPLXPartRenderer] = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Write only one target's audited conditions, never the sequence cache."""
    sequence_assets = Path(sequence_assets)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"condition override already exists: {output_path}")
    if output_path.resolve() == (sequence_assets / "v6_conditions.pt").resolve():
        raise ValueError("single-frame override may not replace v6_conditions.pt")
    smplestx = sequence_assets / "smplestx"
    with (smplestx / "frame_map.json").open("r", encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    matches = [i for i, frame in enumerate(frames) if str(frame["stem"]) == target_stem]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frame with stem {target_stem!r}")
    index = matches[0]
    params = torch.load(smplestx / "params.pt", map_location="cpu", weights_only=False)
    points = project_smplestx_body25_to_openpose(params, index)
    bundle = {
        "sequence": sequence,
        "target_stem": target_stem,
        "kps_25_coords": torch.from_numpy(points),
        "pose_source": "SMPLest-X joint_proj_model named body25",
    }
    if renderer is not None:
        mesh = torch.as_tensor(params["mesh_cam"])[index].cpu().numpy()
        focal_key = "focal_length_xy_raw" if "focal_length_xy_raw" in params else "focal_length_xy"
        principal_key = "princpt_raw" if "princpt_raw" in params else "princpt"
        focal = torch.as_tensor(params[focal_key])[index].float()
        principal = torch.as_tensor(params[principal_key])[index].float()
        width = float(torch.as_tensor(params["width"]).reshape(-1)[0])
        height = float(torch.as_tensor(params["height"]).reshape(-1)[0])
        scale = torch.tensor([renderer.size / width, renderer.size / height])
        focal, principal = focal * scale, principal * scale
        labels = renderer.render_part_map(
            mesh,
            fx=float(focal[0]), fy=float(focal[1]),
            cx=float(principal[0]), cy=float(principal[1]),
        )
        if labels.shape != (renderer.size, renderer.size) or labels.dtype != np.uint8 or labels.max() > 14:
            raise ValueError("invalid rendered part map")
        bundle["part_512"] = torch.from_numpy(labels.copy())
        bundle["part_source"] = "SMPL-X mesh nvdiffrast rasterization"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)
    return points


def main():
    parser = argparse.ArgumentParser(description="Preprocess TikTok dataset for V6.3 Native Adapter")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_dataset/TikTok_dataset",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_3d_assets_native",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument("--sequence", type=str, default="00001", help="Process single sequence ID, e.g. 00001")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing assets")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--single-frame-override-out", type=Path)
    parser.add_argument("--target-stem", default="0074")
    parser.add_argument("--include-part", action="store_true")
    parser.add_argument("--preview-out", type=Path, help="Draw projected 25 points over target RGB")
    parser.add_argument("--part-preview-out", type=Path, help="Save rasterized part labels and RGB overlay")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    assets_root = Path(args.assets_root)

    print("=" * 60)
    print("TikTok V6 Condition Preprocessor")
    print(f"Raw Root    : {raw_root}")
    print(f"Assets Root : {assets_root}")
    print(f"Sequence    : {args.sequence}")
    print("=" * 60)

    renderer = None
    if args.single_frame_override_out is not None and args.include_part and dr is None:
        raise RuntimeError("--include-part requires nvdiffrast; pose-only override remains available")
    needs_renderer = args.single_frame_override_out is None or args.include_part
    if needs_renderer and dr is not None and torch.cuda.is_available() and Path(args.smplx_path).exists():
        print("Initializing SMPLXPartRenderer with nvdiffrast...")
        renderer = SMPLXPartRenderer(args.smplx_path, size=512, device=args.device)
        print("Renderer ready.")
    else:
        print("Note: nvdiffrast not available or CPU only. Part map rasterization will be skipped or stubbed.")

    if args.single_frame_override_out is not None:
        points = write_single_frame_override(
            assets_root / args.sequence,
            sequence=args.sequence,
            target_stem=args.target_stem,
            output_path=args.single_frame_override_out,
            renderer=renderer if args.include_part else None,
            overwrite=args.overwrite,
        )
        print(f"Single-frame override: {args.single_frame_override_out}")
        print(f"Valid body joints: {int((points[:, 2] > 0).sum())}/25")
        if args.preview_out is not None:
            image_path = raw_root / args.sequence / "images" / f"{args.target_stem}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            for number, (x, y, confidence) in enumerate(points):
                if confidence > 0:
                    pixel = (round(float(x) * image.shape[1]), round(float(y) * image.shape[0]))
                    cv2.circle(image, pixel, 5, (0, 255, 255), -1)
                    cv2.putText(image, str(number), (pixel[0] + 5, pixel[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            args.preview_out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.preview_out), image):
                raise OSError(f"could not write preview {args.preview_out}")
            print(f"Pose preview: {args.preview_out}")
        if args.part_preview_out is not None:
            bundle = torch.load(args.single_frame_override_out, map_location="cpu", weights_only=True)
            if "part_512" not in bundle:
                raise ValueError("--part-preview-out requires --include-part")
            labels = bundle["part_512"].numpy()
            image_path = raw_root / args.sequence / "images" / f"{args.target_stem}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            image = cv2.resize(image, (labels.shape[1], labels.shape[0]))
            colors = cv2.applyColorMap((labels * 17).astype(np.uint8), cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(image, 0.5, colors, 0.5, 0)
            overlay[labels == 0] = image[labels == 0]
            args.part_preview_out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.part_preview_out), overlay):
                raise OSError(f"could not write part preview {args.part_preview_out}")
            print(f"Part preview: {args.part_preview_out}")
        return

    if args.sequence:
        seqs = [args.sequence]
    else:
        seqs = sorted([d.name for d in assets_root.iterdir() if d.is_dir() and not d.name.startswith(".")])

    for seq in seqs:
        ok = process_tiktok_sequence(seq, raw_root, assets_root, renderer=renderer, overwrite=args.overwrite)
        print(f"Sequence {seq}: {'SUCCESS' if ok else 'SKIPPED'}")


if __name__ == "__main__":
    main()

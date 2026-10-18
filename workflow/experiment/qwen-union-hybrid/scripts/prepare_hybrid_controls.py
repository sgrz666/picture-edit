#!/usr/bin/env python3
"""准备 Hybrid Qwen Base + Union Control 所需的几何控制资产与纯净深度图。

核心任务：
1. 从 workflow/runs/test/hard_05_torso 提取 A_reference, B_rgb, B_keypoints, B_smplx, B_camera;
2. 使用 SMPL-X 3D Mesh 顶点 (10,475 点) 配合 source_torso 变换 (scale=0.9773, offset=[-198.16, -290.96])
   光栅化生成 640x928 的无背景噪声纯净深度图 depth_source_torso.png;
3. 复制并校验 pose_source_torso.png 与 keypoints_source_torso.npz;
4. 渲染 Pose 与 Depth 的高对比度自检叠加图 overlay_debug.png，验证像素级对齐;
5. 冻结统一原生 PE Prompt (input/prompt_pe_native.txt) 与清单 prep_manifest.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "pic-edit" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.geometry import render_poseguide

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 928


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_smplx_depth(
    mesh_cam: np.ndarray,
    focal_xy: tuple[float, float] | list[float],
    principal_xy: tuple[float, float] | list[float],
    scale: float,
    offset_xy: tuple[float, float] | list[float],
    canvas_wh: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT),
    point_radius: int = 3,
) -> tuple[np.ndarray, Image.Image]:
    """将 SMPL-X 3D Mesh 顶点投影到 Canvas 像素并以 Z-buffer 光栅化渲染深度图。

    返回:
      depth_array: (H, W) float32 [0.0, 1.0]，近处 1.0，远处 0.0，背景 0.0
      depth_image: PIL.Image (RGB, 8-bit)
    """
    W, H = canvas_wh
    fx, fy = focal_xy
    cx, cy = principal_xy
    ox, oy = offset_xy

    # 1. 相机坐标系投影
    X = mesh_cam[:, 0]
    Y = mesh_cam[:, 1]
    Z = mesh_cam[:, 2]
    Z_safe = np.clip(Z, a_min=1e-4, a_max=None)

    u_cam = fx * (X / Z_safe) + cx
    v_cam = fy * (Y / Z_safe) + cy

    # 2. 应用 source_torso 空间变换到生成画布
    u_canvas = scale * u_cam + ox
    v_canvas = scale * v_cam + oy

    # 3. Z-buffering 光栅化渲染 (点云球状 splatting)
    z_buffer = np.full((H, W), np.inf, dtype=np.float32)

    r_sq = point_radius * point_radius
    offsets = [
        (du, dv)
        for du in range(-point_radius, point_radius + 1)
        for dv in range(-point_radius, point_radius + 1)
        if du * du + dv * dv <= r_sq
    ]

    for uc, vc, z_val in zip(u_canvas, v_canvas, Z):
        if not (np.isfinite(uc) and np.isfinite(vc) and np.isfinite(z_val)):
            continue
        iu = int(round(uc))
        iv = int(round(vc))
        if iu < -point_radius or iu >= W + point_radius or iv < -point_radius or iv >= H + point_radius:
            continue

        for du, dv in offsets:
            nu = iu + du
            nv = iv + dv
            if 0 <= nu < W and 0 <= nv < H:
                if z_val < z_buffer[nv, nu]:
                    z_buffer[nv, nu] = z_val

    # 4. 深度归一化 (近处更亮，远处更暗，背景纯黑)
    valid_mask = ~np.isinf(z_buffer)
    depth_norm = np.zeros((H, W), dtype=np.float32)

    if np.any(valid_mask):
        z_min = float(np.min(z_buffer[valid_mask]))
        z_max = float(np.max(z_buffer[valid_mask]))
        z_span = z_max - z_min if z_max > z_min else 1.0
        # 逆线性深度：近处 1.0 (255), 远处 0.05, 背景 0.0
        depth_norm[valid_mask] = 1.0 - 0.95 * ((z_buffer[valid_mask] - z_min) / z_span)

    depth_uint8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_rgb = np.stack([depth_uint8] * 3, axis=-1)
    depth_image = Image.fromarray(depth_rgb, mode="RGB")

    return depth_norm, depth_image


def main():
    parser = argparse.ArgumentParser(description="Prepare Hybrid Qwen + Union Controls")
    parser.add_argument(
        "--workflow-run",
        type=Path,
        default=Path("/home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05"),
        help="hard_05 原始 workflow 目录",
    )
    parser.add_argument(
        "--workflow-run-torso",
        type=Path,
        default=Path("/home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05_torso"),
        help="hard_05_torso 对齐 workflow 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/prepared_controls",
        help="控制条件产物输出目录",
    )
    parser.add_argument("--save-debug", action="store_true", help="保存叠加检查大图")
    args = parser.parse_args()

    wf_dir = args.workflow_run.resolve()
    wf_torso_dir = args.workflow_run_torso.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir.parents[1]

    print(">> [Step 1] 读取源资产与相机元数据...")
    a_ref_path = wf_dir / "inputs/A_reference.png"
    b_rgb_path = wf_dir / "inputs/B_rgb.png"
    prompt_path = wf_dir / "prompt/generation_prompt.txt"

    camera_json_path = wf_torso_dir / "geometry/B_camera.json"
    if not camera_json_path.is_file():
        raise FileNotFoundError(f"未找到 B_camera.json: {camera_json_path}")
    camera_meta = json.loads(camera_json_path.read_text(encoding="utf-8"))

    smplx_pt_path = wf_torso_dir / "geometry/B_smplx.pt"
    if not smplx_pt_path.is_file():
        raise FileNotFoundError(f"未找到 B_smplx.pt: {smplx_pt_path}")

    # 1. 复制统一的原生 Native PE Prompt
    input_dir = exp_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    frozen_prompt_path = input_dir / "prompt_pe_native.txt"
    frozen_prompt_path.write_text(prompt_text, encoding="utf-8")
    prompt_sha = sha256_file(frozen_prompt_path)
    print(f"   Native PE Prompt 冻结: {frozen_prompt_path} (SHA256: {prompt_sha[:12]}...)")

    # 2. 准备 pose_source_torso.png 与 keypoints
    pose_torso_src = wf_torso_dir / "geometry/B_3d_dwpose.png"
    kp_torso_src = wf_torso_dir / "geometry/B_keypoints.npz"

    pose_dest = out_dir / "pose_source_torso.png"
    kp_dest = out_dir / "keypoints_source_torso.npz"

    shutil.copyfile(pose_torso_src, pose_dest)
    shutil.copyfile(kp_torso_src, kp_dest)
    pose_sha = sha256_file(pose_dest)
    kp_sha = sha256_file(kp_dest)
    print(f"   Pose 图复制就绪: {pose_dest} (SHA256: {pose_sha[:12]}...)")

    # 3. 从 SMPL-X 3D Mesh 渲染纯净对齐深度图
    print(">> [Step 2] 渲染 SMPL-X 3D 稠密对齐深度图 (depth_source_torso.png)...")
    import torch
    smplx_params = torch.load(smplx_pt_path, map_location="cpu", weights_only=False)
    mesh_cam = smplx_params["mesh_cam"][0].numpy()  # (10475, 3)

    focal_xy = camera_meta["focal_xy"]
    principal_xy = camera_meta["principal_xy"]
    b2gen = camera_meta["B_to_generation"]
    scale = float(b2gen["scale"])
    offset_xy = [float(b2gen["offset_xy"][0]), float(b2gen["offset_xy"][1])]

    depth_norm, depth_image = render_smplx_depth(
        mesh_cam=mesh_cam,
        focal_xy=focal_xy,
        principal_xy=principal_xy,
        scale=scale,
        offset_xy=offset_xy,
        canvas_wh=(CANVAS_WIDTH, CANVAS_HEIGHT),
        point_radius=3,
    )
    depth_dest = out_dir / "depth_source_torso.png"
    depth_image.save(depth_dest)
    depth_sha = sha256_file(depth_dest)
    print(f"   Depth 图渲染就绪: {depth_dest} (SHA256: {depth_sha[:12]}...)")

    # 4. 生成自检叠加图 (overlay_debug.png)
    if args.save_debug:
        print(">> [Step 3] 生成 Pose 与 Depth 像素对齐自检图...")
        with Image.open(pose_dest) as p_im:
            p_arr = np.array(p_im.convert("RGB"))
        d_arr = np.array(depth_image)

        # 绿色通道为 Depth，洋红通道为 Pose，完全重合处呈现丰富对比
        overlay = cv2.addWeighted(d_arr, 0.6, p_arr, 0.7, 0)
        overlay_dest = out_dir / "overlay_debug.png"
        Image.fromarray(overlay).save(overlay_dest)
        print(f"   对齐自检大图写入: {overlay_dest}")

    # 5. 构建清单
    manifest = {
        "sample_id": "hard_05",
        "canvas_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "source_assets": {
            "A_reference": {"path": str(a_ref_path), "sha256": sha256_file(a_ref_path)},
            "B_rgb": {"path": str(b_rgb_path), "sha256": sha256_file(b_rgb_path)},
            "B_smplx": {"path": str(smplx_pt_path), "sha256": sha256_file(smplx_pt_path)},
            "B_camera": {"path": str(camera_json_path), "sha256": sha256_file(camera_json_path)},
        },
        "prompt": {
            "path": str(frozen_prompt_path),
            "sha256": prompt_sha,
            "text": prompt_text,
        },
        "controls": {
            "pose": {
                "path": str(pose_dest),
                "sha256": pose_sha,
                "keypoints_npz": str(kp_dest),
                "keypoints_sha256": kp_sha,
                "framing": "source_torso",
                "scale": scale,
                "offset_xy": offset_xy,
            },
            "depth": {
                "path": str(depth_dest),
                "sha256": depth_sha,
                "type": "smplx_mesh_projected",
                "num_vertices": int(len(mesh_cam)),
                "scale": scale,
                "offset_xy": offset_xy,
            },
        },
    }
    manifest_dest = out_dir / "prep_manifest.json"
    manifest_dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ 控制资产准备完成，Manifest 写入: {manifest_dest}\n")


if __name__ == "__main__":
    main()

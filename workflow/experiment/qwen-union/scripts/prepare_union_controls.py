#!/usr/bin/env python3
"""准备 Qwen-Image-2.1-Fun-Controlnet-Union 所需的控制条件与 Mask。

职责：
1. 读取指定 workflow run (优先 hard_05 及 hard_05_torso)。
2. 校验 A、B、关键点及相机数据。
3. 按照 Canvas 统一规范，生成三类 Pose 骨架图 (pose_native, pose_source_person, pose_source_torso)。
4. 使用 YOLO11x-seg 提取 A 人体分割 Mask。
5. 针对三类姿态估计目标活动区域，生成三张对应 Inpainting Mask (白=255重绘, 黑=0保留)。
6. 生成输入与 Mask 的混合可视化叠加图用于自检。
7. 执行 PE Prompt 确定性语义编译，生成并冻结 prompt_pe_union.txt。
8. 保存关键点 NPZ、几何变换元数据与逐文件 SHA256 清单 (prep_manifest.json)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "pic-edit" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pose_edit.geometry import (align_to_source_bbox, render_poseguide,
                                    torso_length, valid_points)
from src.evaluation.no_gt_metrics import mask_from_limbs, torso_px


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_union_prompt(raw_prompt: str) -> str:
    """对包含 <image1>、<image2> 的 PE-I2I 提示词执行确定性语义编译。

    规则：
    - <image1> 映射为 'the source image' (源人物/背景参考图)
    - <image2> 映射为 'the pose guide' (姿态控制骨架)
    - 'On <image1> as the canvas' 规范为 'On the source image canvas'
    - 不新增、删改目标动作约束，不引入 B 的身份或衣服信息。
    """
    text = raw_prompt.strip()
    text = text.replace("On <image1> as the canvas", "On the source image as the canvas")
    text = text.replace("<image1>", "the source image")
    text = text.replace("<image2>", "the pose guide")
    text = text.replace("<image3>", "the pose guide")
    return text.strip()


def run_yolo_person_seg(image_path: Path, seg_model_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """使用 YOLO11x-seg 获得高精细度的人体分割二值掩码 (H, W)。"""
    from ultralytics import YOLO

    model = YOLO(str(seg_model_path))
    results = model(str(image_path), verbose=False)
    r = results[0]

    w, h = target_size
    person_indices = [i for i, c in enumerate(r.boxes.cls.cpu().numpy()) if int(c) == 0]
    if not person_indices or r.masks is None:
        raise RuntimeError(f"YOLO11x-seg 未能在 {image_path} 中检测到有效人体实例")

    # 聚合所有检测到的人体 mask
    masks_data = r.masks.data[person_indices].float()
    combined_mask = masks_data.sum(dim=0, keepdim=True).clamp(0, 1)

    # 插值恢复到标准 Canvas 尺寸 (H, W)
    mask_up = F.interpolate(combined_mask.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    mask_bool = (mask_up > 0.5).cpu().numpy().astype(bool)
    return mask_bool


def main():
    parser = argparse.ArgumentParser(description="Prepare Union Pose & Mask Controls")
    parser.add_argument("--workflow-run", type=Path, required=True,
                        help="目标 workflow run 路径 (例如 workflow/runs/test/hard_05)")
    parser.add_argument("--workflow-run-torso", type=Path, default=None,
                        help="躯干对齐版 run 路径 (例如 workflow/runs/test/hard_05_torso)")
    parser.add_argument("--output", type=Path, default=Path("data/prepared_controls"),
                        help="产物输出目录")
    parser.add_argument("--seg-model", type=Path, default=Path("/home/shangguanrz/project/pic-edit/add/yolo11x-seg.pt"),
                        help="人体分割模型路径")
    parser.add_argument("--save-debug", action="store_true",
                        help="是否保存叠加调试图")
    args = parser.parse_args()

    run_dir = args.workflow_run.resolve()
    torso_run_dir = args.workflow_run_torso.resolve() if args.workflow_run_torso else run_dir
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = out_dir.parent.parent / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # 1. 验证源资产
    a_ref_path = run_dir / "inputs/A_reference.png"
    b_rgb_path = run_dir / "inputs/B_rgb.png"
    kp_b_path = run_dir / "geometry/B_keypoints.npz"
    cam_b_path = run_dir / "geometry/B_camera.json"
    src_prep_path = run_dir / "inputs/source_preprocess.json"
    prompt_path = run_dir / "prompt/generation_prompt.txt"

    for p in [a_ref_path, b_rgb_path, kp_b_path, cam_b_path, src_prep_path, prompt_path]:
        if not p.is_file():
            raise FileNotFoundError(f"缺少必要输入资产：{p}")

    with Image.open(a_ref_path) as im_a:
        w_canvas, h_canvas = im_a.size
    with Image.open(b_rgb_path) as im_b:
        w_b, h_b = im_b.size

    if w_canvas % 16 != 0 or h_canvas % 16 != 0:
        raise ValueError(f"Canvas 尺寸 ({w_canvas}, {h_canvas}) 不是 16 的倍数！")

    cam_data = json.loads(cam_b_path.read_text(encoding="utf-8"))
    src_prep = json.loads(src_prep_path.read_text(encoding="utf-8"))

    with np.load(kp_b_path) as d:
        keypoints_b_px = d["keypoints_B_px"]
        valid_b = d["valid_B"]
        bbox_a_canvas_loaded = d.get("source_person_bbox_canvas", None)

    bbox_b = cam_data["detections"]["B"]["bbox_xyxy"]
    if bbox_a_canvas_loaded is not None:
        bbox_a_canvas = bbox_a_canvas_loaded
    else:
        sx, sy = src_prep["scale_xy"]
        ox, oy = src_prep["offset_xy"]
        bbox_a_orig = np.array(cam_data["detections"]["A"]["bbox_xyxy"])
        bbox_a_canvas = bbox_a_orig * np.array([sx, sy, sx, sy]) + np.array([ox, oy, ox, oy])

    # 2. 计算三种 Pose 空间布局
    # (A) Native B letterbox 到 (w_canvas, h_canvas)
    scale_native = min(w_canvas / w_b, h_canvas / h_b)
    offset_x_native = (w_canvas - scale_native * w_b) / 2.0
    offset_y_native = (h_canvas - scale_native * h_b) / 2.0
    k_native = keypoints_b_px * scale_native + np.array([offset_x_native, offset_y_native])
    k_native[~valid_b] = np.nan
    meta_native = {
        "framing": "native",
        "scale": float(scale_native),
        "offset_xy": [float(offset_x_native), float(offset_y_native)],
        "w_b": w_b, "h_b": h_b,
        "canvas_wh": [w_canvas, h_canvas]
    }

    # (B) source_person 对齐
    k_source_person, meta_source_person = align_to_source_bbox(
        keypoints_b_px, bbox_b, bbox_a_canvas, framing="source_person"
    )
    k_source_person[~valid_b] = np.nan

    # (C) source_torso 对齐
    # 优先复用 hard_05_torso 中的精确关键点或计算
    torso_kp_path = torso_run_dir / "geometry/B_keypoints.npz"
    torso_cam_path = torso_run_dir / "geometry/B_camera.json"
    if torso_kp_path.exists() and torso_cam_path.exists():
        with np.load(torso_kp_path) as td:
            k_source_torso = td["keypoints_generation_px"].copy()
        t_cam = json.loads(torso_cam_path.read_text(encoding="utf-8"))
        meta_source_torso = t_cam["B_to_generation"]
    else:
        k_source_torso, meta_source_torso = align_to_source_bbox(
            keypoints_b_px, bbox_b, bbox_a_canvas, framing="source_torso"
        )
    k_source_torso[~valid_b] = np.nan

    layouts = {
        "native": (k_native, meta_native),
        "source_person": (k_source_person, meta_source_person),
        "source_torso": (k_source_torso, meta_source_torso)
    }

    # 3. 渲染骨架图与保存 NPZ
    pose_files = {}
    for name, (kp, meta) in layouts.items():
        pose_img = render_poseguide(kp, (w_canvas, h_canvas))
        pose_path = out_dir / f"pose_{name}.png"
        pose_img.save(pose_path)

        kp_path = out_dir / f"keypoints_{name}.npz"
        v_mask = valid_points(kp, w_canvas, h_canvas)
        np.savez_compressed(
            kp_path,
            keypoints_px=kp,
            valid=v_mask,
            requested_valid=valid_b,
            transform=json.dumps(meta)
        )

        n_req = int(valid_b.sum())
        n_in_frame = int(v_mask.sum())
        oof_ratio = float((n_req - n_in_frame) / n_req) if n_req > 0 else 0.0

        pose_files[name] = {
            "pose_png": str(pose_path),
            "pose_sha256": sha256_file(pose_path),
            "keypoints_npz": str(kp_path),
            "keypoints_sha256": sha256_file(kp_path),
            "joints_requested": n_req,
            "joints_in_frame": n_in_frame,
            "out_of_frame_ratio": oof_ratio,
            "transform": meta
        }

    # 4. 生成 A 的高精度人体分割 Mask
    print(">> Running YOLO11x-seg on A_reference.png...")
    mask_a_bool = run_yolo_person_seg(a_ref_path, args.seg_model, (w_canvas, h_canvas))
    mask_a_path = out_dir / "mask_A_person.png"
    Image.fromarray((mask_a_bool.astype(np.uint8) * 255)).save(mask_a_path)

    # 5. 生成三类姿态活动区域与最终 Inpainting Mask
    mask_files = {}
    for name, (kp, _) in layouts.items():
        # 目标姿态覆盖骨架区域
        t_len = torso_length(kp)
        if not np.isfinite(t_len) or t_len <= 0:
            t_len = torso_px(kp)
        if not np.isfinite(t_len) or t_len <= 0:
            t_len = 0.31 * (bbox_a_canvas[3] - bbox_a_canvas[1])

        target_region = mask_from_limbs(kp, (w_canvas, h_canvas), torso=t_len)
        target_path = out_dir / f"target_region_{name}.png"
        Image.fromarray((target_region.astype(np.uint8) * 255)).save(target_path)

        # 联合掩码：原人物区域 | 目标骨架活动区域
        combined = mask_a_bool | target_region

        # 膨胀半径：max(12, round(0.08 * torso_length))
        radius = max(12, int(round(0.08 * t_len)))
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        dilated = cv2.dilate(combined.astype(np.uint8), elem)
        final_mask = (dilated > 0).astype(np.uint8) * 255

        mask_path = out_dir / f"mask_{name}.png"
        Image.fromarray(final_mask).save(mask_path)

        # 生成叠加自检图
        if args.save_debug:
            with Image.open(a_ref_path) as im_a, Image.open(out_dir / f"pose_{name}.png") as im_pose:
                bg = im_a.convert("RGBA")
                # 红色半透明表示重绘区
                overlay_mask = np.zeros((h_canvas, w_canvas, 4), dtype=np.uint8)
                overlay_mask[final_mask > 0] = [255, 50, 50, 110]
                mask_layer = Image.fromarray(overlay_mask, mode="RGBA")
                blended = Image.alpha_composite(bg, mask_layer)
                # 骨架叠加在最上层
                pose_rgba = im_pose.convert("RGBA")
                pose_arr = np.asarray(pose_rgba)
                pose_valid = (pose_arr[:, :, :3].sum(axis=2) > 0)
                blended_arr = np.asarray(blended).copy()
                blended_arr[pose_valid] = pose_arr[pose_valid]
                overlay_path = out_dir / f"overlay_mask_{name}.png"
                Image.fromarray(blended_arr).save(overlay_path)

        mask_files[name] = {
            "mask_png": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "target_region_png": str(target_path),
            "dilation_radius": radius,
            "torso_length": float(t_len),
            "mask_coverage_ratio": float(final_mask.sum() / (w_canvas * h_canvas * 255))
        }

    # 6. 确定性 Prompt 编译与冻结
    raw_prompt = prompt_path.read_text(encoding="utf-8")
    union_prompt = compile_union_prompt(raw_prompt)
    prompt_out_path = input_dir / "prompt_pe_union.txt"
    prompt_out_path.write_text(union_prompt + "\n", encoding="utf-8")
    prompt_sha256 = sha256_file(prompt_out_path)

    # 7. 汇总并产出清单
    manifest = {
        "sample_id": run_dir.name,
        "canvas_size": [w_canvas, h_canvas],
        "source_preprocess": src_prep,
        "source_assets": {
            "A_reference": {"path": str(a_ref_path), "sha256": sha256_file(a_ref_path)},
            "B_rgb": {"path": str(b_rgb_path), "sha256": sha256_file(b_rgb_path)},
            "generation_prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)}
        },
        "frozen_prompt": {
            "path": str(prompt_out_path),
            "sha256": prompt_sha256,
            "text": union_prompt
        },
        "mask_A_person": {
            "path": str(mask_a_path),
            "sha256": sha256_file(mask_a_path),
            "coverage_ratio": float(mask_a_bool.sum() / (w_canvas * h_canvas))
        },
        "layouts": pose_files,
        "masks": mask_files
    }

    manifest_path = out_dir / "prep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    sample_manifest_path = input_dir / "sample_manifest.json"
    sample_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ 控制条件与 Mask 准备完成！Manifest 写入: {manifest_path}")
    print(f"   Frozen PE Prompt: {prompt_out_path} (SHA256: {prompt_sha256[:12]}...)")


if __name__ == "__main__":
    main()

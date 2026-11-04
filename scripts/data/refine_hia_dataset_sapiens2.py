#!/usr/bin/env python3
"""
Full-scale HIA Dataset Sapiens2 Test-Time Guided SMPL-X Refinement Pipeline.

Refines initial SMPL-X estimations from HIA_geometry_v1_20261029 across the dense pool
using Sapiens2-5B 2D keypoint supervision + upright nvdiffrast rasterization.

Outputs per frame:
  - smplx_params_refined.pt
  - normal.png (upright, background-masked)
  - depth_vis.png (upright, 8-bit normalized)
  - depth_metric.npy (Z-metric depth)
  - dwpose.png (DWPose 134 keypoint skeleton rendered from 3D joints)
  - camera.json
  - record.json (with metrics and SHA256 integrity checks)
"""

import os
import sys
import time
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = "/home/shangguanrz/project/pic-edit"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import smplx
import nvdiffrast.torch as dr
from src.pose_edit.geometry import smplx137_to_dwpose134, render_poseguide

SAPIENS_POSE_DIR = "/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose"
if SAPIENS_POSE_DIR not in sys.path:
    sys.path.insert(0, SAPIENS_POSE_DIR)
    sys.path.insert(0, os.path.join(SAPIENS_POSE_DIR, "tools/vis"))

from sapiens.pose.models import init_model
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from transformers import DetrForObjectDetection, DetrImageProcessor
from sapiens.pose.evaluators import nms

CONFIG_PATH = "/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
CKPT_PATH = "/data/shangguanrz/models/meta/sapiens2-pose-5b/sapiens2_5b_pose.safetensors"
DETR_PATH = "/data/shangguanrz/models/meta/detector/detr-resnet-101-dc5"
SMPLX_DIR = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files"
SMPLX_FACES_NPZ = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"

SMPLX_137_MAPPING = [
    0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63, 64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66,
    25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45,
    73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 56, 57, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143
]

GOLIATH_TO_SMPLX = {
    # Face (5)
    0: 55, 1: 57, 2: 56, 3: 59, 4: 58,
    # Body (6)
    5: 16, 6: 17, 7: 18, 8: 19, 62: 20, 41: 21,
    # Lower Body & Feet (12)
    9: 1, 10: 2, 11: 4, 12: 5, 13: 7, 14: 8,
    15: 60, 16: 61, 17: 62, 18: 63, 19: 64, 20: 65,
    # Right Hand (20)
    24: 52, 23: 53, 22: 54, 21: 71, # thumb
    28: 40, 27: 41, 26: 42, 25: 72, # forefinger
    32: 43, 31: 44, 30: 45, 29: 73, # middle
    36: 49, 35: 50, 34: 51, 33: 74, # ring
    40: 46, 39: 47, 38: 48, 37: 75, # pinky
    # Left Hand (20)
    45: 37, 44: 38, 43: 39, 42: 66, # thumb
    49: 25, 48: 26, 47: 27, 46: 67, # forefinger
    53: 28, 52: 29, 51: 30, 50: 68, # middle
    57: 34, 56: 35, 55: 36, 54: 69, # ring
    61: 31, 60: 32, 59: 33, 58: 70, # pinky
}

HAND_GOLIATH_INDICES = set(range(21, 62))
RHAND_GOLIATH = set(range(21, 41))
LHAND_GOLIATH = set(range(42, 62))

# SMPL-X body pose indices (21 joints, joint_id 1 to 21, each has 3 dims):
# Left arm:
#   joint 13 (left collar) -> body_pose index 12 (slice 36:39)
#   joint 16 (left shoulder) -> body_pose index 15 (slice 45:48)
#   joint 18 (left elbow) -> body_pose index 17 (slice 51:54)
#   joint 20 (left wrist) -> body_pose index 19 (slice 57:60)
# Right arm:
#   joint 14 (right collar) -> body_pose index 13 (slice 39:42)
#   joint 17 (right shoulder) -> body_pose index 16 (slice 48:51)
#   joint 19 (right elbow) -> body_pose index 18 (slice 54:57)
#   joint 21 (right wrist) -> body_pose index 20 (slice 60:63)
L_ARM_SLICES = [(12*3, 13*3), (15*3, 16*3), (17*3, 18*3), (19*3, 20*3)]
R_ARM_SLICES = [(13*3, 14*3), (16*3, 17*3), (18*3, 19*3), (20*3, 21*3)]

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def project_joints_2d(joints_3d, fx, fy, cx, cy):
    u = fx * (joints_3d[:, 0] / (joints_3d[:, 2] + 1e-4)) + cx
    v = fy * (joints_3d[:, 1] / (joints_3d[:, 2] + 1e-4)) + cy
    return torch.stack([u, v], dim=-1)

def render_mesh_nvdiffrast(verts, fx, fy, cx, cy, width, height, glctx, faces_cuda, smplx_faces, device):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    zmin, zmax = float(z.min().item()), float(z.max().item())
    near = max(1e-4, zmin * 0.5)
    far = max(near + 1.0, zmax * 1.5)

    x_clip = (2.0 * fx / width) * x + (2.0 * cx / width - 1.0) * z
    y_clip = (-2.0 * fy / height) * y + (1.0 - 2.0 * cy / height) * z
    A = (far + near) / (far - near)
    B = -2.0 * far * near / (far - near)
    z_clip = A * z + B
    w_clip = z

    clip = torch.stack([x_clip, y_clip, z_clip, w_clip], dim=-1).unsqueeze(0).contiguous()
    rast, _ = dr.rasterize(glctx, clip, faces_cuda, resolution=[height, width])
    valid_mask = (rast[0, :, :, 3] > 0)

    # Depth Map
    depth_interp, _ = dr.interpolate(z.view(1, -1, 1).contiguous(), rast, faces_cuda)
    depth_val = depth_interp[0, :, :, 0]
    metric_depth = depth_val.clone()
    metric_depth[~valid_mask] = 0.0

    if valid_mask.any():
        near_val = float(depth_val[valid_mask].min().item())
        far_val = float(depth_val[valid_mask].max().item())
        norm_depth = torch.clamp((far_val - depth_val) / (far_val - near_val + 1e-8) * 255.0, 0, 255)
        depth_t = torch.where(valid_mask, norm_depth, torch.zeros_like(norm_depth)).byte()
    else:
        depth_t = torch.zeros((height, width), dtype=torch.uint8, device=device)

    # Flip OpenGL NDC to OpenCV top-down raster coordinates
    depth_t = torch.flip(depth_t, dims=[0])
    metric_depth = torch.flip(metric_depth, dims=[0])
    depth_rgb = torch.stack([depth_t, depth_t, depth_t], dim=-1).cpu().numpy()

    # Normal Map
    v0 = verts[smplx_faces[:, 0]]
    v1 = verts[smplx_faces[:, 1]]
    v2 = verts[smplx_faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=-1)
    vertex_normals = torch.zeros_like(verts)
    for k in range(3):
        vertex_normals.index_add_(0, faces_cuda[:, k], face_normals)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

    normal_interp, _ = dr.interpolate(vertex_normals.unsqueeze(0).contiguous(), rast, faces_cuda)
    normal_val = torch.nn.functional.normalize(normal_interp[0], dim=-1)
    normal_rgb = ((normal_val + 1.0) * 0.5 * 255.0).clamp(0, 255).byte()
    normal_rgb[~valid_mask] = 0
    normal_rgb = torch.flip(normal_rgb, dims=[0])
    normal_np = normal_rgb.cpu().numpy()

    return Image.fromarray(depth_rgb), Image.fromarray(normal_np), metric_depth.cpu().numpy()

def render_dwpose_from_joints(joints_3d, fx, fy, cx, cy, width, height):
    j137 = joints_3d[SMPLX_137_MAPPING]
    j2d = project_joints_2d(j137, fx, fy, cx, cy).detach().cpu().numpy()
    dw134 = smplx137_to_dwpose134(j2d)
    return render_poseguide(dw134, (width, height))

def main():
    parser = argparse.ArgumentParser(description="Full-scale HIA Sapiens2-guided SMPL-X Refinement")
    parser.add_argument("--hia-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_processed_v2_full_20261027")
    parser.add_argument("--geom-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029")
    parser.add_argument("--out-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames for smoke test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir = out_dir / "frames"
    frames_out_dir.mkdir(exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    geom_frames_root = Path(args.geom_dir) / "frames"
    dense_images_root = Path(args.hia_dir) / "dense_pool" / "images"

    print("=== Scanning HIA Dataset frames ===")
    frame_manifest = []

    # Iterate through clip directories in geometry v1
    for clip_dir in sorted(geom_frames_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        cid = clip_dir.name
        for frame_dir in sorted(clip_dir.iterdir()):
            if not frame_dir.is_dir():
                continue
            rec_file = frame_dir / "record.json"
            if not rec_file.exists():
                continue
            fid = frame_dir.name
            
            # Read record to get exact paths
            try:
                with open(rec_file, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                frame_split = rec.get("split", "train")
                if args.split != "all" and frame_split != args.split:
                    continue
                
                rgb_path = rec.get("input", {}).get("image_path")
                smplx_pt = rec.get("artifacts", {}).get("smplx_params", {}).get("path")
                
                if not rgb_path or not smplx_pt:
                    continue
                if not Path(rgb_path).exists() or not Path(smplx_pt).exists():
                    continue

                frame_manifest.append({
                    "frame_id": fid,
                    "clip_id": cid,
                    "split": frame_split,
                    "rgb_path": rgb_path,
                    "smplx_pt": smplx_pt,
                    "orig_rec_path": str(rec_file)
                })
            except Exception:
                continue

    print(f"Total valid candidate frames discovered: {len(frame_manifest)}")
    if args.max_frames:
        frame_manifest = frame_manifest[:args.max_frames]
        print(f"Truncated to --max-frames={args.max_frames}")

    # Check resumption
    pending_manifest = []
    for item in frame_manifest:
        cid, fid = item["clip_id"], item["frame_id"]
        target_rec = frames_out_dir / cid / fid / "record.json"
        if not target_rec.exists():
            pending_manifest.append(item)

    print(f"Total frames: {len(frame_manifest)} | Already Completed: {len(frame_manifest) - len(pending_manifest)} | Pending: {len(pending_manifest)}")

    if args.dry_run:
        print("Dry run mode: Exiting without computation.")
        return

    if len(pending_manifest) == 0:
        print("All requested frames are already completed! Done.")
        return

    # Load Models
    print("\n1. Loading DETR Detector...", flush=True)
    det_proc = DetrImageProcessor.from_pretrained(DETR_PATH)
    det_model = DetrForObjectDetection.from_pretrained(DETR_PATH).eval().to(device)

    print("2. Loading Sapiens2-Pose-5B...", flush=True)
    t_load = time.time()
    sapiens_model = init_model(CONFIG_PATH, CKPT_PATH, device=str(device))
    meta_path = os.path.join(SAPIENS_POSE_DIR, "configs/_base_/keypoints308.py")
    sapiens_model.pose_metainfo = parse_pose_metainfo(dict(from_file=meta_path))
    codec_type = sapiens_model.cfg.codec.pop("type")
    sapiens_model.codec = UDPHeatmap(**sapiens_model.cfg.codec)
    print(f"Sapiens2-5B loaded in {time.time() - t_load:.1f}s", flush=True)

    print("3. Loading SMPL-X & nvdiffrast rasterizer...", flush=True)
    smpl_x = smplx.create(
        SMPLX_DIR,
        model_type='smplx',
        gender='neutral',
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
    ).to(device)
    glctx = dr.RasterizeCudaContext()
    smplx_faces = np.load(SMPLX_FACES_NPZ)['f']
    faces_cuda = torch.from_numpy(smplx_faces.astype(np.int32)).to(device)

    print("\n=== Commencing Pipeline Processing ===", flush=True)
    t_start = time.time()
    completed_count = 0
    failed_count = 0
    recent_latencies = []
    recent_reductions = []

    for idx, item in enumerate(pending_manifest):
        cid, fid = item["clip_id"], item["frame_id"]
        rgb_path = item["rgb_path"]
        params_path = item["smplx_pt"]
        frame_dir = frames_out_dir / cid / fid
        frame_dir.mkdir(parents=True, exist_ok=True)
        rec_path = frame_dir / "record.json"

        t_frame_0 = time.time()
        try:
            # 1. Image & Init Params
            img_pil = Image.open(rgb_path).convert("RGB")
            width, height = img_pil.size
            img_bgr = cv2.imread(rgb_path)
            d = torch.load(params_path, map_location='cpu')

            fx = d['focal_length_xy'][0, 0].item()
            fy = d['focal_length_xy'][0, 1].item()
            cx = d['princpt'][0, 0].item()
            cy = d['princpt'][0, 1].item()

            # 2. DETR Detection
            inputs = det_proc(images=img_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = det_model(**inputs)
            target_sizes = torch.tensor([[height, width]], device=device)
            results = det_proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
            person_mask = results["labels"] == 1
            boxes = results["boxes"][person_mask].cpu().numpy()
            scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)

            if len(boxes) > 0:
                bboxes_cat = np.concatenate([boxes, scores], axis=1)
                bboxes = bboxes_cat[nms(bboxes_cat, 0.3), :4]
                box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                best_idx = np.argmax(box_areas)
                primary_bbox = bboxes[best_idx]
                det_score = float(scores[best_idx, 0])
            else:
                primary_bbox = np.array([0, 0, width - 1, height - 1], dtype=np.float32)
                det_score = 0.0

            # 3. Sapiens2 Keypoints
            data_info = dict(img=img_bgr)
            data_info["bbox"] = primary_bbox[None]
            data_info["bbox_score"] = np.ones(1, dtype=np.float32)
            data = sapiens_model.pipeline(data_info)
            data = sapiens_model.data_preprocessor(data)

            with torch.no_grad():
                pred = sapiens_model(data["inputs"])
            pred = pred.cpu().numpy()
            kpts_i, scores_i = sapiens_model.codec.decode(pred[0])
            input_size = data["data_samples"]["meta"]["input_size"]
            bbox_center = data["data_samples"]["meta"]["bbox_center"]
            bbox_scale = data["data_samples"]["meta"]["bbox_scale"]
            kpts_scaled = (kpts_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale)[0]
            scores_arr = scores_i[0]

            s_kpts = torch.tensor(kpts_scaled, dtype=torch.float32, device=device)
            s_scores = torch.tensor(scores_arr, dtype=torch.float32, device=device)

            # 4. Occlusion-Aware Check & Keypoint Mapping
            # Load DWPose to cross-check occlusion
            dw_path = hia_dir / "dense_pool" / "poses" / cid / f"{fid}.json"
            l_hand_occluded = False
            r_hand_occluded = False
            if dw_path.exists():
                with open(dw_path) as f_dw:
                    dw_data = json.load(f_dw)
                dw_conf = np.array(dw_data.get("confidence", []))
                if len(dw_conf) >= 133:
                    l_dw_mean = float(dw_conf[91:112].mean())
                    l_dw_valid = int((dw_conf[91:112] > 0.3).sum())
                    r_dw_mean = float(dw_conf[112:133].mean())
                    r_dw_valid = int((dw_conf[112:133] > 0.3).sum())

                    l_sap_max = max(float(s_scores[g].item()) for g in LHAND_GOLIATH)
                    r_sap_max = max(float(s_scores[g].item()) for g in RHAND_GOLIATH)

                    l_hand_occluded = (l_dw_valid <= 3 and l_sap_max < 0.65) or (l_dw_mean < 0.25 and l_sap_max < 0.60)
                    r_hand_occluded = (r_dw_valid <= 3 and r_sap_max < 0.65) or (r_dw_mean < 0.25 and r_sap_max < 0.60)

            goliath_indices, smplx_indices, weights = [], [], []
            for g_idx, sm_idx in GOLIATH_TO_SMPLX.items():
                if g_idx in LHAND_GOLIATH and l_hand_occluded:
                    continue
                if g_idx in RHAND_GOLIATH and r_hand_occluded:
                    continue
                if g_idx == 62 and l_hand_occluded:
                    continue
                if g_idx == 41 and r_hand_occluded:
                    continue

                score = s_scores[g_idx].item()
                if score > 0.3:
                    goliath_indices.append(g_idx)
                    smplx_indices.append(sm_idx)
                    is_hand = (g_idx in HAND_GOLIATH_INDICES)
                    w = score * (2.5 if is_hand else 1.0)
                    weights.append(w)

            target_2d = s_kpts[goliath_indices]
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
            smplx_indices_t = torch.tensor(smplx_indices, dtype=torch.long, device=device)

            init_body_pose = d['body_pose'].to(device)
            init_lh_pose = d['left_hand_pose'].to(device)
            init_rh_pose = d['right_hand_pose'].to(device)
            global_orient = d['global_orient'].to(device)
            transl = d['transl'].to(device)
            betas = d['betas'].to(device)
            expression = d['expression'].to(device)
            jaw_pose = d['jaw_pose'].to(device)

            with torch.no_grad():
                init_out = smpl_x(
                    global_orient=global_orient,
                    body_pose=init_body_pose,
                    left_hand_pose=init_lh_pose,
                    right_hand_pose=init_rh_pose,
                    jaw_pose=jaw_pose,
                    betas=betas,
                    expression=expression,
                    transl=transl,
                )
                init_j2d = project_joints_2d(init_out.joints[0], fx, fy, cx, cy)
                init_matched_j2d = init_j2d[smplx_indices_t]
                init_errors = torch.norm(init_matched_j2d - target_2d, dim=-1)
                init_mpjpe = (init_errors * weights_t.squeeze(-1)).sum() / weights_t.sum()

            # 5. 35-step Adam Optimization with Occlusion Constraints
            opt_body_pose = torch.nn.Parameter(init_body_pose.clone())
            param_list = [{'params': [opt_body_pose], 'lr': 0.015}]

            if not l_hand_occluded:
                opt_lh_pose = torch.nn.Parameter(init_lh_pose.clone())
                param_list.append({'params': [opt_lh_pose], 'lr': 0.025})
            else:
                opt_lh_pose = init_lh_pose

            if not r_hand_occluded:
                opt_rh_pose = torch.nn.Parameter(init_rh_pose.clone())
                param_list.append({'params': [opt_rh_pose], 'lr': 0.025})
            else:
                opt_rh_pose = init_rh_pose

            opt_transl = torch.nn.Parameter(transl.clone())
            param_list.append({'params': [opt_transl], 'lr': 0.005})

            optimizer = torch.optim.Adam(param_list)

            steps = 35
            for _ in range(steps):
                optimizer.zero_grad()
                with torch.no_grad():
                    if l_hand_occluded:
                        for s, e in L_ARM_SLICES:
                            opt_body_pose[0, s:e] = init_body_pose[0, s:e]
                    if r_hand_occluded:
                        for s, e in R_ARM_SLICES:
                            opt_body_pose[0, s:e] = init_body_pose[0, s:e]

                out = smpl_x(
                    global_orient=global_orient,
                    body_pose=opt_body_pose,
                    left_hand_pose=opt_lh_pose,
                    right_hand_pose=opt_rh_pose,
                    jaw_pose=jaw_pose,
                    betas=betas,
                    expression=expression,
                    transl=opt_transl,
                )
                j2d = project_joints_2d(out.joints[0], fx, fy, cx, cy)
                matched_j2d = j2d[smplx_indices_t]

                reproj_loss = ((matched_j2d - target_2d).pow(2).sum(dim=-1).sqrt() * weights_t.squeeze(-1)).sum() / weights_t.sum()
                reg_body = (opt_body_pose - init_body_pose).pow(2).mean()

                reg_hand = 0.0
                if not l_hand_occluded:
                    reg_hand += (opt_lh_pose - init_lh_pose).pow(2).mean()
                if not r_hand_occluded:
                    reg_hand += (opt_rh_pose - init_rh_pose).pow(2).mean()

                reg_transl = (opt_transl - transl).pow(2).mean()

                loss = reproj_loss + 0.015 * reg_body + 0.005 * reg_hand + 0.1 * reg_transl
                loss.backward()

                if opt_body_pose.grad is not None:
                    if l_hand_occluded:
                        for s, e in L_ARM_SLICES:
                            opt_body_pose.grad[0, s:e] = 0.0
                    if r_hand_occluded:
                        for s, e in R_ARM_SLICES:
                            opt_body_pose.grad[0, s:e] = 0.0

                optimizer.step()

            with torch.no_grad():
                if l_hand_occluded:
                    for s, e in L_ARM_SLICES:
                        opt_body_pose[0, s:e] = init_body_pose[0, s:e]
                if r_hand_occluded:
                    for s, e in R_ARM_SLICES:
                        opt_body_pose[0, s:e] = init_body_pose[0, s:e]

            with torch.no_grad():
                final_out = smpl_x(
                    global_orient=global_orient,
                    body_pose=opt_body_pose,
                    left_hand_pose=opt_lh_pose,
                    right_hand_pose=opt_rh_pose,
                    jaw_pose=jaw_pose,
                    betas=betas,
                    expression=expression,
                    transl=opt_transl,
                )
                final_j2d = project_joints_2d(final_out.joints[0], fx, fy, cx, cy)
                final_matched_j2d = final_j2d[smplx_indices_t]
                final_errors = torch.norm(final_matched_j2d - target_2d, dim=-1)
                final_mpjpe = (final_errors * weights_t.squeeze(-1)).sum() / weights_t.sum()

            # 6. Render Upright Modalities
            ref_depth, ref_normal, ref_metric_depth = render_mesh_nvdiffrast(final_out.vertices[0], fx, fy, cx, cy, width, height, glctx, faces_cuda, smplx_faces, device)
            ref_dwpose = render_dwpose_from_joints(final_out.joints[0], fx, fy, cx, cy, width, height)

            # 7. Atomic Write Outputs
            refined_d = dict(d)
            refined_d['body_pose'] = opt_body_pose.detach().cpu()
            refined_d['left_hand_pose'] = opt_lh_pose.detach().cpu()
            refined_d['right_hand_pose'] = opt_rh_pose.detach().cpu()
            refined_d['transl'] = opt_transl.detach().cpu()

            pt_path = frame_dir / "smplx_params_refined.pt"
            torch.save(refined_d, pt_path)
            ref_normal.save(frame_dir / "normal.png")
            ref_depth.save(frame_dir / "depth_vis.png")
            ref_dwpose.save(frame_dir / "dwpose.png")
            np.save(frame_dir / "depth_metric.npy", ref_metric_depth)

            cam_info = {
                "focal_xy": [fx, fy],
                "principal_xy": [cx, cy],
                "width": width,
                "height": height
            }
            with open(frame_dir / "camera.json", "w", encoding="utf-8") as f_cam:
                json.dump(cam_info, f_cam, indent=2)

            imp_px = init_mpjpe.item() - final_mpjpe.item()
            imp_pct = (imp_px / init_mpjpe.item()) * 100.0
            dur_frame = time.time() - t_frame_0

            frame_rec = {
                "schema": "hia_geometry_sapiens2_refined_v1",
                "status": "complete",
                "frame_id": fid,
                "clip_id": cid,
                "split": item["split"],
                "input": {
                    "image_path": rgb_path,
                    "width": width,
                    "height": height
                },
                "metrics": {
                    "init_2d_mpjpe_px": round(init_mpjpe.item(), 2),
                    "refined_2d_mpjpe_px": round(final_mpjpe.item(), 2),
                    "error_reduction_pct": round(imp_pct, 2),
                    "total_frame_latency_sec": round(dur_frame, 2)
                },
                "artifacts": {
                    "smplx_params": str(pt_path),
                    "normal": str(frame_dir / "normal.png"),
                    "depth_vis": str(frame_dir / "depth_vis.png"),
                    "depth_metric": str(frame_dir / "depth_metric.npy"),
                    "dwpose": str(frame_dir / "dwpose.png")
                },
                "completed_at_unix": time.time()
            }
            with open(rec_path, "w", encoding="utf-8") as f_rec:
                json.dump(frame_rec, f_rec, indent=2)

            completed_count += 1
            recent_latencies.append(dur_frame)
            recent_reductions.append(imp_pct)

        except Exception as e:
            failed_count += 1
            fail_log = logs_dir / "failures.jsonl"
            with open(fail_log, "a", encoding="utf-8") as fl:
                fl.write(json.dumps({"frame_id": fid, "clip_id": cid, "error": str(e), "time": time.time()}) + "\n")
            print(f"Error processing {fid}: {e}", file=sys.stderr)

        # Progress update every 10 frames
        if (idx + 1) % 10 == 0 or idx == len(pending_manifest) - 1:
            elapsed = time.time() - t_start
            fps = completed_count / (elapsed + 1e-6)
            rem_frames = len(pending_manifest) - (idx + 1)
            eta_h = (rem_frames / (fps + 1e-6)) / 3600.0

            avg_lat = np.mean(recent_latencies[-20:]) if recent_latencies else 0.0
            avg_red = np.mean(recent_reductions[-20:]) if recent_reductions else 0.0

            prog_data = {
                "total_frames": len(frame_manifest),
                "processed_in_run": completed_count,
                "failed_in_run": failed_count,
                "remaining_pending": rem_frames,
                "overall_fps": round(fps, 3),
                "avg_frame_latency_sec": round(avg_lat, 2),
                "avg_error_reduction_pct": round(avg_red, 1),
                "eta_hours": round(eta_h, 2),
                "elapsed_hours": round(elapsed / 3600.0, 2),
                "last_processed_frame": fid,
                "timestamp": time.time()
            }
            tmp_prog = out_dir / "progress.tmp"
            with open(tmp_prog, "w", encoding="utf-8") as pf:
                json.dump(prog_data, pf, indent=2)
            os.replace(tmp_prog, out_dir / "progress.json")

            print(f"[{completed_count:05d}/{len(pending_manifest):05d}] {fid} | Avg {avg_lat:.2f}s/frame ({fps:.2f} FPS) | Red: -{avg_red:.1f}% | ETA: {eta_h:.2f}h", flush=True)

        # Periodically clear GPU cache to prevent memory fragmentation
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    print(f"\n=== Refinement Run Finished: {completed_count} frames completed, {failed_count} failures ===", flush=True)

if __name__ == "__main__":
    main()

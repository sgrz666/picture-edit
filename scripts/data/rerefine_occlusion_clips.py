#!/usr/bin/env python3
"""
Re-refine clips with occluded hands (e.g. hands behind back) using Occlusion-Aware Gating,
and refresh their outputs in frames/ and human-check/ with official banners.
"""

import sys
import os
import glob
import json
import time
import argparse
from pathlib import Path

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
    24: 52, 23: 53, 22: 54, 21: 71,
    28: 40, 27: 41, 26: 42, 25: 72,
    32: 43, 31: 44, 30: 45, 29: 73,
    36: 49, 35: 50, 34: 51, 33: 74,
    40: 46, 39: 47, 38: 48, 37: 75,
    # Left Hand (20)
    45: 37, 44: 38, 43: 39, 42: 66,
    49: 25, 48: 26, 47: 27, 46: 67,
    53: 28, 52: 29, 51: 30, 50: 68,
    57: 34, 56: 35, 55: 36, 54: 69,
    61: 31, 60: 32, 59: 33, 58: 70,
}

RHAND_GOLIATH = set(range(21, 41))
LHAND_GOLIATH = set(range(42, 62))

L_ARM_SLICES = [(12*3, 13*3), (15*3, 16*3), (17*3, 18*3), (19*3, 20*3)]
R_ARM_SLICES = [(13*3, 14*3), (16*3, 17*3), (18*3, 19*3), (20*3, 21*3)]

def project_joints_2d(joints_3d, fx, fy, cx, cy):
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2].clamp(min=1e-5)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
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
    depth_np = depth_t.cpu().numpy()

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

    return depth_np, normal_np, metric_depth.cpu().numpy()

def render_dwpose_from_joints(joints_3d, fx, fy, cx, cy, width, height):
    j137 = joints_3d[SMPLX_137_MAPPING]
    j2d = project_joints_2d(j137, fx, fy, cx, cy).detach().cpu().numpy()
    dw134 = smplx137_to_dwpose134(j2d)
    return render_poseguide(dw134, (width, height))

def build_comparison_strip(rgb_bgr, orig_dw_bgr, ref_dw_bgr, orig_norm_bgr, ref_norm_bgr, orig_depth_bgr, ref_depth_bgr, title_text):
    tile_w = 260
    h, w = rgb_bgr.shape[:2]
    tile_h = int(tile_w * (h / w))

    columns = [
        ("1. RGB Input", rgb_bgr, False),
        ("2. Orig 3D DWPose", orig_dw_bgr, False),
        ("3. Refined DWPose", ref_dw_bgr, True),
        ("4. Orig Normal", orig_norm_bgr, False),
        ("5. Refined Normal", ref_norm_bgr, True),
        ("6. Orig Depth", orig_depth_bgr, False),
        ("7. Refined Depth", ref_depth_bgr, True),
    ]

    banner_h = 36
    header_h = 28
    strip_w = tile_w * len(columns)
    strip_h = banner_h + header_h + tile_h

    canvas = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    canvas[:] = (20, 22, 28)

    # Top Banner
    canvas[:banner_h, :] = (28, 32, 42)
    cv2.putText(canvas, title_text, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Column Headers & Resized Panels
    for idx, (label, img, is_refined) in enumerate(columns):
        x0 = idx * tile_w
        x1 = x0 + tile_w
        
        hdr_color = (24, 52, 42) if is_refined else (35, 40, 52)
        canvas[banner_h:banner_h+header_h, x0:x1] = hdr_color
        cv2.putText(canvas, label, (x0 + 8, banner_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        
        r_img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[banner_h+header_h:strip_h, x0:x1] = r_img

    return canvas

def main():
    parser = argparse.ArgumentParser(description="Re-refine clips with occluded hands & refresh human-check")
    parser.add_argument("--clips", nargs="+", default=["000034", "000035", "000037", "000038"])
    parser.add_argument("--hia-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_processed_v2_full_20261027")
    parser.add_argument("--geom-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029")
    parser.add_argument("--refined-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    hia_dir = Path(args.hia_dir)
    geom_dir = Path(args.geom_dir)
    refined_root = Path(args.refined_dir)
    frames_root = refined_root / "frames"
    human_check_dir = refined_root / "human-check"
    human_check_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading DETR, Sapiens 2, SMPL-X & Renderer...")
    det_proc = DetrImageProcessor.from_pretrained(DETR_PATH)
    det_model = DetrForObjectDetection.from_pretrained(DETR_PATH).eval().to(device)

    sapiens_model = init_model(CONFIG_PATH, CKPT_PATH, device=device)
    meta_path = f"{SAPIENS_POSE_DIR}/configs/_base_/keypoints308.py"
    sapiens_model.pose_metainfo = parse_pose_metainfo(dict(from_file=meta_path))
    codec_type = sapiens_model.cfg.codec.pop('type')
    sapiens_model.codec = UDPHeatmap(**sapiens_model.cfg.codec)

    smpl_x = smplx.create(
        SMPLX_DIR,
        model_type='smplx',
        gender='neutral',
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz',
        use_pca=False,
        flat_hand_mean=True
    ).to(device).eval()

    smplx_npz = np.load(SMPLX_FACES_NPZ)
    smplx_faces = smplx_npz['f']
    faces_cuda = torch.from_numpy(smplx_faces.astype(np.int32)).to(device)
    glctx = dr.RasterizeCudaContext()

    target_clips = args.clips
    total_frames = 0
    all_tasks = []

    for cid in target_clips:
        c_frames = sorted([p.name for p in (frames_root / cid).iterdir() if p.is_dir()])
        for fid in c_frames:
            all_tasks.append((cid, fid))

    print(f"\nCommencing re-refinement on {len(all_tasks)} frames across {target_clips}...")

    t_global = time.time()
    for idx, (cid, fid) in enumerate(all_tasks, 1):
        t0 = time.time()
        rgb_path = hia_dir / "dense_pool" / "images" / cid / f"{fid}.png"
        dw_path = hia_dir / "dense_pool" / "poses" / cid / f"{fid}.json"
        v1_frame_dir = geom_dir / "frames" / cid / fid
        v1_rec_path = v1_frame_dir / "record.json"

        if not rgb_path.exists() or not v1_rec_path.exists():
            print(f"Skipping {fid}: missing rgb or v1 record")
            continue

        rgb_bgr = cv2.imread(str(rgb_path))
        H, W = rgb_bgr.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))

        with open(v1_rec_path) as f:
            v1_rec = json.load(f)
        v1_pt = v1_rec["artifacts"]["smplx_params"]["path"]
        d_init = torch.load(v1_pt, map_location=device)

        fx = d_init['focal_length_xy'][0, 0].item()
        fy = d_init['focal_length_xy'][0, 1].item()
        cx = d_init['princpt'][0, 0].item()
        cy = d_init['princpt'][0, 1].item()

        with open(dw_path) as f:
            dw = json.load(f)
        dw_cand = np.array(dw["candidate"])
        dw_conf = np.array(dw["confidence"])

        # DETR Detection
        inputs = det_proc(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = det_model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=device)
        results = det_proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]
        person_mask = results["labels"] == 1
        boxes = results["boxes"][person_mask].cpu().numpy()
        scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)

        bboxes_cat = np.concatenate([boxes, scores], axis=1)
        bboxes = bboxes_cat[nms(bboxes_cat, 0.3), :4]
        box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = np.argmax(box_areas)
        primary_bbox = bboxes[best_idx]

        data_info = dict(img=rgb_bgr)
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

        # Occlusion-Aware Check
        l_dw_mean = float(dw_conf[91:112].mean()) if len(dw_conf) >= 112 else 0.0
        l_dw_valid = int((dw_conf[91:112] > 0.3).sum()) if len(dw_conf) >= 112 else 0
        r_dw_mean = float(dw_conf[112:133].mean()) if len(dw_conf) >= 133 else 0.0
        r_dw_valid = int((dw_conf[112:133] > 0.3).sum()) if len(dw_conf) >= 133 else 0

        l_sap_scores = [float(s_scores[g].item()) for g in LHAND_GOLIATH]
        r_sap_scores = [float(s_scores[g].item()) for g in RHAND_GOLIATH]
        l_sap_max = max(l_sap_scores)
        r_sap_max = max(r_sap_scores)

        l_hand_occluded = (l_dw_valid <= 3 and l_sap_max < 0.65) or (l_dw_mean < 0.25 and l_sap_max < 0.60)
        r_hand_occluded = (r_dw_valid <= 3 and r_sap_max < 0.65) or (r_dw_mean < 0.25 and r_sap_max < 0.60)

        # Build keypoint targets
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
                is_hand = (g_idx in RHAND_GOLIATH or g_idx in LHAND_GOLIATH)
                w = score * (2.5 if is_hand else 1.0)
                weights.append(w)

        target_2d = s_kpts[goliath_indices]
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        smplx_indices_t = torch.tensor(smplx_indices, dtype=torch.long, device=device)

        init_body_pose = d_init['body_pose'].to(device)
        init_lh_pose = d_init['left_hand_pose'].to(device)
        init_rh_pose = d_init['right_hand_pose'].to(device)
        global_orient = d_init['global_orient'].to(device)
        transl = d_init['transl'].to(device)
        betas = d_init['betas'].to(device)
        expression = d_init['expression'].to(device)
        jaw_pose = d_init['jaw_pose'].to(device)

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
            init_mpjpe = float(((init_errors * weights_t.squeeze(-1)).sum() / weights_t.sum()).item())

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
        for step in range(steps):
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
            refined_errors = torch.norm(final_matched_j2d - target_2d, dim=-1)
            refined_mpjpe = float(((refined_errors * weights_t.squeeze(-1)).sum() / weights_t.sum()).item())
            red_pct = float(round((init_mpjpe - refined_mpjpe) / max(init_mpjpe, 1e-4) * 100.0, 2))

            ref_depth_gray, ref_norm_rgb, metric_depth = render_mesh_nvdiffrast(
                final_out.vertices[0], fx, fy, cx, cy, W, H, glctx, faces_cuda, smplx_faces, device
            )
            ref_dw_pil = render_dwpose_from_joints(
                final_out.joints[0], fx, fy, cx, cy, W, H
            )

        # 1. Update frames assets
        frame_dir = frames_root / cid / fid
        frame_dir.mkdir(parents=True, exist_ok=True)

        ref_norm_bgr = cv2.cvtColor(ref_norm_rgb, cv2.COLOR_RGB2BGR)
        ref_depth_bgr = cv2.cvtColor(ref_depth_gray, cv2.COLOR_GRAY2BGR)
        ref_dw_bgr = cv2.cvtColor(np.array(ref_dw_pil), cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(frame_dir / "normal.png"), ref_norm_bgr)
        cv2.imwrite(str(frame_dir / "depth_vis.png"), ref_depth_gray)
        cv2.imwrite(str(frame_dir / "dwpose.png"), ref_dw_bgr)
        np.save(str(frame_dir / "depth_metric.npy"), metric_depth)

        refined_d = dict(d_init)
        refined_d['body_pose'] = opt_body_pose.detach().cpu()
        refined_d['left_hand_pose'] = opt_lh_pose.detach().cpu()
        refined_d['right_hand_pose'] = opt_rh_pose.detach().cpu()
        refined_d['transl'] = opt_transl.detach().cpu()
        torch.save(refined_d, str(frame_dir / "smplx_params_refined.pt"))

        # Update record.json
        rec_data = {
            "clip_id": cid,
            "frame_id": fid,
            "metrics": {
                "init_2d_mpjpe_px": round(init_mpjpe, 2),
                "refined_2d_mpjpe_px": round(refined_mpjpe, 2),
                "error_reduction_pct": red_pct,
                "total_frame_latency_sec": round(time.time() - t0, 2)
            },
            "occlusion": {
                "left_hand_occluded": l_hand_occluded,
                "right_hand_occluded": r_hand_occluded,
            }
        }
        with open(frame_dir / "record.json", "w") as f_rec:
            json.dump(rec_data, f_rec, indent=2)

        # 2. Render Initial mesh upright for comparison
        with torch.no_grad():
            orig_depth_gray, orig_normal_rgb, _ = render_mesh_nvdiffrast(
                init_out.vertices[0], fx, fy, cx, cy, W, H, glctx, faces_cuda, smplx_faces, device
            )
            orig_norm_bgr = cv2.cvtColor(orig_normal_rgb, cv2.COLOR_RGB2BGR)
            orig_depth_bgr = cv2.cvtColor(orig_depth_gray, cv2.COLOR_GRAY2BGR)

        # Read orig dwpose from attempts or render it
        orig_dw_paths = glob.glob(f"/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029/frames/{cid}/{fid}/attempts/*/dwpose.png")
        if orig_dw_paths and os.path.exists(orig_dw_paths[0]):
            orig_dw_bgr = cv2.imread(orig_dw_paths[0])
        else:
            orig_dw_pil = render_dwpose_from_joints(init_out.joints[0], fx, fy, cx, cy, W, H)
            orig_dw_bgr = cv2.cvtColor(np.array(orig_dw_pil), cv2.COLOR_RGB2BGR)

        # 3. Build & save official human-check comparison strip
        title = f"CLIP: {cid} | FRAME: {fid} | 2D MPJPE: {init_mpjpe:.1f}px -> {refined_mpjpe:.1f}px (-{red_pct:.1f}%)"
        strip = build_comparison_strip(rgb_bgr, orig_dw_bgr, ref_dw_bgr, orig_norm_bgr, ref_norm_bgr, orig_depth_bgr, ref_depth_bgr, title)

        out_strip_path = human_check_dir / f"{fid}.jpg"
        cv2.imwrite(str(out_strip_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])

        elapsed = time.time() - t0
        print(f"[{idx}/{len(all_tasks)}] {fid} ({elapsed:.2f}s) | L-Occ: {l_hand_occluded}, R-Occ: {r_hand_occluded} | MPJPE: {init_mpjpe:.1f}->{refined_mpjpe:.1f}px (-{red_pct:.1f}%)")

    total_dur = time.time() - t_global
    print(f"\nAll {len(all_tasks)} frames finished in {total_dur:.1f}s ({total_dur/max(len(all_tasks), 1):.2f}s/frame)!")

if __name__ == "__main__":
    main()

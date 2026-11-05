#!/usr/bin/env python3
"""
Generate all before-and-after comparison images across the HIA dataset.
Order: Follows /data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/frames/
       sorted by clip_id (e.g., 000002, 000016, ...) and frame_id.

Output:
  Flat directly in:
  /data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/human-check/<frame_id>.jpg
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = "/home/shangguanrz/project/pic-edit"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import smplx
import nvdiffrast.torch as dr

SMPLX_DIR = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files"
SMPLX_FACES_NPZ = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"

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
    if valid_mask.any():
        near_val = float(depth_val[valid_mask].min().item())
        far_val = float(depth_val[valid_mask].max().item())
        norm_depth = torch.clamp((far_val - depth_val) / (far_val - near_val + 1e-8) * 255.0, 0, 255)
        depth_t = torch.where(valid_mask, norm_depth, torch.zeros_like(norm_depth)).byte()
    else:
        depth_t = torch.zeros((height, width), dtype=torch.uint8, device=device)

    depth_t = torch.flip(depth_t, dims=[0])
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

    return depth_np, normal_np

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
    parser = argparse.ArgumentParser(description="Generate all HIA comparison strips flat in human-check")
    parser.add_argument("--refined-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1")
    parser.add_argument("--orig-geom-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029/frames")
    parser.add_argument("--out-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/human-check")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    device = torch.device(args.device)
    refined_root = Path(args.refined_dir)
    frames_dir = refined_root / "frames"
    orig_geom_root = Path(args.orig_geom_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== Scanning frames in sorted order ===", flush=True)
    clip_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    print(f"Total clips found: {len(clip_dirs)} (Starting with {clip_dirs[0].name})", flush=True)

    all_frames = []
    for cdir in clip_dirs:
        cid = cdir.name
        fdirs = sorted([d for d in cdir.iterdir() if d.is_dir() and (d / "record.json").exists()], key=lambda x: x.name)
        for fdir in fdirs:
            all_frames.append({
                "clip_id": cid,
                "frame_id": fdir.name,
                "fdir": fdir
            })

    total_count = len(all_frames)
    print(f"Total valid frames discovered: {total_count}", flush=True)

    # Check already completed directly in flat out_root
    pending_frames = []
    for item in all_frames:
        target_path = out_root / f"{item['frame_id']}.jpg"
        if not (target_path.exists() and target_path.stat().st_size > 1024):
            pending_frames.append(item)

    print(f"Frames to generate: {len(pending_frames)} (Already completed: {total_count - len(pending_frames)})", flush=True)

    if len(pending_frames) == 0:
        print("All comparison images are already generated flat in human-check! Done.", flush=True)
        return

    print("\nLoading SMPL-X model and nvdiffrast rasterizer...", flush=True)
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
    print("Models and context initialized successfully.\n", flush=True)

    t_start = time.time()
    processed_count = 0
    recent_latencies = []

    for idx, item in enumerate(pending_frames):
        cid = item["clip_id"]
        fid = item["frame_id"]
        fdir = item["fdir"]

        out_jpg = out_root / f"{fid}.jpg"

        t_f0 = time.time()
        try:
            with open(fdir / "record.json") as f:
                rec = json.load(f)

            orig_rec_path = orig_geom_root / cid / fid / "record.json"
            if not orig_rec_path.exists():
                continue
            with open(orig_rec_path) as of:
                orig_rec = json.load(of)

            rgb_path = rec["input"]["image_path"]
            orig_pt_path = orig_rec["artifacts"]["smplx_params"]["path"]

            rgb_bgr = cv2.imread(rgb_path)
            if rgb_bgr is None:
                continue
            h, w = rgb_bgr.shape[:2]

            # Read refined assets already on disk
            ref_normal_bgr = cv2.imread(str(fdir / "normal.png"))
            ref_depth_gray = cv2.imread(str(fdir / "depth_vis.png"), cv2.IMREAD_GRAYSCALE)
            ref_depth_bgr = cv2.cvtColor(ref_depth_gray, cv2.COLOR_GRAY2BGR)
            ref_dwpose_bgr = cv2.imread(str(fdir / "dwpose.png"))

            # Read original dwpose
            orig_dw_path = orig_rec.get("artifacts", {}).get("dwpose", {}).get("path")
            if orig_dw_path and Path(orig_dw_path).exists():
                orig_dw_bgr = cv2.imread(orig_dw_path)
            else:
                orig_dw_bgr = np.zeros_like(ref_dwpose_bgr)

            # Render initial mesh upright
            d_orig = torch.load(orig_pt_path, map_location="cpu")
            fx = d_orig['focal_length_xy'][0, 0].item()
            fy = d_orig['focal_length_xy'][0, 1].item()
            cx = d_orig['princpt'][0, 0].item()
            cy = d_orig['princpt'][0, 1].item()

            with torch.no_grad():
                init_out = smpl_x(
                    global_orient=d_orig['global_orient'].to(device),
                    body_pose=d_orig['body_pose'].to(device),
                    left_hand_pose=d_orig['left_hand_pose'].to(device),
                    right_hand_pose=d_orig['right_hand_pose'].to(device),
                    jaw_pose=d_orig['jaw_pose'].to(device),
                    betas=d_orig['betas'].to(device),
                    expression=d_orig['expression'].to(device),
                    transl=d_orig['transl'].to(device),
                )
                orig_depth_gray, orig_normal_rgb = render_mesh_nvdiffrast(init_out.vertices[0], fx, fy, cx, cy, w, h, glctx, faces_cuda, smplx_faces, device)
                orig_normal_bgr = cv2.cvtColor(orig_normal_rgb, cv2.COLOR_RGB2BGR)
                orig_depth_bgr = cv2.cvtColor(orig_depth_gray, cv2.COLOR_GRAY2BGR)

            metrics = rec.get("metrics", {})
            init_err = metrics.get("init_2d_mpjpe_px", 0.0)
            ref_err = metrics.get("refined_2d_mpjpe_px", 0.0)
            red_pct = metrics.get("error_reduction_pct", 0.0)

            title = f"CLIP: {cid} | FRAME: {fid} | 2D MPJPE: {init_err:.1f}px -> {ref_err:.1f}px (-{red_pct:.1f}%)"
            panel = build_comparison_strip(rgb_bgr, orig_dw_bgr, ref_dwpose_bgr, orig_normal_bgr, ref_normal_bgr, orig_depth_bgr, ref_depth_bgr, title)

            tmp_jpg = out_root / f".tmp_{fid}.jpg"
            cv2.imwrite(str(tmp_jpg), panel, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            os.replace(tmp_jpg, out_jpg)

            dur = time.time() - t_f0
            processed_count += 1
            recent_latencies.append(dur)

        except Exception as e:
            print(f"Error on {fid}: {e}", file=sys.stderr)

        # Progress reporting every 100 frames
        if (idx + 1) % 100 == 0 or idx == len(pending_frames) - 1:
            elapsed = time.time() - t_start
            fps = processed_count / (elapsed + 1e-6)
            rem_frames = len(pending_frames) - (idx + 1)
            eta_m = (rem_frames / (fps + 1e-6)) / 60.0
            avg_ms = np.mean(recent_latencies[-100:]) * 1000 if recent_latencies else 0.0

            prog_info = {
                "total_frames": total_count,
                "completed": total_count - rem_frames,
                "pending": rem_frames,
                "overall_fps": round(fps, 2),
                "avg_frame_ms": round(avg_ms, 1),
                "eta_minutes": round(eta_m, 1),
                "elapsed_minutes": round(elapsed / 60.0, 1),
                "current_clip": cid,
                "current_frame": fid,
                "timestamp": time.time()
            }
            tmp_p = out_root / ".progress.tmp"
            with open(tmp_p, "w", encoding="utf-8") as pf:
                json.dump(prog_info, pf, indent=2)
            os.replace(tmp_p, out_root / "progress.json")

            print(f"[{processed_count:05d}/{len(pending_frames):05d}] Frame {fid} | {fps:.1f} FPS ({avg_ms:.1f}ms/frame) | ETA: {eta_m:.1f} min", flush=True)

    print(f"\n=== All {processed_count} comparison images generated flat in human-check! Finished in {(time.time()-t_start)/60.0:.2f} min! ===", flush=True)

if __name__ == "__main__":
    main()

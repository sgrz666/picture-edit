#!/usr/bin/env python3
"""
Generate all before-and-after comparison images in strict sequential order.
Order: Follows /data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1/frames/
       sorted by clip_id (e.g. 000002, 000016, ...) and then frame_id (e.g. 000002_000096, ...).

Output:
  Flat directly in:
  /data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_hidden_hand_v2_staging/human-check-full/<frame_id>.jpg
  (No subdirectories).
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = "/home/shangguanrz/project/pic-edit"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SMPLX_DIR = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files"
SMPLX_FACES_NPZ = "/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"


def render_mesh_nvdiffrast(verts, fx, fy, cx, cy, width, height, glctx, faces_cuda, smplx_faces, device):
    import nvdiffrast.torch as dr

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
        canvas[banner_h:banner_h + header_h, x0:x1] = hdr_color
        cv2.putText(canvas, label, (x0 + 8, banner_h + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        r_img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[banner_h + header_h:strip_h, x0:x1] = r_img

    return canvas


def scan_all_frames_in_order(frames_root: Path):
    """Scan and return all (clip_id, frame_id, frame_dir) in strict sequential order."""
    tasks = []
    with os.scandir(frames_root) as it:
        clip_dirs = sorted([e.name for e in it if e.is_dir()])

    for cid in clip_dirs:
        cdir = frames_root / cid
        try:
            with os.scandir(cdir) as it:
                f_dirs = sorted([e.name for e in it if e.is_dir()])
        except Exception:
            continue
        for fid in f_dirs:
            tasks.append((cid, fid, cdir / fid))
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refined-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1")
    parser.add_argument("--orig-geom-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029")
    parser.add_argument("--hia-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_processed_v2_full_20261027")
    parser.add_argument("--out-dir", type=str, default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_hidden_hand_v2_staging/human-check-full")
    parser.add_argument("--watch", action="store_true", help="continuous watch mode until all frames are completed")
    parser.add_argument("--watch-interval", type=int, default=10, help="seconds to sleep in watch mode")
    parser.add_argument("--link", action="store_true", help="use hardlink from existing human-check instead of copy")
    parser.add_argument("--re-render", action="store_true", help="force re-rendering instead of copying existing human-check strip")
    parser.add_argument("--max-frames", type=int, default=None, help="limit total frames (for testing)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    refined_root = Path(args.refined_dir)
    frames_root = refined_root / "frames"
    existing_check_root = refined_root / "human-check"
    orig_geom_root = Path(args.orig_geom_dir)
    hia_root = Path(args.hia_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("==========================================================", flush=True)
    print(" HIA Sequential Comparison Strip Generator (human-check-full)", flush=True)
    print(f" Source frames: {frames_root}", flush=True)
    print(f" Output target: {out_root} (FLAT)", flush=True)
    print(f" Order:         Strict sorted clip_id (starting 000002) -> frame_id", flush=True)
    print(f" Mode:          {'HARDLINK' if args.link else 'FAST COPY / RENDER'}", flush=True)
    print(f" Watch mode:    {'ENABLED' if args.watch else 'SINGLE PASS'}", flush=True)
    print("==========================================================", flush=True)

    all_tasks = scan_all_frames_in_order(frames_root)
    if args.max_frames:
        all_tasks = all_tasks[:args.max_frames]
    total_tasks = len(all_tasks)
    print(f"Total discovered frames across clips: {total_tasks}", flush=True)

    renderer = None
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    def init_renderer():
        nonlocal renderer
        if renderer is not None:
            return renderer
        import smplx
        import nvdiffrast.torch as dr

        print("Initializing SMPL-X & nvdiffrast for fallback rendering...", flush=True)
        smpl_x = smplx.create(
            SMPLX_DIR,
            model_type="smplx",
            gender="neutral",
            use_face_contour=True,
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=False,
        ).to(device)
        glctx = dr.RasterizeCudaContext()
        smplx_faces = np.load(SMPLX_FACES_NPZ)["f"]
        faces_cuda = torch.from_numpy(smplx_faces.astype(np.int32)).to(device)
        renderer = (smpl_x, glctx, faces_cuda, smplx_faces)
        return renderer

    t_start = time.time()
    last_upstream_status = ""

    while True:
        completed_this_cycle = 0
        newly_generated = 0

        for idx, (cid, fid, fdir) in enumerate(all_tasks, 1):
            out_jpg = out_root / f"{fid}.jpg"

            out_exists = out_jpg.exists() and out_jpg.stat().st_size > 1024
            src_check = existing_check_root / f"{fid}.jpg"
            src_exists = src_check.exists() and src_check.stat().st_size > 1024

            # If out_jpg already exists, check if src_check has an updated version
            if out_exists:
                if not args.re_render and src_exists:
                    if src_check.stat().st_mtime > out_jpg.stat().st_mtime + 1.0:
                        try:
                            shutil.copy2(src_check, out_jpg)
                            completed_this_cycle += 1
                            newly_generated += 1
                            if newly_generated % 100 == 0 or newly_generated == 1:
                                print(f"[{idx}/{total_tasks}] (sync-update) {fid}.jpg", flush=True)
                            continue
                        except Exception:
                            pass
                completed_this_cycle += 1
                continue

            # Method 1: If existing human-check strip already exists and is fresh, copy/link it directly!
            if not args.re_render and src_exists:
                try:
                    if args.link:
                        os.link(src_check, out_jpg)
                    else:
                        shutil.copy2(src_check, out_jpg)
                    completed_this_cycle += 1
                    newly_generated += 1
                    if newly_generated % 500 == 0 or newly_generated == 1:
                        print(f"[{completed_this_cycle}/{total_tasks}] (sync) {fid}.jpg", flush=True)
                    continue
                except Exception:
                    pass

            # Method 2: If refined frame assets exist, render the comparison strip
            rec_file = fdir / "record.json"
            norm_file = fdir / "normal.png"
            if not (rec_file.exists() and norm_file.exists()):
                # Not yet refined by the background pipeline; skip for now
                continue

            try:
                smpl_x, glctx, faces_cuda, smplx_faces = init_renderer()

                with open(rec_file) as f:
                    rec = json.load(f)

                rgb_path = hia_root / "dense_pool" / "images" / cid / f"{fid}.png"
                if not rgb_path.exists():
                    rgb_path = rec.get("input", {}).get("image_path")
                if not (rgb_path and Path(rgb_path).exists()):
                    continue

                rgb_bgr = cv2.imread(str(rgb_path))
                if rgb_bgr is None:
                    continue
                H, W = rgb_bgr.shape[:2]

                # Read refined modalities
                ref_norm_bgr = cv2.imread(str(fdir / "normal.png"))
                ref_depth_gray = cv2.imread(str(fdir / "depth_vis.png"), cv2.IMREAD_GRAYSCALE)
                ref_depth_bgr = cv2.cvtColor(ref_depth_gray, cv2.COLOR_GRAY2BGR) if ref_depth_gray is not None else None
                ref_dw_bgr = cv2.imread(str(fdir / "dwpose.png"))

                # Read original assets
                orig_rec_path = orig_geom_root / "frames" / cid / fid / "record.json"
                orig_rec = json.load(open(orig_rec_path)) if orig_rec_path.exists() else {}

                orig_dw_p = orig_rec.get("artifacts", {}).get("dwpose", {}).get("path")
                if orig_dw_p and Path(orig_dw_p).exists():
                    orig_dw_bgr = cv2.imread(orig_dw_p)
                else:
                    orig_dw_bgr = np.zeros_like(ref_dw_bgr) if ref_dw_bgr is not None else np.zeros((H, W, 3), dtype=np.uint8)

                # Render initial mesh upright
                orig_pt_p = orig_rec.get("artifacts", {}).get("smplx_params", {}).get("path")
                if orig_pt_p and Path(orig_pt_p).exists():
                    d_orig = torch.load(orig_pt_p, map_location="cpu")
                    fx = d_orig["focal_length_xy"][0, 0].item()
                    fy = d_orig["focal_length_xy"][0, 1].item()
                    cx = d_orig["princpt"][0, 0].item()
                    cy = d_orig["princpt"][0, 1].item()

                    with torch.no_grad():
                        init_out = smpl_x(
                            global_orient=d_orig["global_orient"].to(device),
                            body_pose=d_orig["body_pose"].to(device),
                            left_hand_pose=d_orig["left_hand_pose"].to(device),
                            right_hand_pose=d_orig["right_hand_pose"].to(device),
                            jaw_pose=d_orig["jaw_pose"].to(device),
                            betas=d_orig["betas"].to(device),
                            expression=d_orig["expression"].to(device),
                            transl=d_orig["transl"].to(device),
                        )
                        orig_d_gray, orig_n_rgb = render_mesh_nvdiffrast(
                            init_out.vertices[0], fx, fy, cx, cy, W, H, glctx, faces_cuda, smplx_faces, device
                        )
                        orig_norm_bgr = cv2.cvtColor(orig_n_rgb, cv2.COLOR_RGB2BGR)
                        orig_depth_bgr = cv2.cvtColor(orig_d_gray, cv2.COLOR_GRAY2BGR)
                else:
                    orig_norm_bgr = np.zeros_like(ref_norm_bgr)
                    orig_depth_bgr = np.zeros_like(ref_norm_bgr)

                metrics = rec.get("metrics", {})
                init_err = metrics.get("init_2d_mpjpe_px", 0.0)
                ref_err = metrics.get("refined_2d_mpjpe_px", 0.0)
                red_pct = metrics.get("error_reduction_pct", 0.0)

                title = f"CLIP: {cid} | FRAME: {fid} | 2D MPJPE: {init_err:.1f}px -> {ref_err:.1f}px (-{red_pct:.1f}%)"
                strip = build_comparison_strip(rgb_bgr, orig_dw_bgr, ref_dw_bgr, orig_norm_bgr, ref_norm_bgr, orig_depth_bgr, ref_depth_bgr, title)

                tmp_jpg = out_root / f".tmp_{fid}.jpg"
                cv2.imwrite(str(tmp_jpg), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
                os.replace(tmp_jpg, out_jpg)

                completed_this_cycle += 1
                newly_generated += 1
                if newly_generated % 50 == 0 or newly_generated == 1:
                    print(f"[{completed_this_cycle}/{total_tasks}] (render) {fid}.jpg MPJPE: {init_err:.1f}->{ref_err:.1f}px", flush=True)

            except Exception as e:
                print(f"Error rendering {fid}: {e}", file=sys.stderr, flush=True)
                continue

        # Check upstream refinement daemon status
        upstream_running = False
        upstream_prog = refined_root / "progress.json"
        upstream_status = ""
        if upstream_prog.exists():
            try:
                with open(upstream_prog) as uf:
                    udata = json.load(uf)
                u_curr = udata.get("current_index", 0)
                u_tot = udata.get("total_frames", total_tasks)
                if u_curr < u_tot:
                    upstream_running = True
                    upstream_status = f"upstream rerefine: {u_curr}/{u_tot} ({udata.get('progress_pct', 0)}%)"
            except Exception:
                pass

        # Update progress.json
        elapsed = time.time() - t_start
        all_done = completed_this_cycle >= total_tasks and not upstream_running
        prog = {
            "total_frames": total_tasks,
            "completed_frames": completed_this_cycle,
            "pending_frames": total_tasks - completed_this_cycle,
            "progress_pct": round(completed_this_cycle / max(total_tasks, 1) * 100, 2),
            "newly_generated": newly_generated,
            "elapsed_sec": round(elapsed, 1),
            "elapsed_minutes": round(elapsed / 60.0, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "upstream_status": upstream_status or "idle/completed",
            "status": "completed" if all_done else "in_progress",
        }
        with open(out_root / "progress.json", "w") as pf:
            json.dump(prog, pf, indent=2)

        if not args.watch:
            print(f"\nSingle pass completed: {completed_this_cycle}/{total_tasks} frames ready in {out_root}.", flush=True)
            break

        if all_done:
            print(f"\nAll {total_tasks} frames completed flat and upstream refinement is done!", flush=True)
            break

        # In watch mode, log progress and wait for more frame updates
        if newly_generated > 0 or upstream_status != last_upstream_status:
            print(f"[{time.strftime('%H:%M:%S')}] Full-check frames: {completed_this_cycle}/{total_tasks} | {upstream_status} | Synced {newly_generated} updates | Next scan in {args.watch_interval}s", flush=True)
            last_upstream_status = upstream_status

        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()

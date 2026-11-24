#!/usr/bin/env python3
"""Render SDPose-refined SMPL-X meshes into bilibili-compatible geometry assets.

Reads ``outputs/refined/<fid>.pt`` produced by ``SDpose--full`` and rasterizes
per-frame full-resolution geometry matching ``/data/shangguanrz/datasets/bilibili/geometry``:

    geometry/
      smplx/<method>/<note>/<fid>.pt         SMPL-X params + mesh_cam + joint_proj_model + W/H
      depth/<method>/<note>/<fid>_depth.png  16-bit uint16 metric depth in mm (bg=0)
      normal/<method>/<note>/<fid>_normal.png 24-bit View-Space purple-blue surface normal (bg=0)
      records/<note>/<fid>.json              per-frame metadata + depth/normal sanity stats
      summary.json                           global summary across all processed frames
      comparisons/overlays/<note>/<fid>_overlay.jpg  optional QC strips (RGB|SDPose|Normal|Depth)

Run with ``/home/shangguanrz/miniconda3/envs/deepgen/bin/python``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/shangguanrz/project/pic-edit")
SMPL_ROOT = REPO_ROOT / "add/tools/SMPLest-X"
HM_DIR = SMPL_ROOT / "human_models/human_model_files"
PARAM_NAMES = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "betas",
    "expression",
    "transl",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/shangguanrz/xiaohongshu/SDpose--full/samples/frames_manifest.json"),
        help="JSON list of {id, path} frame entries",
    )
    ap.add_argument(
        "--refined-dir",
        type=Path,
        default=Path("/data/shangguanrz/xiaohongshu/SDpose--full/outputs/refined"),
        help="directory containing SDPose-refined <fid>.pt files",
    )
    ap.add_argument(
        "--out-geometry",
        type=Path,
        default=Path("/data/shangguanrz/datasets/xiaohongshu(single)/geometry"),
        help="dataset geometry root",
    )
    ap.add_argument(
        "--method",
        default="sdpose_opt",
        help="method subdirectory under smplx/, depth/, normal/ (default: sdpose_opt)",
    )
    ap.add_argument(
        "--overlay-stride",
        type=int,
        default=40,
        help="save a 4-panel QC overlay every N frames (0 = off)",
    )
    ap.add_argument("--no-resume", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    return ap.parse_args()


def load_smplx_and_joint_idx(device: torch.device):
    """Build neutral SMPL-X layer and return (layer, SMPLest-X 137 joint indices)."""
    if str(SMPL_ROOT) not in sys.path:
        sys.path.insert(0, str(SMPL_ROOT))
    import smplx
    from human_models.human_models import SMPLX as SMPLX_HM

    joint_idx = list(SMPLX_HM(str(HM_DIR)).joint_idx)
    layer = smplx.create(
        str(HM_DIR),
        "smplx",
        gender="NEUTRAL",
        use_pca=False,
        use_face_contour=True,
        create_global_orient=False,
        create_body_pose=False,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_betas=False,
        create_expression=False,
        create_transl=False,
    )
    return layer.to(device).eval(), torch.tensor(joint_idx, dtype=torch.long, device=device)


def rasterize_geometry(
    glctx: dr.RasterizeCudaContext,
    verts: torch.Tensor,
    faces_cuda: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize metric depth (m) and View-Space surface normal matching bilibili convention."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    zmin, zmax = float(z.min().item()), float(z.max().item())
    near = max(1e-4, zmin * 0.5)
    far = max(near + 1.0, zmax * 1.5)

    x_clip = (2.0 * fx / width) * x + (2.0 * cx / width - 1.0) * z
    y_clip = (-2.0 * fy / height) * y + (1.0 - 2.0 * cy / height) * z
    a_coef = (far + near) / (far - near)
    b_coef = -2.0 * far * near / (far - near)
    clip = torch.stack([x_clip, y_clip, a_coef * z + b_coef, z], dim=-1).unsqueeze(0).contiguous()

    rast, _ = dr.rasterize(glctx, clip, faces_cuda, resolution=[height, width])
    valid_mask = rast[0, :, :, 3] > 0

    depth_interp, _ = dr.interpolate(z.view(1, -1, 1).contiguous(), rast, faces_cuda)
    depth_val = depth_interp[0, :, :, 0]
    metric_depth = depth_val.clone()
    metric_depth[~valid_mask] = 0.0

    if valid_mask.any():
        near_val = float(depth_val[valid_mask].min().item())
        far_val = float(depth_val[valid_mask].max().item())
        norm_depth = torch.clamp((far_val - depth_val) / (far_val - near_val + 1e-8) * 255.0, 0, 255)
        depth_vis = torch.where(valid_mask, norm_depth, torch.zeros_like(norm_depth)).byte()
    else:
        depth_vis = torch.zeros((height, width), dtype=torch.uint8, device=verts.device)

    depth_vis = torch.flip(depth_vis, dims=[0]).cpu().numpy()
    metric_depth = torch.flip(metric_depth, dims=[0]).cpu().numpy()

    fa = faces_cuda.long()
    v0, v1, v2 = verts[fa[:, 0]], verts[fa[:, 1]], verts[fa[:, 2]]
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    vertex_normals = torch.zeros_like(verts)
    for k in range(3):
        vertex_normals.index_add_(0, fa[:, k], face_normals)
    vertex_normals = F.normalize(vertex_normals, dim=-1)

    # OpenCV camera coords (x right, y down, z forward) -> View-Space normal [Nx, -Ny, -Nz]
    nview = torch.stack(
        [vertex_normals[:, 0], -vertex_normals[:, 1], -vertex_normals[:, 2]], dim=-1
    )
    nview = F.normalize(nview, dim=-1, eps=1e-8)
    normal_interp, _ = dr.interpolate(nview.unsqueeze(0).contiguous(), rast, faces_cuda)
    normal_val = F.normalize(normal_interp[0], dim=-1, eps=1e-8)
    normal_rgb = ((normal_val * 0.5 + 0.5).clamp(0, 1) * 255.0)
    normal_rgb[~valid_mask] = 0.0
    normal_rgb = torch.flip(normal_rgb, dims=[0])
    normal_bgr = normal_rgb[..., [2, 1, 0]].byte().cpu().numpy()
    return depth_vis, normal_bgr, metric_depth


def make_qc_strip(
    rgb_bgr: np.ndarray,
    skel_path: Path,
    normal_bgr: np.ndarray,
    depth_vis: np.ndarray,
    target_h: int = 640,
) -> np.ndarray:
    """Create 4-panel horizontal QC strip: RGB | SDPose 2D | Normal | Depth."""
    h, w = rgb_bgr.shape[:2]
    target_w = max(1, int(round(w * (target_h / h))))
    depth_bgr = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    if skel_path.is_file():
        skel_bgr = cv2.imread(str(skel_path))
        if skel_bgr is None:
            skel_bgr = np.zeros_like(rgb_bgr)
    else:
        skel_bgr = np.zeros_like(rgb_bgr)

    panels = [
        ("RGB", cv2.resize(rgb_bgr, (target_w, target_h))),
        ("SDPose 2D", cv2.resize(skel_bgr, (target_w, target_h))),
        ("SDPose-Opt Normal", cv2.resize(normal_bgr, (target_w, target_h))),
        ("SDPose-Opt Depth", cv2.resize(depth_bgr, (target_w, target_h))),
    ]
    bar = 28
    strips = []
    for title, panel in panels:
        canvas = np.zeros((target_h + bar, target_w, 3), dtype=np.uint8)
        canvas[bar:] = panel
        cv2.putText(
            canvas,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        strips.append(canvas)
    return np.concatenate(strips, axis=1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    layer, joint_idx137 = load_smplx_and_joint_idx(device)
    faces_cuda = torch.as_tensor(np.asarray(layer.faces, dtype=np.int32), device=device)
    glctx = dr.RasterizeCudaContext()

    out_geom = args.out_geometry
    method = args.method
    zero_eye = torch.zeros(1, 3, device=device)

    records: list[dict[str, Any]] = []
    missing_refined: list[str] = []
    t_all = time.time()
    rendered_cnt = 0
    skipped_cnt = 0

    for idx, it in enumerate(items):
        fid = it["id"]
        rgb_path = Path(it["path"])
        note = rgb_path.parent.name
        ref_path = args.refined_dir / f"{fid}.pt"
        if not ref_path.is_file():
            missing_refined.append(fid)
            print(f"[MISS] {fid}: missing {ref_path}", flush=True)
            continue

        smplx_out_path = out_geom / "smplx" / method / note / f"{fid}.pt"
        normal_out_path = out_geom / "normal" / method / note / f"{fid}_normal.png"
        depth_out_path = out_geom / "depth" / method / note / f"{fid}_depth.png"
        record_out_path = out_geom / "records" / note / f"{fid}.json"

        if (
            not args.no_resume
            and smplx_out_path.is_file()
            and normal_out_path.is_file()
            and depth_out_path.is_file()
            and record_out_path.is_file()
        ):
            rec = json.loads(record_out_path.read_text(encoding="utf-8"))
            records.append(rec)
            skipped_cnt += 1
            continue

        t0 = time.time()
        rgb_bgr = cv2.imread(str(rgb_path))
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read image: {rgb_path}")
        height, width = rgb_bgr.shape[:2]

        raw_params = torch.load(ref_path, map_location="cpu", weights_only=False)
        gpu_params = {k: raw_params[k].to(device=device, dtype=torch.float32) for k in PARAM_NAMES}
        f_xy = raw_params["focal_length_xy"][0].tolist()
        c_xy = raw_params["princpt"][0].tolist()
        fx, fy = float(f_xy[0]), float(f_xy[1])
        cx, cy = float(c_xy[0]), float(c_xy[1])

        with torch.no_grad():
            out = layer(**gpu_params, leye_pose=zero_eye, reye_pose=zero_eye)
            verts = out.vertices[0]
            j137 = out.joints[0, joint_idx137]
            z137 = j137[:, 2:3].clamp(min=1e-3)
            f_t = torch.tensor([fx, fy], device=device, dtype=torch.float32)
            c_t = torch.tensor([cx, cy], device=device, dtype=torch.float32)
            j_proj = (j137[:, :2] / z137) * f_t + c_t

            depth_vis, normal_bgr, metric_depth = rasterize_geometry(
                glctx, verts, faces_cuda, fx, fy, cx, cy, width, height
            )

        depth_mm_raw = metric_depth * 1000.0
        clipped_px = int((depth_mm_raw > 65535.0).sum())
        depth_mm = depth_mm_raw.clip(0, 65535).astype(np.uint16)
        fg_mask = depth_mm > 0
        fg_pixels = int(fg_mask.sum())

        smplx_payload = {k: raw_params[k].detach().cpu().float() for k in PARAM_NAMES}
        smplx_payload["mesh_cam"] = verts.unsqueeze(0).detach().cpu().float()
        smplx_payload["joint_proj_model"] = j_proj.unsqueeze(0).detach().cpu().float()
        smplx_payload["transl"] = raw_params["transl"].detach().cpu().float()
        smplx_payload["focal_length_xy"] = raw_params["focal_length_xy"].detach().cpu().float()
        smplx_payload["princpt"] = raw_params["princpt"].detach().cpu().float()
        smplx_payload["width"] = torch.tensor([[float(width)]], dtype=torch.float32)
        smplx_payload["height"] = torch.tensor([[float(height)]], dtype=torch.float32)
        smplx_payload["render_bbox"] = torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32)

        for p in (smplx_out_path, normal_out_path, depth_out_path, record_out_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        torch.save(smplx_payload, smplx_out_path)
        if not cv2.imwrite(str(normal_out_path), normal_bgr):
            raise RuntimeError(f"Failed to write {normal_out_path}")
        if not cv2.imwrite(str(depth_out_path), depth_mm):
            raise RuntimeError(f"Failed to write {depth_out_path}")

        overlay_path_str = None
        if args.overlay_stride > 0 and (idx % args.overlay_stride == 0):
            skel_path = out_geom / "skeleton" / "sdpose_2d" / note / f"{fid}_skeleton.png"
            strip = make_qc_strip(rgb_bgr, skel_path, normal_bgr, depth_vis)
            ov_path = out_geom / "comparisons" / "overlays" / note / f"{fid}_overlay.jpg"
            ov_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ov_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
            overlay_path_str = str(ov_path)

        rec = {
            "fid": fid,
            "note": note,
            "width": width,
            "height": height,
            "fg_pixels": fg_pixels,
            "fg_frac": round(fg_pixels / float(width * height), 6),
            "depth_min_mm": int(depth_mm[fg_mask].min()) if fg_pixels > 0 else 0,
            "depth_max_mm": int(depth_mm[fg_mask].max()) if fg_pixels > 0 else 0,
            "clipped_65535_pixels": clipped_px,
            "normals": {method: str(normal_out_path)},
            "depths": {method: str(depth_out_path)},
            "smplx": {method: str(smplx_out_path)},
            "overlay_path": overlay_path_str,
            "elapsed_sec": round(time.time() - t0, 4),
        }
        record_out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        records.append(rec)
        rendered_cnt += 1

        if rendered_cnt % 100 == 0 or rendered_cnt == len(items):
            print(
                f"[{idx + 1}/{len(items)}] rendered={rendered_cnt} skipped={skipped_cnt} "
                f"elapsed={time.time() - t_all:.1f}s",
                flush=True,
            )

    total_elapsed = time.time() - t_all
    summary = {
        "dataset": "xiaohongshu(single)",
        "source_refined_dir": str(args.refined_dir),
        "methods": [method],
        "manifest_frames": len(items),
        "processed_frames": len(records),
        "rendered_this_run": rendered_cnt,
        "skipped_existing": skipped_cnt,
        "missing_refined": missing_refined,
        "empty_fg_frames": [r["fid"] for r in records if r["fg_pixels"] == 0],
        "clipped_depth_frames": [r["fid"] for r in records if r["clipped_65535_pixels"] > 0],
        "total_time_sec": round(total_elapsed, 3),
        "avg_time_per_frame_sec": round(total_elapsed / max(1, len(records)), 4),
    }
    summary_path = out_geom / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import smplx
import nvdiffrast.torch as dr


PROJECT = Path("/home/shangguanrz/project/pic-edit").resolve()

FRAMES_MANIFEST = PROJECT / "datasets/Hi4D_pilot_v1/native_interaction_v2/frames_used_with_dwpose_v2.jsonl"
PAIRS_MANIFEST  = PROJECT / "datasets/Hi4D_pilot_v1/native_interaction_v2/pairs_all_with_dwpose.jsonl"
OUT_ROOT        = PROJECT / "datasets/Hi4D_pilot_v1/final_assets_native_v1"

SMPL_MODEL_DIR  = PROJECT / "models/smpl"
SMPL_MODEL_PATH = SMPL_MODEL_DIR / "SMPL_NEUTRAL.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------

def read_jsonl(p: Path):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            cur = os.readlink(dst)
            if cur == str(src):
                return
        dst.unlink()
    os.symlink(src, dst)


def npz_load(path):
    with np.load(path, allow_pickle=True) as x:
        return {k: x[k] for k in x.files}


def find_first(row, keys):
    for k in keys:
        if k in row and row[k] not in [None, ""]:
            return row[k]
    return None


def parse_rgb_like_path(path_str):
    """
    兼容:
    .../raw/pair01/hug01/images/16/000020.jpg
    """
    p = Path(path_str)
    parts = p.parts
    if len(parts) < 5:
        raise ValueError(f"bad path: {path_str}")

    # 找 images 所在位置
    idx = None
    for i, x in enumerate(parts):
        if x == "images":
            idx = i
            break
    if idx is None or idx < 2 or idx + 2 >= len(parts):
        raise ValueError(f"cannot parse rgb path: {path_str}")

    pair = parts[idx - 2]
    action = parts[idx - 1]
    cam = parts[idx + 1]
    stem = Path(parts[idx + 2]).stem
    return pair, action, str(cam), stem


def parse_frame_key_from_row(row):
    rgb_path = find_first(
        row,
        [
            "rgb",
            "target_rgb",
            "frame_rgb",
        ],
    )
    if rgb_path is None:
        raise ValueError(f"row has no rgb-like path: {row.keys()}")
    return parse_rgb_like_path(rgb_path)


def parse_target_key_from_pair_row(row):
    rgb_path = find_first(
        row,
        [
            "target_rgb",
            "rgb_target",
            "target_frame_rgb",
        ],
    )
    if rgb_path is not None:
        return parse_rgb_like_path(rgb_path)

    # 兜底：如果 pair manifest 里没有 target_rgb
    pair = find_first(row, ["target_pair", "pair"])
    action = find_first(row, ["target_action", "action"])
    cam = find_first(row, ["target_camera_id", "camera_id", "target_cam"])
    stem = find_first(row, ["target_frame_stem", "frame_stem", "target_frame"])
    if None in [pair, action, cam, stem]:
        raise ValueError(f"cannot parse target key from pair row keys={row.keys()}")
    return str(pair), str(action), str(cam), str(stem).zfill(6)


def build_target_set(pair_rows):
    out = set()
    for r in pair_rows:
        out.add(parse_target_key_from_pair_row(r))
    return out


def key_to_str(key):
    pair, action, cam, stem = key
    return f"{pair}/{action}/cam{cam}/{stem}"


def make_output_dir(out_root: Path, key):
    pair, action, cam, stem = key
    return out_root / pair / action / f"cam{cam}"


# ------------------------------------------------------------
# renderer
# ------------------------------------------------------------

def load_smpl_faces():
    model = smplx.create(
        str(SMPL_MODEL_DIR),
        model_type="smpl",
        gender="NEUTRAL",
        batch_size=1,
    )
    faces = torch.tensor(
        model.faces.astype(np.int32),
        dtype=torch.int32,
        device=DEVICE,
    )
    return faces


def compute_vertex_normals(verts, faces):
    # verts: [V,3], faces: [F,3]
    v0 = verts[faces[:, 0].long()]
    v1 = verts[faces[:, 1].long()]
    v2 = verts[faces[:, 2].long()]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F,3]

    vn = torch.zeros_like(verts)
    for i in range(3):
        idx = faces[:, i].long().unsqueeze(-1).expand(-1, 3)
        vn.scatter_add_(0, idx, fn)
    vn = F.normalize(vn, dim=-1, eps=1e-8)
    return vn


def world_to_camera(verts_world, extrinsic):
    # extrinsic: [3,4], direct world->camera
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    cam = (R @ verts_world.T).T + t[None]
    return cam, R


def camera_to_clip(cam, K, H, W):
    """
    cam: [V,3]
    K  : [3,3]
    return clip: [V,4]
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2].clamp(min=1e-4)

    u_ndc = 2.0 * (fx * x / z + cx) / (W - 1.0) - 1.0
    v_ndc = 1.0 - 2.0 * (fy * y / z + cy) / (H - 1.0)

    zmin = torch.min(z).item()
    zmax = torch.max(z).item()
    znear = max(1e-4, zmin - 0.05)
    zfar = zmax + 0.05
    z_ndc = 2.0 * (z - znear) / max(1e-6, (zfar - znear)) - 1.0

    clip = torch.stack(
        [
            u_ndc * z,
            v_ndc * z,
            z_ndc * z,
            z,
        ],
        dim=-1,
    )
    return clip


def rasterize_single(glctx, clip, faces, H, W, attrs_dict):
    """
    clip:  [V,4]
    faces: [F,3] int32
    attrs_dict: {name: [V,C]}
    """
    clip = clip.contiguous()
    faces = faces.contiguous()

    rast, _ = dr.rasterize(glctx, clip[None], faces, resolution=[H, W])
    valid = rast[0, :, :, 3] > 0

    out = {"valid": valid.cpu().numpy().astype(np.uint8)}
    for name, attr in attrs_dict.items():
        attr = attr[None].contiguous()
        val, _ = dr.interpolate(attr, rast, faces)
        out[name] = val[0].detach().cpu().numpy()

    return out


def render_geometry_for_frame(frame_row, faces, glctx):
    rgb_path = Path(find_first(frame_row, ["rgb", "target_rgb"]))
    smpl_path = Path(find_first(frame_row, ["official_smpl", "smpl", "target_smpl"]))
    cam_path = Path(find_first(frame_row, ["camera_file", "target_camera", "camera"]))
    dwA_path = Path(find_first(frame_row, ["dwpose_A", "target_dwpose_A"]))
    dwB_path = Path(find_first(frame_row, ["dwpose_B", "target_dwpose_B"]))

    person_A = int(find_first(frame_row, ["person_A", "target_person_A"]) or 0)
    person_B = int(find_first(frame_row, ["person_B", "target_person_B"]) or 1)

    cam_id = int(find_first(frame_row, ["camera_id", "target_camera_id"]))
    pair, action, cam_str, stem = parse_frame_key_from_row(frame_row)

    rgb = Image.open(rgb_path).convert("RGB")
    W, H = rgb.size

    smpl = npz_load(smpl_path)
    camz = npz_load(cam_path)

    ids = camz["ids"].tolist()
    cam_idx = ids.index(cam_id)

    K = torch.tensor(camz["intrinsics"][cam_idx], dtype=torch.float32, device=DEVICE)
    E = torch.tensor(camz["extrinsics"][cam_idx], dtype=torch.float32, device=DEVICE)

    verts_np = smpl["verts"].astype(np.float32)   # [2,6890,3]
    contact_np = smpl["contact"]                  # [2,6890]

    vertsA_w = torch.tensor(verts_np[person_A], dtype=torch.float32, device=DEVICE)
    vertsB_w = torch.tensor(verts_np[person_B], dtype=torch.float32, device=DEVICE)

    contactA_v = torch.tensor((contact_np[person_A] > 0).astype(np.float32), dtype=torch.float32, device=DEVICE)
    contactB_v = torch.tensor((contact_np[person_B] > 0).astype(np.float32), dtype=torch.float32, device=DEVICE)

    vertsA_c, R = world_to_camera(vertsA_w, E)
    vertsB_c, _ = world_to_camera(vertsB_w, E)

    normalsA = compute_vertex_normals(vertsA_c, faces)
    normalsB = compute_vertex_normals(vertsB_c, faces)

    clipA = camera_to_clip(vertsA_c, K, H, W)
    clipB = camera_to_clip(vertsB_c, K, H, W)

    # 单人 silhouette / contact
    singleA = rasterize_single(
        glctx,
        clipA,
        faces,
        H,
        W,
        {
            "depth": vertsA_c[:, 2:3],
            "normal": normalsA,
            "contact": contactA_v[:, None],
        },
    )
    singleB = rasterize_single(
        glctx,
        clipB,
        faces,
        H,
        W,
        {
            "depth": vertsB_c[:, 2:3],
            "normal": normalsB,
            "contact": contactB_v[:, None],
        },
    )

    maskA = singleA["valid"] > 0
    maskB = singleB["valid"] > 0
    interperson_overlap = maskA & maskB

    # 双人合并 scene，求 front owner / depth / normal
    V = vertsA_c.shape[0]
    clip_all = torch.cat([clipA, clipB], dim=0)
    verts_all_c = torch.cat([vertsA_c, vertsB_c], dim=0)
    normals_all = torch.cat([normalsA, normalsB], dim=0)
    owner_all = torch.cat(
        [
            torch.ones((V, 1), dtype=torch.float32, device=DEVICE),
            torch.ones((V, 1), dtype=torch.float32, device=DEVICE) * 2.0,
        ],
        dim=0,
    )
    facesB = faces + V
    faces_all = torch.cat([faces, facesB], dim=0)

    scene = rasterize_single(
        glctx,
        clip_all,
        faces_all,
        H,
        W,
        {
            "depth": verts_all_c[:, 2:3],
            "normal": normals_all,
            "owner": owner_all,
        },
    )

    scene_valid = scene["valid"] > 0
    depth_scene = scene["depth"][:, :, 0]
    normal_scene = scene["normal"]
    normal_norm = np.linalg.norm(normal_scene, axis=-1, keepdims=True)
    normal_scene = normal_scene / np.clip(normal_norm, 1e-8, None)

    front_owner = np.rint(scene["owner"][:, :, 0]).astype(np.uint8)
    front_owner[~scene_valid] = 0

    A_occ_by_B = interperson_overlap & (front_owner == 2)
    B_occ_by_A = interperson_overlap & (front_owner == 1)

    contactA_img = singleA["contact"][:, :, 0]
    contactB_img = singleB["contact"][:, :, 0]
    contactA_img = (contactA_img > 1e-5) & maskA
    contactB_img = (contactB_img > 1e-5) & maskB

    out = {
        "rgb_path": str(rgb_path),
        "dwpose_A_path": str(dwA_path),
        "dwpose_B_path": str(dwB_path),
        "pair": pair,
        "action": action,
        "camera_id": cam_id,
        "frame_stem": stem,
        "image_width": W,
        "image_height": H,

        "depth_scene": depth_scene.astype(np.float32),
        "normal_scene": normal_scene.astype(np.float32),

        "semantic_A": maskA.astype(np.uint8),
        "semantic_B": maskB.astype(np.uint8),

        "contact_A": contactA_img.astype(np.uint8),
        "contact_B": contactB_img.astype(np.uint8),

        "front_owner": front_owner.astype(np.uint8),
        "interperson_overlap": interperson_overlap.astype(np.uint8),
        "A_occluded_by_B": A_occ_by_B.astype(np.uint8),
        "B_occluded_by_A": B_occ_by_A.astype(np.uint8),
    }
    return out


# ------------------------------------------------------------
# visualization
# ------------------------------------------------------------

def save_gray_mask(mask_u8, out_path):
    img = Image.fromarray((mask_u8.astype(np.uint8) * 255))
    img.save(out_path)


def colorize_depth(depth, valid):
    vis = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    z = depth[valid]
    if len(z) == 0:
        return vis

    lo = np.percentile(z, 2)
    hi = np.percentile(z, 98)
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((depth - lo) / (hi - lo), 0, 1)

    # 简单伪彩：蓝->青->黄->红
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(x - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    vis[valid] = rgb[valid]
    return vis


def colorize_normal(normal, valid):
    vis = np.zeros_like(normal, dtype=np.uint8)
    x = ((normal + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    vis[valid] = x[valid]
    return vis


def colorize_semantic(maskA, maskB):
    H, W = maskA.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    onlyA = (maskA > 0) & ~(maskB > 0)
    onlyB = (maskB > 0) & ~(maskA > 0)
    both  = (maskA > 0) & (maskB > 0)
    vis[onlyA] = np.array([255, 80, 80], dtype=np.uint8)
    vis[onlyB] = np.array([80, 120, 255], dtype=np.uint8)
    vis[both]  = np.array([255, 220, 0], dtype=np.uint8)
    return vis


def colorize_contact(contactA, contactB):
    H, W = contactA.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    onlyA = (contactA > 0) & ~(contactB > 0)
    onlyB = (contactB > 0) & ~(contactA > 0)
    both  = (contactA > 0) & (contactB > 0)
    vis[onlyA] = np.array([255, 80, 80], dtype=np.uint8)
    vis[onlyB] = np.array([80, 120, 255], dtype=np.uint8)
    vis[both]  = np.array([255, 220, 0], dtype=np.uint8)
    return vis


def colorize_directional_occlusion(A_occ_by_B, B_occ_by_A):
    H, W = A_occ_by_B.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[A_occ_by_B > 0] = np.array([255, 80, 80], dtype=np.uint8)
    vis[B_occ_by_A > 0] = np.array([80, 120, 255], dtype=np.uint8)
    both = (A_occ_by_B > 0) & (B_occ_by_A > 0)
    vis[both] = np.array([255, 220, 0], dtype=np.uint8)
    return vis


def draw_dwpose_vis(dwA_path, dwB_path, H, W):
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def draw_one(npz_path, color):
        with np.load(npz_path, allow_pickle=True) as d:
            xy_n = d["xy_normalized"]   # [K,2]
            score = d["scores"]         # [K]
        xy = xy_n.copy()
        xy[:, 0] *= W
        xy[:, 1] *= H

        # 这里不强依赖具体骨架拓扑，直接按相邻点画，重在可视化
        for i in range(len(xy) - 1):
            if score[i] > 0.1 and score[i + 1] > 0.1:
                draw.line(
                    [tuple(xy[i]), tuple(xy[i + 1])],
                    fill=color,
                    width=max(2, int(min(H, W) * 0.005)),
                )
        for i in range(len(xy)):
            if score[i] > 0.1:
                r = max(2, int(min(H, W) * 0.006))
                x, y = xy[i]
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    draw_one(dwA_path, (255, 80, 80))
    draw_one(dwB_path, (80, 120, 255))
    return np.array(canvas)


def save_qc_panel(rgb_path, dwpose_img, depth_img, normal_img, semantic_img, contact_img, occ_img, out_path):
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    H, W = rgb.shape[:2]

    imgs = [
        ("Target RGB", rgb),
        ("Target DWPose", dwpose_img),
        ("Target Depth", depth_img),
        ("Target Normal", normal_img),
        ("Person Semantic", semantic_img),
        ("Contact", contact_img),
        ("Directional Occlusion", occ_img),
    ]

    tile_w = W
    tile_h = H
    cols = 4
    rows = math.ceil(len(imgs) / cols)

    panel = Image.new("RGB", (cols * tile_w, rows * tile_h + rows * 30), (0, 0, 0))
    draw = ImageDraw.Draw(panel)

    for idx, (title, arr) in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * (tile_h + 30)
        draw.text((x0 + 8, y0 + 5), title, fill=(255, 255, 255))
        im = Image.fromarray(arr)
        panel.paste(im, (x0, y0 + 30))

    panel.save(out_path, quality=95)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    print("=" * 72)
    print("Hi4D FINAL TARGET ASSETS MATERIALIZATION")
    print("=" * 72)
    print("DEVICE:", DEVICE)
    print("frames manifest:", FRAMES_MANIFEST)
    print("pairs manifest :", PAIRS_MANIFEST)
    print("out root       :", OUT_ROOT)
    print()

    ensure_dir(OUT_ROOT)

    frame_rows = read_jsonl(FRAMES_MANIFEST)
    pair_rows = read_jsonl(PAIRS_MANIFEST)

    target_keys = build_target_set(pair_rows)
    print("frame rows      :", len(frame_rows))
    print("pair rows       :", len(pair_rows))
    print("unique targets  :", len(target_keys))
    print()

    frame_map = {}
    for r in frame_rows:
        k = parse_frame_key_from_row(r)
        frame_map[k] = r

    missing = sorted(list(target_keys - set(frame_map.keys())))
    if missing:
        print("WARNING: some targets not found in frame manifest:", len(missing))
        for x in missing[:10]:
            print("  ", x)

    faces = load_smpl_faces()
    print("SMPL faces:", tuple(faces.shape))
    glctx = dr.RasterizeCudaContext() if DEVICE == "cuda" else dr.RasterizeGLContext()

    manifest_out = OUT_ROOT / "final_targets_manifest.jsonl"
    summary_out  = OUT_ROOT / "final_targets_summary.json"

    count_done = 0
    count_skip = 0
    stats = Counter()

    with open(manifest_out, "w", encoding="utf-8") as fw:
        for i, key in enumerate(sorted(target_keys), start=1):
            if key not in frame_map:
                stats["missing_in_frame_manifest"] += 1
                continue

            row = frame_map[key]
            pair, action, cam, stem = key
            seq_root = make_output_dir(OUT_ROOT, key)

            rgb_out   = seq_root / "rgb" / f"{stem}.jpg"
            dwA_out   = seq_root / "dwpose" / "A" / f"{stem}.npz"
            dwB_out   = seq_root / "dwpose" / "B" / f"{stem}.npz"
            geom_out  = seq_root / "geometry" / f"{stem}.npz"

            depth_vis_out    = seq_root / "vis" / "depth" / f"{stem}.png"
            normal_vis_out   = seq_root / "vis" / "normal" / f"{stem}.png"
            semantic_vis_out = seq_root / "vis" / "semantic" / f"{stem}.png"
            contact_vis_out  = seq_root / "vis" / "contact" / f"{stem}.png"
            occ_vis_out      = seq_root / "vis" / "directional_occlusion" / f"{stem}.png"
            panel_out        = seq_root / "vis" / "qc_panel" / f"{stem}.jpg"

            if (
                geom_out.exists()
                and depth_vis_out.exists()
                and normal_vis_out.exists()
                and semantic_vis_out.exists()
                and contact_vis_out.exists()
                and occ_vis_out.exists()
                and panel_out.exists()
            ):
                count_skip += 1
                stats["skipped_existing"] += 1
                print(f"[{i}/{len(target_keys)}] SKIP {key_to_str(key)}")
                continue

            print(f"[{i}/{len(target_keys)}] RUN  {key_to_str(key)}")

            out = render_geometry_for_frame(row, faces, glctx)

            ensure_dir(geom_out.parent)
            np.savez_compressed(
                geom_out,
                depth_scene=out["depth_scene"],
                normal_scene=out["normal_scene"],
                semantic_A=out["semantic_A"],
                semantic_B=out["semantic_B"],
                contact_A=out["contact_A"],
                contact_B=out["contact_B"],
                front_owner=out["front_owner"],
                interperson_overlap=out["interperson_overlap"],
                A_occluded_by_B=out["A_occluded_by_B"],
                B_occluded_by_A=out["B_occluded_by_A"],
            )

            # link rgb / dwpose
            safe_symlink(Path(out["rgb_path"]), rgb_out)
            safe_symlink(Path(out["dwpose_A_path"]), dwA_out)
            safe_symlink(Path(out["dwpose_B_path"]), dwB_out)

            # vis
            ensure_dir(depth_vis_out.parent)
            ensure_dir(normal_vis_out.parent)
            ensure_dir(semantic_vis_out.parent)
            ensure_dir(contact_vis_out.parent)
            ensure_dir(occ_vis_out.parent)
            ensure_dir(panel_out.parent)

            valid = out["front_owner"] > 0
            depth_img = colorize_depth(out["depth_scene"], valid)
            normal_img = colorize_normal(out["normal_scene"], valid)
            semantic_img = colorize_semantic(out["semantic_A"], out["semantic_B"])
            contact_img = colorize_contact(out["contact_A"], out["contact_B"])
            occ_img = colorize_directional_occlusion(out["A_occluded_by_B"], out["B_occluded_by_A"])
            dw_img = draw_dwpose_vis(out["dwpose_A_path"], out["dwpose_B_path"], out["image_height"], out["image_width"])

            Image.fromarray(depth_img).save(depth_vis_out)
            Image.fromarray(normal_img).save(normal_vis_out)
            Image.fromarray(semantic_img).save(semantic_vis_out)
            Image.fromarray(contact_img).save(contact_vis_out)
            Image.fromarray(occ_img).save(occ_vis_out)
            save_qc_panel(
                out["rgb_path"],
                dw_img,
                depth_img,
                normal_img,
                semantic_img,
                contact_img,
                occ_img,
                panel_out,
            )

            meta_row = {
                "pair": pair,
                "action": action,
                "camera_id": int(cam),
                "frame_stem": stem,
                "image_width": int(out["image_width"]),
                "image_height": int(out["image_height"]),
                "rgb": str(rgb_out),
                "dwpose_A": str(dwA_out),
                "dwpose_B": str(dwB_out),
                "geometry_npz": str(geom_out),
                "depth_vis": str(depth_vis_out),
                "normal_vis": str(normal_vis_out),
                "semantic_vis": str(semantic_vis_out),
                "contact_vis": str(contact_vis_out),
                "directional_occlusion_vis": str(occ_vis_out),
                "qc_panel": str(panel_out),
                "native_resolution": True,
                "crop_applied": False,
                "resize_applied": False,
            }
            fw.write(json.dumps(meta_row, ensure_ascii=False) + "\n")

            count_done += 1
            stats["generated"] += 1

    summary = {
        "device": DEVICE,
        "frames_manifest": str(FRAMES_MANIFEST),
        "pairs_manifest": str(PAIRS_MANIFEST),
        "out_root": str(OUT_ROOT),
        "unique_target_frames": len(target_keys),
        "generated": count_done,
        "skipped_existing": count_skip,
        "missing_in_frame_manifest": int(stats["missing_in_frame_manifest"]),
        "notes": [
            "native resolution only",
            "no crop",
            "no resize",
            "contact is projected from official Hi4D contact vertex metadata",
            "directional occlusion is derived from semantic overlap + front owner",
        ],
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

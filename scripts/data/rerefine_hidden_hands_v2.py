#!/usr/bin/env python3
"""
Hidden-hand & truncated-limb aware re-refinement (v2) for HIA Sapiens2 geometry.

核心设计：
  1. 可见性以 DWPose + 画面边界 + 双手碰撞检测联合判定：
     - visible: DWPose 手部可用且与 Sapiens2 手心一致 -> 完整优化手臂 + 20 指节。
     - fist_front: 手腕在胸前/上方举起（DWPose 与 Sapiens2 肘/腕均高置信一致，如握拳/运动模糊），
       且非双手碰撞中的伪手 -> 优化肩/肘/腕 2D 位置，冻结手指姿态为 SMPLest-X 握拳初值，dwpose 不画 21 指节。
     - cropped: 手臂伸出画面左/右/下边界被裁切，或半身/特写镜头手在底边外 ->
       丢弃该侧肘/腕/手关键点，整条手臂冻结为 SMPLest-X 初值（若特写镜头初值误把不可见手举在胸前贴纸处，
       则自动归位为自然下垂姿态），绝不往胸前或背后乱拉。
     - behind: 全身/七分身镜头下手在画面内但被躯干遮挡（背手） ->
       丢弃腕/手关键点，通过下躯干掩膜（骨盆+髋+下腰 [0,1,2,3]）与 3D 运动学约束
       （肘低于肩 >=17cm、腕低于肘 >=8cm、肘与腕深度位于骨盆/下腰之后），
       让上臂自然下垂、前臂从腰后收入躯干剪影内，正面法线/深度图零假手、零穿模残肢。
  2. 腿部/脚部截断保护（truncated_leg）：
     - 联合 DWPose 与 Sapiens2 检查髋、膝、踝是否在画面内且满足 y_hip < y_knee < y_ankle。
       一旦膝或踝被底边裁切或低置信度，丢弃下肢截断点并冻结腿部关节。

默认输出到 staging 目录；加 --commit 才直接更新 refined_v1 的 frames/ 与 human-check/。
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
from scipy.ndimage import distance_transform_edt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rerefine_occlusion_clips as v1
from rerefine_occlusion_clips import (
    GOLIATH_TO_SMPLX, RHAND_GOLIATH, LHAND_GOLIATH, SMPLX_137_MAPPING,
    L_ARM_SLICES, R_ARM_SLICES,
    project_joints_2d, render_mesh_nvdiffrast, build_comparison_strip,
)
import smplx
import nvdiffrast.torch as dr
from src.pose_edit.geometry import smplx137_to_dwpose134, render_poseguide
from sapiens.pose.models import init_model
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from transformers import DetrForObjectDetection, DetrImageProcessor
from sapiens.pose.evaluators import nms

DW_LHAND = slice(92, 113)
DW_RHAND = slice(113, 134)

# SMPL-X 55-joint ids
L_HAND_JOINTS = [20] + list(range(25, 40))
R_HAND_JOINTS = [21] + list(range(40, 55))
L_FOREARM_JOINTS = [18]
R_FOREARM_JOINTS = [19]
# 只取下躯干（骨盆 0、左髋 1、右髋 2、下腰 spine1=3）作为背手收纳目标，防止手被拉向后胸导致架肘
LOWER_TORSO_JOINTS = [0, 1, 2, 3]
FULL_TORSO_JOINTS = [0, 1, 2, 3, 6, 9]

L_LEG_SLICES = [(0, 3), (9, 12), (18, 21), (27, 30)]
R_LEG_SLICES = [(3, 6), (12, 15), (21, 24), (30, 33)]

STEPS_DEFAULT = 35
STEPS_HIDDEN = 120
TORSO_REFRESH = 15


def classify_both_hands(dw: dict, kp_sap: np.ndarray, sc_sap: np.ndarray,
                        init_j2d_np: np.ndarray, init_j3d_np: np.ndarray,
                        W: int, H: int) -> tuple[str, dict, str, dict]:
    """
    Jointly classify left and right hands into:
      - 'visible':     21-pt hand visible & consistent between DWPose and Sapiens2
      - 'fist_front':  hand is raised in front of upper body (fist/blurred), optimize arm+wrist, freeze fingers
      - 'cropped':     arm extends out of frame border (or hangs below close-up frame), freeze arm to init
      - 'closeup_down': close-up shot where neither hand is truly visible in DWPose and init had phantom chest hands
      - 'behind':      hand is tucked behind lower back / waist
    """
    conf = np.asarray(dw.get("confidence", []), dtype=float)
    cand = np.asarray(dw.get("candidate", []), dtype=float)
    scale = float(max(W, H))

    def pt_in_frame(pt, mx=0.04, my=0.04):
        return (mx * W <= float(pt[0]) <= (1.0 - mx) * W) and (my * H <= float(pt[1]) <= (1.0 - my) * H)

    # Check if hips/pelvis are visible inside the frame (full/3-quarter body vs chest/head close-up)
    pelvis_below_frame = float(init_j2d_np[0, 1]) > 0.88 * H
    lhip_ok = len(conf) > 11 and conf[11] > 0.45 and pt_in_frame(cand[11, :2], 0.04, 0.08)
    rhip_ok = len(conf) > 8 and conf[8] > 0.45 and pt_in_frame(cand[8, :2], 0.04, 0.08)
    is_closeup = pelvis_below_frame or not (lhip_ok or rhip_ok)

    # Detect Sapiens2 two-hand collision (both hands predicted on the same single visible hand)
    sap_l_wr = kp_sap[62, :2]
    sap_r_wr = kp_sap[41, :2]
    sap_l_center = kp_sap[list(LHAND_GOLIATH), :2].mean(axis=0)
    sap_r_center = kp_sap[list(RHAND_GOLIATH), :2].mean(axis=0)
    hands_collided = (
        np.linalg.norm(sap_l_wr - sap_r_wr) < 0.09 * scale
        or np.linalg.norm(sap_l_center - sap_r_center) < 0.09 * scale
    )
    # If collided, which side is closer to its SMPLest-X init wrist + DWPose wrist?
    collision_loser = None
    if hands_collided:
        l_init_d = np.linalg.norm(sap_l_wr - init_j2d_np[20])
        r_init_d = np.linalg.norm(sap_r_wr - init_j2d_np[21])
        l_dw_wr_c = float(conf[7]) if len(conf) > 7 else 0.0
        r_dw_wr_c = float(conf[4]) if len(conf) > 4 else 0.0
        # Score each side: lower distance to init + higher DWPose wrist confidence wins
        l_score = l_dw_wr_c - (l_init_d / scale) * 2.0
        r_score = r_dw_wr_c - (r_init_d / scale) * 2.0
        collision_loser = "right" if l_score >= r_score else "left"

    out = {}
    for side in ("left", "right"):
        blk_slice = DW_LHAND if side == "left" else DW_RHAND
        key = "hand_b" if side == "left" else "hand_a"
        sh_idx, el_idx, wr_idx = (5, 6, 7) if side == "left" else (2, 3, 4)
        sap_hand_ids = list(LHAND_GOLIATH) if side == "left" else list(RHAND_GOLIATH)
        sap_wr_id = 62 if side == "left" else 41
        smplx_sh_id, smplx_el_id, smplx_wr_id = (16, 18, 20) if side == "left" else (17, 19, 21)

        blk_conf = conf[blk_slice] if len(conf) >= 134 else np.zeros(21)
        blk_xy = cand[blk_slice, :2] if len(cand) >= 134 else np.zeros((21, 2))
        usable = bool(dw.get("hand_raw_pose_usable", {}).get(key, False))
        n_valid = int((blk_conf > 0.3).sum())
        mean_c = float(blk_conf.mean()) if blk_conf.size else 0.0

        wr_conf = float(conf[wr_idx]) if len(conf) > wr_idx else 0.0
        el_conf = float(conf[el_idx]) if len(conf) > el_idx else 0.0
        sh_conf = float(conf[sh_idx]) if len(conf) > sh_idx else 0.0
        wr_xy = cand[wr_idx, :2] if len(cand) > wr_idx else np.array([-1.0, -1.0])
        el_xy = cand[el_idx, :2] if len(cand) > el_idx else np.array([-1.0, -1.0])
        sh_xy = cand[sh_idx, :2] if len(cand) > sh_idx else np.array([-1.0, -1.0])

        sap_pts = kp_sap[sap_hand_ids, :2]
        sap_center = sap_pts.mean(axis=0)
        dw_valid_mask = blk_conf > 0.25
        dw_center = blk_xy[dw_valid_mask].mean(axis=0) if dw_valid_mask.any() else wr_xy
        center_dist = float(np.linalg.norm(sap_center - dw_center)) / scale

        init_wr = init_j2d_np[smplx_wr_id]
        init_el = init_j2d_np[smplx_el_id]
        init_sh = init_j2d_np[smplx_sh_id]
        sap_wr = kp_sap[sap_wr_id, :2]
        is_loser = (collision_loser == side)

        # 1. Clearly visible 21-keypoint hand
        dw_visible = (
            not is_loser
            and (usable or (n_valid >= 8 and mean_c >= 0.40))
            and wr_conf >= 0.30
            and pt_in_frame(wr_xy, 0.02, 0.02)
            and center_dist < 0.14
            and float(sc_sap[sap_hand_ids].max()) > 0.35
        )
        if dw_visible:
            out[side] = ("visible", {
                "state": "visible", "usable": usable, "n_conf_gt_0.3": n_valid,
                "mean_conf": round(mean_c, 3), "sap_dw_dist_norm": round(center_dist, 3),
            })
            continue

        # 2. Raised front fist / blurred hand in front of upper body (e.g. 000078_000088 L, 000155_000080 L)
        #    Requires: not collision loser, high Sapiens2 wrist score, wrist genuinely raised to chest/shoulder/head
        #    level (v_wrist < v_shoulder + 0.14*H), and Sapiens2 wrist agrees with SMPLest-X init raised wrist.
        sap_dw_wr_dist = float(np.linalg.norm(sap_wr - wr_xy)) / scale
        sap_init_wr_dist = float(np.linalg.norm(sap_wr - init_wr)) / scale
        wrist_raised = (
            float(sap_wr[1]) < float(init_sh[1]) + 0.14 * H
            and float(init_wr[1]) < float(init_sh[1]) + 0.16 * H
        )
        if (
            not is_loser
            and float(sc_sap[sap_wr_id]) > 0.50
            and wrist_raised
            and sap_init_wr_dist < 0.14
            and (wr_conf >= 0.30 or el_conf >= 0.35)
            and pt_in_frame(sap_wr, 0.04, 0.04)
        ):
            out[side] = ("fist_front", {
                "state": "fist_front", "usable": usable, "n_conf_gt_0.3": n_valid,
                "mean_conf": round(mean_c, 3), "sap_dw_wr_dist": round(sap_dw_wr_dist, 3),
            })
            continue

        # 3. Hand is NOT visible. Distinguish close-up / border-cropped vs behind-the-back
        if is_closeup:
            # In a chest/head close-up shot (no hips in frame), any unseen hand is below the frame:
            # always place the unseen arm in a natural downward resting posture so 0 phantom hand pixels appear.
            state = "closeup_down"
            out[side] = (state, {
                "state": state, "usable": usable, "n_conf_gt_0.3": n_valid,
                "mean_conf": round(mean_c, 3), "is_closeup": True,
            })
            continue

        # Full / 3-quarter body shot: check if arm stretches outward past image border
        extrap_wr = el_xy + 0.85 * (el_xy - sh_xy) if (el_conf > 0.35 and sh_conf > 0.35) else init_wr
        near_or_out_border = (
            not pt_in_frame(extrap_wr, 0.06, 0.05)
            or not pt_in_frame(init_wr, 0.05, 0.04)
            or (wr_conf > 0.20 and not pt_in_frame(wr_xy, 0.06, 0.05))
            or (not is_loser and not pt_in_frame(sap_wr, 0.05, 0.04))
        )
        state = "cropped" if near_or_out_border else "behind"
        out[side] = (state, {
            "state": state, "usable": usable, "n_conf_gt_0.3": n_valid,
            "mean_conf": round(mean_c, 3), "is_closeup": False, "collision_loser": is_loser,
        })

    return out["left"][0], out["left"][1], out["right"][0], out["right"][1]


def check_leg_truncated(dw: dict, kp_sap: np.ndarray, sc_sap: np.ndarray, W: int, H: int, side: str) -> bool:
    """Return True if the leg (hip -> knee -> ankle) is truncated by frame border or unreliable."""
    conf = np.asarray(dw.get("confidence", []), dtype=float)
    cand = np.asarray(dw.get("candidate", []), dtype=float)
    hip_i, knee_i, ank_i = (11, 12, 13) if side == "left" else (8, 9, 10)
    g_hip, g_knee, g_ank = (9, 11, 13) if side == "left" else (10, 12, 14)

    def in_box(pt, mx=0.03, my=0.04):
        return (mx * W <= float(pt[0]) <= (1.0 - mx) * W) and (my * H <= float(pt[1]) <= (1.0 - my) * H)

    if len(conf) <= ank_i:
        return True
    if conf[hip_i] < 0.35 or conf[knee_i] < 0.35 or conf[ank_i] < 0.35:
        return True
    if not (in_box(cand[hip_i, :2]) and in_box(cand[knee_i, :2]) and in_box(cand[ank_i, :2])):
        return True
    if not (cand[knee_i, 1] > cand[hip_i, 1] + 0.04 * H and cand[ank_i, 1] > cand[knee_i, 1] + 0.04 * H):
        return True
    if not (sc_sap[g_knee] > 0.35 and sc_sap[g_ank] > 0.35 and in_box(kp_sap[g_knee, :2]) and in_box(kp_sap[g_ank, :2])):
        return True
    if not (kp_sap[g_knee, 1] > kp_sap[g_hip, 1] + 0.04 * H and kp_sap[g_ank, 1] > kp_sap[g_knee, 1] + 0.04 * H):
        return True
    return False


def part_vertex_mask(lbs_weights: torch.Tensor, joint_ids) -> torch.Tensor:
    dom = lbs_weights.argmax(dim=1)
    m = torch.zeros_like(dom, dtype=torch.bool)
    for j in joint_ids:
        m |= dom == j
    return m


def rasterize(verts, faces_cuda, fx, fy, cx, cy, W, H, glctx):
    """Return (tri_id[H,W] int, depth[H,W]) in OpenCV top-down raster; tri_id=-1 for background."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    near = max(1e-4, zmin * 0.5)
    far = max(near + 1.0, zmax * 1.5)
    x_clip = (2.0 * fx / W) * x + (2.0 * cx / W - 1.0) * z
    y_clip = (-2.0 * fy / H) * y + (1.0 - 2.0 * cy / H) * z
    A = (far + near) / (far - near)
    B = -2.0 * far * near / (far - near)
    clip = torch.stack([x_clip, y_clip, A * z + B, z], dim=-1).unsqueeze(0).contiguous()
    rast, _ = dr.rasterize(glctx, clip, faces_cuda, resolution=[H, W])
    depth, _ = dr.interpolate(z.view(1, -1, 1).contiguous(), rast, faces_cuda)
    tri = rast[0, :, :, 3].long() - 1
    tri = torch.flip(tri, dims=[0])
    depth = torch.flip(depth[0, :, :, 0], dims=[0])
    return tri, depth


class TorsoTarget:
    """
    Lower-torso target for pulling hidden hands into the lower back/hips silhouette,
    plus full-torso depth map so neither hand nor forearm ever pierces through the front of the body.
    """

    def __init__(self, verts, lower_torso_faces, full_torso_faces, fx, fy, cx, cy, W, H, glctx, device):
        tri_low, _ = rasterize(verts, lower_torso_faces, fx, fy, cx, cy, W, H, glctx)
        tri_full, depth_full = rasterize(verts, full_torso_faces, fx, fy, cx, cy, W, H, glctx)
        mask = (tri_low >= 0).cpu().numpy().astype(np.uint8)
        if mask.sum() < 50:
            mask = (tri_full >= 0).cpu().numpy().astype(np.uint8)
        erode_px = max(6, int(round(0.022 * W)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask_e = cv2.erode(mask, k)
        if mask_e.sum() < 50:
            mask_e = mask
        _, idx = distance_transform_edt(mask_e == 0, return_indices=True)
        self.near_v = torch.from_numpy(idx[0]).to(device)
        self.near_u = torch.from_numpy(idx[1]).to(device)

        # For full torso depth lookup anywhere on the body
        mask_full = (tri_full >= 0).cpu().numpy().astype(np.uint8)
        _, idx_f = distance_transform_edt(mask_full == 0, return_indices=True)
        self.full_near_v = torch.from_numpy(idx_f[0]).to(device)
        self.full_near_u = torch.from_numpy(idx_f[1]).to(device)
        self.depth = depth_full.detach()
        self.W, self.H = W, H
        self.ok = bool(mask_full.sum() > 0)

    def loss(self, uv_hand: torch.Tensor, z_hand: torch.Tensor,
             uv_forearm: torch.Tensor, z_forearm: torch.Tensor, margin_m: float = 0.05):
        uh = uv_hand[:, 0].detach().round().long().clamp(0, self.W - 1)
        vh = uv_hand[:, 1].detach().round().long().clamp(0, self.H - 1)
        tuh = self.near_u[vh, uh].float()
        tvh = self.near_v[vh, uh].float()
        l2d = torch.sqrt((uv_hand[:, 0] - tuh) ** 2 + (uv_hand[:, 1] - tvh) ** 2 + 1e-6).mean()
        d_th = self.depth[tvh.long(), tuh.long()]
        lz_hand = torch.relu(d_th + margin_m - z_hand).mean()

        uf = uv_forearm[:, 0].detach().round().long().clamp(0, self.W - 1)
        vf = uv_forearm[:, 1].detach().round().long().clamp(0, self.H - 1)
        tuf = self.full_near_u[vf, uf].float()
        tvf = self.full_near_v[vf, uf].float()
        d_tf = self.depth[tvf.long(), tuf.long()]
        lz_forearm = torch.relu(d_tf + margin_m - z_forearm).mean()

        return l2d, lz_hand + 0.8 * lz_forearm


def render_dwpose_masked(joints, fx, fy, cx, cy, W, H, hide_l_fingers, hide_r_fingers, hide_l_arm=False, hide_r_arm=False):
    j137 = joints[SMPLX_137_MAPPING]
    j2d = project_joints_2d(j137, fx, fy, cx, cy).detach().cpu().numpy().astype(float)
    dw134 = smplx137_to_dwpose134(j2d)
    if hide_l_fingers:
        dw134[DW_LHAND] = np.nan
    if hide_r_fingers:
        dw134[DW_RHAND] = np.nan
    if hide_l_arm:
        dw134[[6, 7]] = np.nan
    if hide_r_arm:
        dw134[[3, 4]] = np.nan
    return render_poseguide(dw134, (W, H))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", default=["all"])
    ap.add_argument("--frames", nargs="*", default=None, help="optional explicit frame ids")
    ap.add_argument("--resume", action="store_true", help="skip frames already refined with v2")
    ap.add_argument("--max-frames", type=int, default=None, help="optional limit on total frames to run")
    ap.add_argument("--hia-dir", default="/data/shangguanrz/pic-edit-datasets/HIA_processed_v2_full_20261027")
    ap.add_argument("--geom-dir", default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_v1_20261029")
    ap.add_argument("--refined-dir", default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_refined_v1")
    ap.add_argument("--staging-dir", default="/data/shangguanrz/pic-edit-datasets/HIA_geometry_sapiens2_hidden_hand_v2_staging")
    ap.add_argument("--commit", action="store_true", help="overwrite refined_v1 frames/ and human-check/")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    hia_dir, geom_dir = Path(args.hia_dir), Path(args.geom_dir)
    refined_root = Path(args.refined_dir)
    out_root = refined_root if args.commit else Path(args.staging_dir)
    frames_out = out_root / "frames"
    check_out = out_root / "human-check"
    check_out.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {out_root} ({'COMMIT' if args.commit else 'staging'})", flush=True)
    det_proc = DetrImageProcessor.from_pretrained(v1.DETR_PATH)
    det_model = DetrForObjectDetection.from_pretrained(v1.DETR_PATH).eval().to(device)
    sap = init_model(v1.CONFIG_PATH, v1.CKPT_PATH, device=device)
    sap.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{v1.SAPIENS_POSE_DIR}/configs/_base_/keypoints308.py"))
    sap.cfg.codec.pop("type")
    sap.codec = UDPHeatmap(**sap.cfg.codec)

    smpl_x = smplx.create(v1.SMPLX_DIR, model_type="smplx", gender="neutral", use_face_contour=True,
                          num_betas=10, num_expression_coeffs=10, ext="npz", use_pca=False,
                          flat_hand_mean=True).to(device).eval()
    smplx_faces = np.load(v1.SMPLX_FACES_NPZ)["f"]
    faces_cuda = torch.from_numpy(smplx_faces.astype(np.int32)).to(device)
    faces_long = faces_cuda.long()
    glctx = dr.RasterizeCudaContext()

    lbs_w = smpl_x.lbs_weights
    lhand_v = part_vertex_mask(lbs_w, L_HAND_JOINTS)
    rhand_v = part_vertex_mask(lbs_w, R_HAND_JOINTS)
    lforearm_v = part_vertex_mask(lbs_w, L_FOREARM_JOINTS)
    rforearm_v = part_vertex_mask(lbs_w, R_FOREARM_JOINTS)
    lower_torso_v = part_vertex_mask(lbs_w, LOWER_TORSO_JOINTS)
    full_torso_v = part_vertex_mask(lbs_w, FULL_TORSO_JOINTS)
    lower_torso_faces = faces_cuda[lower_torso_v[faces_long].all(dim=1)].contiguous()
    full_torso_faces = faces_cuda[full_torso_v[faces_long].all(dim=1)].contiguous()
    lhand_face = lhand_v[faces_long].any(dim=1)
    rhand_face = rhand_v[faces_long].any(dim=1)
    lhand_idx = torch.nonzero(lhand_v).squeeze(1)[::2]
    rhand_idx = torch.nonzero(rhand_v).squeeze(1)[::2]
    lforearm_idx = torch.nonzero(lforearm_v).squeeze(1)[::2]
    rforearm_idx = torch.nonzero(rforearm_v).squeeze(1)[::2]

    tasks = []
    frames_dir = refined_root / "frames"
    if "all" in args.clips:
        with os.scandir(frames_dir) as it:
            clips_to_run = sorted([entry.name for entry in it if entry.is_dir()])
    else:
        clips_to_run = args.clips

    for cid in clips_to_run:
        cdir = frames_dir / cid
        if not cdir.exists():
            continue
        try:
            with os.scandir(cdir) as it:
                frame_dirs = sorted([e.name for e in it if e.is_dir()])
        except Exception:
            continue
        for fname in frame_dirs:
            if not args.frames or fname in args.frames or any(f in fname for f in args.frames):
                if args.resume:
                    rec_p = cdir / fname / "record.json"
                    if rec_p.exists():
                        try:
                            rj = json.load(open(rec_p))
                            if rj.get("method") == "sapiens2_refine_hidden_hand_v2":
                                continue
                        except Exception:
                            pass
                tasks.append((cid, fname))
                if args.max_frames and len(tasks) >= args.max_frames:
                    break
        if args.max_frames and len(tasks) >= args.max_frames:
            break
    print(f"{len(tasks)} frames queued across {len(clips_to_run)} clips", flush=True)

    summary = []
    t_all = time.time()
    for i, (cid, fid) in enumerate(tasks, 1):
        t0 = time.time()
        try:
            rgb_path = hia_dir / "dense_pool" / "images" / cid / f"{fid}.png"
            dw_path = hia_dir / "dense_pool" / "poses" / cid / f"{fid}.json"
            v1_rec_path = geom_dir / "frames" / cid / fid / "record.json"
            if not v1_rec_path.exists() or not rgb_path.exists() or not dw_path.exists():
                continue
            v1_rec = json.load(open(v1_rec_path))
            d_init = torch.load(v1_rec["artifacts"]["smplx_params"]["path"], map_location=device)
            rgb_bgr = cv2.imread(str(rgb_path))
            if rgb_bgr is None:
                continue
            H, W = rgb_bgr.shape[:2]
            fx, fy = d_init["focal_length_xy"][0, 0].item(), d_init["focal_length_xy"][0, 1].item()
            cx, cy = d_init["princpt"][0, 0].item(), d_init["princpt"][0, 1].item()
            dw = json.load(open(dw_path))
            dw_conf = np.asarray(dw.get("confidence", []), dtype=float)
        except Exception as err:
            print(f"[{i}/{len(tasks)}] SKIP {cid}/{fid}: {err}", flush=True)
            continue

        P = {k: d_init[k].clone().to(device) for k in ["body_pose", "left_hand_pose", "right_hand_pose", "global_orient",
                                                       "transl", "betas", "expression", "jaw_pose"]}

        def run(body, lh, rh, tr):
            return smpl_x(global_orient=P["global_orient"], body_pose=body, left_hand_pose=lh, right_hand_pose=rh,
                          jaw_pose=P["jaw_pose"], betas=P["betas"], expression=P["expression"], transl=tr)

        with torch.no_grad():
            init_out = run(P["body_pose"], P["left_hand_pose"], P["right_hand_pose"], P["transl"])
            init_j3d_np = init_out.joints[0].detach().cpu().numpy()
            init_j2d_np = project_joints_2d(init_out.joints[0], fx, fy, cx, cy).detach().cpu().numpy()

        # --- Sapiens2 keypoints ---
        img_pil = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        inputs = det_proc(images=img_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            res = det_proc.post_process_object_detection(det_model(**inputs), target_sizes=torch.tensor([[H, W]], device=device), threshold=0.3)[0]
        pm = res["labels"] == 1
        boxes = res["boxes"][pm].cpu().numpy()
        scs = res["scores"][pm].cpu().numpy().reshape(-1, 1)
        if len(boxes):
            bb = np.concatenate([boxes, scs], 1)
            bb = bb[nms(bb, 0.3), :4]
            bbox = bb[np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]))]
        else:
            bbox = np.array([0, 0, W - 1, H - 1], np.float32)
        data = sap.pipeline(dict(img=rgb_bgr, bbox=bbox[None], bbox_score=np.ones(1, np.float32)))
        data = sap.data_preprocessor(data)
        with torch.no_grad():
            pred = sap(data["inputs"]).cpu().numpy()
        kp, sc = sap.codec.decode(pred[0])
        meta = data["data_samples"]["meta"]
        kp = (kp / meta["input_size"] * meta["bbox_scale"] + meta["bbox_center"] - 0.5 * meta["bbox_scale"])[0]
        s_kpts = torch.tensor(kp, dtype=torch.float32, device=device)
        s_scores = sc[0]

        # Classify left/right hands & legs
        l_state, l_info, r_state, r_info = classify_both_hands(dw, kp, s_scores, init_j2d_np, init_j3d_np, W, H)

        # If close-up shot had phantom chest hands in SMPLest-X init (e.g. 000190 cat sticker),
        # reset that arm's init_body_pose to a natural downward resting posture at the side
        with torch.no_grad():
            if l_state == "closeup_down":
                for s0, s1 in L_ARM_SLICES:
                    P["body_pose"][0, s0:s1] = 0.0
                P["body_pose"][0, 15 * 3: 16 * 3] = torch.tensor([0.0, 0.0, -1.25], device=device)
            if r_state == "closeup_down":
                for s0, s1 in R_ARM_SLICES:
                    P["body_pose"][0, s0:s1] = 0.0
                P["body_pose"][0, 16 * 3: 17 * 3] = torch.tensor([0.0, 0.0, 1.25], device=device)

        hide_l_fingers = (l_state != "visible")
        hide_r_fingers = (r_state != "visible")
        hide_l_wrist = l_state in ("behind", "cropped", "closeup_down")
        hide_r_wrist = r_state in ("behind", "cropped", "closeup_down")
        behind_l = (l_state == "behind")
        behind_r = (r_state == "behind")
        freeze_l_arm = l_state in ("cropped", "closeup_down")
        freeze_r_arm = r_state in ("cropped", "closeup_down")

        trunc_l_leg = check_leg_truncated(dw, kp, s_scores, W, H, "left")
        trunc_r_leg = check_leg_truncated(dw, kp, s_scores, W, H, "right")
        lhip_vis = len(dw_conf) > 11 and dw_conf[11] > 0.4
        rhip_vis = len(dw_conf) > 8 and dw_conf[8] > 0.4

        L_LEG_G = {11, 13, 15, 16, 17} | ({9} if not lhip_vis else set())
        R_LEG_G = {12, 14, 18, 19, 20} | ({10} if not rhip_vis else set())

        frozen_slices = []
        if trunc_l_leg:
            frozen_slices += L_LEG_SLICES
        if trunc_r_leg:
            frozen_slices += R_LEG_SLICES
        if freeze_l_arm:
            frozen_slices += L_ARM_SLICES
        if freeze_r_arm:
            frozen_slices += R_ARM_SLICES

        def in_frame(g, frac=0.02):
            u, v = float(kp[g, 0]), float(kp[g, 1])
            m = frac * max(W, H)
            return m <= u < W - m and m <= v < H - m

        g_idx, sm_idx, wts = [], [], []
        n_out_of_frame = 0
        for g, s in GOLIATH_TO_SMPLX.items():
            if hide_l_fingers and g in LHAND_GOLIATH:
                continue
            if hide_r_fingers and g in RHAND_GOLIATH:
                continue
            if hide_l_wrist and g == 62:
                continue
            if hide_r_wrist and g == 41:
                continue
            # When arm is frozen (cropped / closeup_down) or behind back, do not let Sapiens2 elbow pull it forward
            if g == 7 and (freeze_l_arm or behind_l):
                continue
            if g == 8 and (freeze_r_arm or behind_r):
                continue
            if (trunc_l_leg and g in L_LEG_G) or (trunc_r_leg and g in R_LEG_G):
                continue
            if not in_frame(g):
                n_out_of_frame += 1
                continue
            if s_scores[g] > 0.3:
                g_idx.append(g)
                sm_idx.append(s)
                if g in LHAND_GOLIATH or g in RHAND_GOLIATH or g in (62, 41):
                    w = float(s_scores[g]) * 2.5
                else:
                    w = float(s_scores[g]) * 1.0
                wts.append(w)

        target = s_kpts[g_idx]
        w_t = torch.tensor(wts, device=device)
        sm_t = torch.tensor(sm_idx, dtype=torch.long, device=device)

        def mpjpe(out):
            j2d = project_joints_2d(out.joints[0], fx, fy, cx, cy)[sm_t]
            return float(((j2d - target).norm(dim=-1) * w_t).sum() / w_t.sum())

        with torch.no_grad():
            init_mpjpe = mpjpe(init_out)

        body = torch.nn.Parameter(P["body_pose"].clone())
        tr = torch.nn.Parameter(P["transl"].clone())
        groups = [{"params": [body], "lr": 0.02 if (behind_l or behind_r) else 0.015}, {"params": [tr], "lr": 0.005}]
        lh = P["left_hand_pose"] if hide_l_fingers else torch.nn.Parameter(P["left_hand_pose"].clone())
        rh = P["right_hand_pose"] if hide_r_fingers else torch.nn.Parameter(P["right_hand_pose"].clone())
        if not hide_l_fingers:
            groups.append({"params": [lh], "lr": 0.025})
        if not hide_r_fingers:
            groups.append({"params": [rh], "lr": 0.025})
        opt = torch.optim.Adam(groups)

        steps = STEPS_HIDDEN if (behind_l or behind_r) else STEPS_DEFAULT
        torso = None
        for step in range(steps):
            opt.zero_grad()
            out = run(body, lh, rh, tr)
            verts = out.vertices[0]
            j3d = out.joints[0]
            j2d = project_joints_2d(j3d, fx, fy, cx, cy)[sm_t]
            reproj = ((j2d - target).pow(2).sum(-1).add(1e-6).sqrt() * w_t).sum() / w_t.sum()
            loss = reproj + 0.015 * (body - P["body_pose"]).pow(2).mean() + 0.1 * (tr - P["transl"]).pow(2).mean()
            if not hide_l_fingers:
                loss = loss + 0.005 * (lh - P["left_hand_pose"]).pow(2).mean()
            if not hide_r_fingers:
                loss = loss + 0.005 * (rh - P["right_hand_pose"]).pow(2).mean()

            if behind_l or behind_r:
                if torso is None or step % TORSO_REFRESH == 0:
                    with torch.no_grad():
                        torso = TorsoTarget(verts.detach(), lower_torso_faces, full_torso_faces, fx, fy, cx, cy, W, H, glctx, device)
                if torso.ok:
                    z_pelvis = j3d[0, 2].detach()
                    x_pelvis = j3d[0, 0].detach()
                    for is_behind, h_idx, f_idx, sh_j, el_j, wr_j in (
                        (behind_l, lhand_idx, lforearm_idx, 16, 18, 20),
                        (behind_r, rhand_idx, rforearm_idx, 17, 19, 21),
                    ):
                        if is_behind:
                            hv = verts[h_idx]
                            fv = verts[f_idx]
                            l2d, lz = torso.loss(
                                project_joints_2d(hv, fx, fy, cx, cy), hv[:, 2],
                                project_joints_2d(fv, fx, fy, cx, cy), fv[:, 2],
                            )
                            # 3D kinematic chain:
                            #   1) Upper arm hangs downward: y_el >= y_sh + 0.18m (note: +y in j3d is downward)
                            #   2) Forearm goes downward/inward behind lower waist: y_wr >= y_el + 0.08m
                            #   3) Elbow & wrist stay behind shoulder/pelvis coronal plane in z
                            z_sh = j3d[sh_j, 2].detach()
                            y_sh = j3d[sh_j, 1].detach()
                            z_el, y_el = j3d[el_j, 2], j3d[el_j, 1]
                            z_wr, y_wr = j3d[wr_j, 2], j3d[wr_j, 1]
                            l_el_y = torch.relu(y_sh + 0.18 - y_el)
                            l_wr_y = torch.relu(y_el.detach() + 0.08 - y_wr)
                            l_el_z = torch.relu(torch.maximum(z_sh, z_pelvis) + 0.03 - z_el)
                            l_wr_z = torch.relu(torch.maximum(z_el.detach(), z_pelvis) + 0.06 - z_wr)
                            l_wr_x = torch.relu((j3d[wr_j, 0] - x_pelvis).abs() - 0.07)
                            loss = loss + 0.6 * l2d + 350.0 * lz + 200.0 * (l_el_y + l_wr_y + l_el_z + l_wr_z + 0.5 * l_wr_x)

            loss.backward()
            if frozen_slices and body.grad is not None:
                for s0, s1 in frozen_slices:
                    body.grad[0, s0:s1] = 0.0
            opt.step()
            if frozen_slices:
                with torch.no_grad():
                    for s0, s1 in frozen_slices:
                        body[0, s0:s1] = P["body_pose"][0, s0:s1]

        with torch.no_grad():
            fin = run(body, lh, rh, tr)
            ref_mpjpe = mpjpe(fin)
            red = round((init_mpjpe - ref_mpjpe) / max(init_mpjpe, 1e-4) * 100, 2)
            tri, _ = rasterize(fin.vertices[0], faces_cuda, fx, fy, cx, cy, W, H, glctx)
            vis = tri[tri >= 0]
            l_vis_px = int(lhand_face[vis].sum()) if hide_l_fingers else None
            r_vis_px = int(rhand_face[vis].sum()) if hide_r_fingers else None
            ref_depth, ref_norm, metric = render_mesh_nvdiffrast(fin.vertices[0], fx, fy, cx, cy, W, H, glctx, faces_cuda, smplx_faces, device)
            ref_dw = render_dwpose_masked(
                fin.joints[0], fx, fy, cx, cy, W, H,
                hide_l_fingers, hide_r_fingers,
                hide_l_arm=(l_state == "closeup_down"),
                hide_r_arm=(r_state == "closeup_down"),
            )
            o_depth, o_norm, _ = render_mesh_nvdiffrast(init_out.vertices[0], fx, fy, cx, cy, W, H, glctx, faces_cuda, smplx_faces, device)

        fdir = frames_out / cid / fid
        fdir.mkdir(parents=True, exist_ok=True)
        ref_norm_bgr = cv2.cvtColor(ref_norm, cv2.COLOR_RGB2BGR)
        ref_dw_bgr = cv2.cvtColor(np.array(ref_dw), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fdir / "normal.png"), ref_norm_bgr)
        cv2.imwrite(str(fdir / "depth_vis.png"), ref_depth)
        cv2.imwrite(str(fdir / "dwpose.png"), ref_dw_bgr)
        np.save(str(fdir / "depth_metric.npy"), metric)
        rd = dict(d_init)
        rd.update(body_pose=body.detach().cpu(), left_hand_pose=lh.detach().cpu(),
                  right_hand_pose=rh.detach().cpu(), transl=tr.detach().cpu())
        torch.save(rd, str(fdir / "smplx_params_refined.pt"))
        rec = {
            "clip_id": cid, "frame_id": fid, "method": "sapiens2_refine_hidden_hand_v2",
            "metrics": {
                "init_2d_mpjpe_px": round(init_mpjpe, 2), "refined_2d_mpjpe_px": round(ref_mpjpe, 2),
                "error_reduction_pct": red, "steps": steps, "dropped_out_of_frame_kpts": n_out_of_frame,
                "truncated_leg_frozen": {"left": trunc_l_leg, "right": trunc_r_leg},
                "total_frame_latency_sec": round(time.time() - t0, 2),
            },
            "hand_visibility": {
                "left": {"hidden": hide_l_fingers, **l_info, "rendered_visible_px_after": l_vis_px},
                "right": {"hidden": hide_r_fingers, **r_info, "rendered_visible_px_after": r_vis_px},
            },
        }
        json.dump(rec, open(fdir / "record.json", "w"), indent=2)

        orig_dw_paths = glob.glob(f"{geom_dir}/frames/{cid}/{fid}/attempts/*/dwpose.png")
        orig_dw_bgr = cv2.imread(orig_dw_paths[0]) if orig_dw_paths else ref_dw_bgr * 0
        title = (
            f"CLIP: {cid} | FRAME: {fid} | 2D MPJPE: {init_mpjpe:.1f}px -> {ref_mpjpe:.1f}px (-{red:.1f}%) "
            f"| L:{l_state} R:{r_state}"
        )
        strip = build_comparison_strip(rgb_bgr, orig_dw_bgr, ref_dw_bgr,
                                       cv2.cvtColor(o_norm, cv2.COLOR_RGB2BGR), ref_norm_bgr,
                                       cv2.cvtColor(o_depth, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ref_depth, cv2.COLOR_GRAY2BGR), title)
        cv2.imwrite(str(check_out / f"{fid}.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
        summary.append(rec)
        print(f"[{i}/{len(tasks)}] {fid} L={l_state}({l_vis_px}) R={r_state}({r_vis_px}) "
              f"leg={'L' if trunc_l_leg else ''}{'R' if trunc_r_leg else '-'} "
              f"MPJPE {init_mpjpe:.1f}->{ref_mpjpe:.1f} ({time.time() - t0:.1f}s)", flush=True)

        if i % 10 == 0 or i == 1 or i == len(tasks):
            elapsed = time.time() - t_all
            avg_lat = elapsed / max(i, 1)
            eta_sec = avg_lat * (len(tasks) - i)
            prog_data = {
                "current_index": i,
                "total_frames": len(tasks),
                "progress_pct": round(i / len(tasks) * 100, 2),
                "current_frame": f"{cid}/{fid}",
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta_sec, 1),
                "eta_hours": round(eta_sec / 3600, 2),
                "avg_frame_latency_sec": round(avg_lat, 2),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                with open(out_root / "progress.json", "w") as fp:
                    json.dump(prog_data, fp, indent=2)
            except Exception:
                pass

    json.dump(summary, open(out_root / f"hidden_hand_v2_summary_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=2)
    print(f"Done {len(tasks)} frames in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()

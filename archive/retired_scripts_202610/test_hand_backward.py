import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path("/home/shangguanrz/project/pic-edit")
sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.execution import (
    new_adapter, load_adapter_checkpoint, parameter_groups,
    BODY_PREFIXES
)
from src.pose_control.qwen21_full.checkpoint import FACE_ONLY_PREFIXES, HAND_ONLY_PREFIXES
from src.pose_control.qwen21_full.hand.conditions import HandFineCondition, HandReferenceFeatures
from src.pose_control.qwen21_full.hand.geometry import split_iper_134_hand_points
from src.data.letterbox import NativePad

# 1. Load adapter
adapter = new_adapter("cuda")
INITIAL = Path("/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_003000.pt")
load_adapter_checkpoint(adapter, INITIAL)

# Initialize hand zero_heads and up weights to 0.01 so gradients flow
with torch.no_grad():
    for h in adapter.legacy.control_interface.hand_adapter.zero_heads:
        if h.weight.norm() == 0:
            h.weight.fill_(0.01)
    for sa in adapter.legacy.control_interface.hand_adapter.stage_adapters:
        if sa.up.weight.norm() == 0:
            sa.up.weight.fill_(0.01)

# 2. Build hand inputs for pair 0
pairs_path = Path("/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/data/common256_pairs.jsonl")
p0 = json.loads(pairs_path.read_text().splitlines()[0])
seq_id = p0["sequence_id"]
source_stem = p0["source_stem"]
target_stem = p0["target_stem"]
dw_src_p = Path(p0["source_dwpose_json"])
dw_tgt_p = Path(p0["target_dwpose_json"])
smpl_dir = Path(f"/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_3d_assets_native/{seq_id}/smplestx")

dw_src = json.load(open(dw_src_p))
dw_tgt = json.load(open(dw_tgt_p))
smpl = torch.load(smpl_dir / "params.pt", map_location="cpu", weights_only=False)
frame_map = json.load(open(smpl_dir / "frame_map.json"))
stem_to_idx = {item['stem']: item['index'] for item in frame_map.get('frames', [])}

import numpy as np
src_kps = np.array(dw_src['keypoints_norm'], dtype=np.float32)
src_sc = np.array(dw_src['scores'], dtype=np.float32)
tgt_kps = np.array(dw_tgt['keypoints_norm'], dtype=np.float32)
tgt_sc = np.array(dw_tgt['scores'], dtype=np.float32)

src_pts134 = torch.from_numpy(np.concatenate([src_kps, src_sc[:, None]], axis=-1))
tgt_pts134 = torch.from_numpy(np.concatenate([tgt_kps, tgt_sc[:, None]], axis=-1))

pad = NativePad.from_size((604, 1080))
src_pts134 = pad.points(src_pts134)
tgt_pts134 = pad.points(tgt_pts134)

src_hands = split_iper_134_hand_points(src_pts134)
tgt_hands = split_iper_134_hand_points(tgt_pts134)

def get_boxes(hands: torch.Tensor):
    boxes = torch.zeros(2, 4, dtype=torch.float32)
    valid = torch.zeros(2, dtype=torch.bool)
    for side in range(2):
        pts = hands[side]
        vis = pts[:, 2] > 0.15
        if int(vis.sum()) >= 3:
            xy = pts[vis, :2]
            lo = xy.amin(dim=0)
            hi = xy.amax(dim=0)
            span = (hi - lo).clamp_min(1e-3)
            margin = span * 0.2 + 0.01
            b = torch.cat([(lo - margin).clamp(0, 1), (hi + margin).clamp(0, 1)])
            if bool((b[2:] > b[:2]).all()):
                boxes[side] = b
                valid[side] = True
        if not valid[side]:
            boxes[side] = torch.tensor([0.0, 0.0, 0.1, 0.1])
    return boxes, valid

src_boxes, src_valid = get_boxes(src_hands)
tgt_boxes, tgt_valid = get_boxes(tgt_hands)
src_boxes = torch.where(src_valid[:, None], src_boxes, tgt_boxes)

source_boxes = torch.zeros(1, 2, 2, 4, dtype=torch.float32)
target_boxes = torch.zeros(1, 2, 2, 4, dtype=torch.float32)
region_valid = torch.zeros(1, 2, 2, dtype=torch.bool)
source_boxes[0, 0] = src_boxes
target_boxes[0, 0] = tgt_boxes
region_valid[0, 0] = tgt_valid

hand_keypoints = torch.zeros(1, 2, 2, 21, 3, dtype=torch.float32)
hand_keypoints[0, 0] = tgt_hands

hand_pose = torch.zeros(1, 2, 2, 45, dtype=torch.float32)
if target_stem in stem_to_idx:
    t_idx = stem_to_idx[target_stem]
    hand_pose[0, 0, 0] = smpl['left_hand_pose'][t_idx].float().reshape(-1)
    hand_pose[0, 0, 1] = smpl['right_hand_pose'][t_idx].float().reshape(-1)

condition = HandFineCondition(
    hand_keypoints=hand_keypoints.cuda(),
    hand_pose=hand_pose.cuda(),
    source_boxes=source_boxes.cuda(),
    target_boxes=target_boxes.cuda(),
    region_valid=region_valid.cuda(),
    source_indices=torch.tensor([[0, 1]], device="cuda", dtype=torch.long),
    mesh_bps=torch.zeros(1, 2, 2, 1024, device="cuda"),
    mesh_valid=torch.zeros(1, 2, 2, device="cuda", dtype=torch.bool),
    depth=torch.zeros(1, 2, 2, 1, 16, 16, device="cuda"),
    visibility=region_valid.cuda().float(),
).validate()

references = HandReferenceFeatures(
    dino_patches=torch.zeros(1, 2, 2, 1, 16, 1536, device="cuda"),
    reference_valid=torch.zeros(1, 2, 2, 1, device="cuda", dtype=torch.bool),
    source_indices=torch.tensor([[0, 1]], device="cuda", dtype=torch.long),
).validate()

# 3. Forward pass through hand_preparer and hand_adapter
preparer = adapter.legacy.hand_preparer
prepared = preparer(condition, references)

adapter_layer = adapter.legacy.control_interface.hand_adapter
hand_residuals = adapter_layer(prepared, target_token_hw=(68, 38), timestep=torch.tensor([500.0], device="cuda"))

# Project all 6 through output_bridges['hand']
projected_all = sum(adapter.output_bridges['hand'](r).sum() for r in hand_residuals)

# Backward
projected_all.backward()

# Check groups
groups = {'heads_and_interface': 0.0, 'encoders': 0.0, 'control_core': 0.0}
for name, p in adapter.named_parameters():
    if p.grad is not None and "hand" in name:
        gn = float(p.grad.float().norm().item())
        if "output_bridges.hand" in name or "zero_heads" in name or "hand_log_group_scale" in name:
            groups['heads_and_interface'] += gn**2
        elif "control_interface.hand_adapter" in name:
            groups['control_core'] += gn**2
        elif "hand_preparer" in name:
            groups['encoders'] += gn**2

for g in groups:
    groups[g] = groups[g] ** 0.5
    print(f"Group {g}: norm={groups[g]:.4f}")

print("Test completed.")

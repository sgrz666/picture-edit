import json
import time
from pathlib import Path
import torch

ROOT = Path("/home/shangguanrz/project/pic-edit")
pairs_path = Path("/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008/data/common256_pairs.jsonl")
pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]

t0 = time.perf_counter()
smpl_cache = {}
dw_cache = {}
for p in pairs:
    seq = p["sequence_id"]
    if seq not in smpl_cache:
        s_dir = Path(f"/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_3d_assets_native/{seq}/smplestx")
        smpl_cache[seq] = torch.load(s_dir / "params.pt", map_location="cpu")
        smpl_cache[f"{seq}_map"] = json.load(open(s_dir / "frame_map.json"))
    for stem in (p["source_stem"], p["target_stem"]):
        key = (seq, stem)
        if key not in dw_cache:
            dw_p = ROOT / f"datasets/TikTokDataset/TikTok_v3/dwpose_v1/{seq}/dwpose_json/{stem}.json"
            dw_cache[key] = json.load(open(dw_p))

t1 = time.perf_counter()
print(f"Loaded all {len(pairs)} pairs hand data in {t1 - t0:.2f}s! Unique seqs: {len(smpl_cache)//2}, unique dw frames: {len(dw_cache)}")

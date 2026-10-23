"""P0 source-asset audit, header-only: are the condition maps stored at native aspect?

`TikTokV3PoseDataset._native_item` rasterises every condition through the target canvas and
resizes an asset whenever its stored size differs from the frame. If an asset was ever produced
at a square 512x512 while the frame is 604x1080, that resize is a non-uniform stretch and the
"native" path would be distorting normal/part/mask/depth while looking perfectly rectangular.

Reading only PNG/JPEG headers keeps this cheap, and results print per manifest as they go so a
timeout cannot swallow the run.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/shangguanrz/project/pic-edit")

from PIL import Image

ROOT = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
MANIFESTS = ["train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl"]
SIDES = ["source_rgb", "target_rgb", "target_normal", "target_person_mask",
         "target_depth_vis", "target_dwpose"]


def size_of(path):
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return "missing"
    with Image.open(p) as image:      # header only, no pixel decode
        return tuple(image.size)      # (W, H)


total = Counter()
for name in MANIFESTS:
    manifest = ROOT / "TikTok_v3/manifests/pairs" / name
    if not manifest.is_file():
        print(f"{name}: MISSING {manifest}", flush=True)
        continue
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_side = {side: Counter() for side in SIDES}
    part_sizes = Counter()
    aspect_mismatch = Counter()
    frames = Counter()
    for row in rows:
        frame = size_of(row.get("target_rgb"))
        if not isinstance(frame, tuple):
            frames[str(frame)] += 1
            continue
        frames[frame] += 1
        frame_ratio = frame[0] / frame[1]
        for side in SIDES:
            size = size_of(row.get(side))
            per_side[side][size] += 1
            if isinstance(size, tuple):
                ratio = size[0] / size[1]
                if abs(ratio - frame_ratio) > 0.01:
                    aspect_mismatch[side] += 1
        seq = str(row.get("sequence_id") or row.get("appearance"))
        stem = str(row.get("target_stem"))
        for candidate in (ROOT / "TikTok_3d_assets_native" / seq / "part_512" / f"{stem}.png",
                          ROOT / "TikTok_3d_assets_native" / seq / "part_512" / f"{stem}.jpg"):
            if candidate.is_file():
                with Image.open(candidate) as image:
                    part_sizes[tuple(image.size)] += 1
                break
        else:
            part_sizes["missing"] += 1

    print(f"\n=== {name}: {len(rows)} pairs ===", flush=True)
    print(f"  target frame sizes: {dict(frames.most_common(4))}")
    for side in SIDES:
        top = dict(per_side[side].most_common(3))
        print(f"  {side:<20} sizes={top}  aspect-mismatch-vs-frame={aspect_mismatch.get(side, 0)}")
    print(f"  part_512           sizes={dict(part_sizes.most_common(3))}")
    total.update({f"{name}:{s}": c for s, c in aspect_mismatch.items()})
    print(flush=True)

print("=== verdict ===")
if not total:
    print("no condition asset deviates from its frame aspect by >1%  (square-stretch hazard absent)")
else:
    print(f"assets whose stored aspect differs from the frame: {dict(total)}")

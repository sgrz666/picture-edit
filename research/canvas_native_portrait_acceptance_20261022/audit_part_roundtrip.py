"""Does part_512 round-trip consistently onto the native canvas?

part_512 is stored 512x512 for every pair while frames are 604x1080, and `_native_item` resizes it
back to the frame. That is geometrically sound only if the asset was made by stretching the frame
(not padding or cropping it). The normal map is rendered at native size, so it is the right
reference: measure the IoU between the part foreground and the normal foreground, both on the
frame canvas, over many pairs.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/shangguanrz/project/pic-edit/datasets/TikTokDataset")
CACHE = ROOT / "TikTok_3d_assets_native"
FRAME_W, FRAME_H = 604, 1080
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 120

rows = []
for name in ("train_pairs.jsonl", "val_pairs.jsonl", "test_pairs.jsonl"):
    path = ROOT / "TikTok_v3/manifests/pairs" / name
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append((name, json.loads(line)))
    if len(rows) >= LIMIT:
        break

ious, offsets = [], []
skipped = 0
for name, row in rows[:LIMIT]:
    part_path = CACHE / str(row["sequence_id"]) / "part_512" / f"{row['target_stem']}.png"
    if not part_path.is_file() or not Path(row["target_normal"]).is_file():
        skipped += 1
        continue
    with Image.open(part_path) as image:
        part = np.asarray(image.convert("L"), dtype=np.uint8)
    # Exactly what _native_item does: NEAREST resize of the square asset onto the frame.
    part_on_frame = np.asarray(Image.fromarray(part).resize((FRAME_W, FRAME_H), Image.NEAREST),
                               dtype=np.uint8)
    with Image.open(row["target_normal"]) as image:
        normal = np.asarray(image.convert("RGB"), dtype=np.float32)
    part_fg = part_on_frame > 0
    normal_fg = normal.max(axis=2) > 8.0
    union = np.logical_or(part_fg, normal_fg).sum()
    if union == 0:
        skipped += 1
        continue
    ious.append(float(np.logical_and(part_fg, normal_fg).sum()) / float(union))
    rows_idx = np.where(part_fg.any(axis=1))[0]
    cols_idx = np.where(part_fg.any(axis=0))[0]
    n_rows = np.where(normal_fg.any(axis=1))[0]
    n_cols = np.where(normal_fg.any(axis=0))[0]
    if len(rows_idx) and len(n_rows):
        offsets.append((abs(rows_idx[0] - n_rows[0]) / FRAME_H,
                        abs(cols_idx[0] - n_cols[0]) / FRAME_W))

ious = np.array(ious)
offsets = np.array(offsets) if offsets else np.zeros((1, 2))
print(f"pairs compared: {len(ious)}  skipped: {skipped}")
print(f"part-vs-normal foreground IoU: mean {ious.mean():.4f}  min {ious.min():.4f}  "
      f"p05 {np.percentile(ious, 5):.4f}")
print(f"top-left bbox offset (normalised): mean dy {offsets[:, 0].mean():.4f}  "
      f"mean dx {offsets[:, 1].mean():.4f}  max dy {offsets[:, 0].max():.4f}  "
      f"max dx {offsets[:, 1].max():.4f}")
if ious.mean() > 0.9:
    print("VERDICT: part_512 is a square-STRETCHED render; resizing it back to the frame is "
          "geometrically consistent, so the native path does not misalign part against normal/RGB.")
else:
    print("VERDICT: part_512 does NOT round-trip onto the frame (padded or cropped at creation); "
          "the native path misaligns part labels and the asset must be re-rendered at native size.")

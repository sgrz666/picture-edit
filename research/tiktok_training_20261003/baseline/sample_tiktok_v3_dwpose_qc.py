import json
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

MANIFEST = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "dwpose_results.jsonl"
)

OUT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/qc/"
      "dwpose_sample"
)

N = 24
SEED = 2026


rows = [
    json.loads(x)
    for x in MANIFEST.read_text(encoding="utf-8").splitlines()
    if x.strip() and json.loads(x)["status"] == "PASS"
]

rng = random.Random(SEED)
chosen = rng.sample(rows, min(N, len(rows)))

OUT.mkdir(parents=True, exist_ok=True)

tiles = []

for i, r in enumerate(chosen, 1):
    rgb = cv2.imread(r["rgb"])
    pose = cv2.imread(r["dwpose_png"])

    if rgb is None or pose is None:
        continue

    H, W = rgb.shape[:2]

    if pose.shape[:2] != (H, W):
        pose = cv2.resize(
            pose,
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = cv2.addWeighted(
        rgb,
        0.60,
        pose,
        0.90,
        0,
    )

    target_h = 300
    scale = target_h / H

    rgb = cv2.resize(
        rgb,
        (int(W * scale), target_h),
        interpolation=cv2.INTER_AREA,
    )

    overlay = cv2.resize(
        overlay,
        (int(W * scale), target_h),
        interpolation=cv2.INTER_AREA,
    )

    panel = np.concatenate(
        [rgb, overlay],
        axis=1,
    )

    title = (
        f"{i:02d} {r['sequence_id']} {r['stem']} "
        f"body={r['body_mean_conf']:.2f}"
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1], 30),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        panel,
        title,
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    tiles.append(panel)


if not tiles:
    raise RuntimeError("no valid samples")

cell_h = 300
cell_w = max(x.shape[1] for x in tiles)

fixed = []

for x in tiles:
    c = np.zeros(
        (cell_h, cell_w, 3),
        dtype=np.uint8,
    )
    c[:x.shape[0], :x.shape[1]] = x
    fixed.append(c)

sheet_rows = []

for i in range(0, len(fixed), 4):
    row = fixed[i:i+4]

    while len(row) < 4:
        row.append(np.zeros_like(fixed[0]))

    sheet_rows.append(
        np.concatenate(row, axis=1)
    )

sheet = np.concatenate(sheet_rows, axis=0)

out = OUT / "dwpose_contact_sheet_24.jpg"

cv2.imwrite(str(out), sheet)

print("samples :", len(tiles))
print("output  :", out)

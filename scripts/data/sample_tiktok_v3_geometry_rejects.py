
# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import json
import random
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/shangguanrz/project/pic-edit")

INV = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "condition_inventory.jsonl"
)

OUT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/"
      "qc/geometry_reject_sample"
)

N = 12
SEED = 2026


rows = [
    json.loads(x)
    for x in INV.read_text(encoding="utf-8").splitlines()
    if x.strip()
]

reject = [
    r for r in rows
    if not r.get("core_existing_complete", False)
]

by_seq = {}

for r in reject:
    by_seq.setdefault(
        r["sequence_id"],
        []
    ).append(r)

rng = random.Random(SEED)

seqs = sorted(by_seq)

rng.shuffle(seqs)

chosen = []

for seq in seqs:
    if len(chosen) >= N:
        break

    chosen.append(
        rng.choice(
            by_seq[seq]
        )
    )

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

tiles = []
manifest = []

for i, r in enumerate(chosen, 1):
    seq = r["sequence_id"]
    stem = r["stem"]

    rgb_path = Path(
        r["paths"]["rgb"]
    )

    mask_path = Path(
        r["paths"]["person_mask"]
    )

    img = cv2.imread(
        str(rgb_path)
    )

    if img is None:
        continue

    H, W = img.shape[:2]

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is not None:
        if mask.shape != (H, W):
            mask = cv2.resize(
                mask,
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = img.copy()

        fg = mask > 127

        overlay[fg] = (
            0.65 * overlay[fg]
            +
            0.35 * np.array(
                [0, 255, 0]
            )
        ).astype(np.uint8)

        panel = np.concatenate(
            [img, overlay],
            axis=1,
        )
    else:
        panel = img

    target_h = 360

    scale = target_h / panel.shape[0]

    panel = cv2.resize(
        panel,
        (
            int(panel.shape[1] * scale),
            target_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    title = (
        f"{i:02d}  seq={seq}  frame={stem}"
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1], 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        panel,
        title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_img = (
        OUT
        / f"{i:02d}_{seq}_{stem}.jpg"
    )

    cv2.imwrite(
        str(out_img),
        panel,
    )

    tiles.append(panel)

    manifest.append({
        "sequence_id": seq,
        "stem": stem,
        "rgb": str(rgb_path),
        "person_mask": str(mask_path),
        "preview": str(out_img),
    })


# contact sheet
if tiles:
    cell_w = max(
        x.shape[1]
        for x in tiles
    )

    prepared = []

    for tile in tiles:
        canvas = np.zeros(
            (
                360,
                cell_w,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            :tile.shape[0],
            :tile.shape[1],
        ] = tile

        prepared.append(canvas)

    rows_sheet = []

    for i in range(
        0,
        len(prepared),
        3,
    ):
        row = prepared[i:i+3]

        while len(row) < 3:
            row.append(
                np.zeros_like(
                    prepared[0]
                )
            )

        rows_sheet.append(
            np.concatenate(
                row,
                axis=1,
            )
        )

    sheet = np.concatenate(
        rows_sheet,
        axis=0,
    )

    cv2.imwrite(
        str(
            OUT
            / "contact_sheet_12.jpg"
        ),
        sheet,
    )

(
    OUT
    / "sample_manifest.json"
).write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("reject total       :", len(reject))
print("reject sequences   :", len(by_seq))
print("sampled            :", len(manifest))
print(
    "contact sheet      :",
    OUT / "contact_sheet_12.jpg"
)

#!/usr/bin/env python3
"""Export InterHand frame pairs to the official Diffusers I2I dataset format."""

import argparse
import hashlib
import io
import json
from itertools import islice
from pathlib import Path

from PIL import Image


HAND_PHRASES = {"left": "the left hand", "right": "the right hand", "interacting": "both hands"}
SPLITS = ("train", "val", "test")


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_image(root, asset):
    path = root / asset["path"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != asset["sha256"]:
        raise ValueError(f"Source manifest SHA256 mismatch: {path}")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        size = image.size
    return {"path": path.name, "bytes": data}, size


def pair_rows(manifest, split, *, with_pose=False, rejected=None):
    subjects, image_splits, ids, selected = {}, {}, set(), []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_split = row["split"]
        if row_split not in SPLITS or row["id"] in ids:
            raise ValueError(f"Invalid split or repeated pair ID: {row['id']}")
        ids.add(row["id"])
        group = row["group"]
        if not group or subjects.setdefault(group, row_split) != row_split:
            raise ValueError(f"Missing subject or subject occurs in multiple splits: {group}")
        source, = row["source_refs"]
        target = row["target"]
        for asset in (source, target):
            digest = asset["sha256"]
            if image_splits.setdefault(digest, row_split) != row_split:
                raise ValueError(f"Image SHA256 occurs in multiple splits: {digest}")
        if source["sha256"] == target["sha256"]:
            raise ValueError(f"Source and target are identical: {row['id']}")
        if row_split == split:
            selected.append((row, source, target))

    if not selected:
        raise ValueError(f"{manifest} has no {split} pairs")
    for row, source, target in selected:
        source_image, source_size = read_image(manifest.parent, source)
        target_image, target_size = read_image(manifest.parent, target)
        if source_size != target_size:
            raise ValueError(f"Source and target dimensions differ: {row['id']}")
        hand_type = row["interhand"]["hand_type"]
        caption = row.get("caption", "").strip()
        if not caption:
            caption = (f"Change the gesture of {HAND_PHRASES[hand_type]} while preserving "
                       "the skin appearance, lighting, and background.")
        record = {
            "id": row["id"], "group": row["group"], "hand_type": hand_type,
            "source": source_image, "target": target_image, "caption": caption,
            "source_sha256": source["sha256"], "target_sha256": target["sha256"],
        }
        if with_pose:
            from hand_pose import POSE_CAPTIONS, PRESERVE, UnrenderablePose, aligned_triplet

            try:
                a, b, pose, keypoints = aligned_triplet(manifest.parent, row, source_image, target_image,
                                                      target_size, file_sha256)
            except UnrenderablePose as error:
                if rejected is None:
                    raise
                rejected.append({"id": row["id"], "reason": str(error)})
                continue
            record.update(source=a, target=b, pose=pose, caption=POSE_CAPTIONS[hand_type] + PRESERVE,
                          pose_sha256=hashlib.sha256(pose["bytes"]).hexdigest(),
                          pose_joints=json.dumps(keypoints, separators=(",", ":")))
        yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", required=True, type=Path, help="New dataset directory")
    parser.add_argument("--with-pose", action="store_true", help="Export A RGB and aligned B Pose as two images")
    args = parser.parse_args()

    from datasets import Features, Image as DatasetImage, Value
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = Features({
        "id": Value("string"), "group": Value("string"), "hand_type": Value("string"),
        "source": DatasetImage(), "target": DatasetImage(), "caption": Value("string"),
        "source_sha256": Value("string"), "target_sha256": Value("string"),
    })
    if args.with_pose:
        features.update(pose=DatasetImage(), pose_sha256=Value("string"), pose_joints=Value("string"))
    manifest = args.manifest.resolve(strict=True)
    manifest_sha = file_sha256(manifest)
    # Hugging Face Datasets recognizes the official split names from the Parquet filename.
    split = "validation" if args.split == "val" else args.split
    rejected = []
    rows = pair_rows(manifest, args.split, with_pose=args.with_pose, rejected=rejected)
    batch = list(islice(rows, 128))
    if not batch:
        raise ValueError(f"No runnable {split} pairs; rejected {len(rejected)} target poses")
    args.out.mkdir()
    parquet = args.out / f"{split}-00000-of-00001.parquet"
    count = 0
    with parquet.open("xb") as handle, pq.ParquetWriter(handle, features.arrow_schema) as writer:
        while batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=features.arrow_schema))
            count += len(batch)
            batch = list(islice(rows, 128))
    if file_sha256(manifest) != manifest_sha:
        raise ValueError(f"Source manifest changed during export: {manifest}")
    record = {
        "manifest": str(manifest), "manifest_sha256": manifest_sha, "split": split,
        "pairs": count, "parquet": parquet.name, "parquet_sha256": file_sha256(parquet),
        "condition_columns": ["source", "pose"] if args.with_pose else ["source"],
    }
    if args.with_pose:
        record["pose_rejections"] = rejected
    with args.out.with_name(args.out.name + ".json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()

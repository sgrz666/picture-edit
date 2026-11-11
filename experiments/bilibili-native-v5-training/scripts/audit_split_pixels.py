"""Decode every published RGB and check exact pixel isolation, without models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image
from split_dataset import rows, write, SPLITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    pixels = defaultdict(list)
    counts = {}
    for split in SPLITS:
        frames = rows(args.root / split / "frames.jsonl")
        counts[split] = len(frames)
        for frame in frames:
            with Image.open(frame["rgb_path"]) as image:
                image = image.convert("RGB")
                if (image.height, image.width) != tuple(frame["original_hw"]):
                    raise ValueError("Original RGB canvas metadata differs")
                digest = hashlib.sha256(str(image.size).encode() + b"\0" + image.tobytes()).hexdigest()
            pixels[digest].append({"frame_id": frame["frame_id"], "clip_id": frame["clip_id"], "split": split})
        print(json.dumps({"event": "pixels", "split": split, "frames_decoded": len(frames)}), flush=True)
    conflicts = [values for values in pixels.values() if len({v["split"] for v in values}) > 1]
    result = {"passed": not conflicts, "frames_decoded": sum(counts.values()), "split_counts": counts,
        "unique_pixel_images": len(pixels), "hash_method": "sha256(repr((width,height))+NUL+decoded_RGB_bytes)",
        "declared_pixel_sha256_fields_trusted": False, "cross_split_pixel_conflicts": conflicts,
        "input_images_modified": False, "training_started": False}
    write(args.report, result)
    print(json.dumps(result), flush=True)
    if conflicts:
        raise ValueError("Cross-split identical RGB pixels found; preserve all rows and regroup their videos")


if __name__ == "__main__":
    main()

"""Select a fixed tiny paired subset and explicit Depth records; copy no media."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=2)
    args = p.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    with args.pairs.open() as stream:
        rows = [json.loads(line) for line in itertools.islice(stream, args.count)]
    records = []
    for row in rows:
        for key in ("target_depth_vis", "target_smplx_mask", "target_rgb", "source_rgb"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        records.append({"pair_id": row["pair_id"], "target_stem": row["target_stem"],
                        "kind": "relative_png", "polarity": "near_one",
                        "depth_path": row["target_depth_vis"], "valid_path": row["target_smplx_mask"],
                        "camera_id": f"tiktok_smplestx:{row['sequence_id']}:{row['target_stem']}",
                        "units": "unknown", "scale_status": "unknown"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pairs.jsonl", "depth.jsonl"):
        if (args.output_dir/name).exists():
            raise FileExistsError(args.output_dir/name)
    for name, entries in (("pairs.jsonl", rows), ("depth.jsonl", records)):
        with (args.output_dir/name).open("x", encoding="utf8") as stream:
            for entry in entries:
                stream.write(json.dumps(entry)+"\n")
    print(json.dumps({"pairs": len(rows), "depth_records": len(records), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()

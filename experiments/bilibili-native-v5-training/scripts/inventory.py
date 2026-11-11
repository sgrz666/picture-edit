"""Read-only inventory for Bilibili manifests and the Native V5 preparation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return {"length": len(value), "first": compact(value[0]) if value else None}
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    frames = read_rows(root / "frames/frames.jsonl")
    pairs = read_rows(root / "pairs/pairs.jsonl")
    ids = {f["frame_id"] for f in frames}
    paths = {str(v) for row in frames + pairs for k, v in row.items()
             if isinstance(v, str) and k.endswith("_path")}
    records = sorted((root / "geometry/records").glob("*/*.json"))
    geometry = {}
    for category in ("smplx", "depth", "normal"):
        geometry[category] = {d.name: sum(1 for x in d.rglob("*") if x.is_file())
                              for d in (root / "geometry" / category).iterdir() if d.is_dir()}
    report = {
        "root": str(root), "frames": len(frames), "pairs": len(pairs),
        "clips": len({f["clip_id"] for f in frames}),
        "frame_schema": compact(frames[0]), "pair_schema": pairs[0],
        "pair_counts": dict(Counter(x["clip_id"] for x in pairs)),
        "frame_counts": dict(Counter(x["clip_id"] for x in frames)),
        "path_count": len(paths), "missing_paths": sorted(x for x in paths if not Path(x).is_file()),
        "duplicate_frame_ids": len(frames) - len(ids),
        "duplicate_pair_ids": len(pairs) - len({x["pair_id"] for x in pairs}),
        "pair_frame_refs_missing": sum(p[k] not in ids for p in pairs for k in ("source_frame_id", "target_frame_id")),
        "original_loss_masks": {k: dict(Counter(str(x.get("loss_mask", {}).get(k)) for x in frames))
                                for k in ("body", "face", "hand_a", "hand_b")},
        "geometry_counts": geometry, "geometry_record_count": len(records),
        "geometry_example": json.loads(records[0].read_text()),
        "geometry_summary_keys": list(json.loads((root / "geometry/summary.json").read_text())),
        "pose_example": compact(json.loads(Path(pairs[0]["source_pose_path"]).read_text())),
        "source_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                          for x in (root / "frames/frames.jsonl", root / "pairs/pairs.jsonl")},
    }
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
        if report["torch"]["cuda_available"]:
            report["torch"]["device"] = torch.cuda.get_device_name(0)
    except Exception as error:
        report["torch"] = {"error": repr(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in
          ("frame_schema", "pair_schema", "geometry_example", "pose_example", "pair_counts", "frame_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

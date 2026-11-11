"""Partition all published rows by video; never move media or overwrite outputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "val", "test")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def assign_clips(clips, seed):
    ordered = sorted(set(clips), key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).digest())
    n = len(ordered)
    if n < 3:
        raise ValueError("At least three independent videos are required")
    held = max(1, round(n * .1))
    return {clip: "val" if i < held else "test" if i < 2 * held else "train"
            for i, clip in enumerate(ordered)}


def partition(frames, pairs, mapping):
    index = {f["frame_id"]: f for f in frames}
    if len(index) != len(frames) or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate frame/pair IDs")
    by_frame = {s: [] for s in SPLITS}
    by_pair = {s: [] for s in SPLITS}
    for f in frames:
        by_frame[mapping[f["clip_id"]]].append({**f, "original_split": f.get("split"),
                                               "split": mapping[f["clip_id"]], "group_id": f["clip_id"]})
    for p in pairs:
        clip = p["clip_id"]
        if any(index[p[k]]["clip_id"] != clip for k in ("source_frame_id", "target_frame_id")):
            raise ValueError("Cross-video pair: " + p["pair_id"])
        by_pair[mapping[clip]].append({**p, "original_split": p.get("split"),
                                      "split": mapping[clip], "group_id": clip})
    return by_frame, by_pair


def write(path, value, jsonl=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def verify(root, report):
    source_frames = rows(root / "frames/frames.jsonl")
    source_pairs = rows(root / "pairs/pairs.jsonl")
    split_frames = {s: rows(root / s / "frames.jsonl") for s in SPLITS}
    split_pairs = {s: rows(root / s / "pairs.jsonl") for s in SPLITS}
    frame_ids, pair_ids, clips, pixels = {}, {}, {}, {}
    for position, split in enumerate(SPLITS):
        frame_ids[split] = {r["frame_id"] for r in split_frames[split]}
        pair_ids[split] = {r["pair_id"] for r in split_pairs[split]}
        clips[split] = {r["clip_id"] for r in split_frames[split]}
        pixels[split] = {r["pixel_sha256"] for r in split_frames[split] if r.get("pixel_sha256")}
        groups = json.loads((root / split / "groups.json").read_text(encoding="utf-8"))
        expected_groups = {p["pair_id"]: "bilibili:" + p["clip_id"] for p in split_pairs[split]}
        if groups != expected_groups:
            raise ValueError("Training group index differs: " + split)
        for frame in split_frames[split]:
            if frame.get("split") != split or frame.get("group_id") != frame["clip_id"]:
                raise ValueError("Frame split/group assertion differs")
        for p in split_pairs[split]:
            if p["split"] != split or any(p[k] not in frame_ids[split] for k in ("source_frame_id", "target_frame_id")):
                raise ValueError("Split/endpoint mismatch")
        for other in SPLITS[:position]:
            if any(t[split] & t[other] for t in (frame_ids, pair_ids, clips, pixels)):
                raise ValueError("Video/frame/pixel leakage between " + split + " and " + other)
    if (set.union(*frame_ids.values()) != {f["frame_id"] for f in source_frames}
            or set.union(*pair_ids.values()) != {p["pair_id"] for p in source_pairs}
            or sum(map(len, split_frames.values())) != len(source_frames)
            or sum(map(len, split_pairs.values())) != len(source_pairs)):
        raise ValueError("Original rows not exactly conserved")
    original_f = {f["frame_id"]: f for f in source_frames}
    original_p = {p["pair_id"]: p for p in source_pairs}
    for split in SPLITS:
        for row, key, original in [(x, "frame_id", original_f) for x in split_frames[split]] + [(x, "pair_id", original_p) for x in split_pairs[split]]:
            old = original[row[key]]
            if row.get("original_split") != old.get("split") or row.get("group_id") != row["clip_id"]:
                raise ValueError("Original split/group provenance differs")
            if any(row[k] != v for k, v in old.items() if k != "split"):
                raise ValueError("Original row content changed")
    result = {"passed": True, "all_rows_preserved": True, "original_quality_flags_preserved": True,
              "video_frame_pixel_leakage": False, "media_moved_or_deleted": False,
              "splits": {s: {"clips": len(clips[s]), "frames": len(split_frames[s]), "pairs": len(split_pairs[s])} for s in SPLITS}}
    write(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("split", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "verify":
        print(json.dumps(verify(root, args.report), ensure_ascii=False))
        return
    files = [root / s / name for s in SPLITS for name in ("frames.jsonl", "pairs.jsonl", "groups.json", "README.md")]
    if args.report.exists() or any(p.exists() for p in files):
        raise FileExistsError("An output already exists; use a new version, never overwrite")
    frames, pairs = rows(root / "frames/frames.jsonl"), rows(root / "pairs/pairs.jsonl")
    mapping = assign_clips((f["clip_id"] for f in frames), args.seed)
    by_frame, by_pair = partition(frames, pairs, mapping)
    pixels = {}
    for frame in frames:
        digest = frame.get("pixel_sha256")
        if digest:
            pixels.setdefault(digest, set()).add(mapping[frame["clip_id"]])
    if any(len(v) > 1 for v in pixels.values()):
        raise ValueError("Cross-video identical frames need duplicate-group assignment before publishing")
    before = {str(p): sha(p) for p in (root / "frames/frames.jsonl", root / "pairs/pairs.jsonl")}
    artifacts = {}
    for split in SPLITS:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        artifacts[split] = {
            "frames": write(directory / "frames.jsonl", by_frame[split], True),
            "pairs": write(directory / "pairs.jsonl", by_pair[split], True),
            "groups": write(directory / "groups.json", {p["pair_id"]: "bilibili:" + p["clip_id"] for p in by_pair[split]})}
        with (directory / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(f"# Bilibili {split}\n\n按原视频 80/10/10、seed={args.seed} 划分。含 {len(by_frame[split])} 帧、{len(by_pair[split])} 对、{sum(v == split for v in mapping.values())} 个视频。\n\nframes.jsonl 保留全部原帧字段；pairs.jsonl 保留全部原配对字段及质量标记；groups.json 是 pair_id 到原视频组的映射。媒体仍在原路径，未移动。原始 RGB/pose 配对清单需经 Native V5 装配后供正式训练器使用。\n")
    if before != {p: sha(p) for p in before}:
        raise ValueError("Original manifest changed during partition")
    report = {"schema": "bilibili_video_split_v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "seed": args.seed, "ratio_by_video": [.8, .1, .1], "clip_to_split": mapping,
              "source_sha256": before, "artifacts": artifacts,
              "splits": {s: {"clips": sum(v == s for v in mapping.values()), "frames": len(by_frame[s]), "pairs": len(by_pair[s])} for s in SPLITS},
              "quality_filter_applied": False, "strict_subset_created": False, "training_started": False}
    write(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("artifacts", "clip_to_split")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

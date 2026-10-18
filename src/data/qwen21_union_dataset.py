"""Read-only provider adapters and reproducible Plan B pilot manifests.

Source datasets are never modified. Prepared condition assets are written only
under the caller's new workspace; target RGB is opened only by the builder for
dimensions/hash and by the separate training/evaluation supervision consumer.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable

import numpy as np
from PIL import Image

from src.pose_control.qwen21_union.contracts import CanvasTransform, ConditionRecord, SupervisionRecord

PROTOCOL = "pilot_b_v1"
SEED = 20261015
QUOTAS = {"iper": {"train": 512, "val": 24, "test": 32},
          "tiktok": {"train": 512, "val": 24, "test": 32},
          "hi4d": {"train": 512, "val": 24, "test": 32},
          "3dpw": {"train": 0, "val": 0, "test": 32}}
SINGLE_PROMPT = "Change the person in image 1 to the pose shown in image 2. Preserve the person's identity, clothing, and background."
DUAL_PROMPT = "Move the person on the left in image 1 to the pose in image 2, and the person on the right in image 1 to the pose in image 3. Preserve each person's identity and clothing and preserve the background. Keep the two people distinct."


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_image(path: str | Path) -> str:
    """Canonical decoded pixels: PNG metadata/compression cannot evade dedup."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def _stable(value: str) -> str:
    return hashlib.sha256(("union_b_v1:" + value).encode()).hexdigest()


def make_sample_id(dataset, split, source, target, *, camera_id="", person_order=(0,)):
    payload = [dataset, split, Path(source).as_posix(), Path(target).as_posix(), str(camera_id), list(person_order)]
    return f"{dataset}_{split}_{hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()[:24]}"


def _path(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _frame(value) -> int:
    if value is None:
        raise ValueError("missing target frame number")
    match = re.search(r"(\d+)(?:\.[^.]+)?$", str(value))
    if not match:
        raise ValueError(f"cannot parse frame {value!r}")
    return int(match.group(1))


def _manifests(root: Path, dataset: str, splits):
    for split in splits:
        if dataset == "iper":
            base = root / "datasets/iPER/iper_sampled_6src64tgt"
            file = base / "splits" / f"{dict(train='dev_train', val='dev_val', test='test')[split]}_pairs.jsonl"
        elif dataset == "tiktok":
            base = root
            file = root / "datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1" / f"{split}_pairs.jsonl"
        elif dataset == "hi4d":
            base = root
            file = root / "datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests" / f"main_pairs_{split}.jsonl"
        elif dataset == "3dpw":
            base = root / "datasets/3dpw_single_final_v2"
            file = base / f"{'validation' if split == 'val' else split}_pairs.csv"
        else:
            raise ValueError(f"unknown dataset {dataset}")
        yield split, base, file


def read_records(project_root: Path, *, datasets=("iper", "tiktok", "hi4d", "3dpw"), splits=("train", "val", "test")) -> list[dict]:
    """Normalize manifest fields without opening RGB, masks or pose files."""
    root = Path(project_root).resolve()
    records = []
    for dataset in datasets:
        for split, base, manifest in _manifests(root, dataset, splits):
            with manifest.open(encoding="utf-8") as stream:
                raw_rows = list(csv.DictReader(stream)) if dataset == "3dpw" else [json.loads(line) for line in stream if line.strip()]
            for raw in raw_rows:
                embedded_split = raw.get("split", split)
                if {"validation": "val"}.get(embedded_split, embedded_split) != split:
                    raise ValueError(f"split mismatch: manifest {manifest} declares {split}, row declares {embedded_split}")
                row = {"dataset": dataset, "split": split, "manifest": manifest,
                       "target_masks": (), "source_masks": (), "source_keypoints": (), "target_keypoints": (),
                       "person_order": (0,), "metadata": {}}
                if dataset == "iper":
                    appearance = str(raw["appearance"])
                    source, target = raw["source"], raw["target"]
                    home = base / appearance
                    row.update(group_id=appearance, sequence_id=appearance,
                               source_rgb=_path(home, source["rgb_1024"]), target_rgb=_path(home, target["rgb_1024"]),
                               control_image=_path(home, target["dwpose_512"]),
                               target_keypoints=(_path(home, target["keypoints"]),),
                               target_frame=int(target["frame_index0"]), source_frame=int(source["frame_index0"]),
                               source_kind="motion" if "motion" in f"{source.get('role', '')} {source.get('source_type', '')}" else "standard",
                               provenance="iPER DWPose-134 keypoints_134_px; existing native assets")
                    if source.get("keypoints"):
                        row["source_keypoints"] = (_path(home, source["keypoints"]),)
                elif dataset == "tiktok":
                    row.update(group_id=str(raw["source_video_group"]), sequence_id=str(raw["sequence_id"]),
                               source_rgb=_path(base, raw["source_rgb"]), target_rgb=_path(base, raw["target_rgb"]),
                               control_image=_path(base, raw["target_dwpose"]),
                               target_keypoints=(_path(base, raw["target_dwpose_json"]),),
                               target_frame=_frame(raw["target_stem"]), source_frame=_frame(raw["source_stem"]),
                               provenance="TikTok pairs_strict_v1 DWPose-134 normalized; existing native assets")
                    if raw.get("source_dwpose_json"):
                        row["source_keypoints"] = (_path(base, raw["source_dwpose_json"]),)
                    if raw.get("source_person_mask"):
                        row["source_masks"] = (_path(base, raw["source_person_mask"]),)
                    if raw.get("target_person_mask"):
                        row["target_masks"] = (_path(base, raw["target_person_mask"]),)
                elif dataset == "hi4d":
                    row.update(group_id=str(raw["pair"]), sequence_id=f"{raw['pair']}/{raw['target_action']}/cam{raw['camera_id']}",
                               source_rgb=_path(base, raw["source_rgb"]), target_rgb=_path(base, raw["target_rgb"]),
                               control_image=None, person_order=(0, 1),
                               source_masks=tuple(_path(base, raw[f"source_person_mask_{side}"]) for side in ("A", "B")),
                               target_keypoints=tuple(_path(base, raw[f"target_dwpose_{side}"]) for side in ("A", "B")),
                               target_masks=tuple(_path(base, raw[f"target_person_mask_{side}"]) for side in ("A", "B") if raw.get(f"target_person_mask_{side}")),
                               target_frame=int(raw["target_frame_id"]), source_frame=int(raw["source_frame_id"]),
                               camera_id=str(raw["camera_id"]), target_action=raw["target_action"],
                               same_action=raw["source_action"] == raw["target_action"],
                               contact_count=int(raw.get("target_contact_vertex_count_total", 0)),
                               provenance="Hi4D_mask_bbox_DWPose_official; existing assets used target mask upstream")
                    row["metadata"] = {"action_family": re.sub(r"\d+$", "", raw["target_action"]),
                                       "target_contact_label": bool(raw.get("target_contact", False)),
                                       "upstream_target_mask_assisted": True,
                                       "contact_count": row["contact_count"]}
                else:
                    row.update(group_id=str(raw["sequence"]), sequence_id=str(raw["sequence"]),
                               source_rgb=_path(base, raw["source_rgb"]), target_rgb=_path(base, raw["target_rgb"]),
                               control_image=_path(base, raw["target_dwpose_render"]),
                               target_frame=int(raw["target_frame_index"]), source_frame=int(raw["source_frame_index"]),
                               pose_bin=raw.get("pose_bin", "unknown"),
                               provenance="3DPW official SMPL camera Pose render; evaluation keypoints require frozen GT RGB detector")
                    if raw.get("target_person_mask"):
                        row["target_masks"] = (_path(base, raw["target_person_mask"]),)
                row["sample_id"] = make_sample_id(dataset, split, _relative(row["source_rgb"], root),
                                                   _relative(row["target_rgb"], root), camera_id=row.get("camera_id", ""),
                                                   person_order=row["person_order"])
                records.append(row)
    return records


def audit_records(records: Iterable[dict]) -> dict:
    groups = defaultdict(lambda: defaultdict(set))
    counts = Counter()
    seen = set()
    for row in records:
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id {row['sample_id']}")
        seen.add(row["sample_id"])
        groups[row["dataset"]][row["split"]].add(row["group_id"])
        counts[(row["dataset"], row["split"])] += 1
    intersections = {}
    for dataset, splits in groups.items():
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(splits[left] & splits[right])
            intersections[f"{dataset}:{left}:{right}"] = overlap
            if overlap:
                raise ValueError(f"group overlap in {dataset} {left}/{right}: {overlap}")
    return {"counts": {f"{d}:{s}": value for (d, s), value in sorted(counts.items())},
            "groups": {d: {s: sorted(v) for s, v in splits.items()} for d, splits in groups.items()},
            "group_intersections": intersections}


def _stratum(row):
    dataset = row["dataset"]
    if dataset == "iper":
        return row.get("source_kind", "standard")
    if dataset == "tiktok":
        return row["sequence_id"]
    if dataset == "hi4d":
        return (row.get("metadata", {}).get("action_family", "unknown"),
                "same" if row.get("same_action") else "cross",
                "zero" if row.get("contact_count", 0) == 0 else "positive")
    return row.get("pose_bin", "unknown")


def _round_robin(rows):
    pools = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: _stable(r["sample_id"])):
        pools[row["group_id"]][str(_stratum(row))].append(row)
    queues = {}
    for group in sorted(pools, key=_stable):
        children = [deque(pools[group][key]) for key in sorted(pools[group], key=_stable)]
        values = []
        while any(children):
            for queue in children:
                if queue:
                    values.append(queue.popleft())
        queues[group] = deque(values)
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                yield queue.popleft()


def select_pilot(records: Iterable[dict], *, quotas=None):
    """Group-balanced sampling, exact quotas, target dedup and fixed spacing.

    The complete candidate pass is deterministic regardless of manifest order.
    Image hashes use decoded RGB pixels; no mutable mtime/Python hash is used.
    """
    records = list(records)
    report = audit_records(records)
    quotas = QUOTAS if quotas is None else quotas
    selected, excluded, candidate_counts = [], Counter(), {}
    exclusions = []
    hash_cache, existence_cache, canonical_cache, parent_cache = {}, {}, {}, {}
    def canonical(path):
        path = Path(path)
        if path not in canonical_cache:
            if path.is_symlink():
                canonical_cache[path] = str(path.resolve())
            else:
                if path.parent not in parent_cache:
                    parent_cache[path.parent] = path.parent.resolve()
                canonical_cache[path] = str(parent_cache[path.parent] / path.name)
        return canonical_cache[path]
    for dataset, splits in quotas.items():
        for split, quota in splits.items():
            if not quota:
                continue
            rows = [row for row in records if row["dataset"] == dataset and row["split"] == split]
            targets = defaultdict(list)
            for row in sorted(rows, key=lambda r: _stable(r["sample_id"])):
                required = [row["source_rgb"], row["target_rgb"], *row.get("source_keypoints", ()), *row.get("target_keypoints", ()), *row.get("source_masks", ()), *row.get("target_masks", ())]
                if row.get("control_image"):
                    required.append(row["control_image"])
                missing = []
                for path in required:
                    path = Path(path)
                    if path not in existence_cache:
                        existence_cache[path] = path.is_file()
                    if not existence_cache[path]:
                        missing.append(str(path))
                if missing:
                    excluded["missing_asset"] += 1
                    exclusions.append({"sample_id": row["sample_id"], "reason": "missing_asset", "paths": missing})
                    continue
                if dataset == "3dpw" and abs(row["target_frame"] - row["source_frame"]) < 8:
                    excluded["source_target_spacing"] += 1
                    exclusions.append({"sample_id": row["sample_id"], "reason": "source_target_spacing"})
                    continue
                key = (row["group_id"], row.get("target_action"), row.get("camera_id"), row["target_frame"]) if dataset == "hi4d" else canonical(row["target_rgb"])
                targets[key].append(row)
            representatives = []
            source_counts = Counter()
            for key in sorted(targets, key=lambda k: _stable(str(k))):
                variants = targets[key]
                def source_priority(row):
                    kind = str(_stratum(row))
                    weight = 2 if dataset == "hi4d" and not row.get("same_action") else 1
                    return source_counts[(row["group_id"], kind)] / weight, _stable(row["sample_id"])
                chosen = min(variants, key=source_priority)
                source_counts[(chosen["group_id"], str(_stratum(chosen)))] += 1
                representatives.append(chosen)
                excluded["duplicate_target_path"] += len(variants) - 1
                exclusions.extend({"sample_id": row["sample_id"], "reason": "duplicate_target_path", "retained": chosen["sample_id"]}
                                  for row in variants if row is not chosen)
            # PIL decoding/hashlib release the GIL; four bounded readers avoid
            # serially decoding ~19k native targets, while map retains order.
            unhashed = sorted({canonical(row["target_rgb"]) for row in representatives} - hash_cache.keys())
            with ThreadPoolExecutor(max_workers=4) as pool:
                hash_cache.update(zip(unhashed, pool.map(sha256_image, unhashed)))
            seen_paths, seen_hashes, deduped = set(), set(), []
            for row in sorted(representatives, key=lambda r: _stable(r["sample_id"])):
                path = canonical(row["target_rgb"])
                if path in seen_paths:
                    excluded["duplicate_target_path"] += 1
                    exclusions.append({"sample_id": row["sample_id"], "reason": "duplicate_target_path"})
                    continue
                seen_paths.add(path)
                digest = hash_cache[path]
                if digest in seen_hashes:
                    excluded["duplicate_target_content"] += 1
                    exclusions.append({"sample_id": row["sample_id"], "reason": "duplicate_target_content", "target_content_sha256": digest})
                    continue
                seen_hashes.add(digest)
                deduped.append({**row, "target_content_sha256": digest})
            spaced, last_frame = [], {}
            for row in sorted(deduped, key=lambda r: (r["sequence_id"], r["target_frame"], _stable(r["sample_id"]))):
                sequence = row["sequence_id"]
                minimum = 2 if dataset == "hi4d" else 8
                if dataset != "3dpw" and row["target_frame"] - last_frame.get(sequence, -10**12) < minimum:
                    excluded["temporal_neighbor"] += 1
                    exclusions.append({"sample_id": row["sample_id"], "reason": "temporal_neighbor", "target_frame": row["target_frame"], "previous_retained_frame": last_frame[sequence]})
                    continue
                last_frame[sequence] = row["target_frame"]
                spaced.append(row)
            candidate_counts[f"{dataset}:{split}"] = len(spaced)
            if len(spaced) < quota:
                strata = Counter(f"{r['group_id']}:{_stratum(r)}" for r in spaced)
                raise ValueError(f"insufficient {dataset} {split}: requested={quota}, eligible={len(spaced)}, strata={dict(strata)}, excluded={dict(excluded)}")
            selected.extend(list(_round_robin(spaced))[:quota])
    report.update(seed=SEED, ordering="SHA256(union_b_v1:+sample_id)", quotas=quotas,
                  candidates=candidate_counts, excluded=dict(excluded), exclusions=exclusions,
                  selected_counts=dict(Counter(f"{r['dataset']}:{r['split']}" for r in selected)),
                  selected_strata=dict(Counter(f"{r['dataset']}:{r['split']}:{r['group_id']}:{_stratum(r)}" for r in selected)))
    return selected, report


def transform_keypoints(points, scores, transform: CanvasTransform) -> dict:
    points, scores = np.asarray(points, dtype=float), np.asarray(scores, dtype=float).reshape(-1)
    if points.shape != (len(scores), 2):
        raise ValueError("keypoints and scores shape mismatch")
    native_valid = np.isfinite(points).all(1)
    in_frame = native_valid & (points[:, 0] >= 0) & (points[:, 0] < transform.original_wh[0]) & (points[:, 1] >= 0) & (points[:, 1] < transform.original_wh[1])
    mapped = points * (np.asarray(transform.resized_wh) / transform.original_wh) + transform.offset_xy
    requested = np.isfinite(scores) & (scores >= .25)
    def coordinates(values):
        return [[float(x) if np.isfinite(x) else None for x in point] for point in values]
    return {"layout": "DWPose-134", "coordinate_system": "canvas512", "points": coordinates(mapped),
            "xy_native": coordinates(points), "xy_canvas": coordinates(mapped),
            "scores": [float(x) if np.isfinite(x) else None for x in scores],
            "score_threshold": .25, "requested": requested.tolist(), "in_frame": in_frame.tolist(),
            "valid": (requested & in_frame).tolist(), "transform": transform.to_dict()}


def _letterbox(image: Image.Image, transform: CanvasTransform, *, fill, nearest=False):
    result = Image.new(image.mode, transform.canvas_wh, fill)
    resized = image.resize(transform.resized_wh, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    result.paste(resized, transform.offset_xy)
    return result


def _keypoints(path, dataset, transform):
    if dataset == "tiktok":
        value = json.loads(Path(path).read_text())
        points = np.asarray(value["keypoints_norm"], dtype=float) * [value["image_width"], value["image_height"]]
        scores = value["scores"]
        size = (int(value["image_width"]), int(value["image_height"]))
    else:
        with np.load(path, allow_pickle=False) as value:
            if dataset == "iper":
                points, scores = value["keypoints_134_px"], value["scores_134"]
                size = (int(value["image_width"]), int(value["image_height"]))
            else:
                points, scores = value["xy_native"], value["scores"]
                size = (int(value["width"]), int(value["height"])) if "width" in value else transform.original_wh
    if len(points) != 134:
        raise ValueError(f"expected existing 134-point layout in {path}; refusing neck insertion")
    if size != transform.original_wh:
        raise ValueError(f"keypoint coordinate canvas {size} != target RGB {transform.original_wh}: {path}")
    return transform_keypoints(points, scores, transform)


def _person_order(mask_paths):
    centers = []
    for index, path in enumerate(mask_paths):
        with Image.open(path) as image:
            yy, xx = np.nonzero(np.asarray(image.convert("L")) > 127)
        if not len(xx):
            raise ValueError(f"empty Hi4D source mask {path}")
        centers.append((float(xx.mean()), float(yy.mean()), index))
    order = tuple(value[2] for value in sorted(centers))
    ambiguous = len(centers) > 1 and abs(centers[0][0] - centers[1][0]) < 1e-6
    return order, ambiguous, centers


def prepare_record(row: dict, workspace: Path):
    """Create RGB/Pose conditions independently of target RGB pixel content."""
    workspace = Path(workspace).resolve()
    if any(part.lower() in ("datasets", "models") for part in workspace.parts):
        raise ValueError("workspace is inside a protected source directory")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["sample_id"]) or row["sample_id"] in (".", ".."):
        raise ValueError("sample_id must be a safe single path component")
    destination = workspace / "data/assets" / row["sample_id"]
    if destination.exists():
        raise FileExistsError(f"prepared assets already exist: {destination}")
    # RGB header only: target pixels are not loaded by the conditioning path.
    with Image.open(row["target_rgb"]) as target:
        target_transform = CanvasTransform.from_size(target.size)
    with Image.open(row["source_rgb"]) as source:
        source_transform = CanvasTransform.from_size(source.size)
        source_image = _letterbox(source.convert("RGB"), source_transform, fill=(255, 255, 255))
    order, ambiguous, centers = tuple(row.get("person_order", (0,))), False, []
    if row["dataset"] == "hi4d":
        order, ambiguous, centers = _person_order(row["source_masks"])
    point_payloads = [_keypoints(path, row["dataset"], target_transform) for path in row.get("target_keypoints", ())]
    destination.mkdir(parents=True)
    source_path = destination / "source.png"
    source_image.save(source_path)
    control_path = destination / "control.png"
    poses = []
    control_transform = target_transform
    if row["dataset"] == "hi4d":
        from src.pose_edit.geometry import render_poseguide
        combined = np.zeros((512, 512, 3), dtype=np.uint8)
        for person in order:
            payload = point_payloads[person]
            points = np.array([[np.nan if x is None else x for x in p] for p in payload["xy_canvas"]])
            points[~np.asarray(payload["valid"])] = np.nan
            pose = render_poseguide(points, (512, 512))
            path = destination / f"pose_person_{person}.png"
            pose.save(path)
            poses.append(path)
            combined = np.maximum(combined, np.asarray(pose))
        Image.fromarray(combined).save(control_path)
    else:
        with Image.open(row["control_image"]) as pose:
            control_transform = CanvasTransform.from_size(pose.size)
            # Native provider coordinates are verified above; iPER's square
            # 512 pose is a proportional render of its square 1024 RGB.
            if abs(pose.width / pose.height - target_transform.original_wh[0] / target_transform.original_wh[1]) > .005:
                raise ValueError("pose/RGB aspect mismatch: refuse to guess existing letterbox metadata")
            _letterbox(pose.convert("RGB"), control_transform, fill=(0, 0, 0)).save(control_path)
        poses = [control_path]
    requested_paths = []
    for person in order:
        if not point_payloads:
            break
        payload = {**point_payloads[person], "person_index": person, "provenance": row["provenance"]}
        path = destination / f"requested_keypoints_person_{person}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        requested_paths.append(path)
    original_paths = [row["source_rgb"], *row.get("source_masks", ()), *row.get("source_keypoints", ()), *row.get("target_keypoints", ())]
    if row.get("control_image"):
        original_paths.append(row["control_image"])
    metadata = {**row.get("metadata", {}), "original_condition_assets": {str(p): sha256_file(p) for p in original_paths},
                "original_target_sha256": row.get("target_sha256") or sha256_file(row["target_rgb"]),
                "original_target_content_sha256": row.get("target_content_sha256") or sha256_image(row["target_rgb"]),
                "original_source_rgb": str(row["source_rgb"]),
                "original_source_masks": [str(row["source_masks"][person]) for person in order] if row.get("source_masks") else [],
                "control_transform": control_transform.to_dict(), "person_order_ambiguous": ambiguous,
                "source_mask_centers": centers, "sequence_id": row["sequence_id"], "target_frame": row["target_frame"]}
    condition = ConditionRecord(row["sample_id"], row["dataset"], row["split"], row["group_id"],
                                source_path, tuple(poses), control_path, DUAL_PROMPT if len(order) == 2 else SINGLE_PROMPT,
                                order, source_transform, target_transform, row["provenance"], metadata)
    masks = row.get("target_masks", ())
    masks = tuple(masks[i] for i in order) if len(masks) == len(order) else tuple(masks)
    supervision = SupervisionRecord(row["sample_id"], Path(row["target_rgb"]), masks, tuple(requested_paths),
                                    target_transform, order,
                                    {"target_sha256": metadata["original_target_sha256"],
                                     "keypoint_source": "GT RGB estimated pose; frozen detector required" if row["dataset"] == "3dpw" else row["provenance"],
                                     "target_keypoint_assets": [str(p) for p in row.get("target_keypoints", ())]})
    return condition, supervision


def _read_payloads(value):
    if isinstance(value, (str, Path)):
        return [json.loads(line) for line in Path(value).read_text().splitlines() if line.strip()]
    return list(value)


class UnionInferenceDataset:
    """Condition-only reader; neither construction nor indexing opens RGB."""
    def __init__(self, conditions):
        self.records = [row if isinstance(row, ConditionRecord) else ConditionRecord.from_dict(row) for row in _read_payloads(conditions)]
        ids = [row.sample_id for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate condition sample_id")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class UnionTrainingDataset(UnionInferenceDataset):
    def __init__(self, conditions, supervision):
        super().__init__(conditions)
        if any(row.split != "train" for row in self.records):
            raise ValueError("training dataset accepts only train conditions")
        supervision = [row if isinstance(row, SupervisionRecord) else SupervisionRecord.from_dict(row) for row in _read_payloads(supervision)]
        self.supervision = {row.sample_id: row for row in supervision}
        if len(self.supervision) != len(supervision) or set(self.supervision) != {row.sample_id for row in self.records}:
            raise ValueError("condition/supervision sample_id mismatch or duplicates")

    def __getitem__(self, index):
        condition = super().__getitem__(index)
        return {"condition": condition, "supervision": self.supervision[condition.sample_id]}


def _write_jsonl(path: Path, records):
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def _assert_workspace(project_root: Path, workspace: Path):
    for protected in ((project_root / "datasets").resolve(), (project_root / "models").resolve()):
        if workspace == protected or protected in workspace.parents:
            raise ValueError(f"workspace is inside protected source directory {protected}")
    if (workspace / "data").exists():
        raise FileExistsError(f"data workspace already exists: {workspace / 'data'}; choose a new workspace")


def _write_prepared_registry(data: Path) -> dict:
    """Bind prepared condition bytes to the immutable original-data lock."""
    destination = data / "prepared_assets_sha256.json"
    if destination.exists():
        raise FileExistsError(f"prepared asset registry already exists: {destination}")
    lock_path = data / "lock.json"
    lock = json.loads(lock_path.read_text())
    for name, expected in lock["selected_manifest_sha256"].items():
        if sha256_file(data / name) != expected:
            raise ValueError(f"selected manifest changed since data lock: {name}")
    assets, count = set(), 0
    for split in ("train", "val", "test"):
        for row in _read_payloads(data / f"{split}_conditions.jsonl"):
            count += 1
            assets.update([row["source_rgb"], row["control_image"], *row["reference_pose_images"]])
        for row in _read_payloads(data / f"{split}_supervision.jsonl"):
            assets.update(row["requested_keypoints"])
    paths = sorted(assets)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(zip(paths, pool.map(sha256_file, paths)))
    registry = {"protocol": PROTOCOL, "lock_sha256": sha256_file(lock_path),
                "condition_count": count, "asset_count": len(paths), "asset_sha256": hashes}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(registry, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return registry


def build_pilot(project_root: Path, workspace: Path) -> dict:
    project_root, workspace = Path(project_root).resolve(), Path(workspace).resolve()
    _assert_workspace(project_root, workspace)
    records = read_records(project_root)
    selected, audit = select_pilot(records)
    conditions, supervision = [], []
    for row in selected:
        condition, truth = prepare_record(row, workspace)
        conditions.append(condition)
        supervision.append(truth)
    data = workspace / "data"
    for split in ("train", "val", "test"):
        split_ids = {row.sample_id for row in conditions if row.split == split}
        _write_jsonl(data / f"{split}_conditions.jsonl", [row for row in conditions if row.split == split])
        _write_jsonl(data / f"{split}_supervision.jsonl", [row for row in supervision if row.sample_id in split_ids])
    overfit_ids = []
    for dataset in ("iper", "tiktok", "hi4d"):
        overfit_ids.extend([row.sample_id for row in conditions if row.dataset == dataset and row.split == "train"][:2])
    _write_jsonl(data / "overfit6_conditions.jsonl", [row for row in conditions if row.sample_id in overfit_ids])
    _write_jsonl(data / "overfit6_supervision.jsonl", [row for row in supervision if row.sample_id in overfit_ids])
    manifests = {str(path): sha256_file(path) for path in sorted({row["manifest"] for row in records})}
    assets = {}
    for row in selected:
        paths = [row["source_rgb"], row["target_rgb"], *row.get("source_keypoints", ()), *row.get("target_keypoints", ()), *row.get("source_masks", ()), *row.get("target_masks", ())]
        if row.get("control_image"):
            paths.append(row["control_image"])
        for path in paths:
            if str(path) not in assets:
                assets[str(path)] = sha256_file(path)
    try:
        commit = subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    lock = {"protocol": PROTOCOL, "seed": SEED, "audit": audit,
            "manifest_sha256": manifests, "required_asset_sha256": assets,
            "selected_manifest_sha256": {path.name: sha256_file(path) for path in sorted(data.glob("*.jsonl"))},
            "code": {"git_commit": commit, "dataset_sha256": sha256_file(__file__),
                     "contracts_sha256": sha256_file(Path(__file__).parents[1] / "pose_control/qwen21_union/contracts.py")},
            "model_revisions": {"videox_fun": "13bb3c625e8b1e457d7124d92207b49e00e4a6fe", "union": "8a4702014d4dabb5f896fcba917e2ee0a961465f"},
            "counts": dict(Counter(row.split for row in conditions)), "overfit6_sample_ids": overfit_ids,
            "supervision_policy": "target RGB remains original; train-only cache applies target_transform separately",
            "provenance": {dataset: sorted({row.provenance for row in conditions if row.dataset == dataset}) for dataset in QUOTAS}}
    (data / "lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_prepared_registry(data)
    return lock

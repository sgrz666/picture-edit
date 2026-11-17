#!/usr/bin/env python3
"""Batch SDPose-Wholebody skeleton + keypoint generation for a frame dataset.

Backend is the project-standard ``src/evaluation/sdpose`` entry
(see ``docs/SDPOSE_POSE_EVAL_SPEC.md``), so every fixed inference parameter is inherited
rather than re-declared here:

* YOLO11x person detection: ``conf=0.25``, ``iou=0.65``, ``classes=[0]``
* ``GetBBoxCenterScale(padding=1.25)`` + ``TopdownAffine(input_size=(768, 1024), use_udp=True)``
* ``timesteps=[999]``, ``mode="predict"``, ``test_cfg={"flip_test": True}``, weights in float32
* validity = finite AND inside ``[0, W) x [0, H)`` AND ``score >= 0.30``
* 134-slot ``geometry`` order (``14=R_eye, 15=R_ear, 16=L_eye, 17=L_ear``)

Outputs (default ``--out-root /data/shangguanrz/datasets/bilibili/geometry/skeleton``)::

    sdpose_2d/<bvid>/<fid>_skeleton.png        black-background 134-slot render, source resolution
    sdpose_2d_kp/<bvid>/<fid>_keypoints.json   points134 / scores / valid mask / detection stats
    sdpose_2d_kp/summary.json                  aggregate stats, rewritten after every video
    sdpose_2d_kp/run_manifest.json             run config, written once at start
    sdpose_2d_overlay/<bvid>/<fid>_overlay.jpg optional eyeball QC overlays (``--overlay-stride``)

Resume is on by default: a frame is skipped only when both its PNG and its JSON exist and the
JSON parses with a ``valid_count`` field. Re-running after an interruption therefore continues
where it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path("/home/shangguanrz/project/pic-edit")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.sdpose import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_YOLO_CONF,
    DEFAULT_YOLO_IOU,
    detect_boxes,
    load_runtime,
    pose_on_boxes,
)
from src.pose_edit.geometry import render_poseguide  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", default="/data/shangguanrz/datasets/bilibili/frames/images",
                    help="root containing <bvid>/<fid>.<ext> frame images")
    ap.add_argument("--out-root", default="/data/shangguanrz/datasets/bilibili/geometry/skeleton",
                    help="dataset skeleton root; sdpose_2d/ + sdpose_2d_kp/ are created under it")
    ap.add_argument("--bvid", action="append", default=None,
                    help="restrict to specific bvid(s); repeatable. Default: all subdirectories")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames total (0 = all)")
    ap.add_argument("--limit-per-video", type=int, default=0, help="max frames per bvid (0 = all)")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="drop YOLO boxes smaller than this area in px^2 (background pedestrians)")
    ap.add_argument("--device", default=None, help="torch device, default cuda:0 when available")
    ap.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                    help=f"keypoint validity threshold (project default {DEFAULT_SCORE_THRESHOLD})")
    ap.add_argument("--overlay-stride", type=int, default=0,
                    help="write an eyeball-QC overlay every N frames (0 = off)")
    ap.add_argument("--no-resume", action="store_true", help="recompute frames even if outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="enumerate frames and exit without inference")
    return ap.parse_args()


def iter_frames(images_root: Path, bvids: Iterable[str] | None, per_video: int) -> list[tuple[str, Path]]:
    """Return ``[(bvid, image_path), ...]`` in deterministic bvid/filename order."""
    if not images_root.is_dir():
        raise FileNotFoundError(f"images root not found: {images_root}")
    wanted = set(bvids) if bvids else None
    videos = sorted(p for p in images_root.iterdir() if p.is_dir())
    if wanted is not None:
        missing = sorted(wanted - {p.name for p in videos})
        if missing:
            raise FileNotFoundError(f"requested bvid(s) missing under {images_root}: {missing}")
        videos = [p for p in videos if p.name in wanted]
    frames: list[tuple[str, Path]] = []
    for vdir in videos:
        shots = sorted(p for p in vdir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if per_video > 0:
            shots = shots[:per_video]
        frames.extend((vdir.name, p) for p in shots)
    return frames


def to_json_points(arr: np.ndarray) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for x, y in np.asarray(arr, dtype=float):
        out.append([float(x), float(y)] if np.isfinite(x) and np.isfinite(y) else [None, None])
    return out


def select_box(boxes: np.ndarray, scores: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, bool]:
    """Pick the largest-area foreground box; fall back to the full frame when nothing is detected."""
    if len(boxes) == 0:
        return np.array([0.0, 0.0, float(width), float(height)], dtype=np.float32), 1.0, False
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = int(np.argmax(areas))
    return boxes[best].astype(np.float32), float(scores[best]), True


def write_overlay(image_bgr: np.ndarray, skeleton: Image.Image, path: Path) -> None:
    """Composite the black-background skeleton onto the source frame, black pixels transparent."""
    sk = np.asarray(skeleton.convert("RGB"), dtype=np.uint8)
    mask = sk.any(axis=2)
    out = image_bgr.copy()
    if mask.any():
        out[mask] = sk[mask][:, ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def already_done(png: Path, js: Path) -> bool:
    if not (png.is_file() and js.is_file()):
        return False
    try:
        with open(js, "r", encoding="utf-8") as f:
            return "valid_count" in json.load(f)
    except (json.JSONDecodeError, OSError):
        return False


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    skel_dir = out_root / "sdpose_2d"
    kp_dir = out_root / "sdpose_2d_kp"
    overlay_dir = out_root / "sdpose_2d_overlay"

    frames = iter_frames(Path(args.images_root), args.bvid, args.limit_per_video)
    if args.limit > 0:
        frames = frames[: args.limit]
    if not frames:
        print("[sdpose] no frames matched", flush=True)
        return 1
    n_videos = len({b for b, _ in frames})
    print(f"[sdpose] {len(frames)} frames across {n_videos} videos | out={out_root}", flush=True)
    if args.dry_run:
        for bvid, path in frames[:5]:
            print(f"[sdpose] dry-run sample: {bvid}/{path.name}", flush=True)
        return 0

    skel_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    runtime = load_runtime(device=args.device)
    import torch

    print(
        f"[sdpose] runtime loaded in {time.time() - t_load:.1f}s | device={runtime.device} | "
        f"gpu_alloc={torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB",
        flush=True,
    )

    manifest = {
        "pose_backend": "sdpose",
        "backend_module": "src/evaluation/sdpose.py",
        "slot_order": "geometry",
        "yolo": {"weights": "yolo11x.pt", "conf": DEFAULT_YOLO_CONF, "iou": DEFAULT_YOLO_IOU, "classes": [0]},
        "bbox_padding": 1.25,
        "input_size": [768, 1024],
        "timestep": 999,
        "flip_test": True,
        "score_threshold": args.score_threshold,
        "min_area": args.min_area,
        "images_root": str(Path(args.images_root)),
        "out_root": str(out_root),
        "n_frames": len(frames),
        "n_videos": n_videos,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(kp_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    done = skipped = failed = 0
    valid_counts: list[int] = []
    per_video: dict[str, dict[str, Any]] = {}
    elapsed_by_bvid: dict[str, float] = {}

    for i, (bvid, img_path) in enumerate(frames, 1):
        fid = img_path.stem
        png_out = skel_dir / bvid / f"{fid}_skeleton.png"
        json_out = kp_dir / bvid / f"{fid}_keypoints.json"

        if not args.no_resume and already_done(png_out, json_out):
            skipped += 1
            continue

        t_frame = time.time()
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[sdpose][FAIL] cannot read {img_path}", flush=True)
            failed += 1
            continue
        h, w = bgr.shape[:2]

        boxes, box_scores, raw_count = detect_boxes(
            bgr, runtime, conf=DEFAULT_YOLO_CONF, iou=DEFAULT_YOLO_IOU, min_area=args.min_area
        )
        bbox, bscore, det_found = select_box(boxes, box_scores, w, h)

        people = pose_on_boxes(bgr, [bbox], runtime, box_scores=[bscore], order="geometry", flip_test=True)
        if not people:
            print(f"[sdpose][FAIL] no pose output for {bvid}/{img_path.name}", flush=True)
            failed += 1
            continue

        pts134 = np.asarray(people[0]["points"], dtype=float)
        sc134 = np.asarray(people[0]["scores"], dtype=float)

        finite = np.isfinite(pts134).all(axis=1)
        in_canvas = np.zeros(134, dtype=bool)
        in_canvas[finite] = (
            (pts134[finite, 0] >= 0)
            & (pts134[finite, 0] < w)
            & (pts134[finite, 1] >= 0)
            & (pts134[finite, 1] < h)
        )
        valid = finite & in_canvas & (sc134 >= args.score_threshold)

        pts_render = np.full((134, 2), np.nan, dtype=float)
        pts_render[valid] = pts134[valid]
        skeleton = render_poseguide(pts_render, (w, h))
        png_out.parent.mkdir(parents=True, exist_ok=True)
        skeleton.save(png_out, format="PNG", compress_level=1)

        if args.overlay_stride > 0 and (done % args.overlay_stride == 0):
            write_overlay(bgr, skeleton, overlay_dir / bvid / f"{fid}_overlay.jpg")

        payload = {
            "pose_backend": "sdpose",
            "slot_order": "geometry",
            "bvid": bvid,
            "frame_id": fid,
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "score_threshold": args.score_threshold,
            "det_person_found": bool(det_found),
            "raw_persons_detected": int(raw_count),
            "fg_persons_detected": int(len(boxes)),
            "bbox_xyxy": [float(v) for v in bbox.tolist()],
            "bbox_score": float(bscore),
            "points134": to_json_points(pts_render),
            "points134_raw": to_json_points(pts134),
            "scores134": [float(v) for v in np.where(valid, sc134, 0.0)],
            "scores134_raw": [float(v) for v in sc134],
            "valid_mask134": [bool(v) for v in valid],
            "valid_count": int(valid.sum()),
            "skeleton_png": str(png_out),
            "elapsed_sec": round(time.time() - t_frame, 3),
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        done += 1
        valid_counts.append(int(valid.sum()))
        stats = per_video.setdefault(bvid, {"frames": 0, "valid_sum": 0, "valid_min": 134, "no_det": 0})
        stats["frames"] += 1
        stats["valid_sum"] += int(valid.sum())
        stats["valid_min"] = min(stats["valid_min"], int(valid.sum()))
        stats["no_det"] += 0 if det_found else 1
        elapsed_by_bvid[bvid] = elapsed_by_bvid.get(bvid, 0.0) + (time.time() - t_frame)

        if done % 25 == 0 or i == len(frames):
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(frames) - i) / max(rate, 1e-6) / 60.0
            print(
                f"[sdpose] {i}/{len(frames)} | new={done} skip={skipped} fail={failed} | "
                f"{rate:.2f} f/s | eta={eta:.1f} min",
                flush=True,
            )
            summary = {
                "pose_backend": "sdpose",
                "score_threshold": args.score_threshold,
                "total_frames": len(frames),
                "processed": done,
                "skipped_existing": skipped,
                "failed": failed,
                "valid_count_mean": float(np.mean(valid_counts)) if valid_counts else None,
                "valid_count_min": int(np.min(valid_counts)) if valid_counts else None,
                "valid_count_max": int(np.max(valid_counts)) if valid_counts else None,
                "valid_count_ge_100": int(sum(1 for v in valid_counts if v >= 100)),
                "no_detection_frames": int(sum(s["no_det"] for s in per_video.values())),
                "elapsed_sec": round(time.time() - t0, 1),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "per_video": {
                    k: {
                        "frames": v["frames"],
                        "valid_mean": round(v["valid_sum"] / v["frames"], 2),
                        "valid_min": v["valid_min"],
                        "no_detection": v["no_det"],
                        "elapsed_sec": round(elapsed_by_bvid.get(k, 0.0), 1),
                    }
                    for k, v in sorted(per_video.items())
                },
            }
            with open(kp_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    total = time.time() - t0
    print(
        f"[sdpose] DONE in {total / 60:.1f} min | new={done} skipped={skipped} failed={failed} | "
        f"valid_mean={np.mean(valid_counts) if valid_counts else 'n/a'}",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

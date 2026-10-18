#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm


ROOT = Path("/home/shangguanrz/project/pic-edit")

DWROOT = (
    ROOT
    / "add/tools/dwpose_mimicmotion_official"
)

DET = DWROOT / "models/yolox_l.onnx"
POSE = DWROOT / "models/dw-ll_ucoco_384.onnx"

sys.path.insert(0, str(DWROOT))

from dwpose.wholebody import Wholebody
from dwpose.util import (
    draw_bodypose,
    draw_handpose,
    draw_facepose,
)


DEFAULT_MANIFEST = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "candidate_pool_geometry_valid.jsonl"
)

OUT_ROOT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/dwpose_v1"
)

OUT_MANIFEST = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "dwpose_results.jsonl"
)

OUT_SUMMARY = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/"
      "dwpose_summary.json"
)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def select_person(keypoints, scores):
    """
    多人时，选择 body mean confidence 最高者。
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)

    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]

    if scores.ndim != 2:
        raise RuntimeError(f"unexpected score shape: {scores.shape}")

    body_conf = scores[:, :18].mean(axis=1)
    idx = int(np.argmax(body_conf))

    return (
        idx,
        keypoints[idx].copy(),
        scores[idx].copy(),
        float(body_conf[idx]),
        int(keypoints.shape[0]),
    )


def normalize_points(
    keypoints,
    width,
    height,
):
    kp = keypoints.astype(np.float32).copy()
    kp[:, 0] /= float(width)
    kp[:, 1] /= float(height)
    return kp


def build_official_pose(
    kp_norm,
    scores,
    H,
    W,
):
    """
    复用你们之前成熟脚本的 wholebody 画法：
    body + hands + face，黑底。
    """
    body = kp_norm[:18]
    body_score = scores[:18]

    face = kp_norm[24:92][None]
    face_score = scores[24:92][None]

    hands = np.stack(
        [
            kp_norm[92:113],
            kp_norm[113:134],
        ],
        axis=0,
    )

    hands_score = np.stack(
        [
            scores[92:113],
            scores[113:134],
        ],
        axis=0,
    )

    ref_w = 2160
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1.0

    canvas = np.zeros(
        (
            int(H * sr),
            int(W * sr),
            3,
        ),
        dtype=np.uint8,
    )

    subset = np.full(
        (1, 18),
        -1,
        dtype=np.float32,
    )

    for j in range(18):
        if body_score[j] > 0.3:
            subset[0, j] = j

    canvas = draw_bodypose(
        canvas,
        body,
        subset,
        score=body_score[None],
    )

    canvas = draw_handpose(
        canvas,
        hands,
        hands_score,
    )

    canvas = draw_facepose(
        canvas,
        face,
        face_score,
    )

    canvas = cv2.resize(
        canvas,
        (W, H),
        interpolation=cv2.INTER_LINEAR,
    )

    return canvas


def build_record(
    row,
    status,
    reason,
    out_png,
    out_json,
    body_mean_conf=None,
    body_valid_03=None,
    hand_mean_conf=None,
    face_mean_conf=None,
    detected_persons=None,
):
    return {
        "sequence_id": row["sequence_id"],
        "stem": row["stem"],
        "rgb": row["rgb"],
        "status": status,
        "reason": reason,
        "dwpose_png": str(out_png) if out_png is not None else None,
        "dwpose_json": str(out_json) if out_json is not None else None,
        "body_mean_conf": body_mean_conf,
        "body_valid_03": body_valid_03,
        "hand_mean_conf": hand_mean_conf,
        "face_mean_conf": face_mean_conf,
        "detected_persons": detected_persons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    if not DET.exists():
        raise RuntimeError(f"missing detector: {DET}")

    if not POSE.exists():
        raise RuntimeError(f"missing pose model: {POSE}")

    rows = load_rows(manifest_path)

    if args.limit is not None:
        rows = rows[:args.limit]

    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_MANIFEST.parent)

    providers = ort.get_available_providers()

    print("=" * 72, flush=True)
    print("TikTok V3 DWPose WHOLEBODY", flush=True)
    print("=" * 72, flush=True)
    print("manifest:", manifest_path, flush=True)
    print("input rows:", len(rows), flush=True)
    print("providers:", providers, flush=True)

    device = (
        "cuda"
        if "CUDAExecutionProvider" in providers
        else "cpu"
    )

    print("device:", device, flush=True)

    estimator = Wholebody(
        str(DET),
        str(POSE),
        device=device,
    )

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for i, row in enumerate(tqdm(rows), start=1):
        seq = row["sequence_id"]
        stem = row["stem"]

        rgb_path = Path(row["rgb"])

        seq_root = OUT_ROOT / seq
        out_png = seq_root / "dwpose" / f"{stem}.png"
        out_json = seq_root / "dwpose_json" / f"{stem}.json"

        ensure_dir(out_png.parent)
        ensure_dir(out_json.parent)

        if (
            not args.overwrite
            and out_png.is_file()
            and out_json.is_file()
        ):
            skip_count += 1
            continue

        try:
            bgr = read_rgb(rgb_path)
            H, W = bgr.shape[:2]

            result = estimator(bgr)

            if (
                not isinstance(result, (tuple, list))
                or len(result) != 2
            ):
                raise RuntimeError(
                    f"unexpected DWPose output type: {type(result)}"
                )

            keypoints, scores = result

            chosen = select_person(
                keypoints,
                scores,
            )

            if chosen is None:
                record = build_record(
                    row=row,
                    status="FAIL",
                    reason="NO_PERSON_DETECTED",
                    out_png=None,
                    out_json=out_json,
                )

                out_json.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                fail_count += 1
                continue

            idx, kp, sc, body_mean, detected_n = chosen

            kp_norm = normalize_points(
                kp,
                W,
                H,
            )

            pose_img = build_official_pose(
                kp_norm,
                sc,
                H,
                W,
            )

            Image.fromarray(
                cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
            ).save(out_png)

            body_valid_03 = int((sc[:18] > 0.3).sum())
            hand_mean_conf = float(sc[92:134].mean())
            face_mean_conf = float(sc[24:92].mean())

            record = build_record(
                row=row,
                status="PASS",
                reason="OK",
                out_png=out_png,
                out_json=out_json,
                body_mean_conf=float(body_mean),
                body_valid_03=body_valid_03,
                hand_mean_conf=hand_mean_conf,
                face_mean_conf=face_mean_conf,
                detected_persons=detected_n,
            )

            record["image_width"] = int(W)
            record["image_height"] = int(H)
            record["selected_person_index"] = int(idx)
            record["keypoints_norm"] = kp_norm.tolist()
            record["scores"] = sc.astype(np.float32).tolist()

            out_json.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            pass_count += 1

        except Exception as e:
            record = build_record(
                row=row,
                status="FAIL",
                reason=str(e),
                out_png=None,
                out_json=out_json,
            )

            out_json.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            fail_count += 1

        if i % 500 == 0 or i == len(rows):
            print(
                f"[{i}/{len(rows)}] pass={pass_count} fail={fail_count} skip={skip_count}",
                flush=True,
            )

    # 汇总当前 manifest 对应样本的结果
    final_records = []

    for row in rows:
        seq = row["sequence_id"]
        stem = row["stem"]
        out_json = OUT_ROOT / seq / "dwpose_json" / f"{stem}.json"

        if out_json.is_file():
            final_records.append(
                json.loads(
                    out_json.read_text(encoding="utf-8")
                )
            )
        else:
            final_records.append(
                build_record(
                    row=row,
                    status="FAIL",
                    reason="MISSING_OUTPUT_JSON",
                    out_png=None,
                    out_json=None,
                )
            )

    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = sum(r["status"] == "PASS" for r in final_records)
    n_fail = sum(r["status"] != "PASS" for r in final_records)

    summary = {
        "version": "tiktok_v3_dwpose_v1",
        "input_manifest": str(manifest_path),
        "input_rows": len(rows),
        "pass": n_pass,
        "fail": n_fail,
        "skip_existing_outputs": skip_count,
        "dwpose_root": str(OUT_ROOT),
        "result_manifest": str(OUT_MANIFEST),
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72, flush=True)
    print("DWPose COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

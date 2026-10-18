#!/usr/bin/env python3
"""Convert an existing face_supervision.json (letterbox or resize) to native_pad_v1."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.face_supervision import detection_record, evaluate_detections, SCHEMA


def convert(input_path: Path, output_path: Path, target_stems: set[str] | None = None) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    thresholds = data.get("thresholds", {})
    score_threshold = thresholds.get("min_score", 0.5)
    iou_threshold = thresholds.get("min_geometry_iou", 0.2)
    min_output_short_side = thresholds.get("min_short_side_at_actual_output_px", 32.0)

    records = data.get("records", {})
    new_records = {}

    for key, rec in records.items():
        stem = rec.get("stem") or rec.get("target_stem")
        if target_stems is not None and stem not in target_stems:
            continue

        width = rec["original_width"]
        height = rec["original_height"]
        image_hw = (height, width)
        geometry_box = rec["geometry_box_original_px"]
        geometry_valid = rec.get("geometry_valid", True)

        candidates = []
        for det in rec.get("detections", []):
            box = [*det["bbox_original_px"], det["score"]]
            kps = det.get("landmarks5_original_px")
            cand = detection_record(
                box, kps, geometry_box, image_hw,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                align_mode="native"
            )
            candidates.append(cand)

        eligibility = evaluate_detections(
            candidates,
            geometry_valid=geometry_valid,
            min_output_short_side=min_output_short_side
        )

        new_rec = dict(rec)
        new_rec["detections"] = candidates
        new_rec["by_resolution"] = eligibility
        new_records[key] = new_rec

    summary_keys = ["512", "1024", "native"]
    summary = {
        str(res): dict(
            total=len(new_records),
            valid=sum(r["by_resolution"].get(str(res), {}).get("valid", False) for r in new_records.values()),
            reasons=dict(Counter(r["by_resolution"].get(str(res), {}).get("reason", "unknown") for r in new_records.values()))
        )
        for res in summary_keys
    }

    out_payload = dict(
        schema=SCHEMA,
        align_mode="native",
        purpose="Target-RGB eligibility for offline ID/perceptual losses only (native resolution)",
        prohibited_uses=data.get("prohibited_uses", []),
        source_validity=data.get("source_validity", ""),
        detector=data.get("detector", {}),
        thresholds=thresholds,
        visibility_limit=data.get("visibility_limit", ""),
        pair_manifest_sha256=data.get("pair_manifest_sha256", {}),
        records=new_records,
        summary=summary,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(output_path)
    return out_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", nargs="*", type=Path, help="Optional pair JSONLs to filter target stems")
    args = parser.parse_args()

    target_stems = None
    if args.pairs:
        target_stems = set()
        for p in args.pairs:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    target_stems.add(item.get("target_stem"))

    print(f"Converting face supervision to native. Filtering stems: {len(target_stems) if target_stems else 'All'}")
    payload = convert(args.input, args.output, target_stems=target_stems)
    print(f"Saved {len(payload['records'])} records to {args.output}")
    print("Summary:", json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

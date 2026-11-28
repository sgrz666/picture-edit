#!/usr/bin/env python3
"""Score subject-disjoint LoRA comparisons and export an anonymous anatomy review set."""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image

from dataset_contract import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / "src").is_dir():
    sys.path.insert(0, str(REPO_ROOT))


def reference_joints(hands):
    if len(hands) != 2 or any(len(hand) != 21 for hand in hands):
        raise ValueError("Expected left/right 21-joint geometry")
    points = np.full((134, 2), np.nan)
    for offset, joints in zip((92, 113), hands):
        for index, xy in enumerate(joints):
            if xy is not None:
                points[offset + index] = xy
    valid = np.isfinite(points).all(1) & (points >= 0).all(1) & (points < 512).all(1)
    return points, valid


def hand_score(target, visible, pred, pred_valid):
    from src.pose_control.qwen21_full.region_rewards import LEFT21, RIGHT21, pose_metrics

    existing = pose_metrics(target, visible, pred, pred_valid, (512, 512))
    scores = {}
    for side, ids in (("left", LEFT21), ("right", RIGHT21)):
        if not visible[ids[0]] or visible[ids].sum() < 8:
            continue
        wanted = np.array(ids)[visible[ids]]
        scale = np.linalg.norm(np.ptp(target[wanted], axis=0))
        if scale < 8:
            continue
        matched = np.zeros(len(wanted), dtype=bool) if pred is None else pred_valid[wanted]
        pck = 0.
        if matched.any():
            distances = np.linalg.norm(pred[wanted[matched]] - target[wanted[matched]], axis=1)
            pck = float((distances <= .10 * scale).sum() / len(wanted))
        scores[side] = {
            "points": len(wanted), "matched": int(matched.sum()), "hand_pck_010": pck,
            "wrist_relative": existing[side + "_hand"]["gesture_reward"],
            "hand_detected": bool(pred is not None and pred_valid[ids[0]] and matched.sum() >= 8),
        }
    return scores


def mean_or_none(items):
    values = [item for item in items if item is not None]
    return float(np.mean(values)) if values else None


def photo_metrics(source, generated, masks, root):
    if masks is None:
        return {"background_mae_0_255": None, "skin_rgb_hist_similarity": None}

    def read(name):
        asset = masks[name]
        path = root / asset["path"]
        if file_sha256(path) != asset["sha256"]:
            raise ValueError(f"Evaluation mask SHA256 changed: {path}")
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
        if mask.shape != (512, 512):
            raise ValueError(f"Expected an aligned evaluation mask: {path}")
        return mask >= 128

    def histogram(pixels):
        return np.array([np.histogram(values, bins=16, range=(0, 256))[0] / len(pixels) for values in pixels.T])

    a = np.asarray(source, np.float32)
    result = np.asarray(generated, np.float32)
    bg = ~(read("source_foreground") | read("target_foreground"))
    skin_a, skin_b = read("source_skin"), read("target_skin")
    return {
        "background_mae_0_255": float(np.abs(result[bg] - a[bg]).mean()) if bg.any() else None,
        "skin_rgb_hist_similarity": float(1 - np.abs(histogram(result[skin_b]) - histogram(a[skin_a])).sum() / 6)
        if skin_a.any() and skin_b.any() else None,
    }


def aggregate_blind(key, reviews):
    by_image = {}
    fields = ("extra_finger", "missing_finger", "finger_fusion")
    for row in reviews:
        name = row["image"]
        if name not in key or name in by_image:
            raise ValueError(f"Unknown or repeated blind image: {name}")
        if any(row[field] is not None and type(row[field]) is not bool for field in (*fields, "occlusion_uncertain")):
            raise ValueError(f"Annotations must be true, false, or null: {name}")
        by_image[name] = row
    results = {}
    for arm in ("baseline", "lora"):
        images = [name for name in key if key[name]["arm"] == arm]
        labels = [by_image[name] for name in images if name in by_image]
        summary = {"images": len(images), "submitted": len(labels),
                   "occlusion_uncertain": sum(row["occlusion_uncertain"] is True for row in labels)}
        for field in fields:
            values = [row[field] for row in labels
                      if row["occlusion_uncertain"] is False and row[field] is not None]
            summary[field] = {"rated": len(values), "failures": sum(values), "rate": mean_or_none(values)}
        results[arm] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--comparison", type=Path, action="append")
    mode.add_argument("--marked-review", type=Path)
    parser.add_argument("--blind-key", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.marked_review:
        if args.blind_key is None:
            parser.error("--marked-review requires --blind-key")
        votes = [json.loads(line) for line in args.marked_review.read_text(encoding="utf-8").splitlines() if line]
        key = json.loads(args.blind_key.read_text(encoding="utf-8"))
        rates = aggregate_blind(key, votes)
        args.out.mkdir()
        with (args.out / "anatomy-rates.json").open("x", encoding="utf-8") as handle:
            json.dump({"blind_key_sha256": file_sha256(args.blind_key), "review_sha256": file_sha256(args.marked_review),
                       "summary": rates}, handle, indent=2)
        print(json.dumps(rates))
        return
    from src.evaluation import sdpose

    cases = []
    for root in args.comparison:
        receipt = json.loads((root / "verification.json").read_text(encoding="utf-8"))
        if receipt["split"] not in ("validation", "test"):
            raise ValueError(f"Independent evaluation requires a held-out split: {root}")
        plans = receipt["target_joints"]
        for name, digest in receipt["png_sha256"].items():
            if file_sha256(root / f"{name}.png") != digest:
                raise ValueError(f"Comparison PNG changed: {root}/{name}")
        cases.append((root, receipt, plans))
    if len({(r["split"], r["id"], r["seed"]) for _, r, _ in cases}) != len(cases):
        raise ValueError("Repeated held-out case and seed")
    for key in ("base_manifest_sha256", "comparison_adapter_sha256"):
        if len({r[key] for _, r, _ in cases}) != 1:
            raise ValueError(f"Independent comparisons must share {key}")

    evaluator = sdpose.scoring_identity()
    runtime = sdpose.load_scoring_runtime(device=args.device)
    reports, blind_rows = [], []
    for root, receipt, plans in cases:
        source = Image.open(root / "source.png").convert("RGB")
        masks_path = root / "regions.json"
        masks = json.loads(masks_path.read_text(encoding="utf-8")) if masks_path.is_file() else {}
        for name, joints in plans.items():
            target, visible = reference_joints(joints)
            suffix = "_swap" if name == "target_swap" else ""
            for arm in ("baseline", "lora"):
                filename = f"{arm}{suffix}"
                path = root / f"{filename}.png"
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                people = sdpose.score_people(np.asarray(rgb)[..., ::-1].copy(), runtime)
                person = people[0] if people else None
                pred = None if person is None else person["points"]
                validity = None if person is None else sdpose.valid_joints(pred, person["scores"], 512, 512)
                reports.append({
                    "case": receipt["id"], "group": receipt["group"], "target": name, "arm": arm,
                    "image_sha256": receipt["png_sha256"][filename], "person_detected": bool(people),
                    "hands": hand_score(target, visible, pred, validity),
                    **photo_metrics(source, rgb, masks.get(name), root),
                })
                blind_rows.append((path, receipt["id"], name, arm))

    args.out.mkdir()
    totals = {}
    for arm in ("baseline", "lora"):
        images = [row for row in reports if row["arm"] == arm]
        hands = [hand for row in images for hand in row["hands"].values()]
        totals[arm] = {"images": len(images), "person_detected": sum(row["person_detected"] for row in images),
                       "eligible_hands": len(hands)}
        totals[arm].update({key: mean_or_none(hand[key] for hand in hands)
                            for key in ("hand_detected", "hand_pck_010", "wrist_relative")})
        totals[arm].update({key: mean_or_none(row[key] for row in images)
                            for key in ("background_mae_0_255", "skin_rgb_hist_similarity")})
    # Keep the image-to-arm key outside the directory reviewers receive.
    random.Random(42).shuffle(blind_rows)
    blind = args.out / "blind"
    blind.mkdir()
    key = {}
    with (blind / "review.jsonl").open("x", encoding="utf-8") as handle:
        for number, (path, case, target, arm) in enumerate(blind_rows):
            name = f"{number + 1:05d}.png"
            with (blind / name).open("xb") as destination, path.open("rb") as original:
                shutil.copyfileobj(original, destination)
            key[name] = {"case": case, "target": target, "arm": arm, "sha256": file_sha256(path)}
            handle.write(json.dumps(dict(image=name, extra_finger=None, missing_finger=None,
                                         finger_fusion=None, occlusion_uncertain=None, notes="")) + "\n")
    with (args.out / "blind-key.json").open("x", encoding="utf-8") as handle:
        json.dump(key, handle, indent=2)
    result = {"pose_backend": "sdpose", "evaluator": evaluator, "evaluator_code_sha256": file_sha256(sdpose.__file__),
              "subjects": sorted({r["group"] for _, r, _ in cases}), "summary": totals, "images": reports,
              "anatomy_blind_labels": "pending", "contact_success": None}
    with (args.out / "scores.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()

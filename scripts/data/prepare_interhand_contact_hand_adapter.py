#!/usr/bin/env python3
"""Export InterHand2.6M frame pairs and target geometry for Contact Hand Adapter."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="Root containing subject.txt and split JSON directories")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--images", type=Path, help="Extracted RGB root containing train, val and test directories")
    inputs.add_argument("--archive-dir", type=Path, help="Official 5 fps .tar.part* directory")
    parser.add_argument("--allow-partial-archive", action="store_true", help="Use the existing contiguous archive prefix")
    parser.add_argument("--archive-max-images", type=int, help="Limit matching archive candidates for a small conversion")
    parser.add_argument("--out", type=Path, required=True, help="New export directory")
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--capture", help="Numeric capture ID without the Capture prefix")
    parser.add_argument("--camera", help="Camera ID without the cam prefix")
    parser.add_argument("--sequence")
    parser.add_argument("--hand-type", choices=("left", "right", "interacting"))
    parser.add_argument("--min-gap", type=int, default=12, help="Minimum native frame index gap")
    parser.add_argument("--max-gap", type=int, default=180, help="Maximum native frame index gap")
    parser.add_argument("--min-motion", type=float, default=8., help="Minimum RMS displacement in source pixels")
    parser.add_argument("--max-pairs", type=int, help="Maximum emitted pairs per requested split")
    parser.add_argument("--targets-per-source", type=int, default=1, help="Maximum eligible targets per source RGB")
    parser.add_argument("--hand-prior-quality", action="store_true", help="Screen articulation, sharpness and hand area")
    parser.add_argument("--quality-seed", type=int, default=42)
    parser.add_argument("--quality-max-candidates", type=int, help="Optional bounded screening pilot")
    parser.add_argument("--geometry-only", action="store_true", help="Keep target geometry without Adapter crops")
    parser.add_argument("--mano-model-dir", type=Path, help="Optional directory with MANO_LEFT.pkl and MANO_RIGHT.pkl")
    parser.add_argument("--mano-joint-regressor", type=Path, help="Optional official J_regressor_mano_ih26m.npy")
    parser.add_argument("--max-fit-mm", type=float, default=15.)
    parser.add_argument("--occlusion-gap-mm", type=float, default=10.)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.hand_prior_quality and args.max_pairs is None:
        parser.error("--hand-prior-quality requires --max-pairs for its stratified quota")
    if args.quality_max_candidates is not None and (not args.hand_prior_quality or args.quality_max_candidates < 1):
        parser.error("--quality-max-candidates requires --hand-prior-quality and a positive limit")

    if (args.min_gap < 1 or args.max_gap < args.min_gap or not math.isfinite(args.min_motion)
            or args.min_motion < 0 or args.cpu_threads < 1 or args.targets_per_source < 1
            or (args.max_pairs is not None and args.max_pairs < 1)
            or (args.archive_max_images is not None and args.archive_max_images < 2)
            or not math.isfinite(args.max_fit_mm) or args.max_fit_mm <= 0
            or not math.isfinite(args.occlusion_gap_mm) or args.occlusion_gap_mm <= 0
            or (args.mano_joint_regressor is not None and args.mano_model_dir is None)):
        parser.error("Invalid pairing, archive candidate, CPU or MANO fitting settings")

    import torch
    from src.pose_control.contact_hand_adapter.interhand import (
        SPLITS, TarImages, archive_parts, convert_split, load_subject_splits,
    )

    torch.set_num_threads(args.cpu_threads)
    dataset = args.dataset_root.resolve(strict=True)
    annotation_root = args.annotations or dataset / "annotations"
    if args.annotations is None and not (annotation_root / "subject.txt").is_file():
        annotation_root /= "annotations"
    annotation_root = annotation_root.resolve(strict=True)
    owners, excluded = load_subject_splits(annotation_root / "subject.txt")

    image_dir = args.images or (dataset / "images" if args.archive_dir is None else None)
    if args.images is not None or (image_dir is not None and image_dir.is_dir()):
        image_dir = image_dir.resolve(strict=True)
        rgb = nullcontext(image_dir)
        archive_report = None
    else:
        archive_dir = (args.archive_dir or dataset / "images_5fps").resolve(strict=True)
        parts, missing, incomplete = archive_parts(archive_dir, allow_partial=args.allow_partial_archive)
        partial = bool(missing) or incomplete is not None
        rgb = TarImages(parts, partial=partial)
        archive_report = {"directory": str(archive_dir), "readable_prefix": [part.name for part in parts],
                          "missing_parts": missing, "first_incomplete_part": incomplete,
                          "part_count": len(parts), "partial": partial}

    mano = None
    if args.mano_model_dir is not None:
        from src.pose_control.contact_hand_adapter.interhand_geometry import ManoGeometry

        mano = ManoGeometry(args.mano_model_dir.resolve(strict=True), joint_regressor=args.mano_joint_regressor,
                            max_fit_mm=args.max_fit_mm, occlusion_gap_mm=args.occlusion_gap_mm)

    splits = tuple(split for split in SPLITS if args.split is None or split in args.split)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    filters = {field: getattr(args, field) for field in ("capture", "camera", "sequence", "hand_type")}
    rows, stats, used_splits = [], {}, {}

    def archive_candidate(name):
        split, capture, sequence, camera, _ = name.split("/")
        return (split in splits and f"{split}/{capture}" not in excluded
                and (args.capture is None or capture == f"Capture{args.capture}")
                and (args.camera is None or camera == f"cam{args.camera}")
                and (args.sequence is None or sequence == args.sequence))

    with rgb as source:
        if isinstance(source, TarImages):
            indexed = dict.fromkeys(splits, 0)
            for name in source.members(accept=lambda name: archive_candidate(name)
                                       and (args.archive_max_images is None
                                            or indexed[name.split("/", 1)[0]] < args.archive_max_images)):
                indexed[name.split("/", 1)[0]] += 1
                if args.archive_max_images is not None and all(n >= args.archive_max_images for n in indexed.values()):
                    break
            archive_report["indexed_images"] = indexed
        for split in splits:
            quality = None
            if args.hand_prior_quality:
                from src.pose_control.contact_hand_adapter.interhand_quality import HandPriorScreen

                quality = HandPriorScreen(args.max_pairs, seed=args.quality_seed, max_candidates=args.quality_max_candidates)
            converted, stats[split] = convert_split(
                annotation_root, source, out, split, owners, excluded, min_gap=args.min_gap,
                max_gap=args.max_gap, min_motion=args.min_motion, limit=args.max_pairs, filters=filters,
                mano=mano, used_splits=used_splits, targets_per_source=args.targets_per_source, quality=quality,
                geometry_only=args.geometry_only,
            )
            rows.extend(converted)

    report = {
        "annotations": str(annotation_root),
        "archive": archive_report,
        "image_root": str(image_dir) if archive_report is None else None,
        "splits": stats,
        "subject_ownership": owners,
        "excluded_capture_subjects": excluded,
        "filters": filters,
        "pair_policy": {"minimum_frame_gap": args.min_gap, "maximum_frame_gap": args.max_gap,
                        "minimum_rms_motion_pixels": None if args.hand_prior_quality else args.min_motion,
                        "max_pairs_per_split": args.max_pairs, "hand_prior_quality": args.hand_prior_quality,
                        "targets_per_source": args.targets_per_source},
        "mano": mano.provenance if mano is not None else None,
        "geometry_only": args.geometry_only,
        "visibility": "Cross-hand occlusion is mesh-derived when both MANO fits pass; other visibility is unknown.",
        "contact_supervision": "unknown",
    }
    if not rows:
        raise SystemExit(f"No eligible frame pairs. {json.dumps(report, ensure_ascii=False)}")
    with (out / "audit.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (out / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(out / "manifest.jsonl"), "pairs": len(rows), "splits": stats},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

"""方案 B 的姿态对应、严格计分与按组汇总；检测器只在评估时加载。"""
from __future__ import annotations
from collections import defaultdict
from itertools import permutations
import json
from pathlib import Path
import numpy as np


def _inside(points, size=(512, 512)):
    xy = np.asarray(points, dtype=float)
    return np.isfinite(xy).all(axis=-1) & (xy[:, 0] >= 0) & (xy[:, 0] < size[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < size[1])


def torso_length(points):
    xy = np.asarray(points, dtype=float)
    if len(xy) > 11 and np.isfinite(xy[[1, 8, 11]]).all():
        distance = float(np.linalg.norm(xy[1] - (xy[8] + xy[11]) / 2))
        if distance > 1:
            return distance
    valid = xy[np.isfinite(xy).all(axis=-1)]
    return max(1.0, float(np.ptp(valid[:, 1])) * .3) if len(valid) else 1.0


def strict_pose_metrics(requested, scores, generated=None, generated_scores=None, *, torso=None, size=(512, 512)):
    requested = np.asarray(requested, dtype=float)
    valid_requested = (np.asarray(scores) >= .25) & np.isfinite(requested).all(axis=-1)
    n = int(valid_requested.sum())
    in_frame = _inside(requested, size)
    if generated is None:
        generated = np.full_like(requested, np.nan)
        generated_scores = np.zeros(len(requested))
    generated = np.asarray(generated, dtype=float)
    if generated.shape != requested.shape:
        raise ValueError("requested and generated landmark shapes differ")
    valid_generated = _inside(generated, size) & (np.asarray(generated_scores) >= .25)
    shared = valid_requested & in_frame & valid_generated
    denominator = torso_length(requested) if torso is None else max(float(torso), 1.0)
    errors = np.linalg.norm(requested - generated, axis=-1) / denominator
    result = dict(requested=n, requested_outside=int((valid_requested & ~in_frame).sum()), detected=int(shared.sum()),
                  detector_coverage=float(shared.sum() / n) if n else None,
                  mpjpe_norm=float(errors[shared].mean()) if shared.any() else None)
    for key, threshold in (("01", .1), ("02", .2), ("05", .5)):
        hits = int((shared & (errors <= threshold)).sum())
        result[f"pck_{key}_strict"] = hits / n if n else None
        result[f"pck_{key}_in_frame"] = hits / int((valid_requested & in_frame).sum()) if (valid_requested & in_frame).any() else None
    return result


def _centre(points, scores):
    points, scores = np.asarray(points), np.asarray(scores)
    if len(points) > 11 and (scores[[8, 11]] >= .25).all() and np.isfinite(points[[8, 11]]).all():
        return points[[8, 11]].mean(axis=0)
    valid = np.isfinite(points[:24]).all(axis=-1) & (scores[:24] >= .25)
    return np.median(points[:24][valid], axis=0) if valid.any() else np.array([np.nan, np.nan])


def assign_people(requested, detected):
    """几何匹配；不以外观选择检测人物，不重复使用同一检测。"""
    if not requested:
        return dict(mapping=[], ambiguous=False, cost_margin=None)
    source_centres = [_centre(*one) for one in requested]
    centres = [_centre(one["points"], one["scores"]) for one in detected]
    candidates = list(range(len(centres)))
    candidates += [None] * max(0, len(requested) - len(centres))
    costs = []
    for mapping in set(permutations(candidates, len(requested))):
        cost = 0.0
        for centre, index in zip(source_centres, mapping):
            cost += float(np.linalg.norm(centre - centres[index])) if index is not None and np.isfinite(centre).all() and np.isfinite(centres[index]).all() else 1e6
        costs.append((cost, mapping))
    costs.sort(key=lambda one: (one[0], tuple(-1 if x is None else x for x in one[1])))
    margin = costs[1][0] - costs[0][0] if len(costs) > 1 else None
    return dict(mapping=list(costs[0][1]), ambiguous=margin is not None and margin < 5,
                cost=float(costs[0][0]), cost_margin=float(margin) if margin is not None else None)


def aggregate_metrics(rows):
    """先平均 seed，再平均样本，再按独立 group 做宏平均。"""
    samples = defaultdict(list)
    for row in rows:
        samples[(row["dataset"], row["arm"], row["group_id"], row["sample_id"])].append(row)
    groups = defaultdict(list)
    metrics = ("pck_01_strict", "pck_02_strict", "mpjpe_norm", "detector_coverage", "request_available", "person_count_ok", "appearance_mean",
               "arcface_mean", "arcface_compare_coverage", "background_mae_outside_person_boxes", "relative_hip_error_norm", "psnr_gt", "ssim_gt", "lpips_gt", "seconds")
    for (dataset, arm, group, sample), values in samples.items():
        record = {}
        for key in metrics:
            numbers = [float(v[key]) for v in values if v.get(key) is not None and np.isfinite(v[key])]
            record[key] = float(np.mean(numbers)) if numbers else None
        groups[(dataset, arm, group)].append(record)
    datasets = defaultdict(list)
    for (dataset, arm, group), values in groups.items():
        record = dict(group_id=group, samples=len(values))
        for key in metrics:
            numbers = [v[key] for v in values if v[key] is not None]
            record[key] = float(np.mean(numbers)) if numbers else None
        datasets[(dataset, arm)].append(record)
    result = []
    for (dataset, arm), values in sorted(datasets.items()):
        record = dict(dataset=dataset, arm=arm, groups=len(values), samples=sum(v["samples"] for v in values), per_group=values)
        for key in metrics:
            numbers = [v[key] for v in values if v[key] is not None]
            output_key = "pck_02_group_macro" if key == "pck_02_strict" else key + "_group_macro"
            record[output_key] = float(np.mean(numbers)) if numbers else None
        result.append(record)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _requested(supervision, detector, target_image):
    result = []
    for path in supervision.get("requested_keypoints", []):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result.append((np.asarray(data.get("points", data.get("xy_canvas")), dtype=float), np.asarray(data["scores"], dtype=float)))
    if result:
        return result, "provided_DWPose134_canvas512"
    # 3DPW has rendered poses but no numeric landmarks. This is explicitly a GT
    # RGB estimate for evaluation, never an inference or text-encoder input.
    people = detector(target_image)
    if not people:
        return [], "GT_RGB_estimator_failed"
    person = max(people, key=lambda p: (p["box_xyxy"][2] - p["box_xyxy"][0]) * (p["box_xyxy"][3] - p["box_xyxy"][1]))
    return [(np.asarray(person["points"]), np.asarray(person["scores"]))], "GT_RGB_DWPose_estimate"


def _prepared_target(supervision):
    from PIL import Image
    with Image.open(supervision["target_rgb"]) as image:
        image = image.convert("RGB")
    transform = supervision["target_transform"]
    canvas = Image.new("RGB", tuple(transform["canvas_wh"]), "white")
    canvas.paste(image.resize(tuple(transform["resized_wh"]), Image.Resampling.LANCZOS), tuple(transform["offset_xy"]))
    return canvas


def evaluate_artifacts(workspace, run_dir, project_root):
    """Evaluate immutable generation outputs into a new timestamped directory."""
    import csv
    from datetime import datetime
    import cv2
    from PIL import Image
    from src.pose_edit.dwpose import detect_people, load_sessions
    from src.pose_edit.appearance import load_embedder, crop_person, judge_appearance
    from .runner import write_json, append_jsonl

    work, run, project = Path(workspace), Path(run_dir), Path(project_root)
    config = json.loads((project / "workflow-multiple/config.json").read_text())
    sessions = load_sessions(config)
    def detect(image):
        return detect_people(config, cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR), sessions=sessions)
    embed, embed_error = load_embedder(config["dinov2_repo"], device="cuda")
    lpips_model, lpips_error = None, None
    try:
        import torch
        import lpips
        if not (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").is_file():
            raise FileNotFoundError("local pretrained AlexNet weights unavailable")
        lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False).to("cuda")
    except Exception as error:
        lpips_error = f"{type(error).__name__}: {error}"
    try:
        from src.evaluation.joint_pilot_metrics import IdentityMetric
        identity, identity_error = IdentityMetric(project / "models"), None
    except Exception as error:
        identity, identity_error = None, f"{type(error).__name__}: {error}"
    conditions, supervision = {}, {}
    for split in ("train", "val", "test"):
        conditions.update({r["sample_id"]: r for r in read_jsonl(work / "data" / f"{split}_conditions.jsonl")})
        supervision.update({r["sample_id"]: r for r in read_jsonl(work / "data" / f"{split}_supervision.jsonl")})
    output = run / ("evaluation_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    output.mkdir(exist_ok=False)
    write_json(output / "availability.json", dict(dinov2=embed_error or "available", arcface=identity_error or "available", lpips=lpips_error or "available", dwpose="CPUExecutionProvider"))
    reference_cache, target_cache, geometry_cache, rows = {}, {}, {}, []
    for index, record in enumerate(read_jsonl(run / "records.jsonl")):
        sid = record["sample_id"]
        condition, truth = conditions[sid], supervision[sid]
        with Image.open(record["out"]) as image:
            generated = image.convert("RGB")
        if sid not in target_cache:
            target = _prepared_target(truth)
            target_cache[sid] = (target, _requested(truth, detect, target))
        target, (requested, basis) = target_cache[sid]
        people = detect(generated)
        assignment = assign_people(requested, people)
        person_metrics, appearances, face_cosines, face_coverage = [], [], [], []
        if sid not in reference_cache:
            with Image.open(condition["source_rgb"]) as image:
                source = image.convert("RGB")
            masks = condition.get("metadata", {}).get("original_source_masks", [])
            crops, source_boxes = [], []
            if masks:
                transform = condition["source_transform"]
                sx = transform["resized_wh"][0] / transform["original_wh"][0]
                sy = transform["resized_wh"][1] / transform["original_wh"][1]
                left, top = transform["offset_xy"]
                for mask_path in masks:
                    with Image.open(mask_path) as image:
                        yy, xx = np.nonzero(np.asarray(image.convert("L")) > 0)
                    box = (xx.min() * sx + left, yy.min() * sy + top, (xx.max() + 1) * sx + left, (yy.max() + 1) * sy + top) if len(xx) else (0, 0, 512, 512)
                    crops.append(crop_person(source, box))
                    source_boxes.append(box)
            else:
                detected_source = detect(source)
                if detected_source:
                    source_person = max(detected_source, key=lambda p: (p["box_xyxy"][2] - p["box_xyxy"][0]) * (p["box_xyxy"][3] - p["box_xyxy"][1]))
                    crops = [crop_person(source, source_person["box_xyxy"])]
                    source_boxes = [source_person["box_xyxy"]]
                else:
                    crops = [None]
            features = [embed(c) if embed and c is not None else None for c in crops]
            faces = [identity.embedding(c) if identity and c is not None else None for c in crops]
            reference_cache[sid] = (crops, features, faces, source, source_boxes)
        crops, source_features, source_faces, source, source_boxes = reference_cache[sid]
        generated_crops = [crop_person(generated, p["box_xyxy"]) for p in people]
        generated_features = [embed(c) if embed and c is not None else None for c in generated_crops]
        for person_index, ((points, scores), matched) in enumerate(zip(requested, assignment["mapping"])):
            person = people[matched] if matched is not None else None
            normalizer = torso_length(points)
            body = strict_pose_metrics(points[:24], scores[:24], person["points"][:24] if person else None,
                     person["scores"][:24] if person else None, torso=normalizer)
            for name, region in (("face", slice(24, 92)), ("hand", slice(92, 134))):
                region_result = strict_pose_metrics(points[region], scores[region], person["points"][region] if person else None,
                     person["scores"][region] if person else None, torso=normalizer)
                body[name] = region_result
            person_metrics.append(body)
            appearance = None
            if matched is not None and person_index < len(source_features) and source_features[person_index] is not None and generated_features[matched] is not None:
                appearance = float(source_features[person_index] @ generated_features[matched])
            appearances.append(appearance)
            face = identity.embedding(generated_crops[matched]) if identity and matched is not None and generated_crops[matched] is not None else None
            reference_face = source_faces[person_index] if person_index < len(source_faces) else None
            face_coverage.append(dict(source=reference_face is not None, generated=face is not None))
            face_cosines.append(float(reference_face @ face) if reference_face is not None and face is not None else None)
        def mean(key):
            values = [m[key] for m in person_metrics if m.get(key) is not None]
            return float(np.mean(values)) if values else None
        appearance_numbers = [x for x in appearances if x is not None]
        row = {**record, "pose_basis": basis, "persons_detected": len(people), "persons_requested": len(requested),
               "request_available": bool(requested), "person_count_ok": bool(requested) and len(people) == len(requested),
               "assignment": assignment, "per_person": person_metrics, "appearance_cosines": appearances,
               "appearance_mean": float(np.mean(appearance_numbers)) if appearance_numbers else None,
               "arcface_cosines": face_cosines, "arcface_coverage": face_coverage,
               "contact_count": condition.get("metadata", {}).get("contact_count"),
               "action_family": condition.get("metadata", {}).get("action_family"),
               **{key: mean(key) for key in ("pck_01_strict", "pck_02_strict", "mpjpe_norm", "detector_coverage")}}
        face_numbers = [value for value in face_cosines if value is not None]
        row["arcface_mean"] = float(np.mean(face_numbers)) if face_numbers else None
        row["arcface_compare_coverage"] = len(face_numbers) / len(requested) if requested else None
        if condition["dataset"] == "hi4d":
            if sid not in geometry_cache:
                path = Path(truth["target_rgb"]).with_name("geometry.npz")
                geometry = dict(available=False, basis="Hi4D target mesh geometry, evaluation only")
                if path.is_file():
                    from .runner import file_sha256
                    with np.load(path, allow_pickle=False) as values:
                        human = (values["semantic_A"] > 0) | (values["semantic_B"] > 0)
                        occluded = values["occlusion_mask"] > 0
                        ratio = float((human & occluded).sum() / max(1, human.sum()))
                        geometry.update(available=True, occlusion_ratio=ratio, geometry_sha256=file_sha256(path),
                                        occlusion_bin="heavy_gt_015" if ratio >= .15 else "light_gt_015")
                geometry_cache[sid] = geometry
            row["occlusion_geometry"] = geometry_cache[sid]
            count = row.get("contact_count")
            row["contact_bin"] = "zero" if count == 0 else "1_99" if count is not None and count < 100 else "100_999" if count is not None and count < 1000 else "1000_plus" if count is not None else "unavailable"
        if record.get("arm") == "wrong_pose":
            wrong_id = record["control_sample_id"]
            if wrong_id not in target_cache:
                wrong_target = _prepared_target(supervision[wrong_id])
                target_cache[wrong_id] = (wrong_target, _requested(supervision[wrong_id], detect, wrong_target))
            wrong_requested = target_cache[wrong_id][1][0]
            wrong_assignment = assign_people(wrong_requested, people)
            wrong_scores = []
            for (points, scores), match in zip(wrong_requested, wrong_assignment["mapping"]):
                person = people[match] if match is not None else None
                score = strict_pose_metrics(points[:24], scores[:24], person["points"][:24] if person else None,
                     person["scores"][:24] if person else None, torso=torso_length(points))["pck_02_strict"]
                if score is not None:
                    wrong_scores.append(score)
            row["wrong_control_pck_02"] = float(np.mean(wrong_scores)) if wrong_scores else None
            row["wrong_minus_correct_pose_pck"] = row["wrong_control_pck_02"] - row["pck_02_strict"] if row["wrong_control_pck_02"] is not None and row["pck_02_strict"] is not None else None
        if len(source_features) == 2 and len(generated_features) == 2 and all(x is not None for x in source_features + generated_features) and None not in assignment["mapping"] and not assignment["ambiguous"]:
            similarity = np.asarray([[float(a @ b) for b in generated_features] for a in source_features])
            row["appearance_assignment"] = judge_appearance(similarity, assignment["mapping"])
        pixels, gt = np.asarray(generated, dtype=float), np.asarray(target, dtype=float)
        mse = float(np.square(pixels - gt).mean())
        row["psnr_gt"] = float(10 * np.log10(255**2 / max(mse, 1e-12)))
        try:
            from skimage.metrics import structural_similarity
            row["ssim_gt"] = float(structural_similarity(pixels, gt, channel_axis=-1, data_range=255))
        except ImportError:
            row["ssim_gt"] = None
        row["lpips_gt"] = None
        row["lpips_status"] = lpips_error or "available"
        if lpips_model is not None:
            import torch
            x = torch.from_numpy(pixels.astype(np.float32)).permute(2, 0, 1)[None].to("cuda") / 127.5 - 1
            y = torch.from_numpy(gt.astype(np.float32)).permute(2, 0, 1)[None].to("cuda") / 127.5 - 1
            with torch.no_grad():
                row["lpips_gt"] = float(lpips_model(x, y).item())
        row["background_mae_outside_person_boxes"] = None
        row["background_fraction"] = None
        if source_boxes:
            background = np.ones((512, 512), dtype=bool)
            for transform in (condition["source_transform"], condition["target_transform"]):
                region = np.zeros((512, 512), dtype=bool)
                x0, y0 = transform["offset_xy"]
                width, height = transform["resized_wh"]
                region[y0:y0 + height, x0:x0 + width] = True
                background &= region
            content_pixels = int(background.sum())
            for box in source_boxes + [p["box_xyxy"] for p in people]:
                x0, y0, x1, y1 = np.asarray(box, dtype=float)
                x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
                x1, y1 = min(512, int(np.ceil(x1))), min(512, int(np.ceil(y1)))
                background[y0:y1, x0:x1] = False
            row["background_fraction"] = float(background.sum() / content_pixels) if content_pixels else 0.
            if background.any():
                row["background_mae_outside_person_boxes"] = float(np.abs(np.asarray(source, dtype=float) - pixels)[background].mean())
        if len(requested) == 2 and None not in assignment["mapping"]:
            req_centres = [_centre(*one) for one in requested]
            gen_centres = [_centre(people[i]["points"], people[i]["scores"]) for i in assignment["mapping"]]
            difference = (gen_centres[1] - gen_centres[0]) - (req_centres[1] - req_centres[0])
            row["relative_hip_error_norm"] = float(np.linalg.norm(difference) / max(1., np.mean([torso_length(r[0]) for r in requested]))) if np.isfinite(difference).all() else None
        append_jsonl(output / "metrics.jsonl", row)
        if not requested or None in assignment["mapping"] or len(people) != len(requested):
            append_jsonl(output / "detection_failures.jsonl", dict(sample_id=sid, arm=record["arm"], seed=record["seed"],
                 requested=len(requested), detected=len(people), mapping=assignment["mapping"], pose_basis=basis,
                 reason="requested_GT_estimator_failed" if not requested else "missing_merged_or_extra_person"))
        rows.append(row)
        if (index + 1) % 12 == 0:
            print(json.dumps(dict(stage="evaluation", completed=index + 1)), flush=True)
    aggregate = aggregate_metrics(rows)
    strata = []
    for row in rows:
        if row["dataset"] != "hi4d":
            continue
        strata.append({**row, "dataset": "hi4d/contact_" + row.get("contact_bin", "unavailable")})
        geometry = row.get("occlusion_geometry", {})
        strata.append({**row, "dataset": "hi4d/occlusion_" + geometry.get("occlusion_bin", "unavailable")})
    write_json(output / "summary.json", dict(images=len(rows), pose_request_unavailable=sum(not r["request_available"] for r in rows),
                 person_count_failures=sum(not r["person_count_ok"] for r in rows), aggregate=aggregate,
                 contact_occlusion_strata=aggregate_metrics(strata), occlusion_heavy_threshold=.15,
                 identity_limit="DINOv2 is an appearance proxy; ArcFace is reported with detection coverage"))
    with (output / "groups.csv").open("x", encoding="utf-8", newline="") as stream:
        simple = [{k: v for k, v in row.items() if k != "per_group"} for row in aggregate]
        if simple:
            writer = csv.DictWriter(stream, fieldnames=list(simple[0]))
            writer.writeheader()
            writer.writerows(simple)
    return output

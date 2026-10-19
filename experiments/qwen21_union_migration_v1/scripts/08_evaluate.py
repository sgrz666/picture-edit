#!/usr/bin/env python
"""指标与面板：对矩阵产物计算姿态、人物对应、外观与运行代价，并输出逐样本 JSONL、分组 CSV 与面板图。

姿态与外观全部复用仓库既有实现（src/pose_edit/dwpose、src/evaluation/no_gt_metrics、
workflow-multiple/evaluate_dual.py、src/pose_edit/appearance.py），不另起一套口径。
检测失败保留在结果里：strict 口径把漏检记为不命中，同时单独报检测覆盖率。
身份互换判定用 judge_appearance：姿态髋部配对与外观最优配对不一致、且超过阈值才判定。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path("/home/shangguanrz/project/pic-edit")
EXP = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation.no_gt_metrics import pose_error_stats  # noqa: E402
from src.pose_edit.appearance import crop_person, judge_appearance, load_embedder  # noqa: E402
from src.pose_edit.dwpose import detect_people  # noqa: E402


def load_dual_helpers():
    p = REPO / "workflow-multiple/evaluate_dual.py"
    spec = importlib.util.spec_from_file_location("evaluate_dual_for_union", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def requested_points(kind: str, entry: dict):
    """返回 ([(134,2), (134,)bool, 坐标系说明], ...)。单人一组、双人两组。

    单人必须用 keypoints_generation_px（画布坐标系）。keypoints_B_px 是裁剪前的 B 帧坐标，
    画布 928 高时它的 y 能到 1184，拿它比画布会把工作流自己的产物也算成 PCK=0。
    """
    kp_file = None
    for item in entry["eval_only"]:
        if item["role"] == "requested_keypoints":
            kp_file = item["path"]
    if kp_file is None:
        return [], "missing"
    z = np.load(kp_file, allow_pickle=True)
    if kind == "single":
        if "keypoints_generation_px" in z.files:
            return [(z["keypoints_generation_px"], z["valid_generation"])], "keypoints_generation_px(canvas)"
        return [(z["keypoints_B_px"], z["valid_B"])], "keypoints_B_px(pre-crop)"
    offset = np.asarray(z["content_box"][:2], dtype=float)
    return [(np.asarray(cp) - offset, z["canvas_valid"][i])
            for i, cp in enumerate(z["canvas_points"])], "canvas_points-content_box"


def person_points(person):
    return np.asarray(person["points"], dtype=float), np.asarray(person["scores"], dtype=float)


def source_reference(kind: str, entry: dict, prov: dict, config: dict, cache: dict):
    """按槽位顺序返回 (源人物裁剪图列表, 依据说明)；找不到参照时给出原因，不猜。

    双人：14_attach_provenance.py 已把每个画布槽位（npz 的 canvas_points / hi4d_sides）对应的源人物
    从 source_A.png 上裁好，这里按槽位顺序读回。
    单人：源图里最大的那个人框；没检测到人就退回整图，并在依据里写明。
    """
    key = f"{kind}:{entry['sample']}"
    if key in cache:
        return cache[key]
    if kind == "dual":
        crops = []
        for p in (prov or {}).get("persons", []):
            crop = (p.get("identity_crop") or {}).get("path")
            if not crop or not pathlib.Path(crop).exists():
                cache[key] = (None, f"槽位 {p.get('slot')} 缺少身份参照裁剪图")
                return cache[key]
            with Image.open(crop) as im:
                crops.append(im.convert("RGB"))
        basis = (f"source_A.png 按 pair.json 的 hi4d_side 人框裁剪；槽位顺序 "
                 f"{[p.get('slot') for p in prov['persons']]}；pair kind={prov.get('pair_kind')}")
        cache[key] = (crops, basis)
        return cache[key]
    path = entry["files"]["source"]["path"]
    with Image.open(path) as im:
        img = im.convert("RGB")
    people = detect_people(config, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    if people:
        best = max(people, key=lambda p: max(0.0, p["box_xyxy"][2] - p["box_xyxy"][0])
                                 * max(0.0, p["box_xyxy"][3] - p["box_xyxy"][1]))
        crop = crop_person(img, best["box_xyxy"])
        if crop is not None:
            cache[key] = ([crop], "source.png 内最大人物检测框裁剪")
            return cache[key]
    cache[key] = ([img], "source.png 整图（未检测到人，退化为全图参照）")
    return cache[key]


def appearance_metrics(embed, sources, gen_image, people, mapping, kind, basis, assignment=None):
    """源人物 vs 生成人物的 DINOv2 余弦相似度、逐人身份分与身份互换判定。

    行 = 源人物（槽位顺序），列 = 生成图检测到人（检测顺序）。双人且行列都是 2 时用
    judge_appearance：把姿态髋部配对当作 pose 配对，与外观最优配对比较，超过阈值才判「疑似互换」。
    髋部配对本身标了 ambiguous（最优与次优成本差小于阈值）时，把互换判定置空并说明原因：
    姿态对应都不确定，外观与它不一致不能算互换证据。
    单人或人数不对齐时只报每位源人物的最佳匹配，并明确不做互换判定。
    """
    if embed is None:
        return {"available": False, "reason": "DINOv2 嵌入器不可用", "reference_basis": basis}
    if not sources:
        return {"available": False, "reason": basis or "没有源人物参照裁剪图", "reference_basis": basis}
    if not people:
        return {"available": False, "reason": "生成图未检测到人", "source_people": len(sources),
                "generated_people": 0, "reference_basis": basis}
    gen_crops = [crop_person(gen_image, p["box_xyxy"]) for p in people]
    if any(c is None for c in gen_crops):
        return {"available": False, "reason": "生成图某个人的裁剪框退化", "source_people": len(sources),
                "generated_people": len(people), "reference_basis": basis}
    src_feats = [embed(one) for one in sources]
    gen_feats = [embed(one) for one in gen_crops]
    sim = np.array([[float(np.dot(s, g)) for g in gen_feats] for s in src_feats])

    if kind == "dual" and sim.shape == (2, 2) and mapping and len(mapping) == 2:
        verdict = judge_appearance(sim, [int(v) for v in mapping])
        verdict["pose_assignment"] = [int(v) for v in mapping]
        verdict["source_people"] = 2
        verdict["generated_people"] = 2
        verdict["reference_basis"] = basis
        verdict["basis"] = verdict.get("basis", "") + "；行=源槽位顺序，列=检测顺序，DINOv2 ViT-S/14 余弦相似度"
        if (assignment or {}).get("ambiguous"):
            verdict["identity_swap_suspected_raw"] = verdict.get("identity_swap_suspected")
            verdict["identity_swap_suspected"] = None
            verdict["swap_suppressed_reason"] = (
                "髋部配对本身歧义（最优与次优成本差 "
                f"{assignment.get('cost_margin_to_next')} px），外观与姿态配对不一致不算互换证据")
        return verdict
    best = sim.max(axis=1)
    return {"available": True,
            "identity_mean": float(np.mean(best)),
            "identity_similarity": [float(v) for v in best],
            "identity_swap_suspected": None,
            "swap_reason": "只有一位源人物参照或人数未对齐，互换判定需要双人且人数对齐",
            "source_people": int(sim.shape[0]), "generated_people": int(sim.shape[1]),
            "similarity_matrix": [[float(v) for v in row] for row in sim],
            "reference_basis": basis,
            "basis": "每位源人物取最相似的生成身体（不做互换判定）"}


def panel(out_path: pathlib.Path, cells: list[tuple[str, str]]):
    ims = []
    for label, p in cells:
        im = Image.open(p).convert("RGB")
        im.thumbnail((260, 260))
        ims.append((label, im))
    w = sum(im.width for _, im in ims) + 10 * (len(ims) + 1)
    h = max(im.height for _, im in ims) + 34
    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    x = 10
    for label, im in ims:
        canvas.paste(im, (x, 26))
        d.text((x, 8), label, fill="black")
        x += im.width + 10
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="WORK/runs 下的运行目录名")
    ap.add_argument("--panels", action="store_true")
    args = ap.parse_args()

    run_dir = WORK / "runs" / args.run
    records = [json.loads(l) for l in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))
    prov_all = {}
    prov_path = EXP / "input_provenance.json"
    if prov_path.exists():
        prov_all = json.loads(prov_path.read_text(encoding="utf-8"))
    by_sample = {s["sample"]: s for s in manifest["samples"]}
    dh = load_dual_helpers()
    config = json.loads((REPO / "workflow-multiple/config.json").read_text(encoding="utf-8"))
    threshold = config["keypoint_score_threshold"]

    embed, embed_err = load_embedder(config["dinov2_repo"], device="cuda")
    if embed is None:
        print(f"[appearance] 嵌入器不可用，外观指标记为不可用：{embed_err}", flush=True)
    ref_cache: dict = {}

    rows, per_image = [], []
    for rec in records:
        if not rec.get("out"):
            per_image.append({**rec, "pose": None, "error": rec.get("error", "no output")})
            continue
        entry = by_sample[rec["sample"]]
        req, guide_basis = requested_points(entry["kind"], entry)
        with Image.open(rec["out"]) as im:
            pil = im.convert("RGB")
        rgb = np.asarray(pil)
        size = (rgb.shape[1], rgb.shape[0])
        people = detect_people(config, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        row = {"sample": rec["sample"], "kind": rec["kind"], "config": rec["config"],
               "seed": rec["seed"], "seconds": rec.get("seconds"), "peak_GiB": rec.get("peak_GiB"),
               "persons_detected": len(people), "persons_requested": len(req),
               "detector_ok": bool(people)}
        mapping = assignment = None
        if people:
            pts, scores = person_points(people[0])
            keep = np.isfinite(pts).all(axis=1) & (scores >= threshold)
        if entry["kind"] == "single" and req and people:
            guide, gvalid = req[0]
            est = np.where(keep[:, None], pts, np.nan)
            stats = pose_error_stats(guide, gvalid, est, keep, size)
            row["pose"] = stats["summary"]
            row["pose_groups"] = {k: {kk: vv for kk, vv in v.items() if kk.startswith(("pck", "mpjpe", "joints"))}
                                  for k, v in stats["groups"].items()}
            row["joints_requested"] = stats["joints_requested"]
            row["joints_compared"] = stats["joints_compared"]
            body = stats["groups"].get("body", {})
            row["pck_basis"] = f"single/whole-image DWPose-134 vs {guide_basis}, torso-normalised"
            row["pck@0.2"] = body.get("pck@0.2")
            row["pck@0.5"] = body.get("pck@0.5")
            row["pck@0.2_strict"] = body.get("pck@0.2_strict")
            row["pck@0.5_strict"] = body.get("pck@0.5_strict")
            row["mpjpe_norm"] = body.get("mpjpe_norm")
            row["face_mpjpe_norm"] = stats["summary"].get("face_mpjpe_norm")
            row["hand_mpjpe_norm"] = stats["summary"].get("hand_mpjpe_norm")
            row["detector_coverage"] = (round(stats["joints_compared"] / stats["joints_requested"], 4)
                                        if stats.get("joints_requested") else None)
        elif entry["kind"] == "dual":
            mapping, assignment = dh.assign(people, [r[0] for r in req], threshold, size)
            row["assignment"] = assignment
            persons = []
            for slot, (guide, gvalid) in enumerate(req):
                person = people[mapping[slot]] if mapping and mapping[slot] is not None else None
                persons.append(dh.score_person(guide, person, threshold, size))
            row["per_person"] = persons
            comp = [p for p in persons if p.get("comparable")]
            row["pose"] = {"body_pck@0.2": float(np.mean([p["pck@0.2"] for p in comp])) if comp else None,
                           "body_pck@0.5": float(np.mean([p["pck@0.5"] for p in comp])) if comp else None,
                           "body_mpjpe_norm": float(np.mean([p["mpjpe_norm"] for p in comp])) if comp else None}
            row["pck_basis"] = (f"dual/per-person DWPose-134 vs {guide_basis}; shared-valid denominator, "
                                "strict variant charges an undetected/merged person as 0")
            row["pck@0.2"] = row["pose"]["body_pck@0.2"]
            row["pck@0.5"] = row["pose"]["body_pck@0.5"]
            row["mpjpe_norm"] = row["pose"]["body_mpjpe_norm"]
            row["pck@0.2_strict"] = (float(np.mean([p.get("pck@0.2", 0.0) if p.get("comparable") else 0.0
                                                    for p in persons])) if persons else None)
            row["pck@0.5_strict"] = (float(np.mean([p.get("pck@0.5", 0.0) if p.get("comparable") else 0.0
                                                    for p in persons])) if persons else None)
            row["persons_comparable"] = len(comp)
            row["person_count_ok"] = float(len(people) == len(req))
            # 相对位置：两人髋距与请求髋距之比
            try:
                if comp and len(comp) == 2:
                    a = np.asarray(comp[0]["hip_centre_xy"]); b = np.asarray(comp[1]["hip_centre_xy"])
                    row["hip_distance_px"] = float(np.linalg.norm(a - b))
            except Exception:
                pass
        else:
            mapping = None
        sources, ref_basis = source_reference(entry["kind"], entry, prov_all.get(rec["sample"], {}),
                                              config, ref_cache)
        row["appearance"] = appearance_metrics(embed, sources, pil, people,
                                               mapping, entry["kind"], ref_basis, assignment)
        per_image.append(row)
        rows.append(row)
        print("[eval] " + json.dumps({k: row.get(k) for k in
              ("sample", "config", "seed", "persons_detected")}, ensure_ascii=False)
              + f" identity={row['appearance'].get('identity_mean')}", flush=True)

    out_jsonl = run_dir / "metrics.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_image) + "\n", encoding="utf-8")

    # 分组汇总：每样本先平均两个 seed，再按配置聚合
    agg = {}
    for r in rows:
        key = (r["sample"], r["config"])
        agg.setdefault(key, []).append(r)
    with open(run_dir / "metrics_by_sample_config.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["sample", "kind", "config", "n_seeds", "pck@0.2", "pck@0.5", "pck@0.2_strict",
                      "pck@0.5_strict", "mpjpe_norm", "face_mpjpe_norm", "hand_mpjpe_norm",
                      "detector_coverage", "persons_detected", "detector_ok_rate",
                      "identity_mean", "identity_swap_rate", "seconds_mean", "peak_GiB_max", "pck_basis"])
        for (sample, cfg), rs in sorted(agg.items()):
            def mean(field):
                vals = [r.get(field) for r in rs if r.get(field) is not None]
                return round(float(np.mean(vals)), 4) if vals else None
            ident = [r["appearance"]["identity_mean"] for r in rs
                     if r.get("appearance", {}).get("available") and r["appearance"].get("identity_mean") is not None]
            swap = [r["appearance"]["identity_swap_suspected"] for r in rs
                    if r.get("appearance", {}).get("available")
                    and r["appearance"].get("identity_swap_suspected") is not None]
            basis = sorted({r.get("pck_basis") for r in rs if r.get("pck_basis")})
            wtr.writerow([sample, rs[0]["kind"], cfg, len(rs),
                          mean("pck@0.2"), mean("pck@0.5"), mean("pck@0.2_strict"), mean("pck@0.5_strict"),
                          mean("mpjpe_norm"), mean("face_mpjpe_norm"), mean("hand_mpjpe_norm"),
                          mean("detector_coverage"),
                          round(float(np.mean([r["persons_detected"] for r in rs])), 2),
                          round(float(np.mean([1.0 if r.get("detector_ok") else 0.0 for r in rs])), 2),
                          round(float(np.mean(ident)), 4) if ident else None,
                          round(float(np.mean(swap)), 2) if swap else None,
                          round(float(np.mean([r.get("seconds") or 0 for r in rs])), 1),
                          round(max([r.get("peak_GiB") or 0 for r in rs]), 2),
                          " | ".join(basis) or None])

    if args.panels:
        (run_dir / "panels").mkdir(exist_ok=True)
        for sample in sorted({r["sample"] for r in rows}):
            entry = by_sample[sample]
            for seed in sorted({r["seed"] for r in rows if r["sample"] == sample}):
                cells = []
                if entry["kind"] == "single":
                    cells += [("source", entry["files"]["source"]["path"]),
                              ("pose_ref", entry["files"]["pose_reference"]["path"])]
                else:
                    cells += [("src_A", entry["files"]["source_A"]["path"]),
                              ("src_B", entry["files"]["source_B"]["path"])]
                    for p in (prov_all.get(sample, {}) or {}).get("persons", []):
                        crop = (p.get("identity_crop") or {}).get("path")
                        if crop:
                            cells.append((f"id_slot{p.get('slot')}_{p.get('hi4d_side')}", crop))
                cells.append(("control", entry["files"]["control"]["path"]))
                for cfg in ("native", "union_s0", "union_s05", "union_s10"):
                    hit = [r for r in rows if r["sample"] == sample and r["config"] == cfg and r["seed"] == seed and r.get("out")]
                    if hit:
                        cells.append((cfg, hit[0]["out"]))
                if len(cells) >= 3:
                    panel(run_dir / "panels" / f"{sample}_seed{seed}.png", cells)

    print(json.dumps({"run": str(run_dir), "images_evaluated": len(rows),
                      "records": len(per_image), "metrics": str(out_jsonl),
                      "appearance_available": bool(embed is not None)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

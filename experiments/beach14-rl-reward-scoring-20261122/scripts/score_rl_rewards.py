#!/usr/bin/env python
"""RL reward scoring for 2.beach14 generation results (seed42, 4 arms x 14 templates).

打分内核照搬小红书 runtime-v3 冻结的 GRPO 奖励实现
(``src/pose_control/qwen21_full/native_rl/rewards.py`` 的 ``score_candidate`` /
``DUAL_WEIGHTS`` / PART_POINTS / OKS sigmas)，按用户 2026-11-22 决定把姿态后端从
DWPose ONNX 换成项目统一 SDPose-Wholebody + YOLO11x（``src/evaluation/sdpose.py``）；
身份后端 antelopev2（scrfd_10g_bnkps + glintr100）不变。

双人五项 reward：pose .30 / identity .25 / binding .20 / layout .15 / contact .10；
某项缺资格时复合分按剩余权重归一化。

contact 不做人工标注：从 ``pose_1120/beach_target_pose_14/<t>/maps_3d.npz`` 的
``part``（V6_part_0..13，与 RL PART_POINTS 组 0..13 一一对应）+ ``owner`` 数组自动
推导两人部位区域接触对，期望距离取 SDPose 目标姿态对应部位关节均值在打分画布
（669x896）下的实际距离。

运行解释器：SDPose 评测环境
``/data/shangguanrz/pic-edit-experiments/姿态估计/four-skeleton-compair/.venv/bin/python``
（含 torch/diffusers/ultralytics/insightface/onnxruntime/scipy/matplotlib）。
产物写 ``results/rl_score/``，不覆盖任何既有产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RUNTIME = Path("/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1/source/runtime-v3")
BEACH14 = Path("/data/shangguanrz/pic-edit-experiments/two/workflow-v2/experiments/2.beach14")
POSE14 = Path("/data/shangguanrz/nihaochen/pose_1120/beach_target_pose_14")
MODELS = Path("/home/shangguanrz/project/pic-edit/models")
SOURCE_IMAGE = Path("/data/shangguanrz/pic-edit-experiments/two/workflow-v2/experiments/1.beach/beach_target_pose/beach.jpg")

TEMPLATES = tuple(f"t{i:02d}" for i in range(1, 15))
ARMS = {
    "exp3_with_b_skel_q4km": "#3 with_b+skel",
    "exp4_with_b_skel_depth_q4km": "#4 with_b+skel+depth",
    "exp7_skel_only_q4km": "#7 skel_only",
    "exp8_skel_depth_q4km": "#8 skel_only+depth",
}

# geometry 134-slot -> COCO17（由 src/pose_edit/geometry.py cocowholebody133_to_dwpose134 的
# body=[0,1,6,8,10,5,7,9,12,14,16,11,13,15,2,4,1,3] 反解；slot1 是肩中点 neck，不进 COCO17）
GEO134_TO_COCO17 = np.asarray((0, 16, 14, 17, 15, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10))
SCORE_THRESHOLD = 0.30       # 与既有 beach14 SDPose 评测、RL 内核 DWPose conf 一致
MIN_AREA = 5000.0            # 与既有 beach14 评测的前景/背景分离口径一致
CONTACT_3D_M = 0.010         # 部位顶点 3D 接触判定阈值（米）：对齐原 fcl 网格碰撞语义（贴合/穿插）
CONTACT_MIN_MASK = 50        # 投票区域面积下限（px^2）
CONTACT_MIN_VERTS = 10       # 接触判定顶点数下限
GPU_WAIT_GIB = 8.0
GPU_WAIT_ROUNDS = 30

sys.path.insert(0, str(RUNTIME))

from src.evaluation import sdpose  # noqa: E402
from src.evaluation.joint_pilot_metrics import IdentityMetric  # noqa: E402
from src.pose_control.qwen21_full.native_rl.rewards import (  # noqa: E402
    DUAL_WEIGHTS,
    PART_POINTS,
    PersonPose,
    RewardTarget,
    _assign,
    _head_box,
    _iou,
    _part,
    score_candidate,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_gpu() -> dict:
    import torch

    for round_index in range(GPU_WAIT_ROUNDS):
        free, total = torch.cuda.mem_get_info()
        free_gib, total_gib = free / 2**30, total / 2**30
        if free_gib >= GPU_WAIT_GIB:
            return {"free_gib": free_gib, "total_gib": total_gib, "waited_rounds": round_index}
        print(f"gpu busy: {free_gib:.2f} GiB free, round {round_index + 1}/{GPU_WAIT_ROUNDS}; sleep 60s", flush=True)
        time.sleep(60)
    raise RuntimeError(f"GPU never had {GPU_WAIT_GIB} GiB free")


def coco17_person(person: dict) -> PersonPose:
    """SDPose 134-slot person dict -> RL PersonPose（COCO17 序）。"""
    points = np.asarray(person["points"], dtype=float)[GEO134_TO_COCO17]
    scores = np.asarray(person["scores"], dtype=float)[GEO134_TO_COCO17]
    valid = (scores > SCORE_THRESHOLD) & np.isfinite(points).all(axis=1)
    return PersonPose(points, valid, np.asarray(person["box_xyxy"], dtype=float))


def detect_people(image: Image.Image, runtime) -> list[PersonPose]:
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    people = sdpose.detect_people(bgr, runtime, min_area=MIN_AREA, sort_by="x_center")
    return [coco17_person(person) for person in people]


def faces_for_poses(identity: IdentityMetric, image: Image.Image, poses) -> tuple:
    """与冻结 rewards.ImageRewardBank._faces 相同的语义：IoU>=.10 盲配人脸框。"""
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    detected, landmarks = identity.det.detect(bgr)
    if not len(detected):
        return (None,) * len(poses)
    head_boxes = [_head_box(pose) for pose in poses]
    scores = np.array(
        [[_iou(box, face[:4]) if box is not None else 0.0 for face in detected] for box in head_boxes]
    )
    output = []
    for index in _assign(scores, 0.10):
        embedding = None
        if index is not None:
            crop = identity.face_align.norm_crop(bgr, landmark=landmarks[index])
            feature = identity.recognition.get_feat(crop).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid glintr100 embedding")
            embedding = feature / norm
        output.append(embedding)
    return tuple(output)


def source_embeddings(identity: IdentityMetric, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """beach.jpg 两位主角（左男右女）的 antelopev2 嵌入，按 x 中心排序。"""
    bgr = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    detected, landmarks = identity.det.detect(bgr)
    if len(detected) < 2:
        raise ValueError(f"source image {path} yields {len(detected)} faces, need >=2")
    areas = (detected[:, 2] - detected[:, 0]) * (detected[:, 3] - detected[:, 1])
    top2 = np.argsort(-areas)[:2]
    top2 = top2[np.argsort(detected[top2, 0])]  # 左 -> slot0，右 -> slot1
    embeddings = []
    for index in top2:
        crop = identity.face_align.norm_crop(bgr, landmark=landmarks[index])
        feature = identity.recognition.get_feat(crop).reshape(-1).astype(np.float32)
        embeddings.append(feature / float(np.linalg.norm(feature)))
    return tuple(embeddings)


def label_votes(part: np.ndarray, owner: np.ndarray, coco_xy: np.ndarray) -> dict[int, int]:
    """每人独立投票的 part 标签(1..14) -> RL 部位组(0..13) 诊断值（不作映射来源）。"""
    height, width = part.shape
    votes: dict[int, int] = {}
    for label in range(1, 15):
        counts: dict[int, int] = {}
        for person in (0, 1):
            mask = (part == label) & (owner == person)
            if int(mask.sum()) < CONTACT_MIN_MASK:
                continue
            joints: set[int] = set()
            for joint in range(17):
                x, y = coco_xy[person, joint]
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < width and 0 <= yi < height and mask[yi, xi]:
                    joints.add(joint)
            if not joints:
                continue
            best_group, best_score = None, 0.0
            for group, ids in enumerate(PART_POINTS):
                overlap = len(set(ids) & joints) / len(joints)
                if overlap > best_score:
                    best_group, best_score = group, overlap
            if best_group is not None:
                counts[best_group] = counts.get(best_group, 0) + 1
        if counts:
            votes[label] = max(counts.items(), key=lambda item: item[1])[0]
    return votes


def derive_contacts(npz_path: Path, target_poses) -> tuple[tuple, list[dict]]:
    """从重定向网格的 3D 顶点证据自动推导接触对 (a, b, expected_px)。

    映射用权威 nominal：part 标签 ``V6_part_{L}``（即数组值 L）对应 RL PART_POINTS 组 ``L-1``；
    每人独立的关节投票只作诊断记录。接触判定：两人各自标签区域的网格顶点最小 3D 距离
    ≤ ``CONTACT_3D_M``（真实表面接触量级，与 check.json 的穿模检查同尺度）。
    期望距离按 RL 原语义取 SDPose 目标姿态对应部位关节均值在打分画布下的距离。
    """
    data = np.load(npz_path)
    part = np.asarray(data["part"])
    owner = np.asarray(data["owner"])
    height, width = part.shape
    verts = np.asarray(data["vertices_cam"], dtype=float)  # (2, N, 3) 相机坐标（米）
    intrinsic = np.asarray(data["K"], dtype=float)
    uv = (intrinsic @ verts.reshape(-1, 3).T).T
    uv = (uv[:, :2] / uv[:, 2:3]).reshape(2, verts.shape[1], 2)

    # 拥抱等姿势两人 2D 互相遮挡：顶点只认 owner==本人的像素标签，被遮挡面剔除
    vertex_labels = np.zeros((2, verts.shape[1]), dtype=int)
    for person in (0, 1):
        xi = np.clip(np.round(uv[person, :, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.round(uv[person, :, 1]).astype(int), 0, height - 1)
        sampled = part[yi, xi].astype(int)
        owned = owner[yi, xi] == person
        vertex_labels[person] = np.where(owned, sampled, 0)

    coco_xy = np.asarray(data["dwpose134_xy"], dtype=float)[:, GEO134_TO_COCO17, :]  # (2,17,2)
    votes = label_votes(part, np.asarray(data["owner"]), coco_xy)
    disagreements = {label: vote for label, vote in votes.items() if vote != label - 1}

    contacts: list[dict] = []
    for label_a in range(1, 15):
        cloud_a = verts[0][vertex_labels[0] == label_a]
        if len(cloud_a) < CONTACT_MIN_VERTS:
            continue
        for label_b in range(1, 15):
            cloud_b = verts[1][vertex_labels[1] == label_b]
            if len(cloud_b) < CONTACT_MIN_VERTS:
                continue
            gap = float(np.linalg.norm(cloud_a[:, None, :] - cloud_b[None, :, :], axis=2).min())
            if gap > CONTACT_3D_M:
                continue
            group_a, group_b = label_a - 1, label_b - 1
            p0, p1 = _part(target_poses[0], group_a), _part(target_poses[1], group_b)
            if p0 is None or p1 is None:
                continue
            contacts.append(
                {
                    "a": group_a,
                    "b": group_b,
                    "labels": [label_a, label_b],
                    "gap_m": gap,
                    "expected_px": float(np.linalg.norm(p0 - p1)),
                }
            )
    unique: dict[tuple[int, int], dict] = {}
    for contact in contacts:
        key = (contact["a"], contact["b"])
        if key not in unique or contact["gap_m"] < unique[key]["gap_m"]:
            unique[key] = contact
    ordered = sorted(unique.values(), key=lambda item: (item["a"], item["b"]))
    reward_contacts = tuple((c["a"], c["b"], c["expected_px"]) for c in ordered)
    diagnostics = {
        "threshold_m": CONTACT_3D_M,
        "votes": {str(k): v for k, v in sorted(votes.items())},
        "nominal_disagreements": {str(k): v for k, v in sorted(disagreements.items())},
        "contacts": ordered,
    }
    return reward_contacts, diagnostics


def composite(reward: dict) -> tuple[float, list[str]]:
    keys = [key for key in DUAL_WEIGHTS if key in reward]
    weight_sum = sum(DUAL_WEIGHTS[key] for key in keys)
    return sum(DUAL_WEIGHTS[key] * reward[key] for key in keys) / weight_sum, keys


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BEACH14 / "results" / "rl_score")
    parser.add_argument("--templates", nargs="*", default=list(TEMPLATES))
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output dir: {args.output}")
    args.output.mkdir(parents=True)

    gpu = wait_for_gpu()
    runtime = sdpose.load_runtime(device="cuda:0")
    identity = IdentityMetric(models_root=MODELS)
    source_faces = source_embeddings(identity, SOURCE_IMAGE)
    print(f"source embeddings ready: {[float(e @ source_faces[1]) for e in source_faces]}", flush=True)

    results: dict = {}
    for template in args.templates:
        template_dir = BEACH14 / "results" / template
        inputs_dir = template_dir / "_inputs"
        b_reference = Image.open(inputs_dir / "B_reference.png").convert("RGB")
        preprocess = json.loads((inputs_dir / "source_preprocess.json").read_text())
        # 与 final_image 同一裁剪基准（A 画布 content_box），保证目标/候选同一像素空间
        x0, y0, x1, y1 = preprocess["a_content_box"]
        target_image = b_reference.crop((x0, y0, x1, y1))
        content_hw = (target_image.width, target_image.height)

        entry: dict = {"content_hw": list(content_hw), "arms": {}}
        try:
            target_poses = detect_people(target_image, runtime)
            if len(target_poses) != 2:
                raise ValueError(f"target needs exactly 2 foreground persons, got {len(target_poses)}")
            target_faces = faces_for_poses(identity, target_image, target_poses)
            identities = tuple(
                source_faces[slot] if gt is not None else None
                for slot, gt in enumerate(target_faces)
            )
            npz_path = POSE14 / template / "maps_3d.npz"
            contacts, contact_diag = derive_contacts(npz_path, target_poses)
            target = RewardTarget(tuple(target_poses), identities, content_hw, contacts)
            entry.update(
                {
                    "target": {
                        "persons": len(target_poses),
                        "id_eligible": sum(embedding is not None for embedding in identities),
                        "x_centers_px": [float(0.5 * (pose.box[0] + pose.box[2])) for pose in target_poses],
                        "contacts": contact_diag,
                    }
                }
            )
        except ValueError as error:
            entry["unscorable"] = f"{type(error).__name__}: {error}"
            results[template] = entry
            print(f"{template}: UNSCORABLE target ({error})", flush=True)
            continue

        for arm in args.arms:
            arm_dir = template_dir / arm
            image_path = arm_dir / "generation" / "final_image.png"
            if not image_path.is_file():
                entry["arms"][arm] = {"missing": True}
                continue
            candidate = Image.open(image_path).convert("RGB")
            if candidate.size != tuple(content_hw):
                entry["arms"][arm] = {
                    "missing": True,
                    "note": f"size {candidate.size} != content {tuple(content_hw)}",
                }
                continue
            poses = detect_people(candidate, runtime)
            faces = faces_for_poses(identity, candidate, poses)
            reward, details = score_candidate(target, tuple(poses), faces)
            score, keys = composite(reward)
            entry["arms"][arm] = {
                "reward": reward,
                "details": details,
                "composite": score,
                "composite_keys": keys,
                "detections": len(poses),
            }
            print(
                f"{template} {arm}: composite {score:.4f} "
                + " ".join(f"{k}={reward[k]:.3f}" for k in sorted(reward)),
                flush=True,
            )
        results[template] = entry

    # ---- 汇总 ----
    scorable = [template for template, entry in results.items() if "unscorable" not in entry]
    summary: dict = {"scorable_templates": scorable, "arms": {}}
    for arm in args.arms:
        rows = [results[t]["arms"][arm] for t in scorable if "reward" in results[t]["arms"].get(arm, {})]
        summary["arms"][arm] = {
            "label": ARMS.get(arm, arm),
            "composite": aggregate([row["composite"] for row in rows]),
            "components": {
                key: aggregate([row["reward"][key] for row in rows if key in row["reward"]])
                for key in DUAL_WEIGHTS
            },
        }
    ranking = sorted(
        summary["arms"], key=lambda arm: (summary["arms"][arm]["composite"]["mean"] or 0.0), reverse=True
    )
    summary["ranking_by_mean_composite"] = ranking

    # ---- 产物 ----
    manifest = {
        "kernel": {
            "rewards_py_sha256": sha256_file(RUNTIME / "src/pose_control/qwen21_full/native_rl/rewards.py"),
            "runtime_v3": str(RUNTIME),
        },
        "pose_backend": {
            "implementation": str(RUNTIME / "src/evaluation/sdpose.py"),
            "sdpose_sha256": sha256_file(RUNTIME / "src/evaluation/sdpose.py"),
            "ckpt_dir": str(sdpose.DEFAULT_SDPOSE_CKPT_DIR),
            "ood_root": str(sdpose.DEFAULT_SDPOSE_OOD_ROOT),
            "score_threshold": SCORE_THRESHOLD,
            "min_area": MIN_AREA,
        },
        "identity_backend": {"antelopev2_root": str(MODELS / "antelopev2"), "source_image": str(SOURCE_IMAGE)},
        "contact_derivation": {
            "part_maps": str(POSE14),
            "vertex3d_threshold_m": CONTACT_3D_M,
            "min_verts": CONTACT_MIN_VERTS,
            "mapping": "nominal V6_part_{L} -> RL group {L-1}; per-person joint votes recorded as diagnostics",
        },
        "weights": DUAL_WEIGHTS,
        "gpu": gpu,
    }
    (args.output / "rl_score.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (args.output / "summary.json").write_text(
        json.dumps({"summary": summary, "manifest": manifest}, ensure_ascii=False, indent=1)
    )

    lines = [
        "# 2.beach14 RL reward 打分排名（seed42，SDPose 后端）",
        "",
        f"打分内核：runtime-v3 `native_rl/rewards.py` 冻结 `score_candidate` + 双人权重 "
        + " / ".join(f"{k} {w:.2f}" for k, w in DUAL_WEIGHTS.items())
        + "；姿态后端 SDPose-Wholebody + YOLO11x；身份 antelopev2；contact 由 part 图自动推导。",
        f"可打分模板：{len(scorable)}/14"
        + ("" if not results else "；不可打分：" + "、".join(t for t, e in results.items() if "unscorable" in e) if len(scorable) < len(results) else ""),
        "",
        "| 排名 | 臂 | composite 均值 | 中位数 | pose | identity | binding | layout | contact |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, arm in enumerate(ranking, 1):
        arm_summary = summary["arms"][arm]
        comp = arm_summary["composite"]
        comps = arm_summary["components"]
        lines.append(
            f"| {rank} | {arm_summary['label']} | {comp['mean']:.4f} | {comp['median']:.4f} | "
            + f"{comps['pose']['mean']:.3f} | {comps['identity']['mean']:.3f} | "
            + f"{comps['binding']['mean']:.3f} | {comps['layout']['mean']:.3f} | "
            + (f"{comps['contact']['mean']:.3f}" if comps["contact"]["mean"] is not None else "-")
            + " |"
        )
    lines += ["", "## 每模板 composite", "", "| 模板 | " + " | ".join(summary["arms"][a]["label"] for a in args.arms) + " |",
              "|---" * (len(args.arms) + 1) + "|"]
    for template, entry in results.items():
        cells = []
        for arm in args.arms:
            arm_entry = entry["arms"].get(arm, {})
            if "reward" in arm_entry:
                cells.append(f"{arm_entry['composite']:.3f}")
            elif entry.get("unscorable"):
                cells.append("target 不合格")
            else:
                cells.append("-")
        lines.append(f"| {template} | " + " | ".join(cells) + " |")
    (args.output / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 面板图 ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (axis_bar, axis_heat) = plt.subplots(1, 2, figsize=(15, 6), width_ratios=[1, 1.6])
    means = [summary["arms"][arm]["composite"]["mean"] or 0.0 for arm in ranking]
    labels = [summary["arms"][arm]["label"] for arm in ranking]
    axis_bar.barh(range(len(ranking)), means, color="#4a7ebb")
    axis_bar.set_yticks(range(len(ranking)), labels)
    axis_bar.invert_yaxis()
    axis_bar.set_xlabel("mean composite (DUAL_WEIGHTS)")
    axis_bar.set_title("RL composite ranking")

    grid, texts = [], []
    for template in TEMPLATES:
        row, row_texts = [], []
        for arm in args.arms:
            arm_entry = results.get(template, {}).get("arms", {}).get(arm, {})
            if "reward" in arm_entry:
                row.append(arm_entry["composite"])
                row_texts.append(f"{arm_entry['composite']:.2f}")
            else:
                row.append(np.nan)
                row_texts.append("x")
        grid.append(row)
        texts.append(row_texts)
    matrix = np.array(grid, dtype=float)
    image = axis_heat.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axis_heat.set_xticks(range(len(args.arms)), [ARMS[a] for a in args.arms], rotation=20, ha="right")
    axis_heat.set_yticks(range(len(TEMPLATES)), TEMPLATES)
    for y in range(len(TEMPLATES)):
        for x in range(len(args.arms)):
            if not np.isnan(matrix[y, x]):
                axis_heat.text(x, y, texts[y][x], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis_heat, label="composite")
    axis_heat.set_title("per-template composite (x = unscorable)")
    figure.tight_layout()
    figure.savefig(args.output / "rl_score_panel.png", dpi=150)

    print(json.dumps({"ranking": ranking, "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

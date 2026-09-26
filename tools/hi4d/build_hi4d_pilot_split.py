#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# ============================================================
# Config
# ============================================================

HI4D_ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D"
).resolve()

OUT_ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
).resolve()

SEED = 2026

# Pilot:
# 10 identity pairs total
# 6 train / 2 val / 2 test
N_TRAIN_PAIRS = 6
N_VAL_PAIRS = 2
N_TEST_PAIRS = 2

# 每个 pair 最多使用 3 个动作序列
ACTIONS_PER_PAIR = 3

# 每个 action 最多取 80 个 temporal frames
FRAMES_PER_ACTION = 80


# ============================================================
# Helpers
# ============================================================

def np_scalar(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0].item()


def frame_id_from_path(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return None


def evenly_pick(ids, n):
    """从排序后的 frame ids 中均匀取 n 个，不随机扎堆。"""
    ids = sorted(set(int(x) for x in ids))

    if n <= 0 or len(ids) == 0:
        return []

    if len(ids) <= n:
        return ids

    idx = np.linspace(0, len(ids) - 1, n)
    idx = np.round(idx).astype(int)

    out = []
    seen = set()

    for i in idx:
        fid = ids[int(i)]
        if fid not in seen:
            seen.add(fid)
            out.append(fid)

    # linspace + round 理论上可能产生重复，再补齐
    if len(out) < n:
        for fid in ids:
            if fid not in seen:
                seen.add(fid)
                out.append(fid)
                if len(out) == n:
                    break

    return sorted(out)


def resolve_camera_dir(parent: Path, cam_id):
    """
    Hi4D camera folder可能表现为 4 / 04 等。
    用整数比较找到真正目录。
    """
    if not parent.exists():
        return None

    cam_int = int(cam_id)

    for p in parent.iterdir():
        if not p.is_dir():
            continue

        try:
            if int(p.name) == cam_int:
                return p
        except ValueError:
            continue

    return None


def load_action_info(pair_name: str, action_dir: Path):
    meta_path = action_dir / "meta.npz"
    if not meta_path.exists():
        return None

    try:
        meta = dict(np.load(meta_path, allow_pickle=True))
    except Exception as e:
        print(f"[WARN] cannot load {meta_path}: {e}")
        return None

    if "mono_cam" not in meta:
        print(f"[WARN] mono_cam missing: {meta_path}")
        return None

    mono_cam = int(np_scalar(meta["mono_cam"]))

    images_root = action_dir / "images"
    mask_root = action_dir / "seg" / "img_seg_mask"
    smpl_root = action_dir / "smpl"

    image_cam_dir = resolve_camera_dir(images_root, mono_cam)
    mask_cam_dir = resolve_camera_dir(mask_root, mono_cam)

    if image_cam_dir is None:
        print(
            f"[WARN] missing mono_cam image directory "
            f"{pair_name}/{action_dir.name}: camera {mono_cam}"
        )
        return None

    if mask_cam_dir is None:
        print(
            f"[WARN] missing mono_cam mask directory "
            f"{pair_name}/{action_dir.name}: camera {mono_cam}"
        )
        return None

    mask0_dir = mask_cam_dir / "0"
    mask1_dir = mask_cam_dir / "1"

    # --------------------------------------------------------
    # RGB frames
    # --------------------------------------------------------
    image_files = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in image_cam_dir.glob(ext):
            fid = frame_id_from_path(p)
            if fid is not None:
                image_files[fid] = p

    # --------------------------------------------------------
    # official SMPL frames
    # --------------------------------------------------------
    smpl_files = {}
    for p in smpl_root.glob("*.npz"):
        fid = frame_id_from_path(p)
        if fid is not None:
            smpl_files[fid] = p

    # --------------------------------------------------------
    # official person masks
    # --------------------------------------------------------
    mask0_files = {}
    if mask0_dir.exists():
        for p in mask0_dir.glob("*.png"):
            fid = frame_id_from_path(p)
            if fid is not None:
                mask0_files[fid] = p

    mask1_files = {}
    if mask1_dir.exists():
        for p in mask1_dir.glob("*.png"):
            fid = frame_id_from_path(p)
            if fid is not None:
                mask1_files[fid] = p

    # 这里只接受四种官方资产全部存在的 frame
    valid_ids = (
        set(image_files.keys())
        & set(smpl_files.keys())
        & set(mask0_files.keys())
        & set(mask1_files.keys())
    )

    valid_ids = sorted(valid_ids)

    if len(valid_ids) == 0:
        print(f"[WARN] zero valid frames: {pair_name}/{action_dir.name}")
        return None

    contact_ids = set()

    if "contact_ids" in meta:
        c = np.asarray(meta["contact_ids"]).reshape(-1)

        for x in c:
            try:
                contact_ids.add(int(x))
            except Exception:
                pass

    contact_valid = sorted(set(valid_ids) & contact_ids)
    noncontact_valid = sorted(set(valid_ids) - contact_ids)

    genders = []
    if "genders" in meta:
        try:
            genders = [
                str(x)
                for x in np.asarray(meta["genders"]).reshape(-1).tolist()
            ]
        except Exception:
            genders = []

    return {
        "pair": pair_name,
        "action": action_dir.name,
        "action_dir": action_dir,
        "meta_path": meta_path,
        "mono_cam": mono_cam,
        "genders": genders,
        "valid_ids": valid_ids,
        "contact_ids": contact_valid,
        "noncontact_ids": noncontact_valid,
        "image_files": image_files,
        "smpl_files": smpl_files,
        "mask0_files": mask0_files,
        "mask1_files": mask1_files,
        "num_frames": len(valid_ids),
        "num_contact": len(contact_valid),
        "contact_ratio": (
            len(contact_valid) / len(valid_ids)
            if valid_ids else 0.0
        ),
    }


def choose_actions(action_infos, k):
    """
    每个身份 pair 尽量选择：
      1. 接触比例最高的 action
      2. 最长 action
      3. 接触帧数量最多的另一个 action

    避免 Pilot 全是同一类型短动作。
    """
    if len(action_infos) <= k:
        return sorted(action_infos, key=lambda x: x["action"])

    selected = []

    def add(info):
        if info is not None and info not in selected:
            selected.append(info)

    # 高接触比例
    add(
        max(
            action_infos,
            key=lambda x: (
                x["contact_ratio"],
                x["num_contact"],
                x["num_frames"],
            ),
        )
    )

    # 最长
    remaining = [x for x in action_infos if x not in selected]
    if remaining:
        add(max(remaining, key=lambda x: x["num_frames"]))

    # 最大 contact 数
    remaining = [x for x in action_infos if x not in selected]
    if remaining:
        add(
            max(
                remaining,
                key=lambda x: (
                    x["num_contact"],
                    x["contact_ratio"],
                    x["num_frames"],
                ),
            )
        )

    # 如果以后 ACTIONS_PER_PAIR > 3
    remaining = [x for x in action_infos if x not in selected]
    remaining = sorted(
        remaining,
        key=lambda x: (
            x["num_frames"],
            x["num_contact"],
        ),
        reverse=True,
    )

    for info in remaining:
        if len(selected) >= k:
            break
        add(info)

    return selected[:k]


def select_frames(info, split, n):
    """
    TRAIN:
      尽量 50% contact / 50% non-contact，
      避免 contact interaction 被普通站立帧淹没。

    VAL / TEST:
      时间轴均匀抽帧，不人为改变真实 contact 比例。
    """

    valid_ids = info["valid_ids"]
    n = min(n, len(valid_ids))

    if split != "train":
        return evenly_pick(valid_ids, n)

    contact = info["contact_ids"]
    noncontact = info["noncontact_ids"]

    if contact and noncontact:
        n_contact = min(len(contact), n // 2)
        n_non = min(len(noncontact), n - n_contact)

        # 如果一边不足，另一边补
        remaining = n - n_contact - n_non

        if remaining > 0:
            extra_c = min(len(contact) - n_contact, remaining)
            n_contact += extra_c
            remaining -= extra_c

        if remaining > 0:
            extra_n = min(len(noncontact) - n_non, remaining)
            n_non += extra_n
            remaining -= extra_n

        picked = (
            evenly_pick(contact, n_contact)
            + evenly_pick(noncontact, n_non)
        )

        return sorted(set(picked))

    return evenly_pick(valid_ids, n)


# ============================================================
# Main
# ============================================================

def main():
    if not HI4D_ROOT.exists():
        raise FileNotFoundError(
            f"Hi4D root does not exist:\n{HI4D_ROOT}"
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Discover pair folders
    # --------------------------------------------------------
    pair_dirs = sorted(
        [
            p
            for p in HI4D_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("pair")
        ],
        key=lambda p: p.name,
    )

    print("\n===== Hi4D discovery =====")
    print("root:", HI4D_ROOT)
    print("pairs found:", len(pair_dirs))
    print([p.name for p in pair_dirs])

    n_total = N_TRAIN_PAIRS + N_VAL_PAIRS + N_TEST_PAIRS

    if len(pair_dirs) < n_total:
        raise RuntimeError(
            f"Need at least {n_total} valid pair directories, "
            f"but found {len(pair_dirs)}"
        )

    # --------------------------------------------------------
    # Minimal audit: load actions once
    # --------------------------------------------------------
    pair_infos = {}

    for pair_dir in pair_dirs:
        action_dirs = sorted(
            [
                p
                for p in pair_dir.iterdir()
                if p.is_dir() and p.name.startswith("action")
            ],
            key=lambda p: p.name,
        )

        infos = []

        for action_dir in action_dirs:
            info = load_action_info(pair_dir.name, action_dir)
            if info is not None:
                infos.append(info)

        if len(infos) > 0:
            pair_infos[pair_dir.name] = infos

    valid_pairs = sorted(pair_infos.keys())

    print("\nvalid pairs:", len(valid_pairs))

    if len(valid_pairs) < n_total:
        raise RuntimeError(
            f"Only {len(valid_pairs)} pairs contain valid "
            f"RGB+SMPL+Mask0+Mask1 data, need {n_total}"
        )

    # --------------------------------------------------------
    # Reproducible pair-level split
    # --------------------------------------------------------
    rng = random.Random(SEED)

    shuffled = valid_pairs.copy()
    rng.shuffle(shuffled)

    selected = shuffled[:n_total]

    train_pairs = sorted(selected[:N_TRAIN_PAIRS])
    val_pairs = sorted(
        selected[
            N_TRAIN_PAIRS:
            N_TRAIN_PAIRS + N_VAL_PAIRS
        ]
    )
    test_pairs = sorted(
        selected[
            N_TRAIN_PAIRS + N_VAL_PAIRS:
        ]
    )

    split_by_pair = {}

    for p in train_pairs:
        split_by_pair[p] = "train"

    for p in val_pairs:
        split_by_pair[p] = "val"

    for p in test_pairs:
        split_by_pair[p] = "test"

    split_json = {
        "seed": SEED,
        "strategy": "pair_level_identity_disjoint",
        "hi4d_root": str(HI4D_ROOT),
        "pilot": {
            "num_pairs": n_total,
            "actions_per_pair": ACTIONS_PER_PAIR,
            "frames_per_action": FRAMES_PER_ACTION,
            "camera_policy": "official_mono_cam",
            "train_frame_sampling": "balanced_contact_noncontact_when_possible",
            "val_test_frame_sampling": "uniform_temporal",
        },
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
    }

    with open(
        OUT_ROOT / "split.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_json, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Create manifests + links
    # --------------------------------------------------------
    sequences = []
    frames = []

    raw_links = OUT_ROOT / "raw_links"

    for split in ("train", "val", "test"):
        (raw_links / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    for pair_name in selected:
        split = split_by_pair[pair_name]

        actions = choose_actions(
            pair_infos[pair_name],
            ACTIONS_PER_PAIR,
        )

        for info in actions:
            chosen_frames = select_frames(
                info,
                split,
                FRAMES_PER_ACTION,
            )

            seq_entry = {
                "split": split,
                "pair": pair_name,
                "action": info["action"],
                "mono_cam": info["mono_cam"],
                "genders": info["genders"],
                "num_valid_frames_total": info["num_frames"],
                "num_contact_frames_total": info["num_contact"],
                "contact_ratio_total": round(
                    info["contact_ratio"], 6
                ),
                "num_selected_frames": len(chosen_frames),
                "raw_action_dir": str(
                    info["action_dir"].resolve()
                ),
            }

            sequences.append(seq_entry)

            # soft-link whole action for easy inspection
            pair_link_root = (
                raw_links / split / pair_name
            )
            pair_link_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            link_path = (
                pair_link_root / info["action"]
            )

            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()

            os.symlink(
                info["action_dir"].resolve(),
                link_path,
                target_is_directory=True,
            )

            contact_set = set(info["contact_ids"])

            for fid in chosen_frames:
                frames.append(
                    {
                        "split": split,
                        "pair": pair_name,
                        "action": info["action"],
                        "frame_id": int(fid),
                        "frame_name": f"{fid:06d}",
                        "camera_id": int(info["mono_cam"]),
                        "is_contact": bool(
                            fid in contact_set
                        ),

                        "rgb": str(
                            info["image_files"][fid].resolve()
                        ),
                        "official_smpl": str(
                            info["smpl_files"][fid].resolve()
                        ),
                        "person_mask_A": str(
                            info["mask0_files"][fid].resolve()
                        ),
                        "person_mask_B": str(
                            info["mask1_files"][fid].resolve()
                        ),

                        "meta": str(
                            info["meta_path"].resolve()
                        ),

                        "camera_file": str(
                            (
                                info["action_dir"]
                                / "cameras"
                                / "rgb_cameras.npz"
                            ).resolve()
                        ),

                        "person_A": 0,
                        "person_B": 1,

                        "correspondence": {
                            "source_A_to_target_A": 0,
                            "source_B_to_target_B": 1,
                        },
                    }
                )

    # --------------------------------------------------------
    # Write jsonl
    # --------------------------------------------------------
    with open(
        OUT_ROOT / "sequences.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for x in sequences:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        OUT_ROOT / "frames.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for x in frames:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    frame_split_counts = Counter(
        x["split"] for x in frames
    )

    seq_split_counts = Counter(
        x["split"] for x in sequences
    )

    contact_split_counts = Counter(
        x["split"]
        for x in frames
        if x["is_contact"]
    )

    summary = {
        "seed": SEED,
        "num_available_valid_pairs": len(valid_pairs),
        "selected_pairs_total": len(selected),
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,
        "num_sequences": dict(seq_split_counts),
        "num_frames": dict(frame_split_counts),
        "num_contact_frames": dict(contact_split_counts),
        "total_frames": len(frames),
        "total_sequences": len(sequences),
        "output_root": str(OUT_ROOT),
    }

    with open(
        OUT_ROOT / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print("\n==============================================")
    print("          Hi4D PILOT SUBSET READY")
    print("==============================================")

    print("\nTRAIN PAIRS:")
    print("  ", train_pairs)

    print("\nVAL PAIRS:")
    print("  ", val_pairs)

    print("\nTEST PAIRS:")
    print("  ", test_pairs)

    print("\nSEQUENCES:")
    print(dict(seq_split_counts))

    print("\nFRAMES:")
    print(dict(frame_split_counts))

    print("\nCONTACT FRAMES:")
    print(dict(contact_split_counts))

    print("\nTOTAL:")
    print("  sequences =", len(sequences))
    print("  frames    =", len(frames))

    print("\nOUTPUT:")
    print(" ", OUT_ROOT)

    print("\nFiles:")
    print(" ", OUT_ROOT / "split.json")
    print(" ", OUT_ROOT / "sequences.jsonl")
    print(" ", OUT_ROOT / "frames.jsonl")
    print(" ", OUT_ROOT / "summary.json")

    print("\nOriginal Hi4D was NOT modified.")
    print("raw_links are symbolic links only.")


if __name__ == "__main__":
    main()

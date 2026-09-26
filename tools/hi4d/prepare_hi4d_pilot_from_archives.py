#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from collections import defaultdict, Counter

import numpy as np


# ============================================================
# CONFIG
# ============================================================

ARCHIVE_ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D"
).resolve()

OUT_ROOT = Path(
    "/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1"
).resolve()

RAW_ROOT = OUT_ROOT / "raw"

# Pair-level identity-disjoint split
SPLITS = {
    "train": [
        "pair01",
        "pair09",
        "pair10",
        "pair12",
        "pair14",
        "pair16",
    ],
    "val": [
        "pair22",
        "pair28",
    ],
    "test": [
        "pair00",
        "pair02",
    ],
}

ACTIONS_PER_PAIR = 3
FRAMES_PER_ACTION = 80


# ============================================================
# HELPERS
# ============================================================

PAIR_RE = re.compile(r"^(pair\d+)(?:_\d+)?\.tar\.gz$")
PAIR_DIR_RE = re.compile(r"^pair\d+$")


def json_safe(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def get_action_from_member(name):
    """
    Hi4D sequence 实际命名不是 actionXX，而是例如：

        pair01/hug01/...
        pair01/highfive01/...
        pair01/basketball01/...
        pair01/fight01/...
        pair01/talk01/...

    因此把 pairXX 后面的一级目录视为 interaction sequence。
    """
    parts = PurePosixPath(name).parts

    if len(parts) < 2:
        return None

    if not PAIR_DIR_RE.fullmatch(parts[0]):
        return None

    seq = parts[1]

    # 排除 pair 根目录下普通文件
    if "." in seq:
        return None

    return seq


def relative_from_action(name):
    """
    pair01/hug01/images/76/000001.jpg
        ->
    hug01/images/76/000001.jpg
    """
    parts = PurePosixPath(name).parts

    if len(parts) < 2:
        return None

    if not PAIR_DIR_RE.fullmatch(parts[0]):
        return None

    return PurePosixPath(*parts[1:])


def evenly_pick(items, n):
    items = sorted(items, key=lambda x: int(x))

    if len(items) <= n:
        return items

    idx = np.linspace(0, len(items) - 1, n)
    idx = np.round(idx).astype(int)

    out = []
    seen = set()

    for i in idx:
        x = items[int(i)]
        if x not in seen:
            seen.add(x)
            out.append(x)

    if len(out) < n:
        for x in items:
            if x not in seen:
                out.append(x)
                seen.add(x)
                if len(out) == n:
                    break

    return sorted(out, key=lambda x: int(x))


def load_npz_from_bytes(data):
    with np.load(io.BytesIO(data), allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def get_scalar_int(x):
    arr = np.asarray(x).reshape(-1)
    if len(arr) == 0:
        return None
    return int(arr[0])


def archive_pair_name(path):
    m = PAIR_RE.match(path.name)
    return m.group(1) if m else None


def ensure_clean_raw_root():
    # 只清理我们自己的 Pilot 输出，不碰原始 Hi4D
    if RAW_ROOT.exists():
        print(f"[INFO] removing stale pilot raw dir: {RAW_ROOT}")
        shutil.rmtree(RAW_ROOT)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER ARCHIVES
# ============================================================

def discover_archives():
    by_pair = defaultdict(list)

    for p in sorted(ARCHIVE_ROOT.glob("pair*.tar.gz")):
        pair = archive_pair_name(p)
        if pair:
            by_pair[pair].append(p)

    selected_pairs = sum(SPLITS.values(), [])

    print("\n===== ARCHIVE DISCOVERY =====")

    for pair in selected_pairs:
        archives = by_pair.get(pair, [])

        if not archives:
            raise RuntimeError(
                f"Missing archive for selected pair: {pair}"
            )

        print(
            f"{pair}: "
            + ", ".join(x.name for x in archives)
        )

    return by_pair


# ============================================================
# PASS 1
# Scan archive structure without extracting huge raw scans
# ============================================================

def new_action_record(pair, action):
    return {
        "pair": pair,
        "action": action,

        "meta": None,
        "meta_source": None,

        "camera": None,

        # camera_id -> frame_stem -> (archive, member)
        "images": defaultdict(dict),

        # camera_id -> subject -> frame_stem -> (archive, member)
        "masks": defaultdict(
            lambda: defaultdict(dict)
        ),

        # frame_stem -> (archive, member)
        "smpl": {},
    }


def scan_pair_archives(pair, archives):
    print(f"\n===== SCANNING {pair} =====")

    actions = {}

    for archive in archives:
        print(f"[SCAN] {archive.name}")

        # streaming mode: does not unpack to disk
        with tarfile.open(archive, mode="r|gz") as tf:

            for member in tf:
                if not member.isfile():
                    continue

                name = member.name
                action = get_action_from_member(name)

                if action is None:
                    continue

                rec = actions.setdefault(
                    action,
                    new_action_record(pair, action)
                )

                rel = relative_from_action(name)
                if rel is None:
                    continue

                rel_parts = rel.parts

                # --------------------------------------------
                # meta.npz
                # --------------------------------------------
                if rel_parts == (action, "meta.npz"):
                    f = tf.extractfile(member)
                    if f is not None:
                        data = f.read()
                        rec["meta"] = load_npz_from_bytes(data)
                        rec["meta_source"] = (
                            str(archive),
                            name,
                        )
                    continue

                # --------------------------------------------
                # camera
                # --------------------------------------------
                if (
                    len(rel_parts) >= 3
                    and rel_parts[1] == "cameras"
                    and rel_parts[-1] == "rgb_cameras.npz"
                ):
                    rec["camera"] = (
                        str(archive),
                        name,
                    )
                    continue

                # --------------------------------------------
                # RGB
                # actionXX/images/<cam>/<frame>.jpg
                # --------------------------------------------
                if (
                    len(rel_parts) >= 4
                    and rel_parts[1] == "images"
                ):
                    cam = rel_parts[2]
                    suffix = Path(rel_parts[-1]).suffix.lower()

                    if suffix in {".jpg", ".jpeg", ".png"}:
                        stem = Path(rel_parts[-1]).stem

                        if stem.isdigit():
                            rec["images"][cam][stem] = (
                                str(archive),
                                name,
                            )
                    continue

                # --------------------------------------------
                # Official 2D instance masks
                # actionXX/seg/img_seg_mask/<cam>/<0|1|all>/<frame>.png
                # --------------------------------------------
                if (
                    len(rel_parts) >= 6
                    and rel_parts[1] == "seg"
                    and rel_parts[2] == "img_seg_mask"
                ):
                    cam = rel_parts[3]
                    subject = rel_parts[4]
                    stem = Path(rel_parts[-1]).stem

                    if (
                        subject in {"0", "1", "all"}
                        and stem.isdigit()
                    ):
                        rec["masks"][cam][subject][stem] = (
                            str(archive),
                            name,
                        )
                    continue

                # --------------------------------------------
                # SMPL
                # actionXX/smpl/<frame>.npz
                # --------------------------------------------
                if (
                    len(rel_parts) >= 3
                    and rel_parts[1] == "smpl"
                    and rel_parts[-1].endswith(".npz")
                ):
                    stem = Path(rel_parts[-1]).stem

                    if stem.isdigit():
                        rec["smpl"][stem] = (
                            str(archive),
                            name,
                        )

    # --------------------------------------------------------
    # Finalize action info using official mono_cam
    # --------------------------------------------------------

    valid_actions = []

    for action, rec in sorted(actions.items()):

        if rec["meta"] is None:
            print(f"[WARN] {pair}/{action}: no meta.npz")
            continue

        if rec["camera"] is None:
            print(
                f"[WARN] {pair}/{action}: "
                "no cameras/rgb_cameras.npz"
            )
            continue

        meta = rec["meta"]

        if "mono_cam" not in meta:
            print(
                f"[WARN] {pair}/{action}: mono_cam missing"
            )
            continue

        mono_cam = str(get_scalar_int(meta["mono_cam"]))

        images = rec["images"].get(mono_cam, {})
        mask0 = (
            rec["masks"]
            .get(mono_cam, {})
            .get("0", {})
        )
        mask1 = (
            rec["masks"]
            .get(mono_cam, {})
            .get("1", {})
        )

        valid = (
            set(images.keys())
            & set(mask0.keys())
            & set(mask1.keys())
            & set(rec["smpl"].keys())
        )

        valid = sorted(valid, key=lambda x: int(x))

        if not valid:
            print(
                f"[WARN] {pair}/{action}: "
                f"0 valid frames for mono_cam={mono_cam}"
            )
            continue

        contact_ids = set()

        if "contact_ids" in meta:
            for x in np.asarray(
                meta["contact_ids"]
            ).reshape(-1):
                try:
                    contact_ids.add(int(x))
                except Exception:
                    pass

        contact_valid = [
            stem
            for stem in valid
            if int(stem) in contact_ids
        ]

        noncontact_valid = [
            stem
            for stem in valid
            if int(stem) not in contact_ids
        ]

        genders = []

        if "genders" in meta:
            try:
                genders = [
                    str(x)
                    for x in np.asarray(
                        meta["genders"]
                    ).reshape(-1).tolist()
                ]
            except Exception:
                pass

        rec["mono_cam"] = mono_cam
        rec["valid_frames"] = valid
        rec["contact_frames"] = contact_valid
        rec["noncontact_frames"] = noncontact_valid
        rec["num_valid"] = len(valid)
        rec["num_contact"] = len(contact_valid)
        rec["contact_ratio"] = (
            len(contact_valid) / len(valid)
        )
        rec["genders"] = genders

        valid_actions.append(rec)

        print(
            f"[OK] {pair}/{action}: "
            f"cam={mono_cam}, "
            f"frames={len(valid)}, "
            f"contact={len(contact_valid)} "
            f"({rec['contact_ratio']:.1%})"
        )

    return valid_actions


# ============================================================
# ACTION + FRAME SELECTION
# ============================================================

def choose_actions(action_infos, k):
    """
    尽量选：
      1) contact ratio 高
      2) sequence 长
      3) contact 数量高
    """

    if len(action_infos) <= k:
        return sorted(
            action_infos,
            key=lambda x: x["action"]
        )

    selected = []

    def add(x):
        if x is not None and x not in selected:
            selected.append(x)

    add(
        max(
            action_infos,
            key=lambda x: (
                x["contact_ratio"],
                x["num_contact"],
                x["num_valid"],
            ),
        )
    )

    rem = [
        x for x in action_infos
        if x not in selected
    ]

    if rem:
        add(
            max(
                rem,
                key=lambda x: x["num_valid"]
            )
        )

    rem = [
        x for x in action_infos
        if x not in selected
    ]

    if rem:
        add(
            max(
                rem,
                key=lambda x: (
                    x["num_contact"],
                    x["contact_ratio"],
                    x["num_valid"],
                ),
            )
        )

    rem = [
        x for x in action_infos
        if x not in selected
    ]

    rem = sorted(
        rem,
        key=lambda x: (
            x["num_valid"],
            x["num_contact"],
        ),
        reverse=True,
    )

    for x in rem:
        if len(selected) >= k:
            break
        add(x)

    return selected[:k]


def select_frames(rec, split, n):
    valid = rec["valid_frames"]

    n = min(n, len(valid))

    # Validation / test:
    # preserve temporal distribution
    if split != "train":
        return evenly_pick(valid, n)

    contact = rec["contact_frames"]
    noncontact = rec["noncontact_frames"]

    # Train:
    # when possible ~50/50 contact / non-contact
    if contact and noncontact:

        n_c = min(len(contact), n // 2)
        n_n = min(len(noncontact), n - n_c)

        remain = n - n_c - n_n

        if remain > 0:
            extra = min(
                len(contact) - n_c,
                remain
            )
            n_c += extra
            remain -= extra

        if remain > 0:
            extra = min(
                len(noncontact) - n_n,
                remain
            )
            n_n += extra

        out = (
            evenly_pick(contact, n_c)
            + evenly_pick(noncontact, n_n)
        )

        return sorted(
            set(out),
            key=lambda x: int(x)
        )

    return evenly_pick(valid, n)


# ============================================================
# BUILD EXTRACTION PLAN
# ============================================================

def add_needed(plan, source_tuple, dst):
    if source_tuple is None:
        return

    archive, member = source_tuple

    plan[archive][member] = str(dst)


def build_plan(selected):
    plan = defaultdict(dict)
    sequences_manifest = []
    frames_manifest = []

    for split, recs in selected.items():

        for rec in recs:

            pair = rec["pair"]
            action = rec["action"]
            cam = rec["mono_cam"]

            frames = select_frames(
                rec,
                split,
                FRAMES_PER_ACTION,
            )

            action_root = (
                RAW_ROOT
                / pair
                / action
            )

            # meta + camera
            add_needed(
                plan,
                rec["meta_source"],
                action_root / "meta.npz",
            )

            add_needed(
                plan,
                rec["camera"],
                action_root
                / "cameras"
                / "rgb_cameras.npz",
            )

            sequences_manifest.append({
                "split": split,
                "pair": pair,
                "action": action,
                "mono_cam": int(cam),
                "genders": rec["genders"],
                "num_valid_frames_total": rec["num_valid"],
                "num_contact_frames_total": rec["num_contact"],
                "contact_ratio_total": rec["contact_ratio"],
                "num_selected_frames": len(frames),
            })

            mask0 = rec["masks"][cam]["0"]
            mask1 = rec["masks"][cam]["1"]
            mask_all = rec["masks"][cam].get("all", {})

            for stem in frames:

                # RGB
                src_rgb = rec["images"][cam][stem]
                rgb_ext = Path(
                    src_rgb[1]
                ).suffix.lower()

                rgb_dst = (
                    action_root
                    / "images"
                    / cam
                    / f"{stem}{rgb_ext}"
                )

                add_needed(
                    plan,
                    src_rgb,
                    rgb_dst,
                )

                # SMPL
                smpl_dst = (
                    action_root
                    / "smpl"
                    / f"{stem}.npz"
                )

                add_needed(
                    plan,
                    rec["smpl"][stem],
                    smpl_dst,
                )

                # Masks A / B
                m0_dst = (
                    action_root
                    / "seg"
                    / "img_seg_mask"
                    / cam
                    / "0"
                    / f"{stem}.png"
                )

                m1_dst = (
                    action_root
                    / "seg"
                    / "img_seg_mask"
                    / cam
                    / "1"
                    / f"{stem}.png"
                )

                add_needed(
                    plan,
                    mask0[stem],
                    m0_dst,
                )

                add_needed(
                    plan,
                    mask1[stem],
                    m1_dst,
                )

                # Combined mask if present
                all_dst = None

                if stem in mask_all:
                    all_dst = (
                        action_root
                        / "seg"
                        / "img_seg_mask"
                        / cam
                        / "all"
                        / f"{stem}.png"
                    )

                    add_needed(
                        plan,
                        mask_all[stem],
                        all_dst,
                    )

                is_contact = (
                    stem in rec["contact_frames"]
                )

                frames_manifest.append({
                    "split": split,
                    "pair": pair,
                    "action": action,
                    "frame_id": int(stem),
                    "frame_stem": stem,
                    "camera_id": int(cam),
                    "is_contact": bool(is_contact),

                    "rgb": str(rgb_dst),
                    "official_smpl": str(smpl_dst),

                    "person_mask_A": str(m0_dst),
                    "person_mask_B": str(m1_dst),

                    "combined_mask": (
                        str(all_dst)
                        if all_dst is not None
                        else None
                    ),

                    "meta": str(
                        action_root / "meta.npz"
                    ),

                    "camera_file": str(
                        action_root
                        / "cameras"
                        / "rgb_cameras.npz"
                    ),

                    "person_A": 0,
                    "person_B": 1,
                    "correspondence": {
                        "A_source_to_A_target": 0,
                        "B_source_to_B_target": 1,
                    },
                })

    return plan, sequences_manifest, frames_manifest


# ============================================================
# EXTRACTION
# ============================================================

def execute_plan(plan):
    print("\n==========================================")
    print("SELECTIVE EXTRACTION")
    print("==========================================")

    for archive_str, wanted in sorted(plan.items()):

        archive = Path(archive_str)

        print(
            f"\n[EXTRACT] {archive.name} "
            f"({len(wanted)} files)"
        )

        remaining = set(wanted.keys())
        extracted = 0

        with tarfile.open(
            archive,
            mode="r|gz"
        ) as tf:

            for member in tf:

                if not member.isfile():
                    continue

                if member.name not in remaining:
                    continue

                dst = Path(wanted[member.name])
                dst.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                src = tf.extractfile(member)

                if src is None:
                    raise RuntimeError(
                        f"Cannot extract "
                        f"{archive.name}:{member.name}"
                    )

                with open(dst, "wb") as out:
                    shutil.copyfileobj(
                        src,
                        out,
                        length=1024 * 1024
                    )

                remaining.remove(member.name)
                extracted += 1

                if extracted % 100 == 0:
                    print(
                        f"  extracted "
                        f"{extracted}/{len(wanted)}"
                    )

        if remaining:
            examples = list(
                sorted(remaining)
            )[:10]

            raise RuntimeError(
                f"{archive.name}: "
                f"{len(remaining)} planned files "
                f"were not found.\nExamples:\n"
                + "\n".join(examples)
            )

        print(
            f"[DONE] {archive.name}: "
            f"{extracted} files"
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_extracted(frames_manifest):
    print("\n===== VALIDATION =====")

    missing = []

    required_keys = [
        "rgb",
        "official_smpl",
        "person_mask_A",
        "person_mask_B",
        "meta",
        "camera_file",
    ]

    for row in frames_manifest:

        for key in required_keys:
            p = Path(row[key])

            if not p.exists():
                missing.append(
                    (key, str(p))
                )

    if missing:
        print(
            f"[FAIL] missing files: {len(missing)}"
        )

        for x in missing[:20]:
            print(x)

        raise RuntimeError(
            "Pilot extraction validation failed."
        )

    print(
        f"[OK] all {len(frames_manifest)} "
        "selected frames have "
        "RGB + SMPL + MaskA + MaskB + Camera."
    )


# ============================================================
# WRITE MANIFESTS
# ============================================================

def write_outputs(
    selected,
    sequences_manifest,
    frames_manifest,
):
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    split_json = {
        "strategy": "pair_level_identity_disjoint",
        "train_pairs": SPLITS["train"],
        "val_pairs": SPLITS["val"],
        "test_pairs": SPLITS["test"],
        "actions_per_pair": ACTIONS_PER_PAIR,
        "frames_per_action": FRAMES_PER_ACTION,
        "camera_policy": "official_mono_cam",
        "train_sampling": (
            "approximately balanced "
            "contact/non-contact when possible"
        ),
        "val_test_sampling": (
            "uniform temporal sampling"
        ),
        "archive_root": str(ARCHIVE_ROOT),
        "raw_subset_root": str(RAW_ROOT),
    }

    with open(
        OUT_ROOT / "split.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            split_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUT_ROOT / "sequences.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for x in sequences_manifest:
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
        for x in frames_manifest:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    split_frames = Counter(
        x["split"]
        for x in frames_manifest
    )

    split_seq = Counter(
        x["split"]
        for x in sequences_manifest
    )

    split_contact = Counter(
        x["split"]
        for x in frames_manifest
        if x["is_contact"]
    )

    selected_actions = {
        split: [
            f"{r['pair']}/{r['action']}"
            for r in recs
        ]
        for split, recs in selected.items()
    }

    summary = {
        "num_pairs": {
            k: len(v)
            for k, v in SPLITS.items()
        },
        "num_sequences": dict(split_seq),
        "num_frames": dict(split_frames),
        "num_contact_frames": dict(
            split_contact
        ),
        "total_sequences": len(
            sequences_manifest
        ),
        "total_frames": len(
            frames_manifest
        ),
        "selected_actions": selected_actions,
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


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "==========================================\n"
        "Hi4D PILOT PREPARATION FROM TAR.GZ\n"
        "=========================================="
    )

    by_pair = discover_archives()

    selected = {
        "train": [],
        "val": [],
        "test": [],
    }

    # --------------------------------------------------------
    # Pass 1
    # --------------------------------------------------------
    for split, pairs in SPLITS.items():

        print(
            f"\n################ {split.upper()} "
            "################"
        )

        for pair in pairs:

            actions = scan_pair_archives(
                pair,
                by_pair[pair],
            )

            if not actions:
                raise RuntimeError(
                    f"No valid actions for {pair}"
                )

            chosen = choose_actions(
                actions,
                ACTIONS_PER_PAIR,
            )

            print(
                f"\n[SELECT] {pair}: "
                + ", ".join(
                    f"{x['action']}"
                    f"(n={x['num_valid']},"
                    f" contact={x['num_contact']})"
                    for x in chosen
                )
            )

            selected[split].extend(
                chosen
            )

    # --------------------------------------------------------
    # Plan
    # --------------------------------------------------------
    plan, seq_manifest, frame_manifest = (
        build_plan(selected)
    )

    total_files = sum(
        len(x)
        for x in plan.values()
    )

    print(
        "\n=========================================="
    )
    print("EXTRACTION PLAN")
    print("==========================================")
    print(
        "archives:",
        len(plan)
    )
    print(
        "files to extract:",
        total_files
    )
    print(
        "selected sequences:",
        len(seq_manifest)
    )
    print(
        "selected frames:",
        len(frame_manifest)
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------
    ensure_clean_raw_root()
    execute_plan(plan)

    # --------------------------------------------------------
    # Validate + manifests
    # --------------------------------------------------------
    validate_extracted(frame_manifest)

    write_outputs(
        selected,
        seq_manifest,
        frame_manifest,
    )

    print(
        "\n=========================================="
    )
    print("Hi4D PILOT READY")
    print("==========================================")

    with open(
        OUT_ROOT / "summary.json",
        "r",
        encoding="utf-8",
    ) as f:
        print(f.read())

    print("\nSplit:")
    print(
        json.dumps(
            {
                "train": SPLITS["train"],
                "val": SPLITS["val"],
                "test": SPLITS["test"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nOriginal .tar.gz archives were NOT modified."
    )
    print(
        "Huge frames/ and frames_vis/ scans "
        "were NOT extracted."
    )


if __name__ == "__main__":
    main()

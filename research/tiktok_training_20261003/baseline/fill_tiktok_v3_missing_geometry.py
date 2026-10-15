#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import subprocess
from pathlib import Path
from collections import defaultdict


ROOT = Path("/home/shangguanrz/project/pic-edit")
PYTHON = ROOT / "add/envs/smplestx_bw/bin/python"

INVENTORY = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/condition_inventory.jsonl"
)

BUILDER = (
    ROOT
    / "add/tools/SMPLest-X/scripts/build_tiktok_3d_assets_512.py"
)

SUMMARY_JSON = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/fill_missing_geometry_summary.json"
)

SEQ_LIST_TXT = (
    ROOT
    / "datasets/TikTokDataset/TikTok_v3/manifests/fill_missing_geometry_sequences.txt"
)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    if not INVENTORY.is_file():
        raise FileNotFoundError(f"missing inventory: {INVENTORY}")

    if not BUILDER.is_file():
        raise FileNotFoundError(f"missing builder: {BUILDER}")

    rows = load_jsonl(INVENTORY)

    missing_rows = []
    seq_to_stems = defaultdict(list)

    for r in rows:
        avail = r["available"]
        need_fix = (
            (not avail["normal"])
            or (not avail["depth_vis"])
            or (not avail["smplx_mask"])
        )
        if need_fix:
            missing_rows.append(r)
            seq = str(r["sequence_id"])
            stem = str(r["stem"])
            seq_to_stems[seq].append(stem)

    sequences = sorted(seq_to_stems.keys())

    summary = {
        "version": "tiktok_v3_fill_missing_geometry_v1",
        "inventory": str(INVENTORY),
        "builder": str(BUILDER),
        "missing_frame_rows": len(missing_rows),
        "missing_sequences": len(sequences),
        "policy": {
            "reuse_existing_smplestx": True,
            "force_smplestx": False,
            "overwrite_assets": False,
            "save_depth_npy": False,
            "unit_of_repair": "sequence"
        },
        "sequences": [
            {
                "sequence_id": seq,
                "missing_frame_count": len(seq_to_stems[seq]),
                "missing_stems_preview": seq_to_stems[seq][:10],
            }
            for seq in sequences
        ],
    }

    write_json(SUMMARY_JSON, summary)

    SEQ_LIST_TXT.parent.mkdir(parents=True, exist_ok=True)
    with SEQ_LIST_TXT.open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(seq + "\n")

    print("=" * 80)
    print("TikTok V3 FILL MISSING GEOMETRY")
    print("=" * 80)
    print("missing frame rows :", len(missing_rows))
    print("missing sequences  :", len(sequences))
    print("summary            :", SUMMARY_JSON)
    print("sequence list      :", SEQ_LIST_TXT)
    print()

    ok = 0
    fail = 0
    failed = []

    for i, seq in enumerate(sequences, start=1):
        miss_n = len(seq_to_stems[seq])

        print("-" * 80)
        print(f"[{i}/{len(sequences)}] sequence={seq} missing_frames={miss_n}")
        print("-" * 80)

        cmd = [
            str(PYTHON),
            str(BUILDER),
            "--sequence", seq,
        ]

        print("CMD:", " ".join(cmd))

        ret = subprocess.run(cmd)

        if ret.returncode == 0:
            ok += 1
            print(f"[PASS] {seq}")
        else:
            fail += 1
            failed.append({
                "sequence_id": seq,
                "missing_frame_count": miss_n,
                "returncode": ret.returncode,
            })
            print(f"[FAIL] {seq} returncode={ret.returncode}")

    final = {
        **summary,
        "run_result": {
            "pass_sequences": ok,
            "fail_sequences": fail,
            "failed_sequences": failed,
        }
    }

    write_json(SUMMARY_JSON, final)

    print()
    print("=" * 80)
    print("FILL MISSING GEOMETRY COMPLETE")
    print("=" * 80)
    print(json.dumps(final["run_result"], ensure_ascii=False, indent=2))

    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

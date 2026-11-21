#!/usr/bin/env python3
"""Build the Xiaohongshu single-person dataset root from the screened pairs export.

Source : /data/shangguanrz/xiaohongshu/pairs/          (export_clean_pairs.py 筛选产物)
         1307 个 pair 目录 + pairs_manifest.json，来自 198 个笔记
Target : /data/shangguanrz/datasets/xiaohongshu(single)/   (bilibili 式数据集根)

目录契约（对照 /data/shangguanrz/datasets/bilibili）::

    frames/images/<note>/<note>_<idx>.jpg   唯一原图（按文件名 + sha256 去重）
    frames/frames.jsonl                     绝对路径帧索引
    frames/frames_relative.jsonl            相对路径帧索引（移植/分布式通用）
    pairs/pairs.jsonl                       清单全量 1307 对（含 disk_status，不丢任何一对）
    pairs/pairs_manifest.json               筛选元数据原样拷贝（provenance）
    pairs/summary.json                      本次构建统计

去重口径：源盘 2520 张 jpg + 若干面板裁剪 png 折叠为唯一原图集合 —— 同名文件（同一 note 同一
图号同一面板号）被多个 pair 复用，构建时逐组做 sha256 校验，内容不一致立即报错终止，不静默
挑选。note ≈ bilibili 的 clip_id，一个 note 目录 ≈ 一个 clip。

姿态模态：只做 SDPose（2026-11-21 决定，不跑 DWPose）。由
scripts/data/generate_sdpose_skeletons.py 事后写入
geometry/skeleton/{sdpose_2d,sdpose_2d_kp,sdpose_2d_overlay}，本脚本不生成任何姿态产物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

SOURCE_PAIRS = Path("/data/shangguanrz/xiaohongshu/pairs")
DEFAULT_OUT = Path("/data/shangguanrz/datasets/xiaohongshu(single)")
# 源文件名三种形态（均为小写扩展名）：
#   主批次      <note>_<idx>.jpg
#   merged 批次 <N>_<note>_<idx>.jpg          （N = 作者序号，oneman/author/N）
#   面板裁剪    [N_] <note>_<idx>[_pNN].png/jpg （多宫格截图切出的单人小图，allow_panels=true）
# 数据集内统一落为 <note>_<idx>[_pNN].<ext>，N 前缀只保留在 screen 元数据里。
ORIG_NAME_RE = re.compile(r"^(?:\d+_)?([0-9a-f]{24})_(\d+)(?:_p(\d+))?(\.(?:jpg|png))$")
SCHEMA = "xhs_single_v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-pairs", type=Path, default=SOURCE_PAIRS,
                    help="export_clean_pairs.py 输出的 pair 目录根")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT, help="数据集根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与统计，不写任何文件")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_source(source: Path, manifest_pairs: dict[str, dict[str, Any]]) -> dict[tuple[str, int, str, str], list[Path]]:
    """收集源盘全部 jpg/png，按 (note, idx, panel, ext) 归组。

    文件名去掉 a_/b_ 前缀与可选数字前缀后必须匹配统一模式，且 note 与清单一致；同组内容必须
    sha256 一致；pair 目录必须与清单对得上。任何漂移都直接 raise，先审计后落地。
    """
    if not source.is_dir():
        raise FileNotFoundError(f"source pairs dir not found: {source}")
    groups: dict[tuple[str, int, str, str], list[Path]] = defaultdict(list)
    manifest_dirs = {e["dir"] for e in manifest_pairs.values()}
    anomalies: list[str] = []

    for pdir in sorted(p for p in source.iterdir() if p.is_dir()):
        if pdir.name not in manifest_dirs:
            anomalies.append(f"dir not in manifest: {pdir.name}")
            continue
        entry = manifest_pairs[pdir.name]
        expected = {entry["a_file"], entry["b_file"]}
        found = {p.name for p in pdir.glob("*.jpg")} | {p.name for p in pdir.glob("*.png")}
        if found - expected:
            anomalies.append(f"{pdir.name}: unexpected files {sorted(found - expected)}")
        for img in sorted(pdir.glob("*.jpg")) + sorted(pdir.glob("*.png")):
            stem = re.sub(r"^[ab]_", "", img.name)
            m = ORIG_NAME_RE.match(stem)
            if not m:
                anomalies.append(f"{img}: stripped name {stem!r} does not match [N_]note_idx[_pNN] pattern")
                continue
            if m.group(1) != entry["note"]:
                anomalies.append(f"{img}: embedded note {m.group(1)} != manifest note {entry['note']}")
                continue
            groups[(m.group(1), int(m.group(2)), m.group(3) or "", m.group(4))].append(img)

    for key, paths in sorted(groups.items()):
        shas = {sha256_file(p) for p in paths}
        if len(shas) > 1:
            anomalies.append(f"{key}: {len(paths)} copies with conflicting sha256")
    if anomalies:
        print(f"[build] {len(anomalies)} anomalies detected:", flush=True)
        for a in anomalies[:30]:
            print(f"  - {a}", flush=True)
        raise SystemExit(f"[build] abort: fix source drift before building ({len(anomalies)} anomalies)")
    return dict(groups)


def orig_name(key: tuple[str, int, str, str]) -> str:
    note, idx, panel, ext = key
    return f"{note}_{idx}" + (f"_p{panel}" if panel else "") + ext


def parse_src(src: str) -> tuple[str, int, str, str]:
    """清单 src_a/src_b -> 数据集内 (note, idx, panel, ext)；merged 批次数字前缀在此剥离。"""
    m = ORIG_NAME_RE.match(src)
    if not m:
        raise ValueError(f"manifest src name does not match [N_]note_idx[_pNN] pattern: {src!r}")
    return m.group(1), int(m.group(2)), m.group(3) or "", m.group(4)


def main() -> int:
    args = parse_args()
    out_root = args.out_root
    manifest_path = args.source_pairs / "pairs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_pairs = {e["pair_id"]: e for e in manifest["pairs"]}
    print(f"[build] manifest: {len(manifest_pairs)} pairs / {manifest['total_notes']} notes | out={out_root}",
          flush=True)

    groups = scan_source(args.source_pairs, manifest_pairs)
    print(f"[build] disk scan: {sum(len(v) for v in groups.values())} jpgs -> {len(groups)} unique originals",
          flush=True)

    plan: list[tuple[str, int, str, str]] = sorted(groups)
    n_notes = len({k[0] for k in plan})

    disk_by_pair: dict[str, set[tuple[str, int, str, str]]] = defaultdict(set)
    for key, paths in groups.items():
        for p in paths:
            disk_by_pair[p.parent.name].add(key)

    frames_rows: list[dict[str, Any]] = []
    frames_rel_rows: list[dict[str, Any]] = []
    n_copied = n_skipped = 0
    bytes_copied = 0

    if args.dry_run:
        complete = sum(1 for e in manifest_pairs.values()
                       if parse_src(e["src_a"]) in disk_by_pair[e["pair_id"]]
                       and parse_src(e["src_b"]) in disk_by_pair[e["pair_id"]])
        for key in plan[:3]:
            print(f"[build] dry-run sample: {groups[key][0]} -> "
                  f"{out_root}/frames/images/{key[0]}/{orig_name(key)}", flush=True)
        print(f"[build] dry-run: {len(plan)} unique frames / {n_notes} notes / "
              f"{complete} complete pairs of {len(manifest_pairs)}; no files written", flush=True)
        return 0

    (out_root / "frames" / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, key in enumerate(plan, 1):
        note = key[0]
        src = groups[key][0]
        target = out_root / "frames" / "images" / note / orig_name(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if sha256_file(target) != sha256_file(src):
                raise SystemExit(f"[build] abort: existing {target} conflicts with source sha256")
            n_skipped += 1
        else:
            shutil.copy2(src, target)
            n_copied += 1
            bytes_copied += target.stat().st_size
        if i % 200 == 0 or i == len(plan):
            print(f"[build] copied {i}/{len(plan)} | new={n_copied} resume_skip={n_skipped} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    # 帧索引（SDPose 之后如需补 pose 字段，在行内追加即可；pose 数值本体在 geometry/skeleton/sdpose_2d_kp）
    for key in plan:
        note, idx, panel, ext = key
        target = out_root / "frames" / "images" / note / orig_name(key)
        with Image.open(target) as im:
            hw = [im.height, im.width]
        entry = {
            "clip_id": note,
            "frame_id": target.stem,
            "image_index": idx,
            "rgb_path": str(target),
            "rgb_sha256": sha256_file(target),
            "original_hw": hw,
            "file_bytes": target.stat().st_size,
            "source_pair_ids": sorted(p.parent.name for p in groups[key]),
        }
        frames_rows.append(entry)
        rel = entry.copy()
        rel["rgb_path"] = str(target.relative_to(out_root))
        frames_rel_rows.append(rel)

    for name, rows in (("frames.jsonl", frames_rows), ("frames_relative.jsonl", frames_rel_rows)):
        with open(out_root / "frames" / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 配对索引：清单全量落盘，缺图对显式标 disk_status，不静默丢弃
    (out_root / "pairs").mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = defaultdict(int)
    with open(out_root / "pairs" / "pairs.jsonl", "w", encoding="utf-8") as f:
        for pid in sorted(manifest_pairs):
            e = manifest_pairs[pid]
            have = disk_by_pair[pid]
            key_a, key_b = parse_src(e["src_a"]), parse_src(e["src_b"])
            if key_a in have and key_b in have:
                status = "complete"
            elif key_a in have:
                status = "missing_b"
            elif key_b in have:
                status = "missing_a"
            else:
                status = "missing_both"
            status_counts[status] += 1
            note = e["note"]
            if key_a[0] != note or key_b[0] != note:
                raise SystemExit(f"[build] abort: {pid} src note mismatch: {key_a[0]} / {key_b[0]} vs {note}")
            row = {
                "pair_id": pid,
                "note_id": note,
                "schema": SCHEMA,
                "source_frame_id": Path(orig_name(key_a)).stem if key_a in have else None,
                "target_frame_id": Path(orig_name(key_b)).stem if key_b in have else None,
                "source_rgb_path": str(out_root / "frames" / "images" / note / orig_name(key_a)) if key_a in have else None,
                "target_rgb_path": str(out_root / "frames" / "images" / note / orig_name(key_b)) if key_b in have else None,
                "distance": e.get("pose_diff"),
                "cross_shot": False,
                "disk_status": status,
                "screen": {k: v for k, v in e.items() if k not in ("pair_id", "dir", "a_file", "b_file", "note")},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    shutil.copy2(manifest_path, out_root / "pairs" / "pairs_manifest.json")

    summary = {
        "schema": SCHEMA,
        "source": str(args.source_pairs),
        "manifest_pairs": len(manifest_pairs),
        "manifest_notes": manifest["total_notes"],
        "disk_jpgs": sum(len(v) for v in groups.values()),
        "unique_frames": len(frames_rows),
        "notes_with_frames": len({r["clip_id"] for r in frames_rows}),
        "copied_this_run": n_copied,
        "resume_skipped": n_skipped,
        "bytes_copied_this_run": bytes_copied,
        "frames_bytes_total": sum(r["file_bytes"] for r in frames_rows),
        "pair_disk_status": dict(status_counts),
        "screen_criteria": manifest.get("criteria"),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_root / "pairs" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[build] DONE: {len(frames_rows)} frames / {n_notes} notes | pairs={len(manifest_pairs)} "
          f"status={dict(status_counts)} | {summary['frames_bytes_total'] / 1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

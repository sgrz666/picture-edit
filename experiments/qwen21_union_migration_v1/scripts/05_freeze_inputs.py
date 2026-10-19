#!/usr/bin/env python
"""冻结六组样本的输入：只搬运现成产物，不重跑几何估计与提示词重写。

产出 input/<sample>/ 与 input_manifest.json：逐文件 SHA256、画布尺寸、32 对齐检查、
来源 run 目录与 stage.json 里的输入指纹。目标 RGB 只登记为评估用，绝不进入生成输入列表。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

PROJ = pathlib.Path("/home/shangguanrz/project/pic-edit")
EXP = pathlib.Path(__file__).resolve().parent.parent
RAW = EXP / "input"

SINGLE_IDS = ["hard_02", "hard_05_torso", "medium_02"]
DUAL_GLOBS = {
    "hug_pair00_084": ["hug_pair00_084_c4_r000", "hug_pair00_084*"],
    "dance_pair02_120": ["dance_pair02_120_c4_r459", "dance_pair02_120*"],
    "cross_backhug_pair02_037": ["cross_backhug_pair02_037_r545_from_pair00",
                                 "backhug_pair02_037_c64_r545", "cross_backhug*"],
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def size(p: pathlib.Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def first_existing(root: pathlib.Path, rels: list[str]) -> pathlib.Path | None:
    for r in rels:
        p = root / r
        if p.exists():
            return p
    return None


def pick_single(sid: str):
    for root in (PROJ / f"workflow/runs/scale_anchor_20261013/{sid}",
                 PROJ / f"workflow/runs/test/{sid}",
                 PROJ / f"workflow/diagnostics/2026-10-13-scale/{sid}"):
        if not root.exists():
            continue
        source = first_existing(root, ["inputs/A_reference.png"])
        control = first_existing(root, ["geometry/B_3d_dwpose.png", "geometry/skeleton_both.png"])
        prompt = first_existing(root, ["prompt/generation_prompt.txt"])
        if source and control and prompt:
            return root, source, control, control
    return None, None, None, None


def pick_dual(globs: list[str]):
    runs = PROJ / "workflow-multiple/runs"
    for g in globs:
        cands = sorted(runs.glob(g)) if "*" not in g else sorted(runs.glob(g))
        for root in cands:
            if not root.is_dir():
                continue
            a = first_existing(root, ["inputs/A_reference.png"])
            b = first_existing(root, ["inputs/B_reference.png"])
            both = first_existing(root, ["geometry/skeleton_both.png"])
            prompt = first_existing(root, ["prompt/generation_prompt.txt"])
            if a and b and both and prompt:
                per = [first_existing(root, ["geometry/skeleton_0.png"]),
                       first_existing(root, ["geometry/skeleton_1.png"])]
                return root, a, b, both, [p for p in per if p], prompt
    return None


def freeze(kind: str, sid: str, root: pathlib.Path, files: dict[str, pathlib.Path],
           prompt_path: pathlib.Path, extra: dict | None = None):
    dst = RAW / kind / sid
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    entry = {"sample": sid, "kind": kind, "run_dir": str(root), "files": {}, "generation_inputs": [],
             "eval_only": []}
    for role, src in files.items():
        out = dst / f"{role}{src.suffix}"
        shutil.copy2(src, out)
        w, h = size(out)
        entry["files"][role] = {"path": str(out), "origin": str(src), "wh": [w, h],
                                "sha256": sha256(out), "aligned32": (w % 32 == 0 and h % 32 == 0)}
    text = prompt_path.read_text(encoding="utf-8").strip()
    (dst / "prompt.txt").write_text(text, encoding="utf-8")
    entry["prompt"] = text
    entry["prompt_origin"] = str(prompt_path)
    entry["prompt_sha256"] = sha256(dst / "prompt.txt")
    if extra:
        entry.update(extra)
    return entry


def main():
    entries, missing = [], []

    for sid in SINGLE_IDS:
        root, source, control, pose_ref = pick_single(sid)
        if root is None:
            missing.append({"sample": sid, "reason": "找不到 inputs/A_reference.png + geometry 骨架图 + prompt"})
            continue
        e = freeze("single", sid, root,
                   {"source": source, "control": control, "pose_reference": pose_ref},
                   root / "prompt/generation_prompt.txt",
                   {"generator_inputs": ["source", "pose_reference"], "control_input": "control"})
        tgt = first_existing(root, ["inputs/B_rgb.png"])
        kp = first_existing(root, ["geometry/B_keypoints.npz"])
        if tgt:
            out = RAW / "single" / sid / f"target_rgb_eval_only{tgt.suffix}"
            shutil.copy2(tgt, out)
            e["eval_only"].append({"role": "target_rgb", "path": str(out), "origin": str(tgt),
                                   "wh": list(size(out)), "sha256": sha256(out),
                                   "note": "仅供评估，不进入生成器"})
        if kp:
            out = RAW / "single" / sid / "requested_keypoints.npz"
            shutil.copy2(kp, out)
            e["eval_only"].append({"role": "requested_keypoints", "path": str(out), "origin": str(kp),
                                   "sha256": sha256(out)})
        entries.append(e)

    for sid, globs in DUAL_GLOBS.items():
        got = pick_dual(globs)
        if not got:
            missing.append({"sample": sid, "reason": f"在 workflow-multiple/runs 下找不到同时含 A/B 参考、skeleton_both 与 prompt 的 run（试过 {globs}）"})
            continue
        root, a, b, both, per_person, prompt_path = got
        files = {"source_A": a, "source_B": b, "control": both, "skeleton_both": both}
        for i, p in enumerate(per_person):
            files[f"pose_reference_{i}"] = p
        e = freeze("dual", sid, root, files, prompt_path,
                   {"generator_inputs": ["source_A", "source_B"] +
                    [f"pose_reference_{i}" for i in range(len(per_person))],
                    "control_input": "control"})
        for role, rel in (("target_A", "inputs/B_rgb.png"),):
            t = first_existing(root, [rel])
            if t:
                out = RAW / "dual" / sid / f"{role}_eval_only{t.suffix}"
                shutil.copy2(t, out)
                e["eval_only"].append({"role": role, "path": str(out), "origin": str(t),
                                       "wh": list(size(out)), "sha256": sha256(out)})
        for rel, role in (("geometry/dual_keypoints.npz", "requested_keypoints"),):
            k = first_existing(root, [rel])
            if k:
                out = RAW / "dual" / sid / f"{role}.npz"
                shutil.copy2(k, out)
                e["eval_only"].append({"role": role, "path": str(out), "origin": str(k), "sha256": sha256(out)})
        entries.append(e)

    manifest = {"samples": entries, "missing": missing,
                "counts": {"single": sum(1 for e in entries if e["kind"] == "single"),
                           "dual": sum(1 for e in entries if e["kind"] == "dual")},
                "policy": {"geometry_reestimated": False, "prompt_rewritten": False,
                           "target_rgb_in_generator": False}}
    out = EXP / "input_manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"frozen {len(entries)} samples -> {out}")
    for e in entries:
        bad = [r for r, v in e["files"].items() if not v["aligned32"]]
        print(f"  {e['kind']:6s} {e['sample']:28s} control={e['files']['control']['wh']} "
              f"run={pathlib.Path(e['run_dir']).name} 非32对齐={bad}")
    if missing:
        print("MISSING:", json.dumps(missing, ensure_ascii=False))
    return 0 if len(entries) == 6 and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

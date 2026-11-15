"""Collect GENERATED images of the bilibili-dwpose-edit-lora runs into LoRA/pictures.

Copy only (project red line: images are never moved or deleted). Every destination
must not pre-exist. Each copy is verified by sha256 against its source; a manifest
is written only if all copies verify. Inputs (letterboxed A / skeleton renders /
preview) are deliberately excluded and counted in the manifest as excluded.
"""
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

WS = "/data/shangguanrz/LoRA"
EXP = os.path.join(WS, "experiments/bilibili-dwpose-edit-lora")
DEST = os.path.join(WS, "pictures")

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# (source dir, run, kind, method-or-None, rename rule)
def plan():
    jobs = []
    for run, comp, rep in (
        ("full_20261111_v1", "output/full_comparison_20261111_v1", "report/full_report_20261111_v1"),
        ("pilot_20261111_v1", "output/comparison_20261111_v1", "report/pilot_report_20261111_v1"),
    ):
        cdir = os.path.join(EXP, comp)
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".png"):
                continue
            method, rest = fn.split("_", 1)
            assert method in ("baseline", "lora"), fn
            jobs.append((os.path.join(cdir, fn), run, "generated", method,
                         os.path.join(run, method, rest)))
        rdir = os.path.join(EXP, rep)
        for fn in sorted(os.listdir(rdir)):
            if fn.endswith(".png"):
                jobs.append((os.path.join(rdir, fn), run, "panel", None,
                             os.path.join(run, "panels", fn)))
    pdir = os.path.join(EXP, "output/probe_inference_20261111_v1")
    for fn in sorted(os.listdir(pdir)):
        if fn.endswith(".png"):
            jobs.append((os.path.join(pdir, fn), "pilot_20261111_v1", "probe", None,
                         os.path.join("pilot_20261111_v1", "probe_inference", fn)))
    return jobs


jobs = plan()
print("planned copies: %d" % len(jobs))
print("by run/kind:", dict(Counter((r, k) for _, r, k, _, _ in jobs)))

# safety: destination tree must be empty of files, and no planned dest may exist
existing = []
for root, _, files in os.walk(DEST):
    for f in files:
        existing.append(os.path.join(root, f))
if existing:
    print("ABORT: destination already contains %d files, e.g. %s" % (len(existing), existing[:3]))
    sys.exit(2)
seen = set()
for src, run, kind, method, rel in jobs:
    d = os.path.join(DEST, rel)
    if d in seen or os.path.exists(d):
        print("ABORT: duplicate or pre-existing destination", d)
        sys.exit(2)
    seen.add(d)
    if not os.path.isfile(src):
        print("ABORT: missing source", src)
        sys.exit(2)

# audit: any image in the workspace we are NOT collecting?
collected = {os.path.realpath(s) for s, _, _, _, _ in jobs}
unexpected = []
excluded = Counter()
for root, dirs, files in os.walk(WS):
    dirs[:] = [d for d in dirs if d not in ("pictures", ".git")]
    for f in files:
        if not f.lower().endswith(IMG_EXT):
            continue
        p = os.path.realpath(os.path.join(root, f))
        if p in collected:
            continue
        rel = os.path.relpath(p, WS)
        top = rel.split(os.sep)
        if top[0] == "experiments" and len(top) > 2 and top[2] == "input":
            excluded["input_condition_images"] += 1
        elif f == "preview.png":
            excluded["input_preview"] += 1
        else:
            unexpected.append(rel)
print("excluded (not generated):", dict(excluded))
if unexpected:
    print("WARNING unclassified images:", unexpected[:20], "total", len(unexpected))

records, failures = [], []
total_bytes = 0
for src, run, kind, method, rel in jobs:
    d = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    s_sha = sha(src)
    shutil.copy2(src, d)
    d_sha = sha(d)
    size = os.path.getsize(d)
    total_bytes += size
    ok = s_sha == d_sha and size == os.path.getsize(src)
    if not ok:
        failures.append(rel)
    records.append({"dest": "pictures/" + rel.replace(os.sep, "/"),
                    "source": src, "run": run, "kind": kind, "method": method,
                    "bytes": size, "sha256": s_sha, "verified": ok})
    print("%s %s" % ("OK  " if ok else "FAIL", rel))

print("\ncopied %d files, %.1f MiB, failures=%d" % (len(records), total_bytes / 1048576, len(failures)))
if failures:
    print("NOT writing manifest; failures:", failures)
    sys.exit(1)

manifest = {
    "purpose": "browsable copy of GENERATED images only; originals untouched in experiments/",
    "created_by": "monitor_20261112/collect_pictures.py",
    "runs": {
        "full_20261111_v1": "9520-update full run, checkpoint-009520 vs base Qwen",
        "pilot_20261111_v1": "300-update pilot run, checkpoint-000300 vs base Qwen",
    },
    "layout": {
        "<run>/baseline/<case>.png": "original Qwen output, no LoRA",
        "<run>/lora/<case>.png": "LoRA output, same case/seed/inputs",
        "<run>/panels/": "5-column comparison panels (A | B skeleton | B GT | base | LoRA) plus overview/curves",
        "pilot_20261111_v1/probe_inference/": "1-step pilot probe inference, native LoRA reload check",
    },
    "case_order": "normal_00..09 = val, normal_10..17 = train; swap_pose / blank_pose are diagnostics",
    "excluded_not_generated": dict(excluded),
    "unclassified_images": unexpected,
    "files": records,
}
mp = os.path.join(DEST, "manifest.json")
with open(mp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("manifest written:", mp, os.path.getsize(mp), "bytes")
print("DONE")
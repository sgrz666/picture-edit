"""Read-only audit: what role did val and test actually play in the LoRA line?

Checks, all from disk:
  1. prepared manifests: names, counts, split fields, video-group counts
  2. source splits at dataset_root: counts and pair_id sets
  3. is pilot_val a subset of source val? is train.jsonl exactly source train?
  4. did ANY LoRA artifact (cache index, inference outputs, per-image metrics,
     validation records) ever reference a test pair_id?
  5. which pairs fed the periodic validation loss vs the final comparison?
"""
import json
import os
from collections import Counter

ROOT = "/data/shangguanrz/LoRA/experiments/bilibili-dwpose-edit-lora"
DATA = os.path.join(ROOT, "data/data_20261111_v1")
DS = "/data/shangguanrz/datasets/bilibili"


def jl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


print("=" * 78)
print("1. PREPARED MANIFESTS in data/data_20261111_v1")
print("=" * 78)
for fn in sorted(os.listdir(DATA)):
    if fn.endswith(".jsonl"):
        rows = jl(os.path.join(DATA, fn))
        sp = Counter(r.get("split") for r in rows)
        gv = len(set(r.get("group_id", r.get("clip_id")) for r in rows))
        print("  %-22s n=%5d splits=%s video_groups=%d" % (fn, len(rows), dict(sp), gv))
    else:
        print("  %-22s (non-jsonl)" % fn)

print()
print("=" * 78)
print("2. SOURCE SPLITS at dataset_root")
print("=" * 78)
src = {}
for split in ("train", "val", "test"):
    p = os.path.join(DS, split, "pairs.jsonl")
    if not os.path.exists(p):
        print("  MISSING", p)
        continue
    rows = jl(p)
    src[split] = {r["pair_id"] for r in rows}
    gv = len(set(r.get("group_id", r.get("clip_id")) for r in rows))
    print("  %-6s n=%5d video_groups=%d" % (split, len(rows), gv))

print()
print("=" * 78)
print("3. SUBSET RELATIONS")
print("=" * 78)
pv = jl(os.path.join(DATA, "pilot_val.jsonl"))
tr = jl(os.path.join(DATA, "train.jsonl"))
pv_ids = {r["pair_id"] for r in pv}
tr_ids = {r["pair_id"] for r in tr}
for name, ids in (("pilot_val", pv_ids), ("train", tr_ids)):
    for split, s in src.items():
        print("  %-9s subset of source %-5s : %s  (%d/%d)"
              % (name, split, ids <= s, len(ids & s), len(ids)))
print("  pilot_val videos:", sorted({r.get("group_id", r.get("clip_id")) for r in pv}))
print("  pilot_val ∩ source test:", len(pv_ids & src.get("test", set())))
print("  train     ∩ source test:", len(tr_ids & src.get("test", set())))
print("  train     ∩ source val :", len(tr_ids & src.get("val", set())))

print()
print("=" * 78)
print("4. DID ANY ARTIFACT TOUCH A TEST PAIR?")
print("=" * 78)
test_ids = src.get("test", set())
val_ids = src.get("val", set())
train_ids = src.get("train", set())


def scan(path, fields=("pair_id", "desired_pose_pair_id")):
    hits = Counter()
    n = 0
    for r in jl(path):
        n += 1
        for f in fields:
            v = r.get(f)
            if v is None:
                continue
            if v in test_ids:
                hits["test"] += 1
            elif v in val_ids:
                hits["val"] += 1
            elif v in train_ids:
                hits["train"] += 1
            else:
                hits["unknown"] += 1
    return n, hits


targets = [
    "output/full_comparison_20261111_v1/outputs.jsonl",
    "output/comparison_20261111_v1/outputs.jsonl",
    "output/probe_inference_20261111_v1/outputs.jsonl",
    "report/full_report_20261111_v1/per_image.jsonl",
    "report/pilot_report_20261111_v1/per_image.jsonl",
]
for rel in targets:
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        n, hits = scan(p)
        print("  %-52s n=%3d %s" % (rel, n, dict(hits)))
    else:
        print("  %-52s MISSING" % rel)

# cache: hash-named .pt files; look for an index/manifest that names pairs
cache = os.path.join(ROOT, "cache/cache_full_20261111_v1")
non_pt = [f for f in os.listdir(cache) if not f.endswith(".pt")]
print("  cache_full non-.pt entries:", non_pt)
for f in non_pt:
    p = os.path.join(cache, f)
    if f.endswith(".jsonl"):
        n, hits = scan(p)
        print("    %-40s n=%5d %s" % (f, n, dict(hits)))
    elif f.endswith(".json"):
        d = json.load(open(p, "r", encoding="utf-8"))
        s = json.dumps(d)
        print("    %-40s keys=%s mentions_test=%s"
              % (f, sorted(d)[:8] if isinstance(d, dict) else type(d), "test" in s))

# validation records: which pairs?
v = json.load(open(os.path.join(ROOT, "output/train_full_20261111_v1/validation_009520.json"),
                   "r", encoding="utf-8"))
vids = {r["pair_id"] for r in v["records"]}
print("  validation_009520 distinct pairs=%d ; equals pilot_val set: %s ; ∩ test=%d"
      % (len(vids), vids == pv_ids, len(vids & test_ids)))

print()
print("=" * 78)
print("5. WHERE 'test' APPEARS IN CONFIG / CODE")
print("=" * 78)
for rel in ("configs/full.json", "configs/pilot.json"):
    d = json.load(open(os.path.join(ROOT, rel), "r", encoding="utf-8"))
    print("  %s: cache_manifests=%s validation_manifest=%s expected_train_pairs=%s"
          % (rel, d.get("cache_manifests"), d.get("validation_manifest"), d.get("expected_train_pairs")))
import subprocess
out = subprocess.run(["grep", "-rn", "test", os.path.join(ROOT, "scripts/prepare.py")],
                     capture_output=True, text=True).stdout
for line in out.splitlines():
    if "test" in line and ("split" in line or "expected" in line or "manifest" in line):
        print("  prepare.py:", line.strip()[:160])
print("\nDONE")
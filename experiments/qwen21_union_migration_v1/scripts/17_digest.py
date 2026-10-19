#!/usr/bin/env python
"""把三关的 JSON 结果压成 README 要用的关键数字，避免手抄错。"""
import json
import pathlib

WORK = pathlib.Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")
EXP = pathlib.Path("/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1")


def dump(tag, obj, depth=0, maxdepth=3):
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < maxdepth:
                print(f"{'  '*depth}{tag}{k}: <{type(v).__name__} len={len(v)}>")
                dump("", v, depth + 1, maxdepth)
            else:
                s = str(v)
                print(f"{'  '*depth}{tag}{k}: {s[:160]}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:6]):
            if isinstance(v, (dict, list)):
                dump(f"[{i}].", v, depth + 1, maxdepth)
            else:
                print(f"{'  '*depth}[{i}] {str(v)[:160]}")


for name in ("device_check.json",):
    for p in sorted(WORK.rglob(name)):
        print(f"===== {p.relative_to(WORK)}")
        dump("", json.loads(p.read_text(encoding="utf-8")), 0, 2)

for p in sorted(WORK.rglob("*.json")):
    if "runs" not in p.parts or p.name in ("records.jsonl",):
        continue
    if p.suffix != ".json" or "manifest" in p.name:
        continue
    print(f"===== {p.relative_to(WORK)}")
    try:
        dump("", json.loads(p.read_text(encoding="utf-8")), 0, 3)
    except Exception as e:
        print("  读取失败:", e)

rec = WORK / "runs/matrix_20261015T134829/records.jsonl"
rows = [json.loads(l) for l in rec.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"===== matrix records: {len(rows)}（成功 {sum(1 for r in rows if r.get('out'))}）")
for r in rows[-3:]:
    print("  ", r["config"], r["sample"], r["seed"], r.get("seconds"), r.get("peak_GiB"),
          (r.get("error") or "")[:80])

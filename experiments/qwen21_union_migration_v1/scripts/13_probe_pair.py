#!/usr/bin/env python
"""探针：双人样本的 pair.json 里，A 参考图的每个人与 Hi4D side 的对应关系是什么。

身份指标要把「请求槽位 i」（npz 里的 hi4d_sides）映射回源参考图里的那个人，
必须确认 pair.json 是否提供 side 标签与人物框；没有就如实标记外观指标不可用。
"""
import json
import pathlib

import numpy as np
from PIL import Image

EXP = pathlib.Path("/home/shangguanrz/project/pic-edit/experiments/2.qwen-test/qwen21_union_migration_v1")
manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))

for s in manifest["samples"]:
    if s["kind"] != "dual":
        continue
    print("=== ", s["sample"], "run_dir:", s["run_dir"])
    npz = np.load(s["eval_only"][0]["path"], allow_pickle=True)
    print("   npz hi4d_sides:", list(npz["hi4d_sides"]), "slots:", list(npz["slots"]))
    pj = pathlib.Path(s["run_dir"]) / "inputs" / "pair.json"
    if not pj.exists():
        pj = pathlib.Path(s["run_dir"]) / "pair.json"
    print("   pair.json:", pj, pj.exists())
    if not pj.exists():
        continue
    pair = json.loads(pj.read_text(encoding="utf-8"))
    print("   pair keys:", list(pair.keys()), "kind:", pair.get("kind"))
    for side_key in ("A", "B"):
        blk = pair.get(side_key, {})
        persons = blk.get("persons")
        print(f"   {side_key}: image={blk.get('image')} persons={len(persons) if persons is not None else None}")
        for i, p in enumerate(persons or []):
            print("      person", i, "keys:", sorted(p.keys()),
                  "side:", p.get("hi4d_side") or p.get("side"),
                  "bbox:", (p.get("region") or {}).get("bbox_xyxy"))
    print("   slot_order:", json.dumps(pair.get("B", {}).get("slot_order"), ensure_ascii=False)[:300])
    for role in ("source_A", "source_B"):
        f = s["files"][role]
        with Image.open(f["path"]) as im:
            print("  ", role, im.size, f["origin"].split("pic-edit/")[-1])

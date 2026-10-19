#!/usr/bin/env python
"""为双人样本补一份「身份参照」sidecar：把每槽位对应的源人物裁出来，放进实验目录。

为什么需要它：双人矩阵要判「身份有没有互换」，就必须知道 canvas_points[i]（槽位 i）对应
源参考图里的哪一个人。npz 只给了 hi4d_side 标签，人框在 pair 目录的 pair.json 里，而 run 目录
随时可能被清理。所以这里把 pair.json / dual_geometry.json / source_preprocess.json 复制进实验，
并按 evaluate_dual.source_boxes_canvas 同样的换算（native 像素 × scale + offset）把每个人的框
映射到画布，从冻结的 source_A.png 上裁出身份参照图，逐文件记 SHA256。

只做新增：不删除、不改写任何既有图片与清单；产物落 input/dual/<sample>/provenance/。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import numpy as np
from PIL import Image

EXP = pathlib.Path(__file__).resolve().parent.parent
RAW = EXP / "input"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def crop_with_pad(image: Image.Image, box, pad_ratio: float = 0.02) -> list | None:
    """与 src.pose_edit.appearance.crop_person 同样的留边裁框，返回整数框坐标。"""
    w, h = image.size
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    px, py = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
    left, top = max(0, int(np.floor(x0 - px))), max(0, int(np.floor(y0 - py)))
    right, bottom = min(w, int(np.ceil(x1 + px))), min(h, int(np.ceil(y1 + py)))
    if right - left < 2 or bottom - top < 2:
        return None
    return [left, top, right, bottom]


def main():
    manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))
    out = {}
    for entry in manifest["samples"]:
        if entry["kind"] != "dual":
            continue
        sid = entry["sample"]
        run = pathlib.Path(entry["run_dir"])
        geom_p = run / "geometry" / "dual_geometry.json"
        prep_p = run / "inputs" / "source_preprocess.json"
        if not geom_p.exists() or not prep_p.exists():
            out[sid] = {"error": f"缺少 {geom_p} 或 {prep_p}"}
            print(f"[{sid}] 跳过：{out[sid]['error']}")
            continue
        geom = json.loads(geom_p.read_text(encoding="utf-8"))
        prep = json.loads(prep_p.read_text(encoding="utf-8"))
        pair_dir = pathlib.Path(geom["pair_dir"])
        pair_p = pair_dir / "pair.json"
        pair = json.loads(pair_p.read_text(encoding="utf-8")) if pair_p.exists() else None

        dst = RAW / "dual" / sid / "provenance"
        dst.mkdir(parents=True, exist_ok=True)
        copied = {}
        for src in (geom_p, prep_p, pair_p):
            if src.exists():
                t = dst / src.name
                shutil.copy2(src, t)
                copied[src.name] = {"path": str(t), "origin": str(src), "sha256": sha256(t)}

        a_ref = entry["files"].get("source_A", {}).get("path")
        scale, offset = prep["A"]["scale_xy"], prep["A"]["offset_xy"]
        persons, side_index = [], {}
        for p in (pair or {}).get("A", {}).get("persons") or []:
            side = p.get("hi4d_side") or p.get("side")
            bbox = (p.get("region") or {}).get("bbox_xyxy")
            if side is not None and bbox:
                side_index[str(side)] = bbox

        with Image.open(a_ref) as im:
            img = im.convert("RGB")
        for i, p in enumerate(geom["persons"]):
            side = str(p.get("hi4d_side"))
            native = side_index.get(side)
            rec = {"canvas_slot_index": i, "slot": p.get("slot"), "hi4d_side": side,
                   "bbox_A_native": native,
                   "transform_A": {"scale_xy": scale, "offset_xy": offset}}
            if native:
                canvas_box = [native[0] * scale[0] + offset[0], native[1] * scale[1] + offset[1],
                              native[2] * scale[0] + offset[0], native[3] * scale[1] + offset[1]]
                box = crop_with_pad(img, canvas_box)
                rec["bbox_A_canvas"] = [round(v, 2) for v in canvas_box]
                if box:
                    f = dst / f"identity_slot{p.get('slot')}_side{side}.png"
                    img.crop(box).save(f)
                    rec["identity_crop"] = {"path": str(f),
                                            "wh": [box[2] - box[0], box[3] - box[1]],
                                            "sha256": sha256(f)}
                else:
                    rec["identity_crop"] = None
                    rec["error"] = "裁框退化（宽或高不足 2 像素）"
            else:
                rec["identity_crop"] = None
                rec["error"] = f"pair.json 的 A.persons 里没有 side={side} 的人框"
            persons.append(rec)

        out[sid] = {"pair_kind": (pair or {}).get("kind"), "framing": geom.get("framing"),
                    "canvas": geom.get("canvas"), "content_box_A": prep["A"]["content_box"],
                    "source_A": a_ref, "source_A_sha256": entry["files"].get("source_A", {}).get("sha256"),
                    "copied": copied, "persons": persons,
                    "note": "same_identity 指 A/B 两张照片是同一组人，不是两个人彼此相同；槽位顺序与 npz 的 canvas_points / hi4d_sides 一致"}
        print(f"[{sid}] kind={out[sid]['pair_kind']} slots=" +
              json.dumps([{"slot": p["slot"], "side": p["hi4d_side"],
                           "crop": bool(p.get("identity_crop"))} for p in persons], ensure_ascii=False))

    p = EXP / "input_provenance.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {p}")
    bad = [s for s, v in out.items() if any(not q.get("identity_crop") for q in v.get("persons", []))]
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/home/shangguanrz/project/pic-edit")
HI4D_ROOT = ROOT / "datasets/Hi4D_pilot_v1/processed_512_1500"

candidates = [
    ("pair02", "backhug02", "000044", "Back-Hug 背后拥抱"),
    ("pair12", "highfive12", "000035", "High-Five 击掌手部接触"),
    ("pair14", "dance14", "000114", "Dance 双人搭肩揽腰跳舞"),
    ("pair22", "kiss22", "000022", "Kiss 亲密贴面接触"),
    ("pair12", "fight12", "000030", "Fight 格斗推搡手臂接触"),
    ("pair00", "hug00", "000089", "Deep-Hug 正面深拥接触"),
]

for pair, act, stem, desc in candidates:
    rgb = HI4D_ROOT / f"{pair}/{act}/rgb/{stem}.png"
    dw = HI4D_ROOT / f"{pair}/{act}/pose_dwpose/combined/{stem}.png"
    old = HI4D_ROOT / f"{pair}/{act}/pose/combined/{stem}.png"
    print(f"[{desc}] {pair}/{act}/{stem}:")
    print(f"  rgb exists: {rgb.is_file()}")
    print(f"  dwpose exists: {dw.is_file()}")
    print(f"  oldpose exists: {old.is_file()}")

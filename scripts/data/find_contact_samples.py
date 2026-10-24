#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/shangguanrz/project/pic-edit")
HI4D_ROOT = ROOT / "datasets/Hi4D_pilot_v1"

with open(HI4D_ROOT / "processed_frames_1500.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

contact_rows = [r for r in rows if r.get("is_contact") == True]
print(f"Total contact frames: {len(contact_rows)} / {len(rows)}")

by_action = defaultdict(list)
for r in contact_rows:
    by_action[f"{r['pair']}/{r['action']}"].append(r["frame_stem"])

for act, stems in sorted(by_action.items()):
    print(f"{act:<25}: {len(stems)} contact frames, samples: {stems[:5]}")

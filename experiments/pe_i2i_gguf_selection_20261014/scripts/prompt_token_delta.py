#!/usr/bin/env python3
"""Why does llama.cpp report ~7400 prompt tokens where HF reports 6803?

Tokenise the identical PE request (same three images, same effective system prompt, same
instruction) through the HF processor on CPU and split the count into text vs visual, so the
comparison against the GGUF arm is not silently comparing different inputs.
"""
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I"
W = "/home/shangguanrz/project/pic-edit/workflow"

proc = AutoProcessor.from_pretrained(MODEL)
merge = getattr(proc.image_processor, "merge_size", 2)

for run in sys.argv[1:] or ["q8s_hard_02"]:
    rd = os.path.join(W, "runs", run)
    ev = json.load(open(os.path.join(rd, "prompt", "pe_result.json"), encoding="utf-8"))
    images, sizes = [], []
    for item in ev["input_roles"]:
        images.append(Image.open(item["path"]).convert("RGB"))
        sizes.append(list(images[-1].size))
    system = open(os.path.join(rd, "prompt", "effective_system_prompt.txt"), encoding="utf-8").read()
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [*[{"type": "image", "image": im} for im in images],
                                        {"type": "text", "text": ev["instruction"]}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                   enable_thinking=True, return_dict=True, return_tensors="pt")
    grid = inp.get("image_grid_thw")
    visual = int((grid.long().prod(dim=1) // (merge ** 2)).sum()) if grid is not None else 0
    total = int(inp["input_ids"].shape[1])
    print("%s: HF prompt_tokens=%d (visual=%d text=%d) sizes=%s grid=%s" % (
        run, total, visual, total - visual, sizes,
        None if grid is None else grid.tolist()))
    print("   instruction chars=%d system chars=%d" % (len(ev["instruction"]), len(system)))

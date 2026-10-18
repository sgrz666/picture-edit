#!/usr/bin/env python3
"""Check target-RGB isolation and target-geometry sensitivity on one Hi4D source."""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image, ImageDraw

from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset
from src.pose_control.qwen21.action_prompts import build_dual_action_prompts
from src.pose_control.qwen21.dual_adapter import Qwen21DualAdapter
from src.pose_control.qwen21.dual_inference import generation_inputs_from_sample, generate_controlled
from src.pose_control.qwen21.runtime import Qwen21IPERCache
from train_hi4d_qwen21_dual import DUAL_TARGET_CACHE_NAMESPACE, ROOT


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-pairs", type=Path, required=True)
    parser.add_argument("--alternate-pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    parser.add_argument("--control-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_controls"))
    parser.add_argument("--latent-cache", type=Path, default=Path("/home/shangguanrz/nihaochen/qwen21_hi4d_latents"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def control_differences(a, b):
    changed = []
    for name in a:
        left, right = a[name], b[name]
        if torch.is_tensor(left) and not torch.equal(left, right):
            changed.append(name)
        elif name == "contact_relations":
            if any(not torch.equal(getattr(left, field), getattr(right, field))
                   for field in ("src_person", "src_part", "dst_person", "dst_part",
                                 "contact_type", "distance", "valid_mask")):
                changed.append(name)
    return changed


def main():
    config = args()
    config.output.mkdir(parents=True, exist_ok=True)
    peak = Hi4DQwen21Dataset(config.peak_pairs, config.control_cache, load_target=False)[0]
    alternate = Hi4DQwen21Dataset(config.alternate_pairs, config.control_cache, load_target=False)[0]
    assert peak["row"]["source_rgb"] == alternate["row"]["source_rgb"]
    assert "tgt_image" not in peak and "tgt_image" not in alternate
    assert torch.equal(peak["src_image"], alternate["src_image"])
    prompt = build_dual_action_prompts(peak["row"]["target_action"],
                                       (peak["source_mask_a"], peak["source_mask_b"]))
    alternate_prompt = build_dual_action_prompts(alternate["row"]["target_action"],
                                                 (alternate["source_mask_a"], alternate["source_mask_b"]))
    assert prompt == alternate_prompt, "geometry comparison must keep prompt fixed"
    _, baseline = generation_inputs_from_sample(peak, *prompt)

    # Inject fake ground truth fields in memory. The generation boundary must
    # discard them, including any target path in the metadata row.
    poisoned = dict(peak)
    poisoned["row"] = {**peak["row"], "target_rgb": "/tmp/fake_target_rgb.png",
                       "target_gt_person_mask_A": "/tmp/fake_gt_A.png",
                       "target_gt_person_mask_B": "/tmp/fake_gt_B.png"}
    poisoned["tgt_image"] = torch.full_like(peak["src_image"], -1)
    poisoned["gt_mask_A"] = torch.ones((1, 512, 512))
    poisoned["gt_mask_B"] = torch.zeros((1, 512, 512))
    _, replaced = generation_inputs_from_sample(poisoned, *prompt)
    assert np.array_equal(np.asarray(baseline.source_image), np.asarray(replaced.source_image))
    assert not control_differences(baseline.controls, replaced.controls)
    _, switched = generation_inputs_from_sample(alternate, *alternate_prompt)
    changed_controls = control_differences(baseline.controls, switched.controls)
    assert changed_controls, "alternate target frame did not change the control tensors"

    saved = torch.load(config.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    pipe = QwenImage21Pipeline.from_pretrained(str(config.model), dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval().requires_grad_(False)
    adapter = Qwen21DualAdapter().to(device="cuda", dtype=torch.float32)
    adapter.load_state_dict(saved["adapter_state_dict"])
    adapter.eval()
    cache = Qwen21IPERCache(config.latent_cache, prompt=DUAL_TARGET_CACHE_NAMESPACE)
    base_image = generate_controlled(pipe, adapter, cache, baseline,
                                     mode="full", steps=config.steps, seed=config.seed)
    replaced_image = generate_controlled(pipe, adapter, cache, replaced,
                                         mode="full", steps=config.steps, seed=config.seed)
    alternate_image = generate_controlled(pipe, adapter, cache, switched,
                                          mode="full", steps=config.steps, seed=config.seed)
    base_array = np.asarray(base_image)
    replaced_array = np.asarray(replaced_image)
    alternate_array = np.asarray(alternate_image)
    equal = bool(np.array_equal(base_array, replaced_array))
    geometry_mae = float(np.abs(base_array.astype(np.float32) - alternate_array).mean())
    base_image.save(config.output / "frame62_baseline.png")
    replaced_image.save(config.output / "frame62_replaced_gt.png")
    alternate_image.save(config.output / "frame34_geometry.png")
    panel = Image.new("RGB", (1024, 544), (30, 30, 30))
    for index, (image, caption) in enumerate(((base_image, "frame62 target geometry"),
                                               (alternate_image, "frame34 target geometry"))):
        panel.paste(image, (index * 512, 32))
        ImageDraw.Draw(panel).text((index * 512 + 8, 8), caption, fill="white")
    panel.save(config.output / "geometry_switch_panel.png")
    report = {"source": peak["row"]["source_rgb"], "prompt": prompt[1],
              "checkpoint": str(config.checkpoint), "seed": config.seed, "steps": config.steps,
              "target_rgb_loaded_for_generation": False,
              "target_rgb_and_gt_mask_replacement_pixel_equal": equal,
              "target_rgb_and_gt_mask_replacement_max_abs_diff": int(np.abs(base_array.astype(np.int16) - replaced_array).max()),
              "changed_control_names": changed_controls,
              "geometry_switch_output_mae": geometry_mae,
              "geometry_switch_output_pixel_equal": bool(np.array_equal(base_array, alternate_array))}
    (config.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not equal or geometry_mae <= 0:
        raise AssertionError("target isolation or geometry sensitivity audit failed")


if __name__ == "__main__":
    main()

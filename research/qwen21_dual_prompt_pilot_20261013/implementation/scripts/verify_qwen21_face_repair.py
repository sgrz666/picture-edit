#!/usr/bin/env python3
"""Check repaired Face candidates without updating weights or generating images."""

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from diffusers import QwenImage21Pipeline
from scripts.run_qwen21_weight_inheritance import verify_gpu
from scripts.verify_qwen21_full_runtime import cpu_checkpoint_comparison
from src.pose_control.qwen21_full.execution import (
    MODEL, RUN_ROOT, FaceConditionStore, QwenFullDataset, Qwen21IPERCache,
    json_write, load_adapter_checkpoint, new_adapter, seed_all,
)

OUT = RUN_ROOT / 'face_repair'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-1024', action='store_true')
    args = p.parse_args()
    seed_all(42)
    torch.set_num_threads(4)
    pipe = QwenImage21Pipeline.from_pretrained(str(MODEL), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    store = FaceConditionStore(RUN_ROOT / 'face_cache')
    cases = [('roi_same_aux', 512, None), ('roi_aligned_aux', 512, None)]
    if args.include_1024:
        cases.append(('roi_same_aux', 1024, (32, 32)))
    reports = []
    for run, resolution, reference_hw in cases:
        checkpoint = OUT / run / 'checkpoint_step_000300.pt'
        load_adapter_checkpoint(adapter, checkpoint)
        data = QwenFullDataset(RUN_ROOT / 'face_single_sample/single_pair.jsonl', resolution=resolution)
        cache = Qwen21IPERCache(RUN_ROOT / 'native_cache', resolution=resolution)
        report = verify_gpu(pipe, adapter, cache, data, store, run,
                            face_reference_token_hw=reference_hw)
        report['frozen_nonface_weights'] = cpu_checkpoint_comparison(
            RUN_ROOT / 'face_single_sample/body_train/checkpoint_step_000500.pt', checkpoint)
        reports.append(report)
        print(json.dumps({'run': run, 'resolution': resolution, 'passed': report['passed']}), flush=True)
    json_write(OUT / 'runtime_acceptance.json', {'checks': reports, 'passed': all(r['passed'] for r in reports),
        'scope': 'Prediction, prefix, ROI, fallback, supervision boundary and frozen weights; no training.'})

if __name__ == '__main__':
    main()

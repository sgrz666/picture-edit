#!/usr/bin/env python3
"""Diagnose one-step CUDA checkpoint reproducibility without extending training.

Reloads the same checkpoint before each trial, restores every RNG state, and
compares model, AdamW state and gradients before/after the same next sample.
No existing checkpoints or logs are overwritten.
"""
from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import QwenImage21Pipeline

from src.pose_control.qwen21_full.execution import (
    FaceConditionStore, QwenFullDataset, Qwen21IPERCache, json_write,
    load_adapter_checkpoint, new_adapter, parameter_groups, run_loss,
    set_training_phase,
)


def restore_rng(saved):
    torch.set_rng_state(saved["torch_rng_state"])
    torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])


@contextmanager
def deterministic_reference():
    """Reference-only equivalent pooling; never used by the training entrypoint.

    Divisible grids produce exactly the same mathematical averages using disjoint
    windows. For non-divisible grids the existing adaptive pooling runs on CPU;
    device transfers preserve its autograd connection. Floating summation order
    can differ from the official CUDA kernels, so this is a checkpoint reference
    check and is not a claim of bitwise determinism of normal CUDA training.
    """
    original = F.adaptive_avg_pool2d
    original_grid_sample = F.grid_sample
    calls = {"equivalent_nonoverlap_pool": 0, "cpu_adaptive_fallback": 0, 'cpu_grid_sample': 0}

    def pool(input, output_size):
        sizes = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        height, width = input.shape[-2:]
        out_h = height if sizes[0] is None else sizes[0]
        out_w = width if sizes[1] is None else sizes[1]
        if height % out_h == 0 and width % out_w == 0:
            calls["equivalent_nonoverlap_pool"] += 1
            stride = (height // out_h, width // out_w)
            return F.avg_pool2d(input, kernel_size=stride, stride=stride)
        calls["cpu_adaptive_fallback"] += 1
        return original(input.to("cpu"), output_size).to(input.device)

    F.adaptive_avg_pool2d = pool
    def grid_sample(input, grid, *args, **kwargs):
        # CUDA grid-sample backward is nondeterministic. Only this reference
        # experiment uses CPU; the formal Face training path is unchanged.
        if input.is_cuda:
            calls['cpu_grid_sample'] += 1
            return original_grid_sample(input.float().cpu(), grid.float().cpu(), *args, **kwargs).to(input)
        return original_grid_sample(input, grid, *args, **kwargs)
    F.grid_sample = grid_sample
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield calls
    finally:
        F.adaptive_avg_pool2d = original
        F.grid_sample = original_grid_sample


def snapshot(adapter, optimizer, *, gradients=False):
    parameters = {name: p.detach().cpu().clone() for name, p in adapter.named_parameters() if p.requires_grad}
    by_id = {id(p): name for name, p in adapter.named_parameters()}
    state = {}
    for p, value in optimizer.state.items():
        for key, item in value.items():
            state[by_id[id(p)] + "/" + key] = item.detach().cpu().clone() if torch.is_tensor(item) else item
    output = {"parameters": parameters, "optimizer": state}
    if gradients:
        output["gradients"] = {name: p.grad.detach().cpu().clone()
                               for name, p in adapter.named_parameters() if p.grad is not None}
    return output


def digest(values):
    result = hashlib.sha256()
    for key, value in sorted(values.items()):
        result.update(key.encode())
        if torch.is_tensor(value):
            result.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            result.update(repr(value).encode())
    return result.hexdigest()


def compare(a, b):
    if a.keys() != b.keys():
        return {"keys_equal": False, "left_only": sorted(a.keys() - b.keys()),
                "right_only": sorted(b.keys() - a.keys())}
    changed = []
    max_abs, max_relative, diff_sq, reference_sq = 0., 0., 0., 0.
    total_values, changed_values = 0, 0
    for key, left in a.items():
        right = b[key]
        if not torch.is_tensor(left):
            if left != right:
                changed.append({"name": key, "left": left, "right": right})
            continue
        total_values += left.numel()
        ref_l2 = float(left.float().square().sum(dtype=torch.float64))
        reference_sq += ref_l2
        if torch.equal(left, right):
            continue
        difference = left.float() - right.float()
        absolute = float(difference.abs().max())
        relative = absolute / max(float(left.float().abs().max()), 1e-30)
        count = int(torch.count_nonzero(difference))
        l2 = float(difference.square().sum(dtype=torch.float64))
        changed_values += count
        diff_sq += l2
        max_abs = max(max_abs, absolute)
        max_relative = max(max_relative, relative)
        changed.append({"name": key, "shape": list(left.shape), "max_abs": absolute,
                        "max_abs_over_tensor_max": relative,
                        "relative_l2": (l2 / max(ref_l2, 1e-60)) ** .5,
                        "changed_values": count})
    changed.sort(key=lambda row: row.get("max_abs", float("inf")), reverse=True)
    return {"keys_equal": True, "exact_equal": not changed, "tensor_count": len(a),
            "total_values": total_values, "changed_values": changed_values,
            "changed_tensors": len(changed), "max_abs": max_abs,
            "max_tensor_relative": max_relative,
            "relative_l2": (diff_sq / max(reference_sq, 1e-60)) ** .5,
            "left_sha256": digest(a), "right_sha256": digest(b),
            "differences": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "/home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "/home/shangguanrz/nihaochen/qwen21_full/acceptance/resume_diagnosis.json"))
    parser.add_argument("--skip-deterministic", action="store_true")
    parser.add_argument("--deterministic-reference-only", action="store_true",
                        help="Only run two reference steps using equivalent pooling and MATH SDPA")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = saved["args"]
    branch = saved["branch"]
    if branch not in ('body', 'face'):
        raise ValueError('Unsupported checkpoint branch')
    if branch == 'face' and (saved['step'] + 1) % 4 == 0:
        raise ValueError('This continuation diagnostic needs a non-auxiliary next step')
    train = QwenFullDataset(config["train_pairs"], resolution=saved["resolution"])
    cache = Qwen21IPERCache(config["cache"], resolution=saved["resolution"], prompt=config["prompt"])
    store = FaceConditionStore(config["features"])
    epoch, offset = divmod(saved["step"], len(train))
    pair_index = int(torch.randperm(len(train), generator=torch.Generator().manual_seed(config["seed"] + epoch))[offset])
    pipe = QwenImage21Pipeline.from_pretrained(config["model"], dtype=torch.bfloat16)
    for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
        module.eval().requires_grad_(False)
    pipe.transformer.to("cuda").enable_gradient_checkpointing()
    adapter = new_adapter("cuda").train()
    groups, names = parameter_groups(adapter, branch)
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    payload = {"checkpoint": str(args.checkpoint), "checkpoint_step": saved["step"], 'branch': branch,
               "next_step": saved["step"] + 1, "pair_index": pair_index,
               "scope": "Repeat one checkpoint continuation step; no training checkpoint changes",
               "torch_version": torch.__version__, "trials": {}}
    started = time.perf_counter()

    def run_trial():
        restored = load_adapter_checkpoint(adapter, args.checkpoint)
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        set_training_phase(adapter, branch, warmup=saved["step"] < config["warmup_steps"])
        optimizer.zero_grad(set_to_none=True)
        restore_rng(restored)
        before = snapshot(adapter, optimizer)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            row = run_loss(pipe, adapter, cache, train, store, pair_index, branch,
                           seed=config["seed"] + saved["step"] + (100000 if branch == 'face' else 0),
                           require_active_face=config.get('require_active_gradients',False),
                           face_flow_loss=config.get('face_flow_loss','global'),
                           face_aux_alignment=config.get('face_aux_alignment','box'),
                           normalized_face_evaluators=config.get('face_evaluator_range')=='normalized')
        gradients = {name: p.grad.detach().cpu().clone()
                     for name, p in adapter.named_parameters() if p.grad is not None}
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0))
        optimizer.step()
        torch.cuda.synchronize()
        return {"before": before, "after": snapshot(adapter, optimizer),
                "gradients": gradients, "row": row, "grad_norm": norm}

    modes = [True] if args.deterministic_reference_only else ([False] if args.skip_deterministic else [False, True])
    for deterministic in modes:
        label = "deterministic_reference" if deterministic else "default_cuda"
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        print(f"START {label}: checkpoint step {saved['step']}, next pair {pair_index}", flush=True)
        try:
            with deterministic_reference() if deterministic else nullcontext() as pool_calls:
                first = run_trial()
                second = run_trial()
            report = {"first_loss": first["row"]["loss"], "second_loss": second["row"]["loss"],
                      "first_grad_norm": first["grad_norm"], "second_grad_norm": second["grad_norm"],
                      "input_reload_parameters": compare(first["before"]["parameters"], second["before"]["parameters"]),
                      "input_reload_optimizer": compare(first["before"]["optimizer"], second["before"]["optimizer"]),
                      "gradients": compare(first["gradients"], second["gradients"]),
                      "parameters": compare(first["after"]["parameters"], second["after"]["parameters"]),
                      "optimizer": compare(first["after"]["optimizer"], second["after"]["optimizer"])}
            if deterministic:
                report["reference_implementation"] = {
                    "adaptive_pool": "disjoint avg_pool2d on divisible grids; original adaptive pooling on CPU otherwise",
                    "sdpa_backend": "MATH", "pool_calls": pool_calls,
                    'grid_sample': 'CPU reference only; formal training remains CUDA',
                    "formal_training_changed": False,
                    "limitation": "Verifies checkpoint continuation against a deterministic reference, not default CUDA bitwise determinism",
                }
            payload["trials"][label] = report
            print(json.dumps({"trial": label, "losses": [report["first_loss"], report["second_loss"]],
                              **{key: {name: report[key][name] for name in ("exact_equal", "max_abs", "relative_l2", "changed_tensors")}
                                 for key in ("gradients", "parameters", "optimizer")}}), flush=True)
            del first, second
        except Exception:
            payload["trials"][label] = {"error": traceback.format_exc()}
            print(payload["trials"][label]["error"], flush=True)
        payload["seconds"] = time.perf_counter() - started
        payload["peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        json_write(args.output, payload)
        gc.collect()
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/shangguanrz/project/pic-edit")
sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.execution import new_adapter, parameter_groups, BODY_PREFIXES
from src.pose_control.qwen21_full.checkpoint import FACE_ONLY_PREFIXES, HAND_ONLY_PREFIXES

def custom_parameter_groups(adapter, branch, *, lr_scale=1.0):
    if not np.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError('lr_scale must be positive and finite')
    groups = {'heads_and_interface': [], 'encoders': [], 'control_core': []}
    names = {key: [] for key in groups}
    for name, parameter in adapter.named_parameters():
        if branch == 'body' and name.startswith('output_bridges.geometry.'):
            group = 'heads_and_interface'
        elif branch == 'face' and name.startswith('output_bridges.face.'):
            group = 'heads_and_interface'
        elif branch == 'hand' and name.startswith('output_bridges.hand.'):
            group = 'heads_and_interface'
        elif branch == 'body' and name.startswith(('latent_bridge.', 'text_bridge.', 'pooled_bridge.')):
            group = 'heads_and_interface'
        elif name.startswith('legacy.'):
            short = name[len('legacy.'):]
            if branch == 'body':
                allowed = BODY_PREFIXES
            elif branch == 'face':
                allowed = FACE_ONLY_PREFIXES
            elif branch == 'hand':
                allowed = HAND_ONLY_PREFIXES
            else:
                raise ValueError(f"Unknown branch: {branch}")
            if not short.startswith(allowed):
                continue
            if '.zero_heads.' in short or '_log_group_scale' in short:
                group = 'heads_and_interface'
            elif short.startswith(('control_interface.control_core.', 'control_interface.face_adapter.', 'control_interface.hand_adapter.')):
                group = 'control_core'
            else:
                group = 'encoders'
        else:
            continue
        groups[group].append(parameter)
        names[group].append(name)
    rates = {'heads_and_interface': 1e-4, 'encoders': 5e-5, 'control_core': 1e-5}
    result = [{'params': groups[key], 'lr': rates[key] * lr_scale, 'name': key} for key in groups if groups[key]]
    if not result:
        raise ValueError(f'No trainable parameters selected for branch {branch}')
    return result, names

def main():
    adapter = new_adapter("cpu")
    for branch in ("body", "face", "hand"):
        groups, names = custom_parameter_groups(adapter, branch, lr_scale=1.0)
        print(f"\n--- Branch: {branch} ---")
        for g in groups:
            name = g['name']
            param_list = names[name]
            total_params = sum(p.numel() for p in g['params'])
            print(f"  Group: {name:20s} | Tensors: {len(param_list):3d} | Params: {total_params:10d} | LR: {g['lr']}")
            print(f"    Sample: {param_list[:2]}")

if __name__ == '__main__':
    main()

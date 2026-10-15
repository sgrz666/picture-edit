"""Verify the saved evidence without a GPU, torch, network access, or file writes."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    remote = json.loads((root / 'remote_audit.json').read_text(encoding='utf-8'))
    source = root / 'source_snapshot'
    header = json.loads((source / 'upstream/union_safetensors_header.json').read_text(encoding='utf-8'))
    weights = {name: value for name, value in header.items() if name != '__metadata__'}
    base_shapes = remote['base_block_0_tensor_shapes']
    assert len(base_shapes) == 9
    shape_checks = 0
    for block in range(16):
        for suffix, shape in base_shapes.items():
            name = f'control_blocks.{block}.{suffix}'
            assert weights[name]['shape'] == shape, name
            shape_checks += 1
        assert weights[f'control_blocks.{block}.after_proj.weight']['shape'] == [4096, 4096]
        assert weights[f'control_blocks.{block}.after_proj.bias']['shape'] == [4096]
    assert weights['control_blocks.0.before_proj.weight']['shape'] == [4096, 4096]
    assert weights['control_blocks.0.before_proj.bias']['shape'] == [4096]
    assert weights['control_img_in.weight']['shape'] == [4096, 129]
    assert weights['control_img_in.bias']['shape'] == [4096]
    assert len(weights) == 180
    assert {value['dtype'] for value in weights.values()} == {'BF16'}
    params = sum(math.prod(value['shape']) for value in weights.values())
    size = sum(value['data_offsets'][1] - value['data_offsets'][0] for value in weights.values())
    assert params == 3_775_479_808
    assert size == params * 2
    snapshot_checks = 0
    for path in sorted((source / 'remote_20261015').rglob('*.py')):
        rel = path.relative_to(source / 'remote_20261015').as_posix()
        if rel.startswith('diffusers/'):
            continue
        source_name = 'src/pose_control/' + rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == remote['source_sha256'][source_name], source_name
        snapshot_checks += 1
    syntax_checks = 0
    for path in sorted(root.rglob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        syntax_checks += 1
    groups = [[index for index in range(32) if 6 * index // 32 == group] for group in range(6)]
    rank = 32
    lora_core = sum(rank * sum(shape) for suffix, shape in base_shapes.items() if len(shape) == 2) * 16
    lora_with_zero_projections = lora_core + 17 * rank * 8192
    manifest_checks = 0
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for entry in manifest['files']:
            path = root / entry['path']
            assert path.stat().st_size == entry['bytes'], entry['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
            manifest_checks += 1
    result = {
        'status': 'PASS',
        'scope': 'Source hashes, Python syntax and checkpoint tensor-shape compatibility only; no learned-weight numerics or GPU generation.',
        'remote_source_hashes_matched': snapshot_checks,
        'python_files_parsed': syntax_checks,
        'control_block_core_tensor_shapes_matched': shape_checks,
        'manifest_files_verified': manifest_checks,
        'control_tensor_count': len(weights),
        'control_parameters': params,
        'control_bf16_gib': size / 2**30,
        'base_parameters': remote['base_transformer_parameters'],
        'base_plus_control_bf16_gib': (size + remote['base_transformer_tensor_bytes']) / 2**30,
        'full_adapter_layer_groups': groups,
        'lora_rank_32_qkvo_and_mlp_parameters': lora_core,
        'lora_rank_32_including_before_after_parameters': lora_with_zero_projections,
        'training_state_estimates_gib': {
            'base_bf16_plus_control_12_bytes_per_parameter': (remote['base_transformer_tensor_bytes'] + params * 12) / 2**30,
            'base_bf16_plus_control_16_bytes_per_parameter': (remote['base_transformer_tensor_bytes'] + params * 16) / 2**30,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

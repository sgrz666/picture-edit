"""Read-only source/config/safetensors-header audit; never imports torch or loads tensors."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import subprocess
from datetime import datetime, timezone

ROOT = Path('/home/shangguanrz/project/pic-edit')
MODEL = ROOT / 'models/Qwen-Image-2.1/transformer'


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}


def main():
    report = {
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Read-only CPU audit. No inference, CUDA initialization, tensor loading, package install, or production edits.',
        'remote_root': str(ROOT),
        'git_log': command(['git', 'log', '-3', '--oneline']),
        'relevant_git_status': command(['git', 'status', '--short', '--', 'src/pose_control/qwen21', 'src/pose_control/qwen21_full']),
        'transformer_config': json.loads((MODEL / 'config.json').read_text()),
        'vae_config': json.loads((ROOT / 'models/Qwen-Image-2.1/vae/config.json').read_text()),
        'model_index': json.loads((ROOT / 'models/Qwen-Image-2.1/model_index.json').read_text()),
        'package_versions': {},
        'source_sha256': {},
        'transformer_weight_headers': [],
        'base_block_0_tensor_shapes': {},
    }
    for package in ('torch', 'diffusers', 'transformers', 'safetensors', 'accelerate'):
        try:
            report['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['package_versions'][package] = None
    for subdir in ('qwen21', 'qwen21_full'):
        for path in sorted((ROOT / 'src/pose_control' / subdir).rglob('*.py')):
            report['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    total_params = 0
    total_bytes = 0
    seen = set()
    for path in sorted(MODEL.glob('*.safetensors')):
        with path.open('rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            if header_size > 20_000_000:
                raise ValueError(f'Unexpected safetensors header length: {path}: {header_size}')
            raw = handle.read(header_size)
        header = json.loads(raw)
        tensor_count = 0
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if name in seen:
                raise ValueError(f'Duplicate tensor {name}')
            seen.add(name)
            tensor_count += 1
            total_params += math.prod(value['shape'])
            total_bytes += value['data_offsets'][1] - value['data_offsets'][0]
            if name.startswith('transformer_blocks.0.'):
                report['base_block_0_tensor_shapes'][name.removeprefix('transformer_blocks.0.')] = value['shape']
        report['transformer_weight_headers'].append({
            'file': str(path), 'file_bytes': path.stat().st_size, 'header_bytes': header_size,
            'header_sha256': hashlib.sha256(raw).hexdigest(), 'tensor_count': tensor_count,
        })
    report['base_transformer_parameters'] = total_params
    report['base_transformer_tensor_bytes'] = total_bytes
    report['nvidia_smi'] = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

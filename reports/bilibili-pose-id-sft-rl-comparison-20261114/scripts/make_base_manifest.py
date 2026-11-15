import hashlib
import json
from pathlib import Path

BASE = Path('/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1/base_qwen_eval_20261114_v1')
INFER = BASE / 'output/base_infer_20261114_v2'
EVAL = BASE / 'eval_base_qwen_20261114_v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


outputs = {row['original_row']['pair_id']: row for row in read_jsonl(INFER / 'outputs.jsonl')}
val = read_jsonl(BASE / 'data_subset/pilot_val.jsonl')
per_image = json.loads((EVAL / 'per_image.json').read_text(encoding='utf-8'))
summary = json.loads((EVAL / 'summary.json').read_text(encoding='utf-8'))
rows = []
for row in val:
    pair_id = row['pair_id']
    record = outputs[pair_id]
    image = Path(record['image'])
    rows.append({
        'pair_id': pair_id,
        'case': record['case'],
        'image': str(image),
        'sha256': sha256(image),
        'bytes': image.stat().st_size,
        'seed': record['seed'],
        'steps': record['steps'],
        'true_cfg_scale': record['true_cfg_scale'],
        'seconds': record['seconds'],
    })
manifest = {
    'schema': 'base_qwen_reference_v1',
    'created_at': '2026-11-14 America/New_York (server 2026-11-15 midday Asia/Shanghai)',
    'purpose': 'Base Qwen visual reference for the 50 fixed validation pairs; not part of training, checkpoint selection or the 7-checkpoint statistics.',
    'source_report': '/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/report/sft-rl-comparison-20261114-v1',
    'pairs': len(rows),
    'protocol': {
        'canvas_wh': [384, 672],
        'seed': 42,
        'steps': 30,
        'true_cfg_scale': 1.0,
        'adapter': None,
        'pipeline': 'upstream QwenImage21Pipeline used by bilibili-dwpose-edit-lora (KV cache default on)',
    },
    'model_root': '/data/shangguanrz/pic-edit-models/Qwen-Image-2.1',
    'config': str(BASE / 'configs/base_infer.json'),
    'config_sha256': sha256(BASE / 'configs/base_infer.json'),
    'data_index_sha256': sha256(BASE / 'data_subset/index.json'),
    'val_manifest_sha256': sha256(BASE / 'data_subset/pilot_val.jsonl'),
    'images': rows,
    'metrics_summary': summary,
    'metrics_per_image': str(EVAL / 'per_image.json'),
    'metrics_per_image_sha256': sha256(EVAL / 'per_image.json'),
    'per_image': per_image,
    'notes': [
        'All 50 generated images are 384x672 RGB PNG, no cropping or filters.',
        'RL runtime evaluation uses use_kv_cache=false; the upstream pipeline used here defaults to KV cache, so Base is a visual/scale reference, not a bit-identical protocol replay.',
        'ID applies to the same fixed 33 pairs; the other 17 remain not applicable and are not imputed as zero.',
    ],
}
(BASE / 'base_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                         encoding='utf-8')
print('wrote', BASE / 'base_manifest.json', 'images', len(rows), 'bytes', (BASE / 'base_manifest.json').stat().st_size)

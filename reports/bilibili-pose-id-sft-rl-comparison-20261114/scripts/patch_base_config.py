import json
from pathlib import Path
p = Path('/data/shangguanrz/LoRA/experiments/bilibili-pose-id-grpo/runs/20261112_v1/base_qwen_eval_20261114_v1/configs/base_infer.json')
d = json.loads(p.read_text(encoding='utf-8'))
d['cache_manifests'] = ['pilot_train.jsonl', 'pilot_val.jsonl']
d['training_manifest'] = 'pilot_train.jsonl'
p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print('cache_manifests =', d['cache_manifests'])
print('training_manifest =', d.get('training_manifest'))

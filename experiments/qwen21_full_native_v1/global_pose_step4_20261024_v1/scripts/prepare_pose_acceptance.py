"""固定 8 TikTok + 8 iPER 帧，含 iPER 的三个 role；只写 JSONL。"""
from pathlib import Path
import json

repo = Path('/home/shangguanrz/project/pic-edit')
out = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1/data')
out.mkdir(exist_ok=True)
frames = []
for i in range(1, 9):
    frames.append({'dataset': 'tiktok', 'sequence_id': '00001', 'stem': f'{i:04d}', 'role': 'target'})
mapping = json.loads((repo / 'datasets/iPER/iper_sampled_6src64tgt/001_8/smplestx/frame_map.json').read_text())['frames']
iper = [mapping[i] for i in (0, 1, 3, 4, 6, 18, 38, 69)]
frames.extend({'dataset': 'iper', 'sequence_id': '001_8', 'stem': f['stem'], 'role': f['role']} for f in iper)
with (out / 'frames.jsonl').open('x') as stream:
    for row in frames:
        stream.write(json.dumps(row) + '\n')
with (out / 'tiktok_pairs.jsonl').open('x') as stream:
    for row in frames[:8]:
        stream.write(json.dumps({'pair_id': 'pose_tiktok_' + row['stem'], 'sequence_id': '00001',
                                 'source_stem': '0080', 'target_stem': row['stem']}) + '\n')
with (out / 'iper_pairs.jsonl').open('x') as stream:
    for frame in iper:
        stream.write(json.dumps({'pair_id': 'pose_iper_' + frame['stem'], 'appearance': '001_8',
                                 'source': mapping[0], 'target': frame}) + '\n')
print('Prepared 16 raw frame identities and 16 paired records')

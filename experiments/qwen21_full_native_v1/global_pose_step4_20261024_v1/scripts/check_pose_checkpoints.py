"""核对真实 v1 重建/v2 迁移；只读权重，不创建 optimizer。"""
from pathlib import Path
import json
import sys
sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
import torch
from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRuntime
from src.pose_control.qwen21_full.native_geometry import POSE_PREFIXES
from src.pose_control.qwen21_full.pose_data_runtime import NativePoseDataset
from src.pose_control.qwen21_full.pose_cache import sha256

torch.set_num_threads(4)
run = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1')
original = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/depth_step3_20261023_v1/training_smoke_v1/checkpoint_last.pt')
old = torch.load(original, map_location='cpu', mmap=True, weights_only=True)
facts = []
for filename, legacy in (('native_legacy_rebuild_v1.pt', True), ('native_body21_v2.pt', False)):
    path = run / 'checkpoints' / filename
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    model = NativeDepthRuntime.from_native_checkpoint(saved, legacy=legacy)
    common = [key for key, value in old['state_dict'].items() if torch.is_tensor(value)]
    assert all(torch.equal(saved['state_dict'][key], old['state_dict'][key]) for key in common)
    assert all(torch.equal(value, saved['state_dict'][key]) for key, value in model.state_dict().items() if torch.is_tensor(value))
    extras = sorted(set(saved['state_dict']) - set(old['state_dict']))
    assert all(key.startswith(tuple('geometry.' + p for p in POSE_PREFIXES)) for key in extras)
    assert 'optimizer_state_dict' not in saved
    if not legacy:
        assert model.geometry.reasoner.person_reasoner.joint_attention.projection.weight.eq(0).all()
    facts.append({'file': str(path), 'architecture': saved['architecture'], 'exact_inherited_tensors': len(common),
                  'declared_new_pose_keys': len(extras), 'all_inherited_tensor_max_abs': 0,
                  'strict_roundtrip': True, 'optimizer_present': False, 'sha256': sha256(path)})

# Bind explicit asset -> original transform for historical iPER 512 depth maps.
for kind in ('tiktok', 'iper'):
    dataset = NativePoseDataset(run / 'data' / (kind + '_pairs.jsonl'), run / 'data/fk_v1/records.jsonl')
    with (run / 'data' / (kind + '_depth.jsonl')).open('x') as stream:
        for i in range(len(dataset)):
            sample = dataset[i]
            raw = dataset.records.raw(sample['pose_provenance']['record_id'])
            meta = raw['metadata']
            sx, sy = meta['condition_to_original']['depth_vis']['scale_xy']
            row = {'pair_id': sample['pair_metadata']['pair_id'], 'target_stem': sample['target_stem'],
                   'kind': 'relative_png', 'polarity': 'near_one',
                   'depth_path': meta['sources']['depth_vis']['path'], 'valid_path': meta['sources']['smpl_mask']['path'],
                   'camera_id': meta['record_id'], 'units': 'unknown', 'scale_status': 'unknown',
                   'asset_to_original': [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]}
            stream.write(json.dumps(row) + '\n')
report = {'schema': 'body21_checkpoint_audit_v2', 'passed': True, 'no_optimizer': True,
          'source_sha256': sha256(original), 'migrations': facts}
with (run / 'report/checkpoint_audit_v1.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))

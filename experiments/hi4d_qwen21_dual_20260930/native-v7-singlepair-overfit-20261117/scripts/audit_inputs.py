"""Inspect existing production assets without modifying them or starting training."""
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path('/home/shangguanrz/project/pic-edit')
if PROJECT.exists():
    sys.path.insert(0, str(PROJECT))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

MANIFEST = Path('/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-v5-grounded-binding-pilot_20261030_r4/data/typed_v5_pairs.jsonl')
INITIALIZATION = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1/initialization/native_v5_init_v2.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_full_assets import sha256
    dataset = NativeFullDatasetV62(MANIFEST, load_target=True, data_profile='hybrid_hi4d')
    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        row = dataset.rows[index]
        local = sample['local_condition_v4']
        refs = sample['local_references_v4']
        v4 = next(json.loads(line) for line in Path(row['v4_manifest']['path']).read_text().splitlines()
                  if json.loads(line)['pair_id'] == row['pair_id'])
        records.append({'pair_id': row['pair_id'], 'manifest': row, 'v4': v4,
            'sample_keys': sorted(sample), 'source_canvas': sample['source_canvas'], 'target_canvas': sample['target_canvas'],
            'local_metadata': local.metadata, 'face_boxes': local.face_boxes.tolist(),
            'face_roi_valid': local.face_roi_valid.tolist(), 'source_indices': local.source_indices.tolist(),
            'arcface_valid': refs.arcface_valid.tolist(), 'face_dino_valid': refs.face_dino_valid.tolist(),
            'hand_dino_valid': refs.hand_dino_valid.tolist(), 'auxiliary_valid': sample['training_supervision_v5'].auxiliary_valid.tolist(),
            'source_mask_shape': list(sample['source_person_mask'].shape),
            'target_grid': list(sample['target_token_hw'])})
    saved = torch.load(INITIALIZATION, map_location='cpu', mmap=True, weights_only=True)
    config = saved.get('runtime_config', {})
    initialization = {'path': str(INITIALIZATION), 'sha256': sha256(INITIALIZATION),
        'architecture': saved.get('architecture'), 'config': config.get('config'),
        'base_config': config.get('base'), 'bps_convention': config.get('bps_convention'),
        'face_parameter_basis': config.get('face_parameter_basis'), 'profile': saved.get('profile'),
        'metadata_keys': sorted(saved.get('metadata', {})), 'has_training_state': 'training_state' in saved,
        'body_heads_nonzero': {key: int(value.count_nonzero()) for key, value in saved['state_dict'].items()
                              if key.startswith('base.body.output_heads.') and torch.is_tensor(value)}}
    evaluators = []
    for name in ('identity_encoder.ts', 'perceptual_encoder.ts'):
        path = PROJECT / 'models/face_eval' / name
        evaluators.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
    result = {'records': records, 'initialization': initialization, 'evaluators': evaluators,
              'cuda_available': torch.cuda.is_available(), 'training_started': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({'report': str(args.output), 'pairs': len(records), 'initialization': initialization,
                      'cuda_available': result['cuda_available']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

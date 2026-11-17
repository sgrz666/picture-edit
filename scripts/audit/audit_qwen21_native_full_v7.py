#!/usr/bin/env python3
"""Audit actual V7 input/identity/contact/repair qualification; never load a model."""
from pathlib import Path
import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def audit_manifest(manifest, *, data_profile='auto', limit=None):
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v6.conditioning import read_condition
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha
    from src.pose_control.qwen21_full.native_v7.training import paper_sample_status
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Audit limit must be a positive integer')
    dataset = NativeFullDatasetV62(manifest, load_target=True, data_profile=data_profile)
    records = []
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        sample = dataset[index]
        row = dataset.rows[index]
        condition = read_condition(row['qwen_condition'], sample)
        local = sample['local_condition_v4']
        supervision = sample['training_supervision_v5']
        support_status=paper_sample_status(sample)
        reference_grids = [image['canvas']['token_hw'] for image in condition['images']]
        target_grid = sample['target_token_hw']
        records.append({'pair_id': row['pair_id'], 'scene_kind': sample['native_scene_kind'],
            'data_profile': sample['data_profile'], 'qwen_condition_profile': condition['profile'],
            'reference_roles': [image['role'] for image in condition['images']],
            'target_person_ids': list(local.metadata[0]['target_person_ids']),
            'auxiliary_valid': supervision.auxiliary_valid[0].tolist(),
            'auxiliary_declaration': supervision.metadata[0].get('declaration'),
            'contrastive_pair_qualified': bool(supervision.auxiliary_valid[0].all() and local.person_valid[0].all()),
            'known_positive_part_pairs': support_status['paper_positive_contacts'],
            'known_forbidden_part_pairs': support_status['paper_forbidden_contacts'],
            'visible_contact_persons':support_status['paper_visible_contact_persons'],
            'source_repair_supplied': bool(row.get('pose_repair_v61')),
            'target_token_hw': list(target_grid), 'reference_token_hws': reference_grids,
            'image_latent_tokens': target_grid[0] * target_grid[1] + sum(h * w for h, w in reference_grids)})
    return {'schema': 'native_v7_input_audit_v1', 'manifest': str(Path(manifest).resolve()),
        'manifest_sha256': file_sha(manifest), 'manifest_rows': len(dataset), 'audited_rows': count,
        'complete': count == len(dataset), 'profiles': dict(Counter(row['qwen_condition_profile'] for row in records)),
        'qualified_contrastive_pairs': sum(row['contrastive_pair_qualified'] for row in records),
        'repair_rows': sum(row['source_repair_supplied'] for row in records), 'records': records,
        'model_constructed': False, 'training_started': False,
        'training_ready': False, 'readiness_reason': 'Input audit does not bind model/cache/budget or production gradient acceptance'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-profile', choices=('auto', 'full_smplx', 'hybrid_hi4d'), default='auto')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    result = audit_manifest(args.manifest, data_profile=args.data_profile, limit=args.limit)
    if args.report is not None:
        from src.pose_control.qwen21_full.native_v6.training.io import write_json
        write_json(args.report, result)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    main()

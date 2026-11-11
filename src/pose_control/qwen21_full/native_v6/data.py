"""V6 manifests retain V5 rows and add explicit independent dual-source records.

A ``native_full_pair_v6`` JSONL row declares pair_id, scene_kind='dual',
data_profile ('full_smplx' by default), and SHA-bound absolute references:
``sample`` (``native_full_sample_v6`` NPZ: raw sample dictionary including
local_condition_v4), ``source_bank`` (``native_v6_source_bank_v1`` NPZ),
``qwen_condition`` (existing ordered five-image JSON), ``source_rgb`` (image0),
``supervision`` (existing V5 training sidecar), and, for training, ``target_rgb``.
Inference never opens target RGB or the training-only supervision sidecar.
V5 rows keep their V4 manifest and frozen detail/reference sidecars unchanged.
"""
import json
from pathlib import Path

import torch

from src.data.canvas_contract import CanvasSpec
from ..native_full_assets import read_record
from ..native_full_data import NativeFullDatasetV4, collate_native_full_v4, _rgb
from ..native_v5.data import NativeFullDatasetV5, PAIR_SCHEMA as V5_PAIR_SCHEMA, verified, read_sidecar
from ..native_v5.conditioning import read_condition
from ..native_v5.contracts import NativeTrainingSupervisionV5, SUPERVISION_SCHEMA
from ..native_v5.source_bank import NativeSourceBankV5
from .source_bank import read_source_bank, validate_target_sample

PAIR_SCHEMA = 'native_full_pair_v6'
SAMPLE_SCHEMA = 'native_full_sample_v6'


class NativeFullDatasetV6(NativeFullDatasetV5):
    def __init__(self, manifest, *, load_target=False, data_profile='auto'):
        self.path = Path(manifest)
        self.load_target = load_target
        self.data_profile = data_profile
        if data_profile not in ('auto', 'full_smplx', 'hybrid_hi4d'):
            raise ValueError('Unknown V6 data profile assertion')
        self.rows = [json.loads(line) for line in self.path.read_text(encoding='utf8').splitlines() if line.strip()]
        ids = [row.get('pair_id') for row in self.rows]
        if not ids or any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError('Unique nonempty V6 pair IDs required')
        self.readers = {}
        self.indices = {}
        for row in self.rows:
            if row.get('schema') == V5_PAIR_SCHEMA:
                if 'detail' not in row or 'supervision' not in row:
                    raise ValueError('Explicit V5 frozen sidecars required')
                path = str(verified(row['v4_manifest']))
                if path not in self.readers:
                    reader = NativeFullDatasetV4(path, load_target=load_target, data_profile=data_profile)
                    self.readers[path] = reader
                    self.indices[path] = {item['pair_id']: i for i, item in enumerate(reader.rows)}
                if row['pair_id'] not in self.indices[path]:
                    raise ValueError('V5 pair absent from bound V4 manifest')
            elif row.get('schema') == PAIR_SCHEMA:
                required = {'sample', 'source_bank', 'supervision', 'qwen_condition', 'source_rgb'}
                if load_target:
                    required.add('target_rgb')
                if row.get('scene_kind') != 'dual' or required - set(row):
                    raise ValueError('Explicit V6 dual-source sample/bank/condition/supervision references required')
                profile = row.get('data_profile', 'full_smplx')
                if profile not in ('full_smplx', 'hybrid_hi4d') or (data_profile != 'auto' and profile != data_profile):
                    raise ValueError('V6 manifest data profile differs from requested profile')
            else:
                raise ValueError('Explicit V5 or V6 manifest schema required')

    def __getitem__(self, index):
        row = self.rows[index]
        if row['schema'] == V5_PAIR_SCHEMA:
            return super().__getitem__(index)
        verified(row['sample'])
        sample = read_record(row['sample'], schema=SAMPLE_SCHEMA)
        if not isinstance(sample, dict) or any(key in sample for key in (
                'src_image', 'tgt_image', 'source_bank_v5', 'local_references_v4',
                'detail_references_v5', 'training_supervision_v5', 'source_latent',
                'reference_latents', 'prepared', 'prepared_control')):
            raise ValueError('V6 sample cache must contain only raw geometry and local conditions')
        local = sample['local_condition_v4'].validate()
        if local.person_valid.shape[0] != 1:
            raise ValueError('V6 manifest entries require B1 raw conditions')
        metadata = local.metadata[0]
        profile = row.get('data_profile', 'full_smplx')
        if metadata.get('data_profile', 'full_smplx') != profile or sample.get('data_profile', profile) != profile:
            raise ValueError('Local/manifest data profiles differ')
        if sample.get('native_scene_kind', 'dual') != 'dual':
            raise ValueError('Raw scene kind differs from the dual-source manifest')
        if profile == 'hybrid_hi4d':
            raise ValueError('Independent V6 source-bank geometry currently requires full_smplx')
        target_sha = metadata.get('driver_rgb_sha256')
        if not target_sha:
            raise ValueError('Explicit target RGB fingerprint required even during inference')
        bank = read_source_bank(row['source_bank'], forbidden_rgb_sha256=target_sha)
        sample.update(native_scene_kind='dual', supervised_target_rgb=self.load_target, data_profile=profile,
            source_bank_v5=bank)
        validate_target_sample(bank, sample)
        pair = sample['pair_metadata']
        if pair.get('pair_id') != row['pair_id']:
            raise ValueError('Raw sample pair differs from manifest')
        for key, expected in (('source', row['source_rgb']['sha256']), ('target', target_sha)):
            if pair.get(key + '_rgb_sha256') != expected:
                raise ValueError('Raw pair RGB declaration differs: ' + key)
        if row['source_rgb']['sha256'] != bank.sources[0].rgb_sha256:
            raise ValueError('Manifest primary Source differs from source0')
        if row.get('target_rgb') is not None and row['target_rgb']['sha256'] != target_sha:
            raise ValueError('Manifest target RGB fingerprint differs from geometry')
        source = CanvasSpec.from_record(sample['source_canvas'])
        target = CanvasSpec.from_record(sample['target_canvas'])
        if not torch.equal(sample['valid_image_mask'], target.valid_mask()):
            raise ValueError('Raw valid image mask must match the exact target Canvas')
        sample['source_person_mask'] = bank.sources[0].person_mask[0]
        sample['source_head_mask'] = bank.sources[0].head_mask[0]
        sample['local_references_v4'] = bank.merged_references()
        if bank.sources[0].detail is not None:
            sample['detail_references_v5'] = bank.sources[0].detail
        read_condition(row['qwen_condition'], sample)
        sample['src_image'] = _rgb(row['source_rgb'], source)
        if self.load_target:
            if pair.get('cross_person', False) or (metadata.get('counterfactual_geometry')
                    and not metadata.get('paired_rgb_compatible', False)):
                raise ValueError('V6 training requires paired compatible target RGB')
            verified(row['supervision'])
            supervision = read_sidecar(row['supervision'], NativeTrainingSupervisionV5, SUPERVISION_SCHEMA)
            if supervision.weights.shape != local.region_masks.shape or not supervision.paired.all():
                raise ValueError('Training supervision must match the exact paired local Canvas')
            for key in ('target_canvas_id', 'driver_rgb_sha256', 'driver_record_sha256', 'target_person_ids'):
                if supervision.metadata[0].get(key) != metadata.get(key):
                    raise ValueError('Training supervision provenance differs: ' + key)
            sample['training_supervision_v5'] = supervision
            sample['tgt_image'] = _rgb(row['target_rgb'], target)
        return sample


def collate_native_full_v6(samples):
    """Keep a typed B1 bank while the shared collator stacks raw geometry."""
    has_bank = ['source_bank_v5' in sample for sample in samples]
    if any(has_bank):
        if len(samples) != 1 or not all(has_bank):
            raise ValueError('Independent Source bank batches require homogeneous B1')
        bank = samples[0]['source_bank_v5']
        if not isinstance(bank, NativeSourceBankV5) or len(bank.target_person_ids) != 1:
            raise ValueError('Independent Source bank must be a typed B1 bank')
        output = collate_native_full_v4([{key: value for key, value in samples[0].items() if key != 'source_bank_v5'}])
        output['source_bank_v5'] = bank
        return output
    return collate_native_full_v4(samples)

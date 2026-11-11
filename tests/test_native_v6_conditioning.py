"""Frozen multi-reference cache must agree with the locked V6 condition."""
from copy import deepcopy

import pytest
import torch


def pair_and_spec():
    from src.data.canvas_contract import CanvasSpec
    canvas = CanvasSpec((32, 64), (32, 64), (32, 64))
    refs = torch.randn(1, 16, 64)
    pair = dict(qwen_condition_profile='single_2ref', reference_latents=refs, source_latent=refs[:, :8],
        source_token_hw=(2, 4), target_token_hw=(2, 4), reference_token_hws=[(2, 4), (2, 4)],
        reference_roles=['source', 'target_pose'], reference_sha256s=['a'*64, 'b'*64], prompt_sha256='c'*64,
        prompt_embeds=torch.zeros(1, 7, 32), prompt_mask=torch.ones(1, 7, dtype=torch.bool),
        img_mask=torch.tensor([[False, True, True, False, True, True, False, True, True]]))
    spec = dict(profile='single_2ref', target_canvas_id=canvas.fingerprint(), prompt={'sha256': 'c'*64},
        images=[dict(role=role, file={'sha256': sha}, canvas=canvas.metadata(), canvas_id=canvas.fingerprint())
                for role, sha in zip(pair['reference_roles'], pair['reference_sha256s'])])
    return pair, spec


def test_frozen_binding_preserves_tensors_and_pins_canvas():
    from src.pose_control.qwen21_full.native_v6.conditioning import bind_pair_provenance
    pair, spec = pair_and_spec()
    bound = bind_pair_provenance(pair, spec, condition_sha256='d'*64)
    assert bound['reference_latents'] is pair['reference_latents']
    assert bound['condition_sha256'] == 'd'*64
    assert bound['reference_canvas_ids'] == (spec['target_canvas_id'],)*2
    assert 'target_canvas_id' not in pair


@pytest.mark.parametrize('field,value', [
    ('reference_sha256s', ['a'*64, 'e'*64]),
    ('reference_canvas_ids', ['source', 'another-canvas']),
    ('condition_sha256', 'f'*64),
    ('reference_token_hws', [(2, 4), (4, 2)]),
])
def test_same_size_reference_substitution_is_rejected(field, value):
    from src.pose_control.qwen21_full.native_v6.conditioning import bind_pair_provenance
    pair, spec = pair_and_spec()
    pair[field] = value
    with pytest.raises(ValueError):
        bind_pair_provenance(pair, spec, condition_sha256='d'*64)


def test_missing_reference_hashes_cannot_be_filled_from_spec():
    from src.pose_control.qwen21_full.native_v6.conditioning import bind_pair_provenance
    pair, spec = pair_and_spec()
    pair.pop('reference_sha256s')
    with pytest.raises(ValueError):
        bind_pair_provenance(pair, spec)

"""Native reference ordering and source-only appearance, without loading Qwen."""
import pytest
import torch

from src.pose_control.qwen21_full.native_v5.conditioning import (
    primary_source, reference_inputs, validate_pair,
)


def pair(profile='single_2ref'):
    grids = [(2, 4), (4, 4)] if profile == 'single_2ref' else [(2, 4)] + [(4, 4)] * 3
    roles = ['source', 'target_pose'] if len(grids) == 2 else ['source', 'combined_pose', 'person_0_pose', 'person_1_pose']
    refs = [torch.full((1, h*w, 64), float(i+1)) for i, (h, w) in enumerate(grids)]
    n = sum(t.shape[1] for t in refs)
    return dict(qwen_condition_profile=profile, reference_latents=torch.cat(refs, 1),
        reference_token_hws=grids, reference_roles=roles, source_latent=refs[0],
        source_token_hw=grids[0], target_token_hw=(4, 4),
        prompt_embeds=torch.zeros(1, n//4+2, 16), prompt_mask=torch.ones(1, n//4+2, dtype=torch.bool),
        img_mask=torch.tensor([[False, False] + [True]*(n//4+4)]))


@pytest.mark.parametrize('profile', ['single_2ref', 'dual_4ref'])
def test_all_reference_prefixes_and_source_only_slice(profile):
    p = pair(profile); noisy = torch.zeros(1, 16, 64)
    values = reference_inputs(p, noisy, torch.tensor([.5]))
    assert values['img_shapes'][0] == [(1, *g) for g in p['reference_token_hws']] + [(1, 4, 4)]
    assert torch.equal(values['hidden_states'][:, :-16], p['reference_latents'])
    assert torch.equal(primary_source(p), p['source_latent'])
    p['reference_latents'][:, 8:] = 123
    assert (primary_source(p) == 1).all()


def test_reference_roles_cannot_silently_reorder_people():
    p = pair('dual_4ref'); p['reference_roles'][-2:] = ['person_1_pose', 'person_0_pose']
    with pytest.raises(ValueError, match='role'):
        validate_pair(p)


def test_primary_source_cannot_contain_skeleton_latents():
    p = pair(); p['source_latent'] = p['reference_latents']
    with pytest.raises(ValueError, match='Source'):
        validate_pair(p)


def test_native_vlm_slot_count_is_checked():
    p = pair(); p['img_mask'] = p['img_mask'][:, :-1]
    with pytest.raises(ValueError, match='slot'):
        validate_pair(p)


def test_legacy_transformer_entry_dispatches_explicit_multiref():
    from src.pose_control.qwen21_full.qwen_runtime import transformer_inputs
    p = pair('dual_4ref')
    actual = transformer_inputs(p, torch.zeros(1, 16, 64), torch.tensor([.5]))
    assert len(actual['img_shapes'][0]) == 5


def test_dual_manifest_explicit_native_canvases_do_not_use_square_reference():
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.dual_data_runtime import pair_canvases
    s = native_pad((1280, 940)); t = native_pad((1080, 604))
    a, b = pair_canvases({'source_canvas': s.metadata(), 'target_canvas': t.metadata()},
        {'original_hw': [1280, 940]}, {'original_hw': [1080, 604]}, 512*512, 512)
    assert a.canvas_hw == (1280, 960) and b.canvas_hw == (1088, 608)
    with pytest.raises(ValueError):
        pair_canvases({'source_canvas': s.metadata()}, {'original_hw': [1280, 940]},
            {'original_hw': [1080, 604]}, 512*512, 512)


def test_prepared_v5_conversion_selection_and_stack_include_v4_core():
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    from src.pose_control.qwen21_full.native_local_condition import PreparedNativeFullControlV4
    from src.pose_control.qwen21_full.native_v5.contracts import PreparedNativeFullControlV5
    def control(v):
        g=torch.full((1,4,512),v)
        f=NativeGeometryFeatures(g,g.clone(),(2,2),torch.tensor([[True,False]]))
        return PreparedNativeFullControlV5(PreparedNativeFullControlV4(f,None,({'id':v},)),
            g.clone(),torch.ones(1,4,dtype=torch.bool),None,({'id':v},)).validate()
    stacked=PreparedNativeFullControlV5.stack([control(1.),control(2.)])
    selected=stacked.index_select(torch.tensor([1,0],dtype=torch.long)).to(dtype=torch.bfloat16)
    assert selected.body.geometry.dtype==torch.float32 and selected.relation.dtype==torch.float32
    assert (selected.body.geometry[0]==2).all() and (selected.relation[0]==2).all()
    assert selected.metadata[0]['id']==2 and selected.core.metadata[0]['id']==2

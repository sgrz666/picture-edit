"""Audit-specific contracts: prompt roles, cross-person sources and resume safety."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.eval import eval_qwen21_native_value as audit
from src.data.letterbox import Letterbox


def test_explicit_reference_roles_do_not_send_B_RGB_to_adapters(tmp_path):
    names = ('A', 'B', 'A_face', 'B_skeleton', 'B_normal', 'B_depth')
    images = {name: Image.new('RGB', (32, 32), (index * 20, 0, 0)) for index, name in enumerate(names)}
    paths = {name: str(tmp_path / f'{name}.png') for name in names}
    for method in ('simple_body', 'simple_body1024', 'full_body', 'full_body_face'):
        values, records = audit.references_for_method(method, images, paths)
        assert values == [images['A']]
        assert [record['role'] for record in records] == ['A']
    values, records = audit.references_for_method('native_ab_crop', images, paths)
    assert values == [images['A'], images['B'], images['A_face']]
    assert [record['index'] for record in records] == [1, 2, 3]
    _, records = audit.references_for_method('native_geometry', images, paths)
    assert [record['role'] for record in records] == ['A', 'B_normal', 'B_depth']


def test_locked_prompt_variant_and_method_alias_reject_short_missing_prompts(tmp_path):
    path = tmp_path / 'prompts.json'
    prompt = 'Image 1 is A; preserve A and use only the pose from B. ' * 4
    path.write_text(json.dumps({'structured': {'case': {'native_ab': prompt, 'simple_body': prompt}}}))
    values = audit.read_prompts(path, [{'pair_id': 'case'}], ['native_ab', 'simple_body1024'], 'structured')
    assert values['case']['simple_body1024'] == prompt.strip()
    with pytest.raises(ValueError, match='missing/too short'):
        audit.read_prompts(path, [{'pair_id': 'case'}], ['full_body'], 'structured')
    with pytest.raises(KeyError):
        audit.read_prompts(path, [{'pair_id': 'unknown'}], ['native_ab'], 'structured')


def test_face_identity_and_texture_from_A_geometry_from_B_independent_letterboxes(monkeypatch, tmp_path):
    source_path, target_path = tmp_path / 'A.pt', tmp_path / 'B.pt'
    def cached(stem, identity, box, geometry_valid):
        return {'stem_to_idx': {stem: 0}, 'valid': torch.tensor([True]),
                'geometry_valid': torch.tensor([geometry_valid]), 'metadata': {'geometry_valid': [geometry_valid]},
                'arcface': torch.full((1, 512), identity),
                'dino_patches': torch.full((1, 256, 1536), identity),
                'face_landmarks': torch.tensor([[[.5, .25, 1.]] * 72]),
                'jaw_pose': torch.full((1, 3), identity), 'expression': torch.full((1, 10), identity),
                'face_boxes': torch.tensor([box])}
    a_cache = cached('A_frame', 2., [.3, .1, .7, .3], False)
    b_cache = cached('B_frame', 9., [.4, .2, .6, .4], True)
    from scripts import precompute_iper_face_features_v65 as precompute
    monkeypatch.setattr(precompute, 'load_v65_face_cache', lambda path: {str(source_path): a_cache, str(target_path): b_cache}[str(path)])
    row = {'source_appearance': 'A_actor', 'target_appearance': 'B_actor',
           'source_stem': 'A_frame', 'target_stem': 'B_frame',
           'source_face_cache': str(source_path), 'target_face_cache': str(target_path)}
    a_transform, b_transform = Letterbox.from_size((100, 200), 512), Letterbox.from_size((200, 100), 512)
    store = audit.AuditFaceStore([row])
    condition, features = store.for_pairs([row], device='cpu', dtype=torch.float32,
                                         alignments=[{'source': a_transform.metadata(), 'target': b_transform.metadata()}])
    assert condition.face_valid.tolist() == [[True, False]]
    assert torch.equal(features.arcface[0, 0, 0], torch.full((512,), 2.))
    assert torch.equal(features.dino_patches[0, 0, 0], torch.full((256, 1536), 2.))
    assert torch.equal(condition.jaw_pose[0, 0], torch.full((3,), 9.))
    assert torch.equal(condition.expression[0, 0], torch.full((10,), 9.))
    torch.testing.assert_close(condition.source_boxes[0, 0], a_transform.points(a_cache['face_boxes'][0].reshape(2, 2)).reshape(4))
    torch.testing.assert_close(condition.target_boxes[0, 0], b_transform.points(b_cache['face_boxes'][0].reshape(2, 2)).reshape(4))
    torch.testing.assert_close(condition.landmarks[0, 0], b_transform.points(b_cache['face_landmarks'][0]))
    assert not features.arcface[:, 1].count_nonzero()
    b_cache['geometry_valid'][0] = False
    b_cache['metadata']['geometry_valid'] = [False]
    condition, features = store.for_pairs([row], device='cpu', dtype=torch.float32)
    assert not condition.face_valid.any()
    assert not features.arcface.count_nonzero()


def test_resume_index_rejects_changed_generation_and_keeps_resolution_steps_separate(tmp_path):
    path = tmp_path / 'index.jsonl'
    row = dict(pair_id='case', method='native_ab', resolution=512, steps=30,
               prompt_variant=None, image='unchanged.png')
    audit.append_record(path, row)
    audit.append_record(path, row)
    assert len(path.read_text().splitlines()) == 1
    with pytest.raises(ValueError, match='Conflicting generation'):
        audit.append_record(path, {**row, 'image': 'changed.png'})
    audit.append_record(path, {**row, 'steps': 40})
    audit.append_record(path, {**row, 'resolution': 1024})
    assert len(path.read_text().splitlines()) == 3


def test_crop_uses_A_original_detection_box_and_records_provenance(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 400), 'red').save(source)
    row = dict(pair_id='case', source_rgb=str(source), source_face_bbox_pixels=[50., 30., 100., 100.])
    image, record = audit.source_face_crop(row, tmp_path / 'output')
    assert image.size == (512, 512)
    assert record['source'] == str(source)
    assert record['expansion'] == 1.8
    assert record['crop_bbox_pixels'] == [12, 2, 138, 128]
    row['source_face_usable'] = False
    image, record = audit.source_face_crop(row, tmp_path / 'output')
    assert image is None and record['available'] is False


def test_sampling_witness_checks_actual_prefix_and_rejects_nonfinite_velocity():
    from test_qwen21_full_bridge import FakeQwen, fake_inputs, make_model, make_prepared
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
    backbone, inputs, adapter = FakeQwen().eval(), fake_inputs(), make_model()
    context = Qwen21FullControlContext(backbone, adapter, make_prepared(adapter))
    baseline = backbone(**inputs)[0]
    with audit.SamplingWitness(backbone, context, steps=1, grid=4) as witness:
        output = backbone(**inputs)[0]
    assert torch.equal(output, baseline)
    proof = witness.record()
    assert proof['control_prefix_bitwise_checks'] == 32
    assert proof['native_prefix_image_token_counts'] == [16]
    assert proof['target_token_count'] == 16
    assert not getattr(backbone, '_pic_edit_qwen21_control_active', False)
    assert all(not block._forward_hooks for block in backbone.transformer_blocks)
    broken = dict(inputs, hidden_states=torch.full_like(inputs['hidden_states'], float('nan')))
    with pytest.raises(FloatingPointError, match='Nonfinite Qwen'):
        with audit.SamplingWitness(backbone, None, steps=1, grid=4):
            backbone(**broken)
    assert not backbone._forward_hooks
    assert not backbone._forward_pre_hooks


def test_direct_prediction_invalid_face_changes_only_face_payload():
    from dataclasses import replace
    from test_qwen21_full_face_diagnostics import learned_face_runtime
    from scripts.audit.check_qwen21_native_value_prediction import invalidate_face_only, compare_body_conditions, parity
    model, runtime = learned_face_runtime()
    prepared = runtime['prepared']
    invalid = invalidate_face_only(prepared)
    assert invalid.geometry_condition is prepared.geometry_condition
    assert invalid.control_state is prepared.control_state
    assert invalid.adapter_tokens is prepared.adapter_tokens
    assert invalid.face.face_valid.count_nonzero() == 0
    assert compare_body_conditions(prepared, invalid)['exactly_equal']
    changed = replace(invalid, geometry_condition=invalid.geometry_condition + 1)
    with pytest.raises(AssertionError, match='changed Body/shared'):
        compare_body_conditions(prepared, changed)
    value = torch.tensor([1.], dtype=torch.float32)
    assert parity(value, value, 1e-3, require_exact=True)['passed']
    assert not parity(value, value + 1e-5, 1e-3, require_exact=True)['passed']
    assert parity(value, value + 1e-5, 1e-3)['passed']

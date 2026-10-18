"""Protect the RGB audit from accidentally reusing the skeleton input contract."""
import copy
from PIL import Image
import pytest

from scripts.eval.eval_qwen21_bfs_rgb import (
    PROTOCOL, EXPECTED_SHA, prepare_rgb, reference_signature, validate_native,
    validate_saved, versions, sha256, digest_text,
)


@pytest.fixture
def case(tmp_path):
    a, b = tmp_path / 'original_A.png', tmp_path / 'original_B.png'
    Image.new('RGB', (60, 100), (220, 20, 30)).save(a)
    Image.new('RGB', (100, 100), (10, 30, 210)).save(b)
    pair = {'pair_id': 'case', 'source_rgb': str(a), 'target_rgb': str(b),
            'target_dwpose': '/does/not/exist', 'source_face_crop': '/does/not/exist'}
    images, refs, _ = prepare_rgb(pair, tmp_path / 'prepared')
    result = tmp_path / 'generated.png'
    Image.new('RGB', (1024, 1024), 'white').save(result)
    row = {**PROTOCOL, 'variant': 'bfs_body', 'method': 'bfs_body',
           'prompt': 'same detailed A+B prompt', 'prompt_sha256': digest_text('same detailed A+B prompt'),
           'references': refs, 'image': str(result), 'image_sha256': sha256(result),
           'runtime_versions': versions(), 'lora_loaded': True, 'lora_strength': 1.,
           'lora_sha256': EXPECTED_SHA['bfs_body'], 'lock_sha256': 'locked'}
    return pair, images, refs, row


def test_only_two_authorized_rgb_inputs_and_letterbox(case):
    _, images, refs, _ = case
    assert reference_signature(refs)
    assert len(images) == 2
    assert images[0].size == images[1].size == (1024, 1024)
    assert images[0].getpixel((0, 0)) == (128, 128, 128)
    assert images[0].getpixel((512, 512)) == (220, 20, 30)
    assert images[1].getpixel((512, 512)) == (10, 30, 210)


@pytest.mark.parametrize('mutation', ['skeleton_role', 'reversed', 'extra_face', 'changed_content'])
def test_reference_contract_rejects_wrong_roles_order_count_and_content(case, mutation):
    _, _, refs, _ = case
    refs = copy.deepcopy(refs)
    if mutation == 'skeleton_role':
        refs[1]['role'] = 'B_skeleton'
    elif mutation == 'reversed':
        refs.reverse()
    elif mutation == 'extra_face':
        refs.append({**refs[0], 'index': 3, 'role': 'A_face'})
    else:
        Image.new('RGB', (1024, 1024), 'black').save(refs[1]['path'])
    with pytest.raises(ValueError):
        reference_signature(refs)


def test_native_reuse_requires_b_rgb_flag_and_prompt(case):
    _, _, refs, row = case
    native = {**row, 'lora_loaded': False}
    validate_native(native, row['prompt'], refs, versions())
    native['target_rgb_in_native_references'] = False
    with pytest.raises(ValueError):
        validate_native(native, row['prompt'], refs, versions())
    native['target_rgb_in_native_references'] = True
    with pytest.raises(ValueError):
        validate_native(native, 'other prompt', refs, versions())


@pytest.mark.parametrize('field,value', [('target_rgb_in_native_references', False),
    ('lora_strength', .5), ('lora_sha256', 'wrong'), ('use_kv_cache', True), ('control_forwards', 1)])
def test_resume_rejects_changed_rgb_or_sampling_protocol(case, field, value):
    _, _, refs, row = case
    validate_saved(row, row['prompt'], refs, 'bfs_body', 'locked')
    row[field] = value
    with pytest.raises(ValueError):
        validate_saved(row, row['prompt'], refs, 'bfs_body', 'locked')

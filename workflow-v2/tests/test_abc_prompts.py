"""The ABC PE and Qwen inputs share one five-image role protocol."""
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def module():
    assert (ROOT / 'abc_prompt_worker.py').is_file(), 'five-role prompt worker is missing'
    return importlib.import_module('abc_prompt_worker')


def test_user_prompt_is_kept_and_roles_are_explicit():
    api = module()
    instruction = api.pe_instruction('Keep their red scarves.', {'A': 'right', 'B': 'left'})
    assert 'Keep their red scarves.' in instruction
    for number in range(1, 6):
        assert f'<image{number}>' in instruction
    assert 'A=right' in instruction
    assert 'B=left' in instruction
    assert 'exactly two' in instruction


def test_compiler_preserves_five_roles_and_rejects_unknown_images():
    api = module()
    text = api.compile_prompt('Let <image1> and <image2> shake hands following <image5>.')
    for number in range(1, 6):
        assert f'<image{number}>' in text
    assert 'scene' in text
    assert 'identity' in text
    with pytest.raises(ValueError):
        api.compile_prompt('Copy <image6>.')


def test_unmeasured_quality_is_reported_as_unavailable():
    api = importlib.import_module('abc_evaluate')
    report = api.unavailable_report('No detector weights')
    assert report['person_count']['status'] == 'unavailable'
    assert report['pose']['status'] == 'unavailable'
    assert report['appearance']['status'] == 'unavailable'
    assert report['background']['status'] == 'unavailable'
    assert report['passed'] is None

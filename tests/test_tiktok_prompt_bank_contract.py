"""Contract tests for the TikTok per-pair PE prompt rewrite.

The rewrite lives in experiments/ (which pytest.ini excludes from collection), so the
rules that make its output usable for training are pinned here: the two compiled forms,
the phrase cleanups, and the shape of the assembled bank.
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    'experiments/tiktok_pe_prompt_rewrite_20261014/scripts/pe_rewrite_tiktok.py',
    'experiments/*/tiktok_pe_prompt_rewrite_20261014/scripts/pe_rewrite_tiktok.py',
]


def _module():
    pytest.importorskip('transformers', reason='the rewrite runner imports the PE stack')
    for pattern in PATTERNS:
        for path in glob.glob(str(ROOT / pattern)):
            spec = importlib.util.spec_from_file_location('tiktok_pe_rewrite', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip('rewrite script not found in either repo layout')


PE = _module()

SKELETON_WORD = 'the target joint arrangement described in this instruction'


def test_training_form_never_keeps_image_numbers():
    compiled = PE.compile_words('Re-pose <image1> to match <image3> exactly.')
    assert '<image' not in compiled and not re.search(r'\bimage[_\s]*\d', compiled, re.I)
    assert 'the source image' in compiled and SKELETON_WORD in compiled


def test_training_form_rejects_target_photograph_reference():
    with pytest.raises(ValueError, match='target photograph'):
        PE.compile_words('Copy the arm position from <image2>.')


def test_guide_lead_in_is_removed_before_substitution():
    compiled = PE.compile_words('Re-pose to match the joint arrangement defined by <image3>.')
    assert 'joint arrangement defined by ' + SKELETON_WORD not in compiled
    assert compiled.count('joint arrangement') == 1


def test_dangling_guide_noun_after_reference_is_removed():
    compiled = PE.compile_words('Use the joint arrangement of the <image3> skeleton here.')
    assert SKELETON_WORD + ' skeleton' not in compiled
    assert 'the the ' not in compiled


def test_target_photograph_wording_becomes_described_pose():
    compiled = PE.compile_words('Transfer only the pose from the target photograph.')
    assert 'target photograph' not in compiled
    assert 'the described target pose' in compiled


def test_inference_form_maps_skeleton_to_second_reference():
    compiled = PE.compile_inference('Match <image3> while keeping <image1> identity.')
    assert '<image2>' in compiled and '<image1>' in compiled and '<image3>' not in compiled
    with pytest.raises(ValueError):
        PE.compile_inference('Copy <image2> exactly.')
    with pytest.raises(ValueError):
        PE.compile_inference('See <image4>.')


def test_official_answer_contract_is_enforced():
    good = '{"rewritten_prompt": "do it", "wh_ratio": "", "ratio_follow": "<image1>"}'
    assert PE.parse_pe_answer(good)['rewritten_prompt'] == 'do it'
    with pytest.raises(ValueError):
        PE.parse_pe_answer('{"rewritten_prompt": "x", "wh_ratio": "", "ratio_follow": "<image3>"}')
    with pytest.raises(ValueError):
        PE.parse_pe_answer('no json here')


def test_leg_chain_flags_a_knee_that_is_not_on_the_leg():
    scores = [0.9] * 18
    # hip above, ankle below, but the stored "knee" sits far to the side: not a leg.
    broken = [[0.5, 0.9], [0.6, 0.9], [0.4, 0.2], [0.95, 0.55], [0.45, 0.5], [0.5, 0.2], [0.55, 0.25]]
    assert PE.leg_chain_coherent(broken, scores, 0, 3, 5) is False
    coherent = [[0.5, 0.9], [0.6, 0.9], [0.4, 0.2], [0.48, 0.55], [0.45, 0.5], [0.45, 0.2], [0.55, 0.25]]
    assert PE.leg_chain_coherent(coherent, scores, 0, 3, 5) is True
    low = list(scores)
    low[3] = 0.1
    assert PE.leg_chain_coherent(coherent, low, 0, 3, 5) is None


def _bank_lines():
    found = []
    for pattern in PATTERNS:
        for script in glob.glob(str(ROOT / pattern)):
            found += glob.glob(str(Path(script).parents[1] / 'prompts' / '*.jsonl'))
    for path in found:
        lines = [line for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        if lines:
            yield path, lines


def _bank_script_and_files():
    for pattern in PATTERNS:
        for script in glob.glob(str(ROOT / pattern)):
            experiment = Path(script).parents[1]
            banks = [path for path in glob.glob(str(experiment / 'prompts' / '*.jsonl'))
                if Path(path).read_text(encoding='utf-8').strip()]
            if banks:
                return experiment, banks
    return None, []


def _loader():
    experiment, _ = _bank_script_and_files()
    if experiment is None:
        pytest.skip('no non-empty prompt bank yet')
    spec = importlib.util.spec_from_file_location('tiktok_prompt_bank',
        experiment / 'scripts/prompt_bank.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('form', ['training', 'inference'])
def test_loader_serves_both_forms_by_pair_id(form):
    module = _loader()
    _, banks = _bank_script_and_files()
    for path in banks:
        bank = module.TiktokPromptBank(path, form=form)
        assert bank.records
        pair_id = next(iter(bank.records))
        assert bank.prompt_for(pair_id) == bank.records[pair_id]
        with pytest.raises(KeyError):
            bank.prompt_for('pair_id_that_does_not_exist')


def test_loader_also_joins_on_the_fields_a_dataset_batch_actually_carries():
    """TikTokV3PoseDataset yields appearance/source_stem/target_stem, never pair_id."""
    module = _loader()
    _, banks = _bank_script_and_files()
    source = Path(banks[0])
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    bank = module.TiktokPromptBank(source)
    for row in rows[:5]:
        sample = {'appearance': row['sequence_id'], 'source_stem': row['source_stem'],
                  'target_stem': row['target_stem']}
        assert bank.prompt_for_sample(sample) == bank.prompt_for(row['pair_id'])
    with pytest.raises(KeyError):
        bank.prompt_for_sample({'appearance': '99999', 'source_stem': '0000', 'target_stem': '0001'})


def test_loader_rejects_a_tampered_prompt(tmp_path):
    module = _loader()
    _, banks = _bank_script_and_files()
    source = Path(banks[0])
    lines = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    lines[0]['compiled_prompt'] = lines[0]['compiled_prompt'] + ' tampered'
    forged = tmp_path / 'forged.jsonl'
    forged.write_text(''.join(json.dumps(row) + '\n' for row in lines), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.TiktokPromptBank(forged, form='training')


def test_loader_coverage_counts_add_up_against_the_manifest():
    module = _loader()
    experiment, banks = _bank_script_and_files()
    manifest_root = ROOT / 'datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1'
    if not manifest_root.is_file() and not (manifest_root / 'test_pairs.jsonl').is_file():
        pytest.skip('manifest not reachable from this checkout')
    for path in banks:
        split = Path(path).stem
        wanted = module.manifest_pair_ids(split, manifest_root)
        bank = module.TiktokPromptBank(path)
        have, total = bank.coverage(wanted)
        assert total == len(wanted) == len(set(wanted))
        assert have + len(bank.missing(wanted)) == total
        assert have == len([pair for pair in wanted if pair in bank.records])


@pytest.mark.parametrize('path,lines', list(_bank_lines()) or [('', [])],
    ids=lambda value: Path(value).name if isinstance(value, str) else '')
def test_assembled_bank_lines_are_self_consistent(path, lines):
    seen = set()
    for line in lines:
        row = json.loads(line)
        assert row['status'] == 'complete'
        assert row['pair_id'] not in seen
        seen.add(row['pair_id'])
        for key in ('compiled_prompt', 'compiled_prompt_inference'):
            assert row[key].strip(), (row['pair_id'], key)
        assert hashlib.sha256(row['compiled_prompt'].encode()).hexdigest() == row['compiled_prompt_sha256']
        assert '<image' not in row['compiled_prompt']
        assert not re.search(r'\bimage[_\s]*\d', row['compiled_prompt'], re.I)
        assert SKELETON_WORD in row['compiled_prompt']
        assert '<image3>' not in row['compiled_prompt_inference']
        assert row['dwpose_quality']['dwpose_json_found'] is True
        assert row['lower_body_extrapolated_joints'] is None or \
            0 <= row['lower_body_extrapolated_joints'] <= 6


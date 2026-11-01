"""Grounded Hi4D identity binding is enforced by the production consumer."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

CONSUMER = Path(os.environ.get('HI4D_V5_CONSUMER_ROOT',
    '/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/hi4d-v5-grounded-binding-pilot_20261030_r3'))
pytestmark = pytest.mark.skipif(not CONSUMER.is_dir(), reason='Requires real immutable r3 Hi4D CPU assets')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(path, value, jsonl=False):
    path.write_text(''.join(json.dumps(r) + '\n' for r in value) if jsonl else json.dumps(value), encoding='utf8')
    return ref(path)


@pytest.fixture(scope='module')
def assets():
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    config = read(CONSUMER / 'configs/consumer_prepare.json')
    diagnostics = rows(config['condition_manifest']['path'])
    typed = rows(config['typed_consumer']['manifest']['path'])
    dataset = NativeFullDatasetV5(config['typed_consumer']['manifest']['path'], load_target=True, data_profile='hybrid_hi4d')
    return diagnostics, typed, [dataset[i] for i in range(len(dataset))]


@pytest.mark.parametrize('kind', ['diagnostic', 'typed'])
def test_real_grounded_conditions_load_with_authoritative_bank(assets, kind):
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    diagnostics, typed, samples = assets
    selected = diagnostics if kind == 'diagnostic' else typed
    for i, row in enumerate(selected):
        read_condition(row['qwen_condition'], None if kind == 'diagnostic' else samples[i])


def mutate(value, mutation):
    if mutation in ('swapped_guides', 'swapped_guides_and_map'):
        value['images'][2]['file'], value['images'][3]['file'] = value['images'][3]['file'], value['images'][2]['file']
        if mutation.endswith('_and_map'):
            guides = value['guide_identity_files']
            guides['A'], guides['B'] = guides['B'], guides['A']
    elif mutation == 'missing_map': value.pop('guide_identity_files')
    elif mutation == 'missing_order': value.pop('person_order')
    elif mutation == 'missing_schema': value['prompt'].pop('schema')
    elif mutation == 'downgraded_schema': value['prompt']['schema'] = 'hi4d_pei2i_q4_plan_a_v4'
    elif mutation == 'changed_prompt':
        value['prompt']['text'] += ' Silently changed grounded text.'
        value['prompt']['sha256'] = hashlib.sha256(value['prompt']['text'].encode()).hexdigest()
    elif mutation == 'forged_order':
        order = value['person_order']
        order['slot0_identity'], order['slot1_identity'] = order['slot1_identity'], order['slot0_identity']
        order['evidence']['left_actor'] = order['slot0_identity']
    else: raise AssertionError(mutation)


MUTATIONS = ('swapped_guides', 'swapped_guides_and_map', 'missing_map', 'missing_order',
             'missing_schema', 'downgraded_schema', 'changed_prompt', 'forged_order')


@pytest.mark.parametrize('kind', ['diagnostic', 'typed'])
@pytest.mark.parametrize('mutation', MUTATIONS)
def test_formal_reader_rejects_rehashed_binding_tamper(assets, tmp_path, kind, mutation):
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    diagnostics, typed, samples = assets
    row = (diagnostics if kind == 'diagnostic' else typed)[0]
    value = read(row['qwen_condition']['path'])
    mutate(value, mutation)
    bound = write(tmp_path / (kind + mutation + '.json'), value)
    with pytest.raises(ValueError):
        read_condition(bound, None if kind == 'diagnostic' else samples[0])


@pytest.mark.parametrize('mutation', MUTATIONS)
def test_real_trainer_dataplan_rejects_rehashed_binding_tamper(assets, tmp_path, mutation):
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, SourceSpec, stage_preset
    from src.pose_control.qwen21_full.native_v5.training.data import DataPlan
    _, typed, _ = assets
    train, observation = copy.deepcopy(typed[:2])
    condition = read(train['qwen_condition']['path'])
    mutate(condition, mutation)
    train['qwen_condition'] = write(tmp_path / 'bad_condition.json', condition)
    tr = write(tmp_path / 'train.jsonl', [train], jsonl=True)
    ob = write(tmp_path / 'observation.jsonl', [observation], jsonl=True)
    tg = write(tmp_path / 'train_groups.json', {train['pair_id']: 'pair12/highfive12/cam4'})
    og = write(tmp_path / 'observation_groups.json', {observation['pair_id']: 'pair12/highfive12/cam4'})
    config = TrainingConfig(sources=(SourceSpec('train', manifest=tr, groups=tg, data_profile='hybrid_hi4d'),
        SourceSpec('observation', role='observation', manifest=ob, groups=og, data_profile='hybrid_hi4d')),
        stage=stage_preset('S0'), experiment_mode='fixed_pair_overfit', qwen_condition_profile='dual_4ref')
    with pytest.raises(ValueError):
        DataPlan(config)


@pytest.mark.parametrize('schema', [None, 'hi4d_pei2i_q4_plan_a_v4'])
def test_historical_bank_compatibility_retains_exact_file_sha(assets, tmp_path, schema):
    from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
    diagnostics, _, _ = assets
    value = read(diagnostics[0]['qwen_condition']['path'])
    row = {'pair_id': value['pair_id'], 'schema': 'hi4d_pei2i_q4_plan_a_v4',
           'compiled_prompt_inference': value['prompt']['text']}
    value['prompt']['bank'] = write(tmp_path / 'legacy_bank.jsonl', [row], jsonl=True)
    if schema is None: value['prompt'].pop('schema')
    else: value['prompt']['schema'] = schema
    value.pop('person_order')
    value.pop('guide_identity_files')
    read_condition(write(tmp_path / 'legacy.json', value))
    value['images'][2]['file']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA'):
        read_condition(write(tmp_path / 'legacy_bad_sha.json', value))


def test_fixedpair_hi4d_actions_require_explicit_root():
    path = Path(__file__).resolve().parents[1] / 'scripts/train/run_qwen21_v5_fixedpair.py'
    spec = importlib.util.spec_from_file_location('fixedpair_root_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='explicit|hi4d-root'):
        module.resolve_hi4d_root(None)

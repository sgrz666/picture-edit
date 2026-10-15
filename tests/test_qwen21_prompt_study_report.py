from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from scripts.reports.build_qwen21_pei2i_study_report import text_block, build_study_report
from src.pose_control.qwen21_full.prompt_study.protocol import ARMS, PROTOCOLS, SCHEMA, INIT_SHA256, sha256_file
from src.pose_control.qwen21_full.prompt_study.results import verify_report_relative_links
from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock as REAL_LOAD_STUDY_LOCK


def test_raw_prompt_is_html_escaped():
    assert '<script>' not in text_block('<script>alert(1)</script>')
    assert '&lt;script&gt;' in text_block('<script>alert(1)</script>')
    assert '&quot;test&quot;' in text_block('pair "test" & <other>')


def test_scoring_error_is_retained_but_never_counted_complete(proven_run):
    from src.pose_control.qwen21_full.prompt_study.results import validate_evaluation_artifacts
    run, evaluation = proven_run
    directory = evaluation('qwen_p0', 0, 'val', 42)
    path = directory / 'scores.json'
    scores = json.loads(path.read_text())
    scores['per_sample'][0]['score_failure'] = 'RuntimeError: detector session failed'
    path.write_text(json.dumps(scores))
    with pytest.raises(ValueError, match='Offline scoring failed'):
        validate_evaluation_artifacts(directory, run, require_scores=True)
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 1
    assert summary['scoring_errors_count'] == 1
    assert summary['is_completed'] is False
    assert 'detector session failed' in report.read_text(encoding='utf-8')
    assert (directory / 'out0.png').is_file()


def test_valid_detector_miss_is_not_a_scoring_program_error(proven_run):
    from src.pose_control.qwen21_full.prompt_study.results import validate_evaluation_artifacts
    run, evaluation = proven_run
    directory = evaluation('qwen_p0', 0, 'val', 42)
    path = directory / 'scores.json'
    scores = json.loads(path.read_text())
    scores['per_sample'][0].update(identity_A=None, pose_pck_002=None,
        pose_failure='no_detected_pose', bg_l1_A=None, bg_l1_reason='missing_source_mask')
    path.write_text(json.dumps(scores))
    validate_evaluation_artifacts(directory, run, require_scores=True)


@pytest.fixture
def proven_run(tmp_path, monkeypatch):
    import scripts.precompute_qwen21_pei2i_prompts as pe_module
    from scripts.precompute_qwen21_pei2i_prompts import PEModelRunner, process_split_prompts
    # The report fixture exercises real PE/PNG evidence while avoiding the fixed
    # production train512 byte hash, which cannot be synthesized from two rows.
    monkeypatch.setattr(pe_module, 'load_study_lock', lambda *a, **k: {'artifacts': {}})
    run = tmp_path / 'run'
    data = run / 'data'
    data.mkdir(parents=True)
    pairs = []
    for i in range(2):
        image = data / f'pair{i}.png'
        Image.new('RGB', (64, 80), 'blue').save(image)
        pairs.append(dict(pair_id=f'pair_{i}', source_rgb=str(image), target_rgb=str(image), target_depth_vis=str(image), target_normal=str(image), filtered_dwpose=str(image), source_video_group='same_video', source_appearance=f'appearance{i}'))
    (data / 'lock.json').write_text('{}', encoding='utf-8')
    for split in ('train', 'val'):
        (data / ('train512_pairs.jsonl' if split == 'train' else 'val24_pairs.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in pairs), encoding='utf-8')
    model_dir = tmp_path / 'pe_model'
    model_dir.mkdir()
    (model_dir / 'system_prompt.txt').write_text('Official test system.')
    (model_dir / 'config.json').write_text('{"_commit_hash":"test"}')
    (model_dir / 'model.safetensors').write_bytes(b'weights')
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return 'prompt'
        def __call__(self, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2]]))
        def decode(self, ids, **kwargs):
            return '<think><script>alert(1)</script></think>{"rewritten_prompt":"Raise both arms. Preserve Image 1 identity."}'
    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])
    pe = PEModelRunner(model_dir, device='cpu', processor=Processor(), model=Model())
    for split in ('train', 'val'):
        process_split_prompts(run, split, list(PROTOCOLS), runner=pe)
    import src.pose_control.qwen21_full.prompt_study.protocol as protocol
    monkeypatch.setattr(protocol, 'load_study_lock', lambda *a, **k: {'scoring_signature_sha256': 'e' * 64, 'artifacts': {'model': {'path': str(tmp_path / 'qwen_model')}, 'head_lora': {'sha256': 'f' * 64}}})
    def evaluation(arm, step, split, seed, *, failure=False):
        edir = run / f'evaluation/{arm}/step_{step:06d}/{split}/seed_{seed}'
        edir.mkdir(parents=True)
        proto = arm.split('_', 1)[1]
        manifest = run / f'prompts/{proto}/{split}.jsonl'
        prompt_records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
        pair_file = data / ('train512_pairs.jsonl' if split == 'train' else 'val24_pairs.jsonl')
        indices = [0] if split == 'train' else [0, 1]
        config = dict(study_schema=SCHEMA, arm=arm, protocol=proto, study_step=step, study_split=split, seed=seed, indices=indices, lock_sha=sha256_file(data / 'lock.json'), pairs_sha256=sha256_file(pair_file), prompt_manifest_sha256=sha256_file(manifest))
        config.update(model=str(tmp_path / 'qwen_model'), lora_sha256='f' * 64 if arm.startswith('head_') else None, resolution='native', align_mode='native', steps=30, generation_rgb_inputs=['Source A'], with_hand=False, checkpoint_sha256='c' * 64, model_config_sha256='b' * 64)
        (edir / 'config.json').write_text(json.dumps(config), encoding='utf-8')
        signature = dict(arm=arm, model=config['model'], lora_sha256=config['lora_sha256'], study_lock_sha256=config['lock_sha'], max_updates=1024, prompt_manifest_sha256=sha256_file(run / f'prompts/{proto}/train.jsonl'), instruction_sha256=sha256_file(run / 'prompts/instruction_v1.json'))
        (edir / 'checkpoint_metadata.json').write_text(json.dumps({'step': step, 'joint_signature': signature, 'study_schema': SCHEMA, 'joint_initialization_sha256': INIT_SHA256, 'arm': arm, 'protocol': proto, 'max_updates': 1024}), encoding='utf-8')
        (edir / 'lora_loading.json').write_text(json.dumps({'file_sha256': config['lora_sha256'], 'checkpoint_sha256': config['checkpoint_sha256']}), encoding='utf-8')
        records, scores = [], []
        for i in indices:
            image = edir / f'out{i}.png'
            Image.new('RGB', (64, 80), 'red').save(image)
            for side in ('source', 'target'):
                Image.new('RGB', (64, 80), 'blue').save(edir / f'{side}{i}.png')
            record = dict(pair_index=i, pair_id=pairs[i]['pair_id'], arm=arm, protocol=proto, study_step=step, study_split=split, seed=seed, status='failed' if failure and i == 1 else 'complete', failure='CUDA failure' if failure and i == 1 else None, source=f'source{i}.png', target=f'target{i}.png', image=image.name, image_size=[64, 80], image_sha256=sha256_file(image), prompt=prompt_records[i]['compiled_prompt'], prompt_sha256=prompt_records[i]['compiled_prompt_sha256'], initialization='fresh_gaussian', metadata=pairs[i], target_face_box=[.2, .2, .6, .6], target_left_hand_box=[0., 0., .2, .2], target_right_hand_box=[.8, .8, 1., 1.])
            records.append(record)
            record['artifact_sha256'] = {key: sha256_file(edir / record[key]) for key in ('source', 'target', 'image')}
            scores.append(dict(pair_index=i, status=record['status'], identity_A=.8 if record['status'] == 'complete' else None, pose_pck_002=.9 if record['status'] == 'complete' else None, bg_l1_A=.04 if record['status'] == 'complete' else None))
            scores[-1].update({key: record.get(key) for key in ('pair_id', 'arm', 'protocol', 'study_step', 'study_split', 'seed', 'prompt_sha256', 'image_sha256', 'image_size')})
        (edir / 'per_sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        (edir / 'scores.json').write_text(json.dumps({'schema': 'qwen21_joint_pilot_scores_v2', 'scoring_signature_sha256': 'e' * 64, 'metric_resolution': 'per_sample_native', 'per_sample': scores, 'config_sha256': sha256_file(edir / 'config.json'), 'per_sample_sha256': sha256_file(edir / 'per_sample.jsonl')}), encoding='utf-8')
        return edir
    return run, evaluation


def test_build_study_report_with_production_provenance(proven_run):
    run, evaluation = proven_run
    for arm in ARMS:
        for step in (0, 1024):
            for seed in (42, 43):
                evaluation(arm, step, 'val', seed, failure=arm == 'head_pbg' and step == 1024)
    report = build_study_report(run)
    content = report.read_text(encoding='utf-8')
    for title in ('TikTok PE-I2I', '实验配置与架构隔离', '配对对比分析', '八组样本并排对比', '异常与失败归档'):
        assert title in content
    assert '<script>' not in content
    assert '&lt;script&gt;' in content
    assert '目标 B 面部 ROI' in content and '左手 ROI' in content and '右手 ROI' in content
    assert '实际 PE 输入' in content and 'Raise both arms' in content
    assert '身份、动作与背景曲线' in content
    assert 'charts/val_identity_A.png' in content
    assert '实际底座加载记录' in content
    assert 'PB 粗粒度外观' not in content and 'lr=5e-6' not in content
    assert 'PE 看 A＋B，转写目标姿态' in content
    assert '2.5e-5 / 1.25e-5 / 2.5e-6' in content
    conditions = json.loads((report.parent / 'conditions.json').read_text(encoding='utf-8'))
    assert conditions['additional_updates_per_arm'] == 1024
    assert conditions['qwen_reference_rgb'] == ['A']
    assert conditions['test_backpropagation'] is False
    assert verify_report_relative_links(report, run) == []
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['pe_success'] == 12  # three PE protocols, two pairs, two splits
    assert summary['p0_records'] == 4
    assert summary['completed_eval_jobs'] == 62
    assert summary['failed_cases_count'] == 2
    assert not summary['is_completed']
    assert summary['provenance_errors_count'] == 0
    comparison = next(r for r in summary['paired_results'] if (r['left'], r['right'], r['step'], r['split'], r['metric']) == ('head_p0', 'head_pbg', 1024, 'val', 'identity_A'))
    assert comparison['expected_n'] == 2 and comparison['valid_pair_n'] == 1
    assert set(comparison['group_means']) == {'same_video'}


def test_intermediate_step_and_missing_seed_are_visible(proven_run):
    run, evaluation = proven_run
    evaluation('qwen_p0', 128, 'train', 42)
    evaluation('qwen_p0', 0, 'val', 43)
    report = build_study_report(run)
    content = report.read_text(encoding='utf-8')
    assert '步数 Step 128' in content
    assert 'seed 43' in content
    for control in ('filter-split', 'filter-step', 'filter-seed', 'filter-pair', 'filter-view'):
        assert f'id="{control}"' in content
    # Seed 43 must never appear as a seed 42 generated cell.
    assert 'seed_43/out0.png' not in content or 'seed 43' in content
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['pe_success'] == 12


def test_prompt_evidence_is_inspectable_before_any_evaluation(proven_run):
    run, _ = proven_run
    report = build_study_report(run)
    content = report.read_text(encoding='utf-8')
    assert '实际 PE 输入' in content
    assert 'PE 原始请求与回答' in content
    assert 'pair_0' in content and 'pair_1' in content


def test_changed_png_is_excluded_from_progress(proven_run):
    run, evaluation = proven_run
    edir = evaluation('qwen_p0', 0, 'val', 42)
    Image.new('RGB', (64, 80), 'yellow').save(edir / 'out0.png')
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0
    assert summary['provenance_errors_count'] > 0


def test_stale_score_snapshot_is_excluded(proven_run):
    run, evaluation = proven_run
    edir = evaluation('qwen_p0', 0, 'val', 42)
    scores_path = edir / 'scores.json'
    scores = json.loads(scores_path.read_text(encoding='utf-8'))
    scores['config_sha256'] = 'f' * 64
    scores_path.write_text(json.dumps(scores), encoding='utf-8')
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0


def test_wrong_common_initialization_is_excluded(proven_run):
    run, evaluation = proven_run
    edir = evaluation('qwen_p0', 0, 'val', 42)
    path = edir / 'checkpoint_metadata.json'
    record = json.loads(path.read_text())
    record['joint_initialization_sha256'] = 'f' * 64
    path.write_text(json.dumps(record))
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0


def test_changed_reference_is_excluded_even_for_retained_failed_attempt(proven_run):
    run, evaluation = proven_run
    edir = evaluation('qwen_p0', 0, 'val', 42, failure=True)
    Image.new('RGB', (64, 80), 'yellow').save(edir / 'source1.png')
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0
    assert summary['provenance_errors_count'] > 0


def test_duplicate_job_across_directories_is_not_counted(proven_run):
    import shutil
    run, evaluation = proven_run
    source = evaluation('qwen_p0', 0, 'val', 42)
    shutil.copytree(source, run / 'evaluation/duplicate')
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0
    assert summary['provenance_errors_count'] > 0


def test_invalid_lock_cannot_support_verified_progress(proven_run, monkeypatch):
    import src.pose_control.qwen21_full.prompt_study.protocol as protocol
    run, evaluation = proven_run
    evaluation('qwen_p0', 0, 'val', 42)
    monkeypatch.setattr(protocol, 'load_study_lock', REAL_LOAD_STUDY_LOCK)
    report = build_study_report(run)
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0


def test_preflight_status_uses_actual_bound_evidence(tmp_path):
    from scripts.reports import build_qwen21_pei2i_study_report as report_module
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/lock.json').write_text('{}')
    digest = sha256_file(tmp_path / 'data/lock.json')
    directory = tmp_path / 'preflight'
    directory.mkdir()
    signature = {'study_lock_sha256': digest}
    (directory / 'selfcheck.json').write_text(json.dumps({'schema': 'qwen21_study_selfcheck_v1', 'passed': True, 'checks': {'actual': True, 'eight_arms_completed': False}, 'study_lock_sha256': digest}))
    for arm in ARMS:
        (directory / arm).mkdir()
        (directory / arm / 'gradient_only.json').write_text(json.dumps({'passed': True, 'frozen_transformer_unchanged': True, 'frozen_adapter_unchanged': True, 'joint_signature': dict(signature, arm=arm)}))
    (directory / 'resume_check').mkdir()
    result_path = directory / 'resume_check/result.json'
    result_path.write_text(json.dumps({'mandatory_exact_gate': {'passed': True}, 'private_optimizer_updates': 10, 'numerical_review_required': False, 'study_lock_sha256': digest, 'joint_signature': dict(signature, arm='qwen_pg')}))
    status = report_module.read_preflight_evidence(tmp_path)
    assert status['passed'] is True and len(status['gradients']) == 8
    result = json.loads(result_path.read_text())
    result['numerical_review_required'] = True
    result_path.write_text(json.dumps(result))
    assert report_module.read_preflight_evidence(tmp_path)['passed'] is False


def test_loss_curves_use_only_validated_committed_rows(tmp_path, monkeypatch):
    from scripts.reports import build_qwen21_pei2i_study_report as report_module
    from scripts.run import run_qwen21_pei2i_study as runner
    directory = tmp_path / 'training/qwen_p0'
    directory.mkdir(parents=True)
    (directory / 'checkpoint_last.pt').write_bytes(b'committed')
    rows = [{'step': i, 'loss': 1 / i, 'identity': .1, 'perceptual': .2} for i in range(1, 4)]
    (directory / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(runner, 'checkpoint_step', lambda *a: 2)
    step, committed = report_module.committed_training_metrics(tmp_path, 'qwen_p0')
    assert step == 2 and committed == rows[:2]
    plots = report_module.build_training_loss_curves({'qwen_p0': committed}, tmp_path / 'report')
    assert {path.name for _, path in plots} == {'training_loss.png', 'training_identity.png', 'training_perceptual.png'}
    assert all(path.is_file() for _, path in plots)
    def invalid(*a):
        raise RuntimeError('trajectory corrupt')
    monkeypatch.setattr(runner, 'checkpoint_step', invalid)
    with pytest.raises(RuntimeError, match='trajectory'):
        report_module.committed_training_metrics(tmp_path, 'qwen_p0')


def test_historical_images_are_hash_bound_and_independent(tmp_path):
    (tmp_path / 'data').mkdir()
    original = tmp_path / 'old.png'
    Image.new('RGB', (64, 80), 'blue').save(original)
    entry = {'id': 'J_common', 'label': '历史 J256', 'prompt': 'Old prompt.', 'budget': {'initial_history_steps': 256, 'additional_updates': 0}, 'loading': {'lora': None}, 'metadata': {'config': {'seed': 42}}, 'images': [{'pair_id': 'pair_a', 'path': 'old.png', 'sha256': sha256_file(original)}]}
    (tmp_path / 'data/historical_references.json').write_text(json.dumps({'schema': 'qwen21_study_historical_references_v1', 'records': [entry]}), encoding='utf-8')
    report = build_study_report(tmp_path)
    content = report.read_text(encoding='utf-8')
    assert 'Old prompt.' in content and '历史 J256' in content
    snapshot = json.loads((report.parent / 'history/references.json').read_text(encoding='utf-8'))
    copied = report.parent / snapshot['records'][0]['images'][0]['path']
    assert copied.read_bytes() == original.read_bytes()
    assert verify_report_relative_links(report, tmp_path) == []
    summary = json.loads((report.parent / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_eval_jobs'] == 0 and summary['pe_success'] == 0
    Image.new('RGB', (64, 80), 'red').save(original)
    report = build_study_report(tmp_path)
    assert '历史图片校验失败' in report.read_text(encoding='utf-8')


def test_detection_audit_separates_no_face_multi_face_and_pose_failure():
    from scripts.reports import build_qwen21_pei2i_study_report as report_module
    entry = {'arm': 'qwen_p0', 'step': 0, 'split': 'val', 'seed': 42, 'status': 'complete', 'score': {'face_count': 0, 'source_face_count': 2, 'identity_A': None, 'pose_failure': 'No person'}}
    audit = report_module.summarize_detection_counts({'case': entry})
    assert len(audit) == 1
    assert audit[0]['output_no_face'] == 1
    assert audit[0]['source_multiple_faces'] == 1
    assert audit[0]['identity_invalid'] == 1
    assert audit[0]['pose_failures'] == {'No person': 1}

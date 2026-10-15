"""Read-only evidence collector. Run through ssh sg with the qwen21 Python.

The script only reads source files and recorded experimental outputs. It does
not load diffusion models, execute training, or modify the remote project.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

CODE = Path('/home/shangguanrz/project/pic-edit')
RUNS = Path('/home/shangguanrz/nihaochen/qwen21_full')


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


files = {}


def record(path):
    files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return read_json(path)


audit = RUNS / 'native_value_audit_20261008'
metrics_path = audit / 'evaluation/metric_summary.json'
all_metrics = record(metrics_path)
metrics = {}
for group, methods in all_metrics['groups'].items():
    if group not in ('iper_512_steps30', 'tiktok_512_steps30'):
        continue
    metrics[group] = {}
    for method in ('native_skeleton', 'simple_body', 'full_body', 'full_body_face'):
        m = methods[method]
        metrics[group][method] = {
            k: m.get(k) for k in (
                'n_images', 'pose_pck_002', 'pose_torso_aligned_pck_002',
                'pose_joint_angle_mae_degrees', 'identity_a_cosine_common',
                'identity_a_cosine_reliable', 'background_mae_source', 'seconds',
            )
        }

preservation = {}
for resolution, filename in (
    ('512', 'acceptance/direct_prediction/results.json'),
    ('1024', 'acceptance/direct_prediction_1024/results.json'),
):
    value = record(audit / filename)
    preservation[resolution] = {
        'passed': value['passed'],
        'cases': [{k: c[k] for k in (
            'pair_id', 'resolution', 'sigma', 'comparisons',
            'all_models_frozen', 'parameter_gradients_absent', 'passed',
        )} for c in value['cases']],
    }

generation_rows = read_rows(audit / 'generations.jsonl')
files[str(audit / 'generations.jsonl')] = hashlib.sha256(
    (audit / 'generations.jsonl').read_bytes()).hexdigest()
protocol = {}
for method in ('native_skeleton', 'simple_body', 'full_body', 'full_body_face'):
    row = next(r for r in generation_rows if r['method'] == method
               and r['pair_id'] == 'tiktok_03' and r['resolution'] == 512)
    protocol[method] = {k: row.get(k) for k in (
        'references', 'prompt', 'prompt_sha256', 'steps', 'seed', 'resolution',
        'use_kv_cache', 'checkpoint', 'checkpoint_sha256',
        'checkpoint_training_resolution', 'last_control_diagnostics',
    )}

pilot = RUNS / 'bfs_body_face_joint_pilot_20261008'
training = {}
for arm in ('J', 'Qwen', 'BFSBody', 'BFSHead', 'BFH'):
    p = pilot / 'training' / arm
    manifest = record(p / 'run_manifest.json')
    rows = read_rows(p / 'metrics.jsonl')
    files[str(p / 'metrics.jsonl')] = hashlib.sha256(
        (p / 'metrics.jsonl').read_bytes()).hexdigest()
    training[arm] = {
        'args': manifest['args'],
        'joint_signature': manifest.get('joint_signature'),
        'initialization_provenance': manifest.get('initialization_provenance'),
        'optimizer_group_names': list(manifest.get('optimizer_parameters', {})),
        'updates': len(rows),
        'unique_training_pairs': len({r['pair_id'] for r in rows}),
        'face_active_updates': sum(bool(r.get('face_active')) for r in rows),
        'hand_active_updates': sum(bool(r.get('hand_active')) for r in rows),
        'aux_requested_updates': sum(bool(r.get('auxiliary_requested')) for r in rows),
        'aux_used_updates': sum(bool(r.get('auxiliary_used')) for r in rows),
        'aux_eligible_updates': sum(r.get('auxiliary_eligible_faces', 0) > 0 for r in rows),
        'mean_first_32_loss': sum(r['loss'] for r in rows[:32]) / min(32, len(rows)),
        'mean_last_32_loss': sum(r['loss'] for r in rows[-32:]) / min(32, len(rows)),
    }

pilot_scores = {}
for name in ('init_initial_tiktok', 'J_common_tiktok', 'Qwen_final_tiktok',
             'BFSBody_final_tiktok', 'BFSHead_final_tiktok', 'BFH_final_tiktok'):
    p = pilot / 'evaluation' / name
    config = record(p / 'config.json')
    scores = record(p / 'scores.json')
    rows = scores['per_sample']
    averages = {}
    for k in ('identity_A', 'pose_pck_002', 'pose_pck_005', 'pose_nme_diagonal'):
        values = [r[k] for r in rows if r.get(k) is not None]
        averages[k] = {'mean': sum(values) / len(values) if values else None,
                       'n': len(values)}
    pilot_scores[name] = {
        'config': config, 'n': len(rows), 'averages': averages,
        'metric_resolution': scores.get('metric_resolution'),
    }

source_hashes = {}
for filename in ('qwen_adapter.py', 'qwen_runtime.py', 'joint_training.py',
                 'execution.py', 'data_runtime.py', 'control_core.py',
                 'frozen_bfs_lora.py'):
    path = CODE / 'src/pose_control/qwen21_full' / filename
    source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

output = {
    'collected_utc': datetime.now(timezone.utc).isoformat(),
    'remote_code_root': str(CODE), 'remote_results_root': str(RUNS),
    'scope': 'Read-only collection of historical outputs; no new images or training.',
    'native_value_metrics': metrics,
    'native_value_observations': record(audit / 'final_observations.json'),
    'historical_single_prediction_preservation': preservation,
    'native_value_protocol_examples': protocol,
    'pilot_training': training, 'pilot_scores': pilot_scores,
    'bfs_skeleton_observations': record(
        RUNS / 'bfs_skeleton_direct_20261010/evaluation/observations.json'),
    'bfs_rgb_observations': record(
        RUNS / 'bfs_rgb_direct_20261010/evaluation/observations.json'),
    'hi4d_native_summary': record(
        RUNS / 'hi4d_native_binding_20261010/evaluation/summary.json'),
    'source_sha256': source_hashes, 'evidence_sha256': files,
}
print(json.dumps(output, ensure_ascii=False, indent=2))

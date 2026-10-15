"""Run the approved phases, including gated final-test prompt/cache preparation."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

run = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python = '/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
state_path = run / 'repair_audit/full_pipeline_state.json'
launch_path = run / 'repair_audit/full_pipeline_v2_launch.json'

# Stage names have distinct logs; phase is the actual production CLI verb.
PHASES = (
    ('rewrite', 'rewrite', ['--splits', 'train', 'val', 'diagnostic']),
    ('cache', 'cache', ['--splits', 'train', 'val', 'diagnostic']),
    ('selfcheck', 'selfcheck', []),
    ('preflight', 'preflight', []),
    ('initial', 'initial', []),
    ('train', 'train', ['--resume']),
    ('rewrite_test', 'rewrite', ['--splits', 'test']),
    ('cache_test', 'cache', ['--splits', 'test']),
    ('finaltest', 'finaltest', []),
    ('report', 'report', []),
    ('export', 'export', []),
)


def save(state):
    temp = state_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(state_path)


state = {
    'schema': 'qwen21_repair_full_pipeline_v2', 'pid': os.getpid(),
    'status': 'validating_signed_qa_review',
    'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'stages': [], 'planned_stages': [stage for stage, _, _ in PHASES],
    'previous_controller_evidence': 'repair_audit/controller_v1_before_test_fix.json',
}
save(state)
launch = json.loads(launch_path.read_text())
launch.update(pid=os.getpid(), status='running', started_at=time.time())
launch_path.write_text(json.dumps(launch, indent=2))
try:
    sys.path.insert(0, str(run / 'code'))
    from scripts.precompute_qwen21_pei2i_prompts import validate_qa8_review
    from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed, sha256_file
    review = run / 'prompts/qa8.json'
    validate_qa8_review(run)
    state.update(status='running', review_sha256=sha256_file(review), lock_sha256=sha256_file(run / 'data/lock.json'))
    save(state)
    runner = run / 'code/scripts/run/run_qwen21_pei2i_study.py'
    for name, phase, extra in PHASES:
        if name in ('rewrite_test', 'cache_test', 'finaltest'):
            require_eight_arms_completed(run)
        stage = {
            'phase': name, 'cli_phase': phase, 'status': 'running', 'started_at': time.time(),
            'argv': [python, str(runner), '--run', str(run), '--phase', phase, *extra],
            'log': f'logs/full_pipeline_v2_{name}.log',
        }
        state['stages'].append(stage)
        state['phase'] = name
        save(state)
        with (run / stage['log']).open('x') as output:
            process = subprocess.Popen(stage['argv'], cwd=run / 'code', stdout=output, stderr=subprocess.STDOUT)
            stage['pid'] = process.pid
            save(state)
            stage['exit_code'] = process.wait()
        stage.update(finished_at=time.time(), status='complete' if stage['exit_code'] == 0 else 'failed')
        save(state)
        if stage['exit_code'] != 0:
            raise RuntimeError(f'Phase failed: {name}, exit {stage["exit_code"]}; retained log: {stage["log"]}')
    summary = json.loads((run / 'report/summary.json').read_text())
    if summary.get('is_completed') is not True:
        raise RuntimeError('Final report does not prove the complete experiment')
    state.update(status='remote_complete_waiting_local_copy', remote_complete=True, finished_at=time.time(),
                 report='report/index.html', export='export/results.tar.gz')
    save(state)
    launch.update(status='complete', exit_code=0, finished_at=time.time())
    launch_path.write_text(json.dumps(launch, indent=2))
except BaseException as exc:
    state.update(status='failed', failure=f'{type(exc).__name__}: {exc}', finished_at=time.time())
    save(state)
    launch.update(status='failed', failure=state['failure'], finished_at=time.time())
    launch_path.write_text(json.dumps(launch, indent=2))
    raise

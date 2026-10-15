"""Replace only this experiment's controller group before any Adapter updates."""
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time

run = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python = '/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
state_path = run / 'repair_audit/full_pipeline_state.json'
state = json.loads(state_path.read_text())
assert state['schema'] == 'qwen21_repair_full_pipeline_v1'
assert state['status'] == 'running' and state['phase'] == 'rewrite', state
assert len(state['stages']) == 1 and state['stages'][0]['phase'] == 'rewrite'
assert not any((run / 'training').glob('*/metrics.jsonl')), 'Adapter updates already exist; do not interrupt this way'
assert not (run / 'repair_audit/full_pipeline_v2_launch.json').exists(), 'Replacement already launched'
pid = state['pid']
command = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert str(run / 'repair_audit/full_pipeline_remote.py') in command
assert pathlib.Path(f'/proc/{pid}/cwd').resolve() == run / 'code'
pgid = os.getpgid(pid)
assert pgid != os.getpgrp(), 'Must not signal the SSH audit process'
leader = pathlib.Path(f'/proc/{pgid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert str(run / 'repair_audit/phase_worker.py') in leader and 'full_pipeline_launch.json' in leader
assert os.getpgid(state['stages'][0]['pid']) == pgid
script = run / 'repair_audit/full_pipeline_v2_remote.py'
assert script.is_file()
compile(script.read_text(), str(script), 'exec')
backup = run / 'repair_audit/controller_v1_before_test_fix.json'
assert not backup.exists()
backup.write_text(json.dumps({
    'state': state, 'process_group': pgid,
    'reason': 'Old orchestration omitted rewrite --splits test and cache --splits test before finaltest; production CLI/geometry/training protocol unchanged.',
    'formal_optimizer_updates': 0, 'stopped_at': time.time(),
    'old_controller_script_sha256': hashlib.sha256((run / 'repair_audit/full_pipeline_remote.py').read_bytes()).hexdigest(),
}, indent=2), encoding='utf-8')
os.killpg(pgid, signal.SIGTERM)
for _ in range(30):
    result = subprocess.run(['ps', '-eo', 'pgid=,stat='], capture_output=True, text=True, check=True)
    live = [line for line in result.stdout.splitlines()
            if len(line.split()) >= 2 and line.split()[0] == str(pgid) and not line.split()[1].startswith('Z')]
    if not live:
        break
    time.sleep(1)
else:
    raise RuntimeError('Old controller group has not stopped; preserve evidence and inspect, do not double-launch')
record = run / 'repair_audit/full_pipeline_v2_launch.json'
record.write_text(json.dumps({
    'argv': [python, str(script)], 'status': 'queued',
    'old_controller_pid': pid, 'old_process_group': pgid,
    'reason': 'Add gated test rewrite/cache phases; preserve completed prompt records and logs',
}, indent=2))
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
with (run / 'logs/full_pipeline_v2.log').open('x') as output:
    process = subprocess.Popen([python, str(script)], cwd=run / 'code', env=env,
                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
(run / 'repair_audit/full_pipeline_v2_pid.json').write_text(json.dumps({'pid': process.pid}))
print(json.dumps({'pid': process.pid, 'old_group_stopped': True,
                  'test_preparation_fixed': True, 'formal_optimizer_updates': 0}, indent=2))

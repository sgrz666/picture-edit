"""One Linux owner, immutable evidence, recoverable committed Native V5 stages.

No torch/model imports: status/stop/verify remain available while CUDA is busy.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from run_plan import DEFAULT_RUN, PHASES, make_plan


class IntegrityError(ValueError):
    pass


class StopRequested(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_ref(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked_ref(ref):
    if (not isinstance(ref, dict) or not ref.get('sha256') or
            not Path(ref.get('path', '')).is_absolute()):
        raise IntegrityError('Every result/input reference must pin an absolute path and SHA256')
    actual = file_ref(ref['path'])
    if actual != ref:
        raise IntegrityError('Artifact SHA changed: '+str(ref['path']))
    return actual


def read_ref(ref):
    checked_ref(ref)
    return json.loads(Path(ref['path']).read_text(encoding='utf8'))


def audit_references(value, seen=None):
    """Explicit verify follows SHA-pinned JSON closures, including generated PNGs."""
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if value.get('path') and value.get('sha256'):
            ref = {k:value[k] for k in ('path','sha256')}
            identity = (ref['path'], ref['sha256'])
            if identity in seen:
                return
            checked_ref(ref)
            seen.add(identity)
            if Path(ref['path']).suffix == '.json':
                audit_references(read_ref(ref), seen)
        for child in value.values():
            audit_references(child, seen)
    elif isinstance(value, list):
        for child in value:
            audit_references(child, seen)


def _write_staged_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write((canonical(value)+'\n').encode())
        f.flush()
        os.fsync(f.fileno())
    if os.name == 'posix':
        fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return file_ref(path)


def write_new_json(path, value):
    """Publish a complete new file atomically; retain abandoned staging evidence."""
    path = Path(path)
    pending = path.parent/'.pending'/(path.stem+'_'+uuid.uuid4().hex+'.json')
    staged = _write_staged_json(pending, value)
    os.link(staged['path'], path)
    if os.name == 'posix':
        fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return file_ref(path)


publish_new_json = write_new_json


CODE_SUFFIXES = {'.py','.sh','.json','.toml','.yaml','.yml'}
NON_CODE_DIRECTORIES = {'.git','__pycache__','.pending','cache','caches','weights','models','datasets',
                        'output','outputs','report','reports','logs'}


def runtime_code_files(roots):
    files = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise IntegrityError('Missing fixed runtime source root: '+str(root))
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in CODE_SUFFIXES and not any(
                    part in NON_CODE_DIRECTORIES for part in path.relative_to(root).parts):
                files.add(path.resolve())
    return sorted(files)


def snapshot_runtime_code(roots, output):
    value = {'schema':'bilibili_runtime_code_manifest_v1','roots':[str(Path(r).resolve()) for r in roots],
             'files':[file_ref(p) for p in runtime_code_files(roots)]}
    if Path(output).exists():
        if digest(json.loads(Path(output).read_text(encoding='utf8'))) != digest(value):
            raise IntegrityError('Fixed runtime code manifest differs')
        return file_ref(output)
    return write_new_json(output,value)


def runtime_manifest_path(root,roots):
    # Different delivered directories retain their own manifests. A running plan
    # still pins exactly one reference and refuses all subsequent code drift.
    return Path(root)/'source'/('runtime_code_manifest_'+digest([str(Path(p).resolve()) for p in roots])[:16]+'.json')


def verify_runtime_code(plan):
    manifest = read_ref(plan['runtime_code_manifest'])
    if manifest.get('schema') != 'bilibili_runtime_code_manifest_v1':
        raise IntegrityError('Invalid fixed runtime code manifest')
    actual = {str(p) for p in runtime_code_files(manifest['roots'])}
    if actual != {r['path'] for r in manifest['files']}:
        raise IntegrityError('Executable/configuration file set changed in the fixed runtime snapshot')
    for ref in manifest['files']:
        checked_ref(ref)
    if plan.get('authorization_ref'):
        checked_ref(plan['authorization_ref'])


class EventJournal:
    """Exclusive numbered records form a durable append-only hash chain.

    A partial journal file is an integrity failure, never an event silently skipped.
    """
    def __init__(self, root):
        self.directory = Path(root)/'journal'

    def read(self):
        rows, previous = [], '0'*64
        for sequence, path in enumerate(sorted(self.directory.glob('*.json')), 1):
            try:
                raw = path.read_bytes()
                if not raw.endswith(b'\n'):
                    raise IntegrityError('Truncated journal record: '+str(path))
                row = json.loads(raw)
                claimed = row.pop('sha256')
                if row['sequence'] != sequence or row['previous'] != previous or digest(row) != claimed:
                    raise IntegrityError('Journal sequence/hash mismatch: '+str(path))
                previous = claimed
                rows.append(row['record'])
            except (KeyError, json.JSONDecodeError) as error:
                raise IntegrityError('Invalid journal record: '+str(path)) from error
        return rows

    def append(self, record):
        rows = self.read()
        previous = '0'*64
        if rows:
            previous = json.loads(sorted(self.directory.glob('*.json'))[-1].read_text(encoding='utf8'))['sha256']
        row = {'sequence': len(rows)+1, 'previous': previous,
               'record': {'time': utc_now(), **record}}
        row['sha256'] = digest(row)
        return publish_new_json(self.directory/f'{len(rows)+1:08d}.json', row)


def replay(rows):
    state = {'accepted': {}, 'attempts': {}, 'failure_counts':{}, 'terminal_tasks':set(), 'status': 'new', 'ignored_stops': set()}
    for row in rows:
        event = row['event']
        if event == 'attempt':
            state['attempts'][row['task']] = row['attempt']
            state.update(status='running', task=row['task'])
        elif event == 'accepted':
            state['accepted'][row['task']] = read_ref(row['result'])
        elif event == 'attempt_failed':
            state['failure_counts'][row['task']] = state['failure_counts'].get(row['task'], 0)+1
            if row.get('failure_class') == 'terminal':
                state['terminal_tasks'].add(row['task'])
        elif event in ('failed', 'completed', 'stopped'):
            state['status'] = event
            state['last_event'] = row
        elif event == 'started':
            state.update(status='running', pid=row['pid'], started=row['time'])
            state['ignored_stops'].update(row.get('ignored_stops', []))
        elif event == 'resource_wait':
            state.update(status='waiting_resources', resources=row['resources'])
    return state


def next_task(state):
    accepted = state['accepted']
    for task in ('prepare_data', 'prepare_cache', 'data_preflight'):
        if task not in accepted:
            return task, None
    for phase in PHASES:
        for action in ('stage_grad', 'stage_train'):
            if phase+'.'+action not in accepted:
                return action, phase
    if 'eval_test' not in accepted:
        return 'eval_test', None
    if 'export_final' not in accepted:
        return 'export_final', None
    return None, None


def latest_committed(root, phase):
    records = []
    for marker in (Path(root)/'stages'/phase/'checkpoints').glob('*.commit.json'):
        try:
            value = json.loads(marker.read_text(encoding='utf8'))
            if value.get('schema') != 'native_v5_training_commit_v1':
                raise IntegrityError('Invalid checkpoint commit schema: '+str(marker))
            ref = checked_ref(value['checkpoint'])
            if Path(ref['path']).with_suffix('.commit.json').resolve() != marker.resolve():
                raise IntegrityError('Commit points outside its checkpoint: '+str(marker))
            if type(value['stage_update']) is not int or type(value['global_update']) is not int:
                raise IntegrityError('Checkpoint update counters must be integers')
            records.append({**value, 'commit': file_ref(marker),'committed_mtime_ns':marker.stat().st_mtime_ns})
        except (KeyError, json.JSONDecodeError) as error:
            raise IntegrityError('Invalid checkpoint commit: '+str(marker)) from error
    return max(records, key=lambda r: (r['global_update'], r['stage_update'],r['committed_mtime_ns'],r['commit']['path'])) if records else None


def completion_errors(plan, accepted):
    errors, total = [], 0
    for task in ('prepare_data', 'prepare_cache', 'data_preflight'):
        if accepted.get(task, {}).get('passed') is not True:
            errors.append('Missing successful '+task)
    for stage in plan['stages']:
        phase, budget = stage['phase'], stage['budget_updates']
        grad, trained = accepted.get(phase+'.stage_grad', {}), accepted.get(phase+'.stage_train', {})
        if grad.get('production_passed') is not True:
            errors.append(phase+' lacks production gradient acceptance')
        total += budget
        if (trained.get('passed') is not True or trained.get('budget_completed') is not True or
                trained.get('stage_update') != budget or trained.get('global_update') != total or not trained.get('checkpoint')):
            errors.append(phase+' did not commit the complete successful update budget')
        validation = trained.get('validation_state', {})
        if not validation.get('passed') or not validation.get('finite') or validation.get('triggered'):
            errors.append(phase+' lacks final passing finite validation')
    test = accepted.get('eval_test', {})
    if test.get('passed') is not True or test.get('technical_only') is not True:
        errors.append('Missing technical-only final held-out test')
    for key in ('loss_report', 'generation_report', 'metrics_report', 'validation_loss_report'):
        if key not in test:
            errors.append('Missing final test '+key)
    exported = accepted.get('export_final', {})
    if (exported.get('passed') is not True or exported.get('successful_updates') != 20000 or
            any(not exported.get(key) for key in ('weights','summary','readme')) or
            exported.get('checkpoint') != accepted.get('S3.stage_train', {}).get('checkpoint')):
        errors.append('Missing final export weights/summary/README bound to the final checkpoint')
    if total != plan['total_updates'] or total != 20000:
        errors.append('Total successful update budget differs from 20000')
    return errors


def classify_failure(message, returncode=1):
    text = message.lower()
    if any(word in text for word in ('out of memory', 'cuda_error_out_of_memory', 'cannot allocate memory')):
        return 'resource'
    if any(word in text for word in ('sha', 'integrity', 'nonfinite', 'non-finite', 'nan', 'frozen',
            'eligib', 'gradient', 'mismatch', 'differs', 'missing', 'not passed', 'budget', 'preflight')):
        return 'terminal'
    if returncode in (-9, -15, 137, 143) or any(word in text for word in ('temporarily unavailable', 'resource busy', 'input/output error')):
        return 'transient'
    return 'terminal'


def may_retry(failure_class, attempt):
    return failure_class in ('resource', 'transient') and attempt <= 2


def request_stop(root):
    return write_new_json(Path(root)/'control'/('stop_'+uuid.uuid4().hex+'.json'),
                          {'action': 'stop', 'time': utc_now(), 'own_child_only': True})


def pending_stops(root, ignored):
    return [ref for p in sorted((Path(root)/'control').glob('stop_*.json'))
            if (ref := file_ref(p))['sha256'] not in ignored]


@contextmanager
def single_owner(root):
    if os.name != 'posix':
        raise RuntimeError('The unattended owner requires Linux fcntl; status/verify remain portable')
    import fcntl
    path = Path(root)/'supervisor.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A supervisor already owns this run') from error
        yield


def resources(gpu):
    probe = 'nvml'
    try:
        result = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                text=True, capture_output=True, check=True)
        gpu_free = float(result.stdout.strip().splitlines()[0])/1024
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        detail = str(getattr(error, 'stderr', '') or error).lower()
        if 'permission denied' in detail:
            raise RuntimeError('Permission denied by GPU resource probe') from error
        # CUDA allocation remains usable on this host although its NVML library
        # and kernel module versions differ. Query CUDA in a short-lived process.
        code = ('import json,sys,torch; '
                'free,total=torch.cuda.mem_get_info(int(sys.argv[1])); '
                'print(json.dumps(dict(free_bytes=free,total_bytes=total)))')
        result = subprocess.run([sys.executable, '-c', code, str(gpu)],
                                text=True, capture_output=True, check=True)
        values = json.loads(result.stdout.strip().splitlines()[-1])
        gpu_free = values['free_bytes']/1024**3
        probe = 'cuda_runtime'
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    host_free = float(memory['MemAvailable'].split()[0])/1024**2
    return {'gpu': str(gpu), 'gpu_free_gib': gpu_free, 'host_available_gib': host_free, 'gpu_probe': probe}


def await_resources(plan, journal, ignored_stops, gpu):
    while True:
        if pending_stops(plan['run_root'], ignored_stops):
            raise StopRequested()
        available = resources(gpu)
        if available['gpu_free_gib'] >= plan['gpu_free_gib'] and available['host_available_gib'] >= plan['host_available_gib']:
            return
        journal.append({'event': 'resource_wait', 'resources': available})
        print(canonical({'status': 'waiting_resources', **available}), flush=True)
        for _ in range(plan['resource_poll_seconds']):
            if pending_stops(plan['run_root'], ignored_stops):
                raise StopRequested()
            time.sleep(1)


def child_environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTHONUNBUFFERED='1')
    return env


def terminate_owned_child(child):
    if child.poll() is not None:
        return
    # Every worker is its own session leader. Never signal a shared GPU owner's
    # process, the supervisor's shell, or an external spatial-cache writer.
    if os.name == 'posix':
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
    else:
        child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait()


def launch_child(command, log_path, env, root, ignored_stops):
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while child.poll() is None:
                if pending_stops(root, ignored_stops):
                    terminate_owned_child(child)
                    raise StopRequested()
                time.sleep(2)
            if pending_stops(root, ignored_stops):
                raise StopRequested()
            return child.returncode
        except BaseException:
            terminate_owned_child(child)
            raise


def validate_result(action, result):
    if result.get('passed') is not True:
        raise IntegrityError('Worker result did not pass: '+action)
    if action == 'stage_grad' and result.get('production_passed') is not True:
        raise IntegrityError('Actual production gradient acceptance did not pass')
    for key in ('prepared', 'config', 'report', 'checkpoint', 'commit', 'loss_report', 'generation_report',
                'metrics_report', 'body_gate', 'validation_report', 'validation_loss_report', 'cache_identity', 'weights', 'summary', 'readme'):
        if result.get(key):
            checked_ref(result[key])


def status(root, *, verify_checkpoints=False):
    root = Path(root)
    state = replay(EventJournal(root).read())
    state['ignored_stops'] = sorted(state['ignored_stops'])
    state['terminal_tasks'] = sorted(state['terminal_tasks'])
    state['run_root'] = str(root)
    state['owner_alive'] = bool(state.get('pid') and Path('/proc',str(state['pid'])).exists()) if os.name=='posix' else None
    if verify_checkpoints:
        for row in EventJournal(root).read():
            if row['event'] == 'plan':
                audit_references(row['plan'])
        audit_references(state['accepted'])
        state['artifact_closure_verified'] = True
    plan_path = root/'run_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf8'))
        state['total_budget'] = plan['total_updates']
        checkpoints = {phase: latest_committed(root, phase) for phase in PHASES} if verify_checkpoints else {}
        # Cheap status reads commit counters; verify additionally hashes every checkpoint.
        if not verify_checkpoints:
            for phase in PHASES:
                values = [json.loads(p.read_text(encoding='utf8')) for p in (root/'stages'/phase/'checkpoints').glob('*.commit.json')]
                checkpoints[phase] = max(values, key=lambda x:x['global_update']) if values else None
        state['committed_updates'] = max((c['global_update'] for c in checkpoints.values() if c), default=0)
        state['stage_commits'] = checkpoints
        telemetry = []
        for path in root.glob('stages/*/segments/*/metrics.jsonl'):
            with path.open('rb') as stream:
                length = stream.seek(0, 2)
                stream.seek(max(0, length-2*1024*1024))
                lines = stream.read().splitlines()
                if length > 2*1024*1024:
                    lines = lines[1:]
                for line in lines:
                    try:
                        record = json.loads(line).get('record', {})
                    except json.JSONDecodeError:
                        continue  # Never claim this uncommitted tail as a checkpoint.
                    if record.get('optimizer_updated'):
                        for key in ('update_seconds', 'elapsed_seconds', 'seconds_per_update'):
                            if record.get(key, 0) > 0:
                                telemetry.append(float(record[key]))
                                break
        state['eta_seconds'] = None
        if telemetry:
            state['seconds_per_successful_update'] = sum(telemetry[-100:])/len(telemetry[-100:])
            state['eta_seconds'] = max(0, plan['total_updates']-state['committed_updates'])*state['seconds_per_successful_update']
        state['eta_note'] = 'measured update throughput; excludes cache/gradient/evaluation overhead' if telemetry else 'not enough measured throughput yet'
    return state


def run(args):
    root = Path(args.run_root).resolve()
    if not str(root).startswith('/data/shangguanrz/'):
        raise ValueError('Production run artifacts must remain under /data/shangguanrz/')
    root.mkdir(parents=True, exist_ok=True)
    with single_owner(root):
        journal = EventJournal(root)
        rows = journal.read()
        plan_path = root/'run_plan.json'
        if args.action == 'run':
            if rows or plan_path.exists():
                raise ValueError('Existing run: use resume to retain its committed state')
            authorization_path = Path(args.authorization_json or root/'authorization.json')
            authorization = json.loads(authorization_path.read_text(encoding='utf8-sig'))
            plan = make_plan(str(root), str(Path(args.repo_root).resolve()), authorization)
            plan['authorization_ref'] = file_ref(authorization_path)
            # The private repo also carries the exact third_party and HIA helpers
            # imported by preparation. Include them without scanning dynamic assets.
            code_roots = [Path(plan['repo_root']),Path(__file__).resolve().parent]
            legacy_root = root/'source'/'legacy'
            if legacy_root.is_dir():
                code_roots.append(legacy_root)
            plan['runtime_code_manifest'] = snapshot_runtime_code(code_roots,runtime_manifest_path(root,code_roots))
            plan_ref = write_new_json(plan_path, plan)
            journal.append({'event': 'plan', 'plan': plan_ref})
        else:
            plan = json.loads(plan_path.read_text(encoding='utf8'))
            recorded = next((r['plan'] for r in rows if r['event'] == 'plan'), None)
            checked_ref(recorded)
            state = replay(rows)
            if state['status'] in ('failed', 'completed'):
                raise ValueError('Terminal '+state['status']+' run cannot resume')
            # A child may finish after its supervisor disappeared. Recover only its
            # explicitly recorded immutable result, never an uncommitted work file.
            previous = next((r for r in reversed(rows) if r['event']=='attempt'), None)
            if previous and state['status'] not in ('stopped',) and previous['task'] not in state['accepted']:
                result_path = previous.get('result_path')
                failed = any(r['event']=='attempt_failed' and r.get('task')==previous['task'] and
                             r.get('attempt')==previous['attempt'] for r in rows)
                if result_path and Path(result_path).exists() and not failed:
                    result = json.loads(Path(result_path).read_text(encoding='utf8'))
                    if result.get('passed') is True:
                        action = previous['task'].split('.')[-1]
                        validate_result(action, result)
                        journal.append({'event':'accepted','task':previous['task'],'result':file_ref(result_path),'recovered_after_owner_restart':True})
                    else:
                        journal.append({'event':'attempt_failed','task':previous['task'],'attempt':previous['attempt'],
                                        'failure_class':result.get('failure_class','terminal'),'error':result.get('error')})
                elif not failed:
                    journal.append({'event':'attempt_failed','task':previous['task'],'attempt':previous['attempt'],
                                    'failure_class':'transient','error':'Supervisor disappeared before child result acceptance; worker lock protects any surviving child'})
        ignored = {r['sha256'] for r in pending_stops(root, set())}
        journal.append({'event': 'started', 'pid': os.getpid(), 'ignored_stops': sorted(ignored), 'action':args.action})
        env = child_environment(args.gpu)
        try:
            while True:
                verify_runtime_code(plan)
                state = replay(journal.read())
                action, phase = next_task(state)
                if action is None:
                    errors = completion_errors(plan, state['accepted'])
                    if errors:
                        raise IntegrityError('; '.join(errors))
                    # Hash every accepted final checkpoint before recording completion.
                    for p in PHASES:
                        committed = latest_committed(root, p)
                        if not committed or committed['checkpoint'] != state['accepted'][p+'.stage_train']['checkpoint']:
                            raise IntegrityError('Final committed checkpoint differs: '+p)
                    audit_references(state['accepted']['eval_test'])
                    audit_references(state['accepted']['export_final'])
                    journal.append({'event':'completed', 'successful_updates':plan['total_updates'], 'technical_only':True})
                    return
                if pending_stops(root, ignored):
                    raise StopRequested()
                task = phase+'.'+action if phase else action
                if task in state['terminal_tasks']:
                    raise IntegrityError('Terminal failure remains recorded for '+task)
                previous_attempt = state['attempts'].get(task, 0)
                if state['failure_counts'].get(task, 0) >= 3:
                    raise IntegrityError('Retry budget exhausted for '+task)
                if action != 'export_final':
                    await_resources(plan, journal, ignored, args.gpu)
                attempt = previous_attempt+1
                identifier = task+'_'+uuid.uuid4().hex
                logs = root/'logs'
                logs.mkdir(exist_ok=True)
                result_path = root/'results'/(identifier+'.json')
                request = {'plan': file_ref(plan_path), 'accepted': state['accepted'], 'action':action,
                           'phase':phase, 'attempt':attempt, 'run_root':str(root), 'repo_root':plan['repo_root'],
                           'ignored_stops':sorted(ignored)}
                request_ref = write_new_json(root/'requests'/(identifier+'.json'), request)
                journal.append({'event':'attempt', 'task':task, 'attempt':attempt, 'request':request_ref,
                                'log':str(logs/(identifier+'.log')), 'result_path':str(result_path),
                                'resources':resources(args.gpu) if action != 'export_final' else {'device':'cpu'}})
                command = [args.python, str(Path(__file__).with_name('stage_worker.py')),
                           '--input', request_ref['path'], '--input-sha256', request_ref['sha256'], '--result', str(result_path)]
                started = time.monotonic()
                returncode = launch_child(command, logs/(identifier+'.log'), env, root, ignored)
                result = json.loads(result_path.read_text(encoding='utf8')) if result_path.exists() else {}
                if returncode == 0:
                    validate_result(action, result)
                    journal.append({'event':'accepted', 'task':task, 'result':file_ref(result_path),
                                    'elapsed_seconds':time.monotonic()-started})
                    print(canonical({'task':task, 'status':'accepted', 'result':str(result_path)}), flush=True)
                    continue
                with (logs/(identifier+'.log')).open('rb') as log:
                    log.seek(max(0, log.seek(0, 2)-16384))
                    tail = log.read().decode('utf8', errors='replace')
                failure = result.get('failure_class') or classify_failure(tail, returncode)
                journal.append({'event':'attempt_failed', 'task':task, 'attempt':attempt, 'returncode':returncode,
                                'failure_class':failure, 'error':result.get('error', tail[-4000:]), 'result_path':str(result_path)})
                failures = state['failure_counts'].get(task, 0)+1
                if not may_retry(failure, failures):
                    raise IntegrityError('Terminal worker failure: '+task+' ('+failure+')')
                # A resource retry always rechecks availability and reuses the same config/last commit.
        except StopRequested:
            journal.append({'event':'stopped', 'reason':'explicit stop request', 'checkpoint_policy':'last committed boundary only'})
        except BaseException as error:
            journal.append({'event':'failed', 'error':str(error), 'error_type':type(error).__name__})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','status','stop','resume','verify'))
    parser.add_argument('--run-root', default=DEFAULT_RUN)
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--authorization-json')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args(argv)
    if args.action in ('run','resume'):
        run(args)
    elif args.action == 'stop':
        print(canonical({'stop_request':request_stop(args.run_root)}))
    else:
        print(canonical(status(args.run_root, verify_checkpoints=args.action=='verify')))


if __name__ == '__main__':
    main()

"""Run one recorded remote job with file-backed output and no SSH pipes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def main():
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text())
    command = config['command']
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError('command must be a nonempty argument list')
    paths = {key: Path(config[key]) for key in ('log', 'process_record', 'result_record')}
    if any(path.exists() for path in paths.values()):
        raise ValueError('job artifacts already exist; inspect the existing job')
    started = time.time()
    with paths['log'].open('x') as log:
        child = subprocess.Popen(command, cwd=config['cwd'],
            env=dict(os.environ, **config.get('environment', {})),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            close_fds=True)
        write_json(paths['process_record'], dict(pid=child.pid, runner_pid=os.getpid(),
            session_id=os.getsid(0), started_unix=started, command=command,
            cwd=config['cwd'], config_sha256=sha256(config_path),
            runner_sha256=sha256(__file__), log=str(paths['log']),
            ssh_pipes_inherited=False))
        exit_code = child.wait()
    write_json(paths['result_record'], dict(exit_code=exit_code,
        elapsed_seconds=time.time() - started, log_sha256=sha256(paths['log']),
        config_sha256=sha256(config_path), runner_sha256=sha256(__file__),
        process_record_sha256=sha256(paths['process_record'])))
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

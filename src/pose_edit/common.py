"""Stage provenance, cache fingerprints, canvas preprocessing and estimator sandboxing.

A stage request has two halves with different lifetimes:

- ``identity``: settings, code hashes, model identity and input content hashes. This is
  what a cached stage is allowed to be reused for, so it carries no absolute paths.
- ``runtime``: absolute paths the worker needs to read and write this run directory.
  Changing them (renaming a run, moving the checkout) must not invalidate content.

Stage records list artifacts relative to the run directory for the same reason.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from PIL import Image, ImageOps

REQUEST_SCHEMA = 'single_person_pose_request_v2'
STAGE_SCHEMA = 'single_person_pose_stage_v2'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_request(stage, identity, run_dir, paths):
    """Split what a cached stage must match from where this run lives."""
    return {'schema': REQUEST_SCHEMA, 'stage': stage,
        'identity': identity,
        'runtime': {'run_dir': str(Path(run_dir).resolve()),
            'paths': {name: str(Path(value).resolve()) for name, value in paths.items()}}}


def request_identity(request):
    return request['identity']


def runtime_paths(request):
    return request['runtime']['paths']


def create_estimator_sandbox(source, destination):
    """Copy root-sensitive entry points; share immutable model/package assets."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Estimator sandbox must be outside the shared estimator checkout')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'main').mkdir()
    for file in (source / 'main').glob('*.py'):
        shutil.copyfile(file, destination / 'main' / file.name)
    (destination / 'scripts').mkdir()
    shutil.copyfile(source / 'scripts/run_smplestx_params_only.py',
        destination / 'scripts/run_smplestx_params_only.py')
    for name in ('models', 'utils', 'human_models', 'datasets', 'configs', 'pretrained_models'):
        (destination / name).symlink_to(source / name, target_is_directory=True)
    (destination / 'demo/input_frames').mkdir(parents=True)
    (destination / 'demo/output_frames').mkdir(parents=True)
    (destination / 'outputs').mkdir()
    return destination


def finish_stage(path, request, outputs, metadata, base_dir):
    """Record artifacts relative to the run directory plus their content hashes."""
    base_dir = Path(base_dir).resolve()
    artifacts = {}
    for name, file in outputs.items():
        resolved = Path(file).resolve()
        artifacts[name] = {'file': resolved.relative_to(base_dir).as_posix(),
            'sha256': sha256_file(resolved)}
    atomic_json(path, {'schema': STAGE_SCHEMA, 'status': 'complete',
        'identity': request_identity(request),
        'request_sha256': fingerprint(request_identity(request)),
        'outputs': artifacts, 'metadata': metadata, 'completed_at_unix': time.time()})


def _record_rows(record, base_dir):
    rows = []
    for name, item in record['outputs'].items():
        path = Path(base_dir).resolve() / item['file']
        rows.append((name, path, item['sha256']))
    return rows


def stage_reasons(path, request, base_dir):
    """Explain every reason a cached stage is not reusable, empty list means reusable."""
    try:
        record = read_json(path)
    except (OSError, ValueError) as error:
        return [f'stage record unreadable: {type(error).__name__}: {error}']
    reasons = []
    if record.get('schema') != STAGE_SCHEMA:
        reasons.append('stage record uses an older schema; rerun this stage once')
    if record.get('status') != 'complete':
        reasons.append(f"stage status is {record.get('status')!r}, not 'complete'")
    stored = record.get('request_sha256')
    current = fingerprint(request_identity(request))
    if stored != current:
        reasons.append(f'request identity changed ({str(stored)[:8]}... != {current[:8]}...)')
        reasons.extend(identity_diff(request_identity(request), record.get('identity')))
    if not record.get('outputs'):
        reasons.append('stage record lists no artifacts')
    for name, file, digest in _record_rows(record, base_dir):
        if not file.is_file():
            reasons.append(f'artifact missing: {name} -> {file}')
        elif sha256_file(file) != digest:
            reasons.append(f'artifact content changed after the stage wrote it: {name} -> {file} '
                f'(recorded {digest[:12]}..., on disk {sha256_file(file)[:12]}...)')
    return reasons


def identity_diff(current, stored):
    """Name the identity fields that differ between this request and the recorded one."""
    if stored is None:
        return ['the request that produced this stage is not available for comparison']
    reasons = []
    for group, value in current.items():
        before = stored.get(group)
        if before == value:
            continue
        if isinstance(value, dict) and isinstance(before, dict):
            for key in sorted(set(value) | set(before)):
                if value.get(key) == before.get(key):
                    continue
                if group == 'code':
                    reasons.append(f'workflow code changed: {key} '
                        f'({str(before.get(key))[:8]}... -> {str(value.get(key))[:8]}...)')
                else:
                    reasons.append(f'{group}.{key} changed: '
                        f'{_brief(before.get(key))} -> {_brief(value.get(key))}')
        else:
            reasons.append(f'{group} changed: {_brief(before)} -> {_brief(value)}')
    return reasons


def _brief(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 60 else text[:57] + '...'


def stage_valid(path, request, base_dir):
    return not stage_reasons(path, request, base_dir)


def stage_outputs(path, base_dir):
    """Absolute artifact paths recorded in a stage, resolved against the run directory."""
    record = read_json(path)
    return {name: str(file) for name, file, _ in _record_rows(record, base_dir)}


def furthest_valid_stage(run_dir, chain):
    """Name of the furthest stage whose record still validates, walking a stage chain.

    Orchestrators use this so a run's status reports how complete the run *is*, rather than
    what the last invocation happened to ask for: rerunning an old run with ``--until
    geometry`` must not silently downgrade a finished generation to ``partial``.

    ``chain`` is ordered ``(stage, record_path, request_or_None)``; a None request means the
    chain could not be rebuilt past that point.
    """
    furthest = None
    for name, record, request in chain:
        if request is None or not Path(record).is_file():
            break
        if stage_reasons(record, request, run_dir):
            break
        furthest = name
    return furthest


def model_signature(path):
    """Record weight stat identity plus hashes of small config/prompt files."""
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(path.rglob('*'))
    rows = []
    for file in files:
        if not file.is_file() or any(part.startswith('.') for part in file.relative_to(path.parent).parts):
            continue
        if file.suffix not in {'.json', '.txt', '.jinja', '.safetensors', '.bin', '.pt',
                '.npz', '.pkl', '.tar', '.py', '.onnx'}:
            continue
        stat = file.stat()
        row = {'name': file.name if path.is_file() else file.relative_to(path).as_posix(),
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if file.suffix in {'.json', '.txt', '.jinja', '.py'} and stat.st_size < 1024 * 1024:
            row['sha256'] = sha256_file(file)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No model/config files under {path}')
    return {'path': str(path), 'files': rows}


def canvas_size(original_size, resolution):
    width, height = original_size
    if min(width, height, resolution) <= 0:
        raise ValueError('Image size and resolution must be positive')
    ratio = width / height
    seen = set()
    for _ in range(32):
        w = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
        h = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        next_ratio = w / h
        next_size = (round(math.sqrt(resolution * resolution * next_ratio) / 32) * 32,
            round(math.sqrt(resolution * resolution / next_ratio) / 32) * 32)
        if next_size == (w, h):
            return w, h
        if (w, h) in seen:
            break
        seen.add((w, h))
        ratio = next_ratio
    raise ValueError('Source aspect ratio has no stable Qwen conditioning canvas')


def letterbox(image, size):
    image = ImageOps.exif_transpose(image).convert('RGB')
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    offset = ((size[0] - resized[0]) // 2, (size[1] - resized[1]) // 2)
    canvas = Image.new('RGB', size, (128, 128, 128))
    canvas.paste(image.resize(resized, Image.Resampling.LANCZOS), offset)
    return canvas, {'original_size': list(image.size), 'canvas_size': list(size),
        'resized_size': list(resized), 'scale_xy': [resized[0] / image.width, resized[1] / image.height],
        'offset_xy': list(offset), 'content_box': [offset[0], offset[1], offset[0] + resized[0], offset[1] + resized[1]]}


@contextmanager
def file_lock(path):
    """Linux process lock, automatically released even if a worker crashes."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another workflow invocation holds {path}') from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f'pid={os.getpid()}\n')
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

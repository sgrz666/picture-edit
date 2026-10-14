from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "qwen21_pei2i_body_face_study_v1"
PROTOCOLS = ("p0", "pg", "pb", "pbg")
P0_PROMPT = "Change the person pose to match the target pose. Preserve identity, clothing, and background."
ARMS = tuple(f"{base}_{p}" for base in ("qwen", "head") for p in PROTOCOLS)
PROMPT_ROLES = {
    "p0": (),
    "pg": ("source_A", "target_depth", "target_normal", "target_dwpose"),
    "pb": ("source_A", "target_B"),
    "pbg": ("source_A", "target_B", "target_depth", "target_normal", "target_dwpose"),
}
EVAL_STEPS = tuple(range(0, 1025, 128))
FULL_VAL_STEPS = (0, 256, 512, 768, 1024)
TRAIN_MONITOR = (0, 64, 128, 192, 256, 320, 384, 448)
VAL_MONITOR = (0, 4, 7, 8, 13, 14, 16, 19, 22)
INIT_SHA256 = "9e0f048c585937ab74be792eb6d9730b0621eccda16e1744a6837b0eb9d24651"
TRAIN_SHA256 = "854133776d2067aa392442553f3003ef9e6d45107ccc9cffe5a4af8600fb929f"
VAL_SHA256 = "cece883b96dbbc9270fd77d4bb1e973470f52f941af85c2a0746947f5d91e10a"
DATA_LOCK_SCHEMA = 'qwen21_pei2i_data_lock_v2'
MODEL_REVISION = '790c92633540aa0cb11d9abf19eb46d861714758'
SPLIT_COUNTS = {'train': 512, 'val': 24, 'test': 24, 'diagnostic': 6}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def validate_arm(arm: str) -> tuple[str, str]:
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")
    base, protocol = arm.split("_", 1)
    return base, protocol


def validate_study_config(config: dict[str, Any]) -> None:
    """Validate study-wide fixed configuration according to the experiment protocol."""
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Invalid schema: {config.get('schema')}")
    if config.get("max_updates") != 1024:
        raise ValueError(f"max_updates must be 1024, got {config.get('max_updates')}")
    if config.get("eval_every") != 128:
        raise ValueError(f"eval_every must be 128, got {config.get('eval_every')}")
    if config.get("resolution") != "native" or config.get("align_mode") != "native":
        raise ValueError("Resolution and align_mode must be 'native'")
    if config.get("with_hand") is True or config.get("hand_active") is True:
        raise ValueError("Hand adapter must remain disabled in this study")
    if config.get("use_depth") is True:
        raise ValueError("Adapter Depth channel must remain disabled in this study")
    arms = config.get("arms")
    if arms is not None and tuple(arms) != ARMS:
        raise ValueError(f"Configured arms must match exact 8 ARMS, got {arms}")


def directory_artifact(path: str | Path, revision: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {}
    for item in sorted(path.rglob('*')):
        rel = item.relative_to(path)
        if any(part.startswith('.') for part in rel.parts) or not item.is_file():
            continue
        if item.suffix not in ('.safetensors', '.json', '.txt', '.jinja'):
            continue
        stat = item.stat()
        files[rel.as_posix()] = dict(sha256=sha256_file(item), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError(f'No actual model weights: {path}')
    digest = text_sha256(json.dumps({k: v['sha256'] for k, v in files.items()}, sort_keys=True))
    return dict(path=str(path), revision=revision, files=files, manifest_sha256=digest)


def verify_artifact(artifact: dict[str, Any], *, full: bool = False) -> None:
    path = Path(artifact['path'])
    if 'files' not in artifact:
        if not path.is_file() or sha256_file(path) != artifact['sha256']:
            raise ValueError(f'Changed or missing artifact: {path}')
        return
    hashes = {k: v['sha256'] for k, v in artifact['files'].items()}
    if text_sha256(json.dumps(hashes, sort_keys=True)) != artifact['manifest_sha256']:
        raise ValueError('Model artifact manifest digest differs')
    for name, expected in artifact['files'].items():
        item = path / name
        if not item.is_file():
            raise ValueError(f'Missing model artifact: {item}')
        stat = item.stat()
        if stat.st_size != expected['size']:
            raise ValueError(f'Changed model artifact size: {item}')
        # Immutable files were fully hashed at lock creation. Rehash changed metadata;
        # callers can request full verification independently before long GPU runs.
        if (full or stat.st_mtime_ns != expected['mtime_ns']) and sha256_file(item) != expected['sha256']:
            raise ValueError(f'Changed model artifact: {item}')


def load_study_lock(path: str | Path, verify_assets: bool = True) -> dict[str, Any]:
    from .sampling import make_order
    path = Path(path)
    lock = json.loads(path.read_text(encoding='utf-8'))
    if lock.get('schema') != DATA_LOCK_SCHEMA:
        raise ValueError('Invalid study lock schema; regenerate the verified data lock')
    validate_study_config(lock.get('config', {}))
    if lock.get('epoch_order') != make_order():
        raise ValueError('Locked epoch order differs from the approved trajectory')
    if lock.get('initial_checkpoint_sha256') != INIT_SHA256:
        raise ValueError('Locked J256 initialization differs')
    if set(lock.get('splits', {})) != set(SPLIT_COUNTS):
        raise ValueError('Locked split set differs')
    split_rows = {}
    for split, count in SPLIT_COUNTS.items():
        record = lock['splits'][split]
        item = path.parent / record['path']
        if not item.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('Split escapes the data directory')
        if record['count'] != count or sha256_file(item) != record['sha256']:
            raise ValueError(f'Changed locked split: {split}')
        rows = [json.loads(line) for line in item.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != count or len({row['pair_id'] for row in rows}) != count:
            raise ValueError(f'Invalid or duplicate locked pairs: {split}')
        split_rows[split] = rows
    if (lock['splits']['train']['sha256'], lock['splits']['val']['sha256']) != (TRAIN_SHA256, VAL_SHA256):
        raise ValueError('Historical train/val bytes differ')
    artifacts = lock.get('artifacts', {})
    if not {'model', 'initial_checkpoint', 'head_lora'} <= artifacts.keys():
        raise ValueError('Missing locked model/initialization/Head artifacts')
    if artifacts['initial_checkpoint']['sha256'] != INIT_SHA256:
        raise ValueError('Initialization artifact differs from common J256')
    if artifacts['model'].get('revision') != MODEL_REVISION:
        raise ValueError('Qwen model revision differs')
    from .source_assets import validate_source_asset_manifest
    validate_source_asset_manifest(path.parent, lock, split_rows, verify_files=verify_assets)
    from .measurement import validate_measurement_artifacts, scoring_signature
    measurement = lock.get('measurement_artifacts', {})
    validate_measurement_artifacts(measurement, verify_files=verify_assets)
    if scoring_signature(measurement) != lock.get('scoring_signature_sha256'):
        raise ValueError('Locked scoring signature differs')
    if verify_assets:
        for artifact in artifacts.values():
            verify_artifact(artifact)
        for record in lock.get('geometry_assets', []):
            if sha256_file(path.parent / record['path']) != record['sha256']:
                raise ValueError(f'Changed geometry asset: {record["path"]}')
    return lock


def validate_study_binding(lock: dict[str, Any], arm: str, lora_sha256: str | None,
                           model_path: str | Path) -> None:
    base, _ = validate_arm(arm)
    if Path(model_path).resolve() != Path(lock['artifacts']['model']['path']).resolve():
        raise ValueError('Model path differs from the locked Qwen artifact')
    if base == 'qwen' and lora_sha256 is not None:
        raise ValueError('Plain Qwen arm must not load a LoRA')
    if base == 'head' and lora_sha256 != lock['artifacts']['head_lora']['sha256']:
        raise ValueError('Head arm must load the exact locked BFS Head from step 0')


def validate_committed_training_state(checkpoint: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    """A completed trajectory must also contain usable saved training state."""
    import math
    import torch
    adapter = checkpoint.get('adapter_state_dict')
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError('Missing committed Adapter state')
    for name, value in adapter.items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Invalid committed Adapter tensor: {name}')
    optimizer = checkpoint.get('optimizer_state_dict')
    if (not isinstance(optimizer, dict) or not optimizer.get('state')
            or not optimizer.get('param_groups')
            or any(not group.get('params') for group in optimizer['param_groups'])):
        raise ValueError('Missing committed optimizer state')
    torch_rng = checkpoint.get('torch_rng_state')
    cuda_rng = checkpoint.get('cuda_rng_state')
    if (not isinstance(torch_rng, torch.Tensor) or torch_rng.numel() == 0
            or not isinstance(cuda_rng, list) or not cuda_rng
            or any(not isinstance(state, torch.Tensor) or state.numel() == 0 for state in cuda_rng)
            or not checkpoint.get('python_rng_state')
            or not isinstance(checkpoint.get('numpy_rng_state'), tuple)
            or len(checkpoint['numpy_rng_state']) != 5):
        raise ValueError('Missing committed RNG state')
    for index, row in enumerate(metrics):
        for key in ('loss', 'grad_norm'):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or (key == 'grad_norm' and value <= 0)):
                raise ValueError(f'Invalid committed {key} at update {index}')


def require_eight_arms_completed(run_dir: str | Path) -> None:
    """Final test is released by recorded training evidence, never summary claims alone."""
    run_dir = Path(run_dir)
    try:
        lock_path = run_dir / 'data/lock.json'
        lock = load_study_lock(lock_path, verify_assets=False)
        lock_sha = sha256_file(lock_path)
        instruction_sha = sha256_file(run_dir / 'prompts/instruction_v1.json')
        face_sha = sha256_file(run_dir / 'data/face_assets.json')
        train_rows = [json.loads(line) for line in (run_dir / 'data/train512_pairs.jsonl').read_text().splitlines() if line.strip()]
        from .prompts import PromptBank
        from .sampling import validate_trajectory
        import torch
        for arm in ARMS:
            root = run_dir / 'training' / arm
            summary = json.loads((root / 'summary.json').read_text())
            if not (summary.get('completed') is True and summary.get('steps') == 1024
                    and summary.get('frozen_transformer_unchanged') is True
                    and summary.get('frozen_adapter_unchanged') is True):
                raise ValueError(f'Incomplete or unfrozen training: {arm}')
            checkpoint = torch.load(root / 'checkpoint_step_001024.pt', map_location='cpu', weights_only=False, mmap=True)
            signature = checkpoint['joint_signature']
            if (checkpoint.get('step'), checkpoint.get('study_schema'), checkpoint.get('arm'), checkpoint.get('protocol')) != (1024, SCHEMA, arm, arm.split('_', 1)[1]):
                raise ValueError(f'Checkpoint arm/step differs: {arm}')
            if checkpoint.get('joint_initialization_sha256') != INIT_SHA256:
                raise ValueError(f'Checkpoint initialization differs: {arm}')
            for key, expected in [('study_lock_sha256', lock_sha), ('instruction_sha256', instruction_sha),
                                  ('face_assets_sha256', face_sha)]:
                if signature.get(key) != expected:
                    raise ValueError(f'Checkpoint {key} differs: {arm}')
            validate_study_binding(lock, arm, signature.get('lora_sha256'), signature['model'])
            protocol = arm.split('_', 1)[1]
            prompt_path = run_dir / 'prompts' / protocol / 'train.jsonl'
            if signature.get('prompt_manifest_sha256') != sha256_file(prompt_path):
                raise ValueError(f'Checkpoint prompts differ: {arm}')
            bank = PromptBank(prompt_path, protocol, [r['pair_id'] for r in train_rows])
            metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line.strip()]
            validate_trajectory(checkpoint, metrics, lock['epoch_order'])
            validate_committed_training_state(checkpoint, metrics)
            for u, row in enumerate(metrics):
                pid = train_rows[lock['epoch_order'][u]]['pair_id']
                if row.get('pair_id') != pid or row.get('prompt_sha256') != bank.records[pid]['compiled_prompt_sha256']:
                    raise ValueError(f'Committed pair/prompt differs: {arm} at {u}')
            del checkpoint
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise RuntimeError(f'Final test blocked: complete matching eight-arm training evidence is required: {exc}') from exc

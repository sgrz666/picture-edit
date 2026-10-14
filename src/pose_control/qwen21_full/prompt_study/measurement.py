"""Lock auxiliary training losses and offline measurement implementation."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .protocol import sha256_file, text_sha256

LOSS_FILES = ('face_eval/identity_encoder.ts', 'face_eval/perceptual_encoder.ts')
METRIC_FILES = ('antelopev2/scrfd_10g_bnkps.onnx', 'antelopev2/glintr100.onnx',
                'DWPose/yolox_l.onnx', 'DWPose/dw-ll_ucoco_384.onnx')
MODEL_FILES = LOSS_FILES + METRIC_FILES
SCORING_CODE = ('scripts/eval/eval_qwen21_joint_pilot.py',
    'src/evaluation/joint_pilot_metrics.py', 'src/evaluation/dwpose_onnx/__init__.py',
    'src/evaluation/dwpose_onnx/onnxdet.py', 'src/evaluation/dwpose_onnx/onnxpose.py',
    'src/pose_control/qwen21_full/prompt_study/measurement.py')
VERSION_PACKAGES = {
    'numpy': ('numpy',), 'opencv': ('opencv-python', 'opencv-python-headless',
        'opencv-contrib-python', 'opencv-contrib-python-headless'),
    'onnxruntime': ('onnxruntime', 'onnxruntime-gpu'), 'insightface': ('insightface',),
    'pillow': ('Pillow',),
}


def measurement_versions() -> dict[str, str]:
    result = {}
    for name, candidates in VERSION_PACKAGES.items():
        installed = {}
        for candidate in candidates:
            try:
                installed[candidate] = importlib.metadata.version(candidate)
            except importlib.metadata.PackageNotFoundError:
                pass
        if not installed:
            raise ValueError(f'Missing measurement dependency: {name}')
        result[name] = json.dumps(installed, sort_keys=True)
    return result


def measurement_artifacts(models_root: Path, *, code_root: Path,
                          versions: dict[str, str] | None = None) -> dict[str, Any]:
    def records(root, names):
        return {name: {'path': str((root / name).resolve()),
                       'sha256': sha256_file(root / name)} for name in names}
    return {'schema': 'qwen21_study_measurement_v1',
            'models': records(Path(models_root), MODEL_FILES),
            'code': records(Path(code_root), SCORING_CODE),
            'versions': versions if versions is not None else measurement_versions()}


def scoring_signature(record: dict[str, Any]) -> str:
    payload = {'schema': record['schema'], 'versions': record['versions'],
               'models': {name: record['models'][name]['sha256'] for name in METRIC_FILES},
               'code': {name: record['code'][name]['sha256'] for name in SCORING_CODE}}
    return text_sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')))


def validate_measurement_artifacts(record: dict[str, Any], *, verify_files: bool) -> None:
    if (record.get('schema') != 'qwen21_study_measurement_v1'
            or set(record.get('models', {})) != set(MODEL_FILES)
            or set(record.get('code', {})) != set(SCORING_CODE)
            or not record.get('versions')):
        raise ValueError('Missing locked loss/scoring measurement artifacts')
    for section, description in (('models', 'measurement model'), ('code', 'scoring code')):
        for name, item in record[section].items():
            if (not isinstance(item.get('path'), str) or not isinstance(item.get('sha256'), str)
                    or len(item['sha256']) != 64):
                raise ValueError(f'Invalid {description} record: {name}')
            if verify_files and sha256_file(item['path']) != item['sha256']:
                raise ValueError(f'Changed {description}: {name}')


def loss_signature(lock: dict[str, Any], models_root: Path) -> dict[str, str]:
    record = lock['measurement_artifacts']
    result = {}
    for name in LOSS_FILES:
        item = record['models'][name]
        actual = Path(models_root) / name
        if actual.resolve() != Path(item['path']).resolve() or sha256_file(actual) != item['sha256']:
            raise ValueError(f'Changed loaded loss model: {name}')
        result[name] = item['sha256']
    return result


def validate_scoring_runtime(lock: dict[str, Any], models_root: Path, code_root: Path) -> str:
    record = lock['measurement_artifacts']
    validate_measurement_artifacts(record, verify_files=True)
    for name in METRIC_FILES:
        if (Path(models_root) / name).resolve() != Path(record['models'][name]['path']).resolve():
            raise ValueError(f'Loaded scoring model path differs: {name}')
    for name in SCORING_CODE:
        if sha256_file(Path(code_root) / name) != record['code'][name]['sha256']:
            raise ValueError(f'Loaded scoring code differs: {name}')
    if measurement_versions() != record['versions']:
        raise ValueError('Scoring dependency versions differ from the lock')
    signature = scoring_signature(record)
    if signature != lock.get('scoring_signature_sha256'):
        raise ValueError('Locked scoring signature differs')
    return signature

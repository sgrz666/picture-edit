from pathlib import Path
import pytest
from src.pose_control.qwen21_full.prompt_study import measurement


@pytest.fixture
def locked_measurement(tmp_path, monkeypatch):
    from src.evaluation import sdpose

    detector = tmp_path / 'yolo11x.pt'
    detector.write_bytes(b'evaluator')
    monkeypatch.setattr(sdpose, 'scoring_model_files', lambda: [detector])
    sdpose.scoring_identity.cache_clear()
    models = tmp_path / 'models'
    for relative in measurement.MODEL_FILES:
        path = models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    code = tmp_path / 'code'
    for relative in measurement.SCORING_CODE:
        path = code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# original\n')
    record = measurement.measurement_artifacts(models, code_root=code,
        versions={'numpy': 'test', 'opencv': 'test', 'onnxruntime': 'test', 'insightface': 'test'})
    yield models, code, record
    sdpose.scoring_identity.cache_clear()


@pytest.mark.parametrize('relative', measurement.MODEL_FILES)
def test_loss_and_scoring_weights_are_locked(locked_measurement, relative):
    models, code, record = locked_measurement
    (models / relative).write_bytes(b'changed')
    with pytest.raises(ValueError, match='measurement model'):
        measurement.validate_measurement_artifacts(record, verify_files=True)


def test_scoring_code_change_invalidates_measurement(locked_measurement):
    models, code, record = locked_measurement
    (code / measurement.SCORING_CODE[0]).write_text('# changed\n')
    with pytest.raises(ValueError, match='scoring code'):
        measurement.validate_measurement_artifacts(record, verify_files=True)


def test_sdpose_weight_change_invalidates_measurement(locked_measurement):
    _, _, record = locked_measurement
    path = Path(next(iter(record['pose']['files'])))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SDPose artifact'):
        measurement.validate_measurement_artifacts(record, verify_files=True)


def test_scoring_signature_binds_versions_and_model_hashes(locked_measurement):
    models, code, record = locked_measurement
    signature = measurement.scoring_signature(record)
    record['versions']['numpy'] = 'changed'
    assert measurement.scoring_signature(record) != signature


def test_loss_signature_checks_actual_loaded_paths(locked_measurement):
    models, code, record = locked_measurement
    lock = {'measurement_artifacts': record}
    assert len(measurement.loss_signature(lock, models)) == 2
    (models / 'face_eval/identity_encoder.ts').write_bytes(b'changed')
    with pytest.raises(ValueError, match='loss model'):
        measurement.loss_signature(lock, models)

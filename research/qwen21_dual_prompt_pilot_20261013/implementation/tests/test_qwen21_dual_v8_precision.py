"""Reversible precision policy without loading any native model weights."""
from contextlib import nullcontext
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from src.pose_control.qwen21_full.dual.precision import precision_policy, qwen21_precision


@pytest.fixture
def original_bf16_reduction_flag():
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    yield original
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original


@pytest.mark.parametrize('initial', [True, False])
def test_precision_context_disables_reduced_bf16_reduction_and_restores(initial, original_bf16_reduction_flag):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = initial
    with qwen21_precision() as policy:
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
        assert policy == precision_policy()
        assert policy['allow_bf16_reduced_precision_reduction'] is False
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is initial


@pytest.mark.parametrize('initial', [True, False])
def test_precision_context_restores_flag_after_exception(initial, original_bf16_reduction_flag):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = initial
    with pytest.raises(RuntimeError, match='aborted'):
        with qwen21_precision():
            assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
            raise RuntimeError('aborted')
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is initial


def test_nested_context_keeps_outer_policy_until_final_exit(original_bf16_reduction_flag):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with qwen21_precision():
        with qwen21_precision():
            assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True


def test_precision_context_can_decorate_repeated_training_calls(original_bf16_reduction_flag):
    @qwen21_precision()
    def training_entry():
        return torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    assert training_entry() is False
    assert training_entry() is False
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_runtime_applies_same_policy_to_native_and_adapter_preparation(controlled, fail, monkeypatch, original_bf16_reduction_flag):
    from src.pose_control.qwen21_full.dual import runtime
    from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
    generation = SimpleNamespace(source=SimpleNamespace(rgb=torch.zeros(1, 3, 16, 16)),
        geometry=SimpleNamespace(person_valid=torch.ones(1, 2, dtype=torch.bool)),
        prompts=('hug',), seeds=(42,), validate=lambda: None)
    generation.to = lambda *args: generation
    preparation_flags = []
    def prepare(adapter, value):
        preparation_flags.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        return object()
    monkeypatch.setattr(runtime, 'prepared_for_qwen', prepare)
    monkeypatch.setattr(runtime, 'create_control_context', lambda *args, **kwargs: nullcontext())
    class Observer:
        summary = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(runtime, 'GenerationDiagnostics', lambda *args, **kwargs: Observer())
    class Pipeline:
        transformer = torch.nn.Linear(1, 1)
        def __call__(self, **kwargs):
            assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
            if fail:
                raise RuntimeError('aborted')
            return SimpleNamespace(images=[Image.new('RGB', (16, 16))])
    pipe = Pipeline()
    adapter = SimpleNamespace() if controlled else None
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if fail:
        with pytest.raises(RuntimeError, match='aborted'):
            runtime.generate(pipe, generation, adapter=adapter, strengths=ControlStrengths())
    else:
        runtime.generate(pipe, generation, adapter=adapter, strengths=ControlStrengths())
        assert pipe.last_dual_v8_precision_policy == precision_policy()
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    assert preparation_flags == ([False] if controlled else [])


def test_inference_sidecar_records_actual_geometry_and_precision(tmp_path, monkeypatch):
    import sys
    import scripts.infer.infer_qwen21_dual_v8 as infer
    generation = SimpleNamespace(geometry=SimpleNamespace(geometry_hash='a' * 64), prompts=('hug',), seeds=(42,))
    class Dataset:
        def __init__(self, path): pass
        def generation(self, **kwargs): return generation
    pipe = SimpleNamespace(last_dual_v8_precision_policy=precision_policy())
    monkeypatch.setattr(infer, 'Hi4DSMPLXDataset', Dataset)
    monkeypatch.setattr(infer, 'load_native', lambda *args: pipe)
    monkeypatch.setattr(infer, 'generate', lambda *args: Image.new('RGB', (16, 16)))
    manifest = tmp_path / 'generation.json'; manifest.write_text('{}')
    output = tmp_path / 'output.png'
    monkeypatch.setattr(sys, 'argv', ['infer', '--generation', str(manifest), '--output', str(output)])
    infer.main()
    import json
    record = json.loads(output.with_suffix('.json').read_text())
    assert record['geometry_sha256'] == 'a' * 64
    assert record['precision_policy'] == precision_policy()

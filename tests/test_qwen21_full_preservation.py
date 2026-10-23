"""Audit the copied control architecture independently of the Qwen bridge.

These CPU tests compare both namespaces with identical weights and inputs.
Nonzero residual heads are used so equality cannot pass only because of zeros.
"""
from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = "src.pose_control.v6"
NEW = "src.pose_control.qwen21_full"
MANIFEST = Path("/home/shangguanrz/nihaochen/qwen21_full/migration_manifest.json")


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(before)


@lru_cache(None)
def suite(name: str, namespace: str = OLD):
    """Read existing fixtures, replacing their control namespace when requested."""
    path = ROOT / "tests" / (name + ".py")
    if not path.exists():
        pytest.skip('Original preservation fixtures live on sg; run this audit there')
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = types.ModuleType(f"_preservation_{namespace.rsplit('.', 1)[-1]}_{name}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    source = path.read_text().replace(OLD, namespace)
    # Repository scripts were moved into scripts/data; the historical fixture
    # still imports its old root path. Resolve the same implementation for both namespaces.
    source = source.replace('from scripts.precompute_iper_hand_features_v66 import',
                            'from scripts.data.precompute_iper_hand_features_v66 import')
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif is_dataclass(a):
        assert [x.name for x in fields(a)] == [x.name for x in fields(b)]
        for field in fields(a):
            assert_equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def assert_gradients(a, b):
    left, right = dict(a.named_parameters()), dict(b.named_parameters())
    assert left.keys() == right.keys()
    nonzero = 0
    for key in left:
        x, y = left[key].grad, right[key].grad
        assert (x is None) == (y is None), key
        if x is not None:
            assert torch.isfinite(x).all(), key
            torch.testing.assert_close(x, y, rtol=0, atol=0, msg=key)
            nonzero += bool(torch.count_nonzero(x))
    assert nonzero > 0


def test_manifest_preserves_every_original_file_and_hash():
    if not MANIFEST.exists():
        pytest.skip('Original source manifest is on sg')
    manifest = json.loads(MANIFEST.read_text())
    original = ROOT / "src/pose_control/v6"
    copied = ROOT / "src/pose_control/qwen21_full"
    v2 = json.loads((copied / 'native_v2_migration_manifest.json').read_text())
    changed = v2['changed_original_files']
    assert set(changed) == {'condition_injector.py', 'conditions.py', 'token_encoders.py',
                            'reasoning/person_geometry.py', 'reasoning/reasoner.py'}
    paths = {item["relative_path"] for item in manifest["files"]}
    assert len(paths) == 47
    assert paths == {str(p.relative_to(original)) for p in original.rglob("*.py")}
    for item in manifest["files"]:
        path = item["relative_path"]
        source_hash = hashlib.sha256((original / path).read_bytes()).hexdigest()
        copied_hash = hashlib.sha256((copied / path).read_bytes()).hexdigest()
        assert source_hash == item["source_sha256"], path
        if path in changed:
            assert changed[path]['baseline_sha256'] == item['copied_sha256'] == source_hash, path
            assert copied_hash == changed[path]['v2_sha256'], path
        else:
            assert copied_hash == item["copied_sha256"] == source_hash, path


def test_copied_modules_do_not_import_old_namespace():
    if not MANIFEST.exists():
        pytest.skip('Original source manifest is on sg')
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest["files"]:
        path = ROOT / "src/pose_control/qwen21_full" / item["relative_path"]
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(OLD), path
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith(OLD) for alias in node.names), path
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert OLD not in node.value, path


@pytest.mark.parametrize("counts", [(1,), (2,), (1, 2)])
def test_reasoner_same_weights_forward_and_gradients(counts):
    original = suite("test_adapter_reasoning_v1")
    copied = suite("test_adapter_reasoning_v1", NEW)
    a, b = original._build_reasoner(), copied._build_reasoner()
    b.load_state_dict(a.state_dict(), strict=True)
    inputs = original._make_bundle(counts, size=4)
    x, y = a(inputs), b(inputs)
    assert_equal(x, y)
    (x.geometry_highres.square().mean() + x.interaction_highres.square().mean()).backward()
    (y.geometry_highres.square().mean() + y.interaction_highres.square().mean()).backward()
    assert_gradients(a, b)
    assert b.__class__.__module__.startswith(NEW)


def test_control_core_same_weights_active_forward_and_gradients():
    original = suite("test_deepgen_control_interface_v1")
    copied = suite("test_deepgen_control_interface_v1", NEW)
    a, b = original._tiny_core().eval(), copied._tiny_core().eval()
    with torch.no_grad():
        for head in [*a.geometry_zero_heads, *a.interaction_zero_heads]:
            head.weight.normal_(0, .01)
    b.load_state_dict(a.state_dict(), strict=True)
    inputs = original._core_inputs()
    x, y = a(**inputs), b(**inputs)
    assert_equal(x, y)
    assert torch.count_nonzero(x.geometry[0]) > 0
    sum(t.square().mean() for t in (*x.geometry, *x.interaction)).backward()
    sum(t.square().mean() for t in (*y.geometry, *y.interaction)).backward()
    assert_gradients(a, b)


def test_appearance_binding_and_strength_same_weights():
    old = importlib.import_module(OLD + ".token_encoders")
    new = importlib.import_module(NEW + ".token_encoders")
    a, b = old.AppearanceTokenEncoder(token_dim=32), new.AppearanceTokenEncoder(token_dim=32)
    c, d = old.PersonTokenBinder(token_dim=32), new.PersonTokenBinder(token_dim=32)
    b.load_state_dict(a.state_dict(), strict=True)
    d.load_state_dict(c.state_dict(), strict=True)
    valid = torch.tensor([[True, False], [True, True]])
    source = torch.tensor([[3, -1], [4, 9]])
    latents, geometry = torch.randn(2, 2, 16, 8, 8), torch.randn(2, 2, 4, 32)
    x, y = a(latents, valid), b(latents, valid)
    assert_equal(x, y)
    left, right = c(geometry, x, valid, source), d(geometry, y, valid, source)
    assert_equal(left, right)
    left[0].square().mean().backward()
    right[0].square().mean().backward()
    assert_gradients(a, b)
    assert_gradients(c, d)

    original = suite("test_deepgen_control_interface_v1")
    copied = suite("test_deepgen_control_interface_v1", NEW)
    a = original.ControlStrengthController(enable_face=True, enable_hand=True)
    b = copied.ControlStrengthController(enable_face=True, enable_hand=True)
    with torch.no_grad():
        a.geometry_log_group_scale.uniform_(-1, 1)
        a.interaction_log_group_scale.uniform_(-1, 1)
    b.load_state_dict(a.state_dict(), strict=True)
    raw = original._branch_residuals()
    for progress in [0., .4, .8, 1.]:
        assert_equal(a(raw, denoise_progress=progress), b(raw, denoise_progress=progress))


def test_face_preprocessing_adapter_same_weights_and_gradients():
    original = suite("test_v65_face_foundation")
    copied = suite("test_v65_face_foundation", NEW)
    a = original.FaceConditioningPreparer(resampler_depth=1, perceiver_depth=1).eval()
    b = copied.FaceConditioningPreparer(resampler_depth=1, perceiver_depth=1).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    condition, reference = original.make_condition(1), original.make_references(1, 1)
    x, y = a(condition, reference), b(condition, reference)
    assert_equal(x, y)
    old_adapter = importlib.import_module(OLD + ".face.adapter").FaceControlAdapter
    new_adapter = importlib.import_module(NEW + ".face.adapter").FaceControlAdapter
    c = old_adapter(face_dim=16, deepgen_dim=8, heads=4, rank=4).eval()
    d = new_adapter(face_dim=16, deepgen_dim=8, heads=4, rank=4).eval()
    with torch.no_grad():
        for head in c.zero_heads:
            head.weight.normal_(0, .01)
    d.load_state_dict(c.state_dict(), strict=True)
    kwargs = dict(target_token_hw=(8, 8), timestep=torch.tensor([300.]), texture_scale=.6)
    left, right = c(x, **kwargs), d(y, **kwargs)
    assert_equal(left, right)
    assert any(torch.count_nonzero(v) > 0 for v in left)
    sum(t.square().mean() for t in left).backward()
    sum(t.square().mean() for t in right).backward()
    assert_gradients(a, b)
    assert_gradients(c, d)


def test_hand_preprocessing_adapter_same_weights_and_gradients():
    original = suite("test_v66_hand_core")
    copied = suite("test_v66_hand_core", NEW)
    a = original.HandConditioningPreparer(dim=32, resampler_depth=1).eval()
    b = copied.HandConditioningPreparer(dim=32, resampler_depth=1).eval()
    b.load_state_dict(a.state_dict(), strict=True)
    condition, references = original._inputs()
    x, y = a(condition, references), b(condition, references)
    assert_equal(x, y)
    c = original.HandControlAdapter(dim=32, deepgen_dim=48, heads=4).eval()
    d = copied.HandControlAdapter(dim=32, deepgen_dim=48, heads=4).eval()
    with torch.no_grad():
        for head in c.zero_heads:
            head.weight.normal_(0, .01)
    d.load_state_dict(c.state_dict(), strict=True)
    kwargs = dict(target_token_hw=(8, 8), timestep=torch.tensor([300.]))
    left, right = c(x, **kwargs), d(y, **kwargs)
    assert_equal(left, right)
    assert any(torch.count_nonzero(v) > 0 for v in left)
    sum(t.square().mean() for t in left).backward()
    sum(t.square().mean() for t in right).backward()
    assert_gradients(a, b)
    assert_gradients(c, d)

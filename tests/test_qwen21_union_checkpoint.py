"""Checkpoint signatures, complete state, and real optimizer resume parity."""
import copy
import importlib
import importlib.util
import random

import numpy as np
import pytest
import torch

from test_qwen21_union_lora import make_transformer, weights_module
from test_qwen21_union_training import training_module


def checkpoint_module():
    name = "src.pose_control.qwen21_union.checkpoint"
    assert importlib.util.find_spec(name) is not None, "checkpoint implementation is missing"
    return importlib.import_module(name)


def metadata():
    return {"scope": "joint", "base_fingerprint": "base-sha", "union_fingerprint": "union-sha",
            "lock_sha256": "data-sha", "preprocessing_version": "letterbox-v1", "rank": 16,
            "alpha": 16, "lora_targets": list(weights_module().control_lora_targets()),
            "dataset_order": ["iper", "tiktok", "hi4d"], "trajectory_sha256": "trajectory-sha"}


def setup_training():
    model = make_transformer()
    weights_module().install_control_lora(model, expected_parameters=None)
    optimizer = training_module().make_optimizer(model, expected_parameters=None)
    scheduler = training_module().make_lr_scheduler(optimizer, warmup_updates=2)
    return model, optimizer, scheduler


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def update(model, optimizer, scheduler, step):
    # Exercise restored Python/NumPy/torch RNG as well as derived per-sample RNG.
    factor = random.random() + float(np.random.random()) + float(torch.rand(()))
    gen = torch.Generator().manual_seed(training_module().deterministic_seed(3407, step, 0, "sample", "noise"))
    x = torch.randn(2, 129, generator=gen)
    target = torch.randn(2, 4, generator=gen)
    optimizer.zero_grad(set_to_none=True)
    loss = ((model.control_img_in(x) - target) * factor).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    scheduler.step()
    return (factor, x.clone(), loss.item())


def test_resume_equivalence_continuous_four_vs_two_plus_two(tmp_path):
    module = checkpoint_module()
    torch.manual_seed(7)
    initial_model, _, _ = setup_training()
    initial = copy.deepcopy(initial_model.state_dict())
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    initial_rng = module.capture_rng_state()
    full_model, full_optimizer, full_scheduler = setup_training()
    full_model.load_state_dict(initial)
    module.restore_rng_state(initial_rng)
    full_trace = [update(full_model, full_optimizer, full_scheduler, i) for i in range(4)]
    full_rng = module.capture_rng_state()
    model, optimizer, scheduler = setup_training()
    model.load_state_dict(initial)
    module.restore_rng_state(initial_rng)
    first_trace = [update(model, optimizer, scheduler, i) for i in range(2)]
    cursor = module.TrainingCursor(optimizer_step=2, microstep=6, next_sample_position=6,
                                   loss_log_cursor=6, gradient_log_cursor=2)
    saved = module.save_checkpoint(tmp_path / "checkpoint_000002", model, optimizer, scheduler,
                                   metadata=metadata(), cursor=cursor)
    assert saved == tmp_path / "checkpoint_000002"
    resumed, resumed_optimizer, resumed_scheduler = setup_training()
    resumed.load_state_dict(initial)
    restored = module.load_checkpoint(saved, resumed, resumed_optimizer, resumed_scheduler,
                                      expected_metadata=metadata())
    assert restored == cursor
    resumed_trace = first_trace + [update(resumed, resumed_optimizer, resumed_scheduler, i) for i in range(2, 4)]
    assert_tree_equal(full_trace, resumed_trace)
    assert_tree_equal(full_model.state_dict(), resumed.state_dict())
    assert_tree_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert_tree_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())
    assert_tree_equal(full_rng, module.capture_rng_state())


@pytest.mark.parametrize("field,value", [("scope", "target"), ("lock_sha256", "changed-data"),
                                        ("base_fingerprint", "changed-base"),
                                        ("union_fingerprint", "changed-union"),
                                        ("preprocessing_version", "new-layout")])
def test_checkpoint_signature_mismatch_refused_before_mutating_model(tmp_path, field, value):
    module = checkpoint_module()
    model, optimizer, scheduler = setup_training()
    directory = tmp_path / "checkpoint_000000"
    module.save_checkpoint(directory, model, optimizer, scheduler, metadata=metadata(),
                           cursor=module.TrainingCursor())
    before = copy.deepcopy(model.state_dict())
    changed = dict(metadata(), **{field: value})
    with pytest.raises(ValueError, match=field):
        module.load_checkpoint(directory, model, optimizer, scheduler, expected_metadata=changed)
    assert_tree_equal(before, model.state_dict())


def test_checkpoint_refuses_overwrite_partial_update_and_insufficient_signatures(tmp_path):
    module = checkpoint_module()
    model, optimizer, scheduler = setup_training()
    directory = tmp_path / "checkpoint"
    module.save_checkpoint(directory, model, optimizer, scheduler, metadata=metadata(),
                           cursor=module.TrainingCursor())
    with pytest.raises(FileExistsError):
        module.save_checkpoint(directory, model, optimizer, scheduler, metadata=metadata(),
                               cursor=module.TrainingCursor())
    with pytest.raises(ValueError, match="boundary"):
        module.save_checkpoint(tmp_path / "partial", model, optimizer, scheduler, metadata=metadata(),
                               cursor=module.TrainingCursor(microstep=1))
    with pytest.raises(ValueError, match="metadata"):
        module.save_checkpoint(tmp_path / "unsigned", model, optimizer, scheduler, metadata={},
                               cursor=module.TrainingCursor())


def test_checkpoint_integrity_rejects_corrupted_state(tmp_path):
    module = checkpoint_module()
    model, optimizer, scheduler = setup_training()
    directory = tmp_path / "checkpoint"
    module.save_checkpoint(directory, model, optimizer, scheduler, metadata=metadata(),
                           cursor=module.TrainingCursor())
    with (directory / "state.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        module.load_checkpoint(directory, model, optimizer, scheduler, expected_metadata=metadata())


def test_checkpoint_refuses_different_lr_warmup_function(tmp_path):
    module = checkpoint_module()
    model, optimizer, scheduler = setup_training()
    directory = tmp_path / "checkpoint"
    module.save_checkpoint(directory, model, optimizer, scheduler, metadata=metadata(),
                           cursor=module.TrainingCursor())
    wrong_lr = training_module().make_lr_scheduler(optimizer, warmup_updates=32)
    with pytest.raises(ValueError, match="warmup"):
        module.load_checkpoint(directory, model, optimizer, wrong_lr, expected_metadata=metadata())

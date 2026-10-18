"""独立 Pro 全量断点：顺序绑定、篡改拒绝与 4=2+2 精确续跑。"""
import copy
import importlib
import json
import random
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from test_qwen21_pro_training import TinyPro, api as training, run


def api():
    return importlib.import_module("src.pose_control.qwen21_pro.checkpoint")


def metadata(model):
    return {"config": model.task_adapter.config.to_dict(), "scope": "target", "task_signature": "task+loss-v1",
            "source_signature": "source-weights", "cache_signature": "cache-v1", "code_signature": "code-v1",
            "data_signature": "locked-data", "trajectory_signature": "noise+time+order+precision-v1",
            "memory_mode": "resident", "control_lora_targets": []}


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def saved(tmp_path):
    torch.manual_seed(5)
    model = TinyPro()
    result, optimizer, scheduler = run(model, stop_after_updates=2)
    path = tmp_path / "step2"
    api().save_checkpoint(path, model, optimizer, scheduler, metadata=metadata(model), cursor=result["cursor"])
    return path, model, optimizer, scheduler, result


def test_cpu_four_updates_equal_two_plus_two_model_optimizer_rng_and_trajectory(tmp_path):
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    first = TinyPro()
    initial = copy.deepcopy(first)
    continuous, opt_a, lr_a = run(first)
    rng_a = api().capture_rng_state()
    split, opt_b, lr_b = run(initial, stop_after_updates=2)
    path = tmp_path / "resume"
    api().save_checkpoint(path, initial, opt_b, lr_b, metadata=metadata(initial), cursor=split["cursor"])
    # Construction consumes global RNG; loading must restore saved RNG exactly.
    resumed = TinyPro()
    resumed.load_state_dict(copy.deepcopy(initial.state_dict()))
    opt_c = training().make_optimizer(resumed, lr=.01)
    lr_c = training().make_lr_scheduler(opt_c, warmup_updates=2)
    cursor = api().load_checkpoint(path, resumed, opt_c, lr_c, expected_metadata=metadata(resumed))
    finish, _, _ = run(resumed, optimizer=opt_c, lr_scheduler=lr_c, cursor=cursor)
    equal_tree(first.state_dict(), resumed.state_dict())
    equal_tree(opt_a.state_dict(), opt_c.state_dict())
    equal_tree(lr_a.state_dict(), lr_c.state_dict())
    equal_tree(rng_a, api().capture_rng_state())
    assert continuous["trajectory"] == split["trajectory"] + finish["trajectory"]
    assert finish["cursor"] == continuous["cursor"]
    assert isinstance(resumed.task_adapter.state_dict()["_extra_state"], dict)
    with pytest.raises(FileExistsError):
        api().save_checkpoint(path, initial, opt_b, lr_b, metadata=metadata(initial), cursor=split["cursor"])


@pytest.mark.parametrize("key", ["config", "scope", "data_signature", "cache_signature", "memory_mode", "trajectory_signature"])
def test_metadata_mismatch_rejected_before_mutation(tmp_path, key):
    path, model, optimizer, scheduler, _ = saved(tmp_path)
    expected = metadata(model)
    expected[key] = {**expected[key], "source_dim": 1600} if key == "config" else "wrong"
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        api().load_checkpoint(path, model, optimizer, scheduler, expected_metadata=expected)
    equal_tree(before, model.state_dict())


def test_frozen_hash_and_optimizer_order_and_lr_rejected(tmp_path):
    path, model, optimizer, scheduler, _ = saved(tmp_path)
    before = copy.deepcopy(model.state_dict())
    model.base.weight.data.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        api().load_checkpoint(path, model, optimizer, scheduler, expected_metadata=metadata(model))
    model.load_state_dict(before)
    optimizer.param_groups[0]["params"] = list(reversed(optimizer.param_groups[0]["params"]))
    with pytest.raises(ValueError, match="binding|order"):
        api().load_checkpoint(path, model, optimizer, scheduler, expected_metadata=metadata(model))
    optimizer.param_groups[0]["params"].reverse()
    other_lr = training().make_lr_scheduler(optimizer, warmup_updates=99)
    with pytest.raises(ValueError, match="LR|scheduler"):
        api().load_checkpoint(path, model, optimizer, other_lr, expected_metadata=metadata(model))


@pytest.mark.parametrize("mutation", ["bytes", "old", "state_shape", "extra_state", "optimizer_lr", "rng"])
def test_corruption_and_rechecksummed_internal_tamper_rejected_before_mutation(tmp_path, mutation):
    path, model, optimizer, scheduler, _ = saved(tmp_path)
    manifest_path = path / "checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "bytes":
        with (path / "state.pt").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "old":
        manifest["format_version"] = "qwen21_union_training_checkpoint_v1"
        manifest_path.write_text(json.dumps(manifest))
    else:
        state = torch.load(path / "state.pt", weights_only=False)
        if mutation == "state_shape":
            state["task"]["query_projection.weight"] = torch.zeros(1)
        elif mutation == "extra_state":
            state["task"]["_extra_state"]["enabled"] = False
        elif mutation == "optimizer_lr":
            state["optimizer"]["param_groups"][0]["lr"] = 99.
        else:
            state["rng"]["torch_cpu"] = torch.ones(1, dtype=torch.uint8)
        torch.save(state, path / "state.pt")
        import hashlib
        manifest["state_sha256"] = hashlib.sha256((path / "state.pt").read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    before, before_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError):
        api().load_checkpoint(path, model, optimizer, scheduler, expected_metadata=metadata(model))
    equal_tree(before, model.state_dict())
    equal_tree(before_opt, optimizer.state_dict())


def test_inference_weights_load_without_optimizer_rng_or_data_memory_binding(tmp_path):
    path, model, _, _, _ = saved(tmp_path)
    restored = TinyPro()
    restored.base.load_state_dict(model.base.state_dict())
    restored.output.load_state_dict(model.output.state_dict())
    restored._pro_memory_mode = "frozen_linear_cpu"
    rng = api().capture_rng_state()
    expected = {k: v for k, v in metadata(model).items() if k in
                {"config", "scope", "task_signature", "code_signature", "source_signature"}}
    manifest = api().load_adapter_weights(path, restored, expected_metadata=expected)
    equal_tree(model.state_dict(), restored.state_dict())
    equal_tree(rng, api().capture_rng_state())
    assert manifest["metadata"]["data_signature"] == "locked-data"
    expected["source_signature"] = "wrong-base"
    with pytest.raises(ValueError, match="source_signature"):
        api().load_adapter_weights(path, restored, expected_metadata=expected)


def test_optional_lora_state_rank_binding_and_gate_round_trip(tmp_path):
    from torch import nn
    from src.pose_control.qwen21_union.weights import ControlLoRALinear
    model = TinyPro()
    model.control_img_in = ControlLoRALinear(nn.Linear(129, 8), rank=2, alpha=3)
    opt = training().make_optimizer(model, control_lora_targets=("control_img_in",))
    lr = training().make_lr_scheduler(opt, warmup_updates=0)
    model.task_adapter.gate.fill_(.75)
    meta = {**metadata(model), "control_lora_targets": ["control_img_in"]}
    path = tmp_path / "with_lora"
    api().save_checkpoint(path, model, opt, lr, metadata=meta, cursor=api().TrainingCursor())
    model.control_img_in.lora_A.data.zero_()
    model.task_adapter.gate.fill_(.5)
    api().load_checkpoint(path, model, opt, lr, expected_metadata=meta)
    assert model.control_img_in.lora_A.count_nonzero() > 0
    assert model.task_adapter.gate == .75
    model.control_img_in.alpha = 2.
    with pytest.raises(ValueError, match="rank/alpha"):
        api().load_checkpoint(path, model, opt, lr, expected_metadata=meta)


@pytest.mark.parametrize("loader", ["inference", "resume"])
@pytest.mark.parametrize("hook_name", ["_hf_hook", "_diffusers_hook", "_group_offload_hook", "_offload_hook", "_auto_device_hooks"])
def test_load_rejects_existing_offload_hooks_without_state_or_rng_mutation(tmp_path, loader, hook_name):
    path, model, optimizer, scheduler, _ = saved(tmp_path)
    marker = [model.task_adapter.query_projection.register_forward_pre_hook(lambda *args: None)] if hook_name == "_auto_device_hooks" else object()
    setattr(model.task_adapter, hook_name, marker)
    with torch.no_grad():
        model.task_adapter.output_heads["0"].bias.add_(3.)
    random.random()
    np.random.rand()
    torch.rand(2)
    before = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = api().capture_rng_state()
    expected = metadata(model)
    with pytest.raises(ValueError, match="hook"):
        if loader == "inference":
            expected = {k: v for k, v in expected.items() if k in
                        {"config", "scope", "task_signature", "code_signature", "source_signature"}}
            api().load_adapter_weights(path, model, expected_metadata=expected)
        else:
            api().load_checkpoint(path, model, optimizer, scheduler, expected_metadata=expected)
    equal_tree(before, model.state_dict())
    equal_tree(before_optimizer, optimizer.state_dict())
    equal_tree(before_rng, api().capture_rng_state())


SOURCE = Path(os.environ.get("QWEN21_UNION_SOURCE", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/source/VideoX-Fun"))


@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.is_dir()
                    or os.environ.get("QWEN21_PRO_CUDA_TESTS") != "1",
                    reason="requires CUDA, pinned VideoX-Fun and QWEN21_PRO_CUDA_TESTS=1")
def test_fixed_native_cuda_offload_four_updates_equal_two_plus_two_exact_checkpoint(tmp_path):
    from src.pose_control.qwen21_union.pipeline import _upstream, scoped_transformer_class
    from src.pose_control.qwen21_pro.integration import install_pro_adapter
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro.offload import place_training_model
    from test_qwen21_pro_training import pair, Scheduler
    module = training()
    module.configure_determinism()
    _, models, _, _ = _upstream(SOURCE)
    cls = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    def construct():
        torch.manual_seed(21)
        net = cls(num_layers=32, num_attention_heads=4, attention_head_dim=8, context_in_dim=8,
                  axes_dims_rope=(2, 2, 4), control_in_dim=129).to(dtype=torch.bfloat16)
        with torch.no_grad():
            for block in net.control_blocks:
                block.after_proj.weight.normal_(std=.02)
        install_pro_adapter(net, ProAdapterConfig(hidden_dim=32, width=16, heads=4, source_dim=1600))
        place_training_model(net, "cuda", memory_mode="frozen_linear_cpu")
        net.gradient_checkpointing = True
        return net
    order = tuple((dataset, f"{dataset}_{i}") for i in range(4) for dataset in ("iper", "tiktok", "hi4d"))
    item = pair(1600)
    def execute(net, optimizer=None, lr=None, cursor=None, stop=None):
        optimizer = optimizer or module.make_optimizer(net, lr=.01)
        lr = lr or module.make_lr_scheduler(optimizer, warmup_updates=2)
        result = module.run_training_updates(net, optimizer=optimizer, lr_scheduler=lr, scheduler=Scheduler(),
            sample_order=order, load_sample=lambda _: item, updates=4, device="cuda", cursor=cursor,
            stop_after_updates=stop, require_verified_conditions=False)
        return result, optimizer, lr
    random.seed(31)
    np.random.seed(31)
    continuous = construct()
    initial_rng = api().capture_rng_state()
    full, opt_a, lr_a = execute(continuous)
    expected_rng = api().capture_rng_state()
    split = construct()
    api().restore_rng_state(initial_rng)
    start, opt_b, lr_b = execute(split, stop=2)
    meta = {**metadata(split), "scope": "joint", "memory_mode": "frozen_linear_cpu",
            "trajectory_signature": module.trajectory_signature(order, memory_mode="frozen_linear_cpu")}
    path = tmp_path / "native_cuda_step2"
    api().save_checkpoint(path, split, opt_b, lr_b, metadata=meta, cursor=start["cursor"])
    resumed = construct()
    original_ids = {n: id(p) for n, p in resumed.named_parameters()}
    opt_c = module.make_optimizer(resumed, lr=.01)
    lr_c = module.make_lr_scheduler(opt_c, warmup_updates=2)
    random.random()
    np.random.rand()
    torch.rand(5)
    torch.rand(5, device="cuda")
    cursor = api().load_checkpoint(path, resumed, opt_c, lr_c, expected_metadata=meta)
    assert {n: id(p) for n, p in resumed.named_parameters()} == original_ids
    end, _, _ = execute(resumed, opt_c, lr_c, cursor)
    equal_tree(continuous.state_dict(), resumed.state_dict())
    equal_tree(opt_a.state_dict(), opt_c.state_dict())
    equal_tree(lr_a.state_dict(), lr_c.state_dict())
    equal_tree(expected_rng, api().capture_rng_state())
    assert full["trajectory"] == start["trajectory"] + end["trajectory"]
    assert full["cursor"] == end["cursor"]

"""CPU 原生控制链、checkpoint、绑定与构建顺序的集成回归。"""
from __future__ import annotations

import copy
import importlib
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from test_qwen21_pro_contracts import make_condition
from src.pose_control.qwen21_pro import ProAdapterConfig


def integration():
    return importlib.import_module("src.pose_control.qwen21_pro.integration")


def pipeline_api():
    return importlib.import_module("src.pose_control.qwen21_pro.pipeline")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class NativeBlock(nn.Module):
    """严格的 native kwargs；保留官方九个 native 参数和因果前缀。"""
    def __init__(self, dim=32):
        super().__init__()
        self.attn = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(self.attn, name, nn.Linear(dim, dim, bias=False))
        self.attn.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.attn.norm_q = nn.LayerNorm(dim, bias=False)
        self.attn.norm_k = nn.LayerNorm(dim, bias=False)
        self.img_mlp = nn.Module()
        for name in ("proj", "gate_layer", "out"):
            setattr(self.img_mlp, name, nn.Linear(dim, dim, bias=False))

    def forward(self, hidden_states, modulation, rotary_emb=None, attention_mask=None,
                target_token_mask=None, layer_cache=None, kv_cache_mode=None,
                cache_write_slice=None, segments=None, key_valid=None):
        q = self.attn.norm_q(self.attn.to_q(hidden_states))
        k = self.attn.norm_k(self.attn.to_k(hidden_states))
        v = self.attn.to_v(hidden_states)
        # A causal operation lets target deltas propagate along the running stream,
        # while preventing them from modifying earlier text/reference positions.
        count = torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None]
        context = v.cumsum(dim=1) / count
        hidden_states = hidden_states + .01 * self.attn.to_out[0](q * k + context)
        return hidden_states + .01 * self.img_mlp.out(
            torch.nn.functional.silu(self.img_mlp.proj(hidden_states))
            * self.img_mlp.gate_layer(hidden_states))


class ControlBlock(NativeBlock):
    def __init__(self, block_id, dim=32):
        super().__init__(dim)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.after_proj.weight.copy_(torch.eye(dim) * .1)
            self.after_proj.bias.zero_()

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c, history = self.before_proj(c) + x, []
        else:
            history = list(torch.unbind(c))
            c = history.pop()
        c = super().forward(c, **kwargs)
        return torch.stack([*history, self.after_proj(c), c])


class UnionModel(nn.Module):
    _no_split_modules = ["NativeBlock", "ControlBlock"]
    _repeated_blocks = ["NativeBlock", "ControlBlock"]

    def __init__(self):
        super().__init__()
        self.inner_dim = 32
        self.config = SimpleNamespace(causal_condition=True)
        self.control_layers = list(range(0, 32, 2))
        self.control_blocks = nn.ModuleList([ControlBlock(i) for i in self.control_layers])
        self.control_img_in = nn.Linear(64, 32)
        self.img_in = nn.Linear(64, 32)
        self.txt_in = nn.Linear(32, 32)
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.fail = False
        self.interleave = None
        self.calls = []

    @staticmethod
    def build_token_metadata(image_pad_mask, img_shapes):
        lengths = [math.prod(shape) for shape in img_shapes]
        positions = image_pad_mask.nonzero(as_tuple=True)[0]
        if sum(lengths) != len(positions):
            raise ValueError("img_shapes accounts for a different image token count")
        mask = torch.zeros_like(image_pad_mask)
        mask[positions[-lengths[-1]:]] = True
        return torch.zeros_like(image_pad_mask, dtype=torch.long), mask

    def forward_control(self, control_joint, x_joint, kwargs):
        c = control_joint
        for block in self.control_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def custom_forward(c, block=block, static_kwargs=dict(kwargs)):
                    return block(c, x_joint, **static_kwargs)
                c = checkpoint(custom_forward, c, use_reentrant=False)
            else:
                c = block(c, x_joint, **kwargs)
        return torch.unbind(c)[:-1]

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                encoder_hidden_states_mask=None, attention_kwargs=None, kv_cache=None,
                kv_cache_mode=None, control_context=None, control_context_scale=1., return_dict=True):
        if self.fail:
            raise RuntimeError("deliberate native failure")
        if self.interleave:
            callback, self.interleave = self.interleave, None
            callback()
        repeats = torch.where(img_mask, 4, 1)[0]
        image_mask = torch.repeat_interleave(img_mask[0], repeats)
        target_slots = math.prod(img_shapes[0][-1]) // 4
        joint = torch.cat([self.txt_in(encoder_hidden_states),
                           encoder_hidden_states.new_zeros(1, target_slots, 32)], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_mask] = self.img_in(hidden_states)
        _, target = self.build_token_metadata(image_mask, img_shapes[0])
        self.calls.append((joint.shape[1], target.clone()))
        if control_context is None:
            return joint
        control = torch.zeros_like(joint)
        control[:, image_mask] = self.control_img_in(control_context)
        native = dict(modulation=timestep, rotary_emb=None, attention_mask=None,
                      target_token_mask=target, layer_cache=None, kv_cache_mode=None,
                      cache_write_slice=None, segments=None, key_valid=None)
        hints = self.forward_control(control, joint, native)
        return joint + torch.stack(hints).sum(dim=0) * control_context_scale


def model():
    torch.manual_seed(11)
    return UnionModel()


def cfg(stages=(0, 4, 8, 12)):
    return ProAdapterConfig(hidden_dim=32, width=16, heads=4, stage_indices=stages)


def inputs(text_length=3):
    torch.manual_seed(12)
    return dict(hidden_states=torch.randn(1, 8, 64),
                encoder_hidden_states=torch.randn(1, text_length, 32),
                timestep=torch.tensor([.4]), img_shapes=[[(1, 2, 2), (1, 2, 2)]],
                img_mask=torch.tensor([[False, True, *([False] * (text_length - 2)), True]]),
                control_context=torch.randn(1, 8, 64))


def installed(stages=(0, 4, 8, 12)):
    result = model()
    integration().install_pro_adapter(result, cfg(stages))
    return result


def test_zero_heads_preserve_all_180_union_keys_parameter_objects_and_reference_output():
    base = model()
    original = dict(base.named_parameters())
    control = {k: v for k, v in original.items() if k.startswith(("control_blocks.", "control_img_in."))}
    assert len(control) == 180
    baseline = copy.deepcopy(base)(**inputs())
    cls, block_cls = type(base), type(base.control_blocks[0])
    audit = integration().install_pro_adapter(base, cfg())
    actual = base(**inputs(), task_condition=make_condition())
    assert torch.equal(actual, baseline)
    assert all(dict(base.named_parameters())[k] is v for k, v in original.items())
    assert all(not p.requires_grad for p in original.values())
    assert all(p.requires_grad and p.dtype == torch.float32 for p in base.task_adapter.parameters())
    assert not any(".base." in k for k in base.state_dict())
    assert type(base) is not cls and type(base.control_blocks[0]) is not block_cls
    assert block_cls.forward is ControlBlock.forward
    assert audit["control_parameter_keys"] == tuple(control)
    assert audit["num_control_blocks"] == 16
    assert len([n for n, _ in base.named_modules() if n.endswith("task_adapter")]) == 1
    for name in ("_no_split_modules", "_repeated_blocks"):
        assert type(base.control_blocks[0]).__name__ in getattr(base, name)
        assert "ProTaskAdapter" in getattr(base, name)


def test_nonzero_task_changes_current_skip_and_running_without_history_or_prefix_writes():
    base = installed((4,))
    captures = []
    handle = base.control_blocks[4].register_forward_hook(lambda m, args, out: captures.append(out.detach().clone()))
    expected = base(**inputs(), task_condition=make_condition())
    with torch.no_grad():
        base.task_adapter.output_heads["4"].bias.fill_(.2)
    actual = base(**inputs(), task_condition=make_condition())
    handle.remove()
    before, after = captures
    target = base.calls[-1][1]
    assert torch.equal(before[:4], after[:4])
    assert not torch.equal(before[4:, :, target], after[4:, :, target])
    assert torch.equal(before[:, :, ~target], after[:, :, ~target])
    assert torch.equal(expected[:, ~target], actual[:, ~target])
    assert not torch.equal(expected[:, target], actual[:, target])


def test_adapter_normal_call_hooks_receive_current_hidden_and_stage_mapping():
    base, seen = installed(tuple(range(16))), []
    handle = base.task_adapter.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((args[0].shape, kwargs["stage"], kwargs["timestep"].shape)), with_kwargs=True)
    base(**inputs(), task_condition=make_condition())
    handle.remove()
    assert [stage for _, stage, _ in seen] == list(range(16))
    assert all(shape == (1, 10, 32) and time_shape == (1,) for shape, _, time_shape in seen)


def test_checkpoint_backward_captures_execution_after_forward_context_was_cleared():
    base = installed()
    base.gradient_checkpointing = True
    pack = make_condition()
    optimizer = torch.optim.SGD(base.task_adapter.parameters(), lr=.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = base(**inputs(), task_condition=pack)
        assert base._pro_execution_context.get() is None
        out.square().mean().backward()
        assert base.task_adapter.output_heads["0"].weight.grad.abs().sum() > 0
        if step:
            for branch in (base.task_adapter.geometry_encoder, base.task_adapter.banks):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
        optimizer.step()
    assert all(p.grad is None for n, p in base.named_parameters() if not n.startswith("task_adapter."))


def test_binding_is_per_model_reentrant_and_restored_after_exception():
    first, second = installed(), installed()
    pack = make_condition()
    with integration().bind_task_condition(first, pack):
        first.interleave = lambda: second(**inputs(), task_enabled=False)
        first(**inputs())
        assert first._pro_execution_context.get() is None
        first.fail = True
        with pytest.raises(RuntimeError, match="deliberate"):
            first(**inputs())
        assert first._pro_execution_context.get() is None
        first.fail = False
        with integration().bind_task_condition(first, None, enabled=False):
            first(**inputs())
        first(**inputs())
    with pytest.raises(ValueError, match="task_condition"):
        first(**inputs())
    with pytest.raises(ValueError, match="task_condition"):
        second(**inputs())


def test_same_model_nested_forward_restores_outer_condition():
    base = installed((0,))
    outer, inner = make_condition(), make_condition()
    inner = replace(inner, source_fingerprint="inner-source")
    seen = []
    handle = base.task_adapter.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(kwargs["conditions"].source_fingerprint), with_kwargs=True)
    base.interleave = lambda: base(**inputs(), task_condition=inner)
    base(**inputs(), task_condition=outer)
    handle.remove()
    assert seen == ["inner-source"] * 16 + [outer.source_fingerprint] * 16
    assert base._pro_execution_context.get() is None


def test_cfg_forwards_rebuild_joint_mask_for_each_prompt_length():
    base = installed()
    pack = make_condition()
    with integration().bind_task_condition(base, pack):
        first, second = base(**inputs()), base(**inputs(5))
    assert first.shape[1] == 10 and second.shape[1] == 12
    assert [int(mask.sum()) for _, mask in base.calls] == [4, 4]
    assert [len(mask) for _, mask in base.calls] == [10, 12]


@pytest.mark.parametrize("case", ["batch", "text_batch", "condition_batch", "mask_dtype", "mask_length", "shapes",
                                  "grid", "time_batch", "time_nan", "kv", "kv_mode", "causal", "parallel", "control"])
def test_rejects_unsupported_batch_layout_grid_cache_and_parallelism(case):
    base, args, pack = installed(), inputs(), make_condition()
    if case == "batch":
        args["hidden_states"] = args["hidden_states"].repeat(2, 1, 1)
    elif case == "text_batch":
        args["encoder_hidden_states"] = args["encoder_hidden_states"].repeat(2, 1, 1)
    elif case == "condition_batch":
        pack = make_condition(batch=2)
    elif case == "mask_dtype":
        args["img_mask"] = args["img_mask"].float()
    elif case == "mask_length":
        args["img_mask"] = args["img_mask"][:, :-1]
    elif case == "shapes":
        args["img_shapes"] = [[(1, 2, 4), (1, 2, 2)]]
    elif case == "grid":
        args["img_shapes"] = [[(1, 2, 2), (1, 1, 4)]]
    elif case == "time_batch":
        args["timestep"] = torch.tensor([.2, .3])
    elif case == "time_nan":
        args["timestep"] = torch.tensor([float("nan")])
    elif case == "kv":
        args["kv_cache"] = object()
    elif case == "kv_mode":
        args["kv_cache_mode"] = "cached"
    elif case == "causal":
        base.config.causal_condition = False
    elif case == "parallel":
        base.sp_world_size = 2
    else:
        args["control_context"] = None
    with pytest.raises(ValueError):
        base(**args, task_condition=pack)
    assert base._pro_execution_context.get() is None


def test_task_requires_condition_and_explicit_disable_restores_union():
    reference = model()
    expected = reference(**inputs())
    integration().install_pro_adapter(reference, cfg())
    with pytest.raises(ValueError, match="task_condition"):
        reference(**inputs())
    assert torch.equal(reference(**inputs(), task_enabled=False), expected)
    reference.task_adapter.enabled = False
    assert torch.equal(reference(**inputs()), expected)


def test_lora_disable_is_explicit_and_install_preserves_its_trainability():
    from src.pose_control.qwen21_union.weights import install_control_lora
    base = model()
    expected = base(**inputs())
    install_control_lora(base, expected_parameters=None)
    lora_parameters = {name: p for name, p in base.named_parameters() if p.requires_grad}
    with torch.no_grad():
        base.control_blocks[0].after_proj.lora_B.fill_(.05)
    integration().install_pro_adapter(base, cfg())
    assert all(p.requires_grad for p in lora_parameters.values())
    assert not torch.equal(base(**inputs(), task_enabled=False), expected)
    report = integration().set_pro_enabled(base, False, include_lora=True)
    assert report["task_enabled"] is False and report["lora_enabled"] is False
    assert torch.equal(base(**inputs()), expected)


def test_install_validation_does_not_partially_mutate_model():
    base, original = model(), model()
    base.control_blocks[-1].block_id = 31
    with pytest.raises(ValueError, match="control"):
        integration().install_pro_adapter(base, cfg())
    assert type(base) is type(original)
    assert not hasattr(base, "task_adapter")
    assert all(p.requires_grad for p in base.parameters())
    good = installed()
    with pytest.raises(ValueError, match="installed"):
        integration().install_pro_adapter(good, cfg())


def test_build_transformer_reuses_verified_union_then_lora_then_task(monkeypatch):
    from src.pose_control.qwen21_union import pipeline as old
    from src.pose_control.qwen21_union import weights
    base, events = model(), []
    def load(*args, **kwargs):
        events.append("load180")
        base.requires_grad_(False)
        return base, {"restored_after_union_load": 180}
    original = weights.install_control_lora
    def lora(model, **kwargs):
        assert not hasattr(model, "task_adapter")
        events.append("lora")
        return original(model, **kwargs)
    monkeypatch.setattr(old, "build_transformer", load)
    monkeypatch.setattr(weights, "install_control_lora", lora)
    actual, audit = pipeline_api().build_pro_transformer(config=cfg(), control_lora=True,
                                                        lora_options={"expected_parameters": None})
    assert actual is base and hasattr(actual, "task_adapter")
    assert events == ["load180", "lora"]
    assert audit["restored_after_union_load"] == 180
    assert audit["pro_adapter"]["num_control_blocks"] == 16
    assert all(p.requires_grad for n, p in base.named_parameters() if "lora_" in n)


class FakePipeline:
    def __init__(self, transformer, text_encoder=object()):
        self.transformer, self.text_encoder = transformer, text_encoder
        self.seen = None
    def __call__(self, prompt=None, image=None, mask_image=None, reference_images=None,
                 num_images_per_prompt=1, prompt_embeds=None, negative_prompt=None,
                 negative_prompt_embeds=None, true_cfg_scale=1., use_kv_cache=False, **kwargs):
        self.seen = (image, mask_image, reference_images)
        return self.transformer(**inputs()), self.transformer(**inputs(5))


def test_pipeline_registers_pro_before_offload_and_delegates_reference_inpaint_args(monkeypatch):
    from src.pose_control.qwen21_union import pipeline as old
    base, events = model(), []
    def build(*args, transformer_setup=None, **kwargs):
        base.requires_grad_(False)
        audit = transformer_setup(base)
        events.append("setup")
        assert hasattr(base, "task_adapter")
        assert all(p.dtype == torch.float32 for p in base.task_adapter.parameters())
        events.append("offload")
        return FakePipeline(base), {"transformer_setup": audit}
    monkeypatch.setattr(old, "build_pipeline", build)
    pipe, audit = pipeline_api().build_pro_pipeline(config=cfg(), device="cuda", cached=False)
    image, mask, refs = object(), object(), [object()]
    outputs = pipe(image=image, mask_image=mask, reference_images=refs, task_condition=make_condition())
    assert events == ["setup", "offload"]
    assert pipe.transformer is base
    assert pipe._pipeline.seen == (image, mask, refs)
    assert [out.shape[1] for out in outputs] == [10, 12]
    assert base._pro_execution_context.get() is None
    with pytest.raises(ValueError, match="task_condition"):
        base(**inputs())


@pytest.mark.parametrize("kwargs", [dict(num_images_per_prompt=2), dict(prompt=["a", "b"]),
                                    dict(prompt_embeds=torch.zeros(2, 3, 32)), dict(use_kv_cache=True)])
def test_pipeline_rejects_implicit_batch_expansion_and_kv_cache(kwargs):
    pipe = pipeline_api().ProPipeline(FakePipeline(installed()))
    with pytest.raises(ValueError):
        pipe(task_condition=make_condition(), **kwargs)


def test_cached_pipeline_rejects_raw_refs_or_missing_embeddings_before_text_encoder():
    pipe = pipeline_api().ProPipeline(FakePipeline(installed(), text_encoder=None))
    with pytest.raises(ValueError, match="cached|text_encoder"):
        pipe(prompt="hello", task_condition=make_condition())
    with pytest.raises(ValueError, match="reference|cached"):
        pipe(prompt_embeds=torch.zeros(1, 3, 32), reference_images=[object()], task_condition=make_condition())
    pipe(prompt_embeds=torch.zeros(1, 3, 32), task_condition=make_condition())


SOURCE = Path(os.environ.get("QWEN21_UNION_SOURCE", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/source/VideoX-Fun"))


@pytest.mark.skipif(not SOURCE.is_dir(), reason="fixed VideoX-Fun source is not mounted locally")
def test_fixed_source_small_real_qwen_zero_equivalence_and_checkpoint_gradient():
    from src.pose_control.qwen21_union.pipeline import _upstream, scoped_transformer_class
    _, models, _, _ = _upstream(SOURCE)
    cls = scoped_transformer_class(models.QwenImage21ControlTransformer2DModel)
    torch.manual_seed(21)
    base = cls(num_layers=32, num_attention_heads=4, attention_head_dim=8, context_in_dim=32,
               axes_dims_rope=(2, 2, 4), control_in_dim=64)
    # The released Union has trained after_proj. Zero projections would hide all
    # task gradients and make this test meaningless.
    with torch.no_grad():
        for block in base.control_blocks:
            block.after_proj.weight.normal_(std=.02)
    args = inputs()
    args["return_dict"] = False
    expected = base(**args)[0]
    originals = dict(base.named_parameters())
    control = {n: p for n, p in originals.items() if n.startswith(("control_blocks.", "control_img_in."))}
    assert len(control) == 180
    integration().install_pro_adapter(base, cfg())
    actual = base(**args, task_condition=make_condition())[0]
    assert torch.equal(actual, expected)
    assert all(dict(base.named_parameters())[n] is p for n, p in originals.items())
    base.gradient_checkpointing = True
    optimizer = torch.optim.SGD(base.task_adapter.parameters(), lr=.2)
    pack = make_condition()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = base(**args, task_condition=pack)[0]
        assert base._pro_execution_context.get() is None
        out.square().mean().backward()
        assert base.task_adapter.output_heads["0"].weight.grad.abs().sum() > 0
        if step:
            for branch in (base.task_adapter.geometry_encoder, base.task_adapter.banks):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
        optimizer.step()


def test_public_build_and_binding_exports_remain_lazy():
    import src.pose_control.qwen21_pro as package
    for name in ("install_pro_adapter", "bind_task_condition", "set_pro_enabled", "build_pro_transformer", "build_pro_pipeline", "ProPipeline"):
        assert callable(getattr(package, name))
    import subprocess
    import sys
    code = "from src.pose_control.qwen21_pro import build_pro_pipeline; import sys; assert 'videox_fun' not in sys.modules; assert 'diffusers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_actual_non_square_target_grid_and_source_dim_1600():
    base = model()
    config = replace(cfg(), source_dim=1600)
    integration().install_pro_adapter(base, config)
    pack = make_condition()
    pack = replace(pack, target_hw=(2, 4),
                   spatial={k: v.repeat_interleave(2, dim=-1) for k, v in pack.spatial.items()},
                   person_mask=pack.person_mask.repeat_interleave(2, dim=-1),
                   owner=pack.owner.repeat_interleave(2, dim=-1),
                   face_mask=pack.face_mask.repeat_interleave(2, dim=-1),
                   hand_mask=pack.hand_mask.repeat_interleave(2, dim=-1),
                   source_tokens=torch.randn(1, 2, 3, 1600))
    args = inputs()
    args.update(hidden_states=torch.randn(1, 12, 64), control_context=torch.randn(1, 12, 64),
                img_shapes=[[(1, 2, 2), (1, 2, 4)]], img_mask=torch.tensor([[False, True, False, True, True]]))
    out = base(**args, task_condition=pack)
    assert out.shape == (1, 14, 32)
    assert int(base.calls[-1][1].sum()) == 8


@pytest.mark.parametrize("bad_config", [False, 0, "invalid"])
def test_install_rejects_non_config_without_mutation(bad_config):
    base = model()
    with pytest.raises(ValueError, match="config"):
        integration().install_pro_adapter(base, bad_config)
    assert type(base) is UnionModel and not hasattr(base, "task_adapter")


def test_cached_cfg_guidance_alias_requires_negative_embeddings():
    pipe = pipeline_api().ProPipeline(FakePipeline(installed(), text_encoder=None))
    with pytest.raises(ValueError, match="negative_prompt_embeds"):
        pipe(prompt_embeds=torch.zeros(1, 3, 32), negative_prompt="negative",
             true_cfg_scale=1., guidance_scale=2., task_condition=make_condition())


def test_install_task_parameters_follow_existing_model_device_without_dtype_cast():
    base = model().to(device="meta", dtype=torch.bfloat16)
    integration().install_pro_adapter(base, cfg())
    assert base.control_blocks[0].after_proj.weight.device.type == "meta"
    assert all(p.device.type == "meta" and p.dtype == torch.float32 for p in base.task_adapter.parameters())



def test_pipeline_to_keeps_pro_wrapper_and_condition_binding_after_chaining():
    class MovablePipeline(FakePipeline):
        def to(self, *args, **kwargs):
            self.to_arguments = (args, kwargs)
            return self

    native = MovablePipeline(installed())
    pipe = pipeline_api().ProPipeline(native)
    moved = pipe.to("cpu", torch.bfloat16, silence_dtype_warnings=True)
    assert moved is pipe
    assert native.to_arguments == (("cpu", torch.bfloat16), {"silence_dtype_warnings": True})
    outputs = moved(task_condition=make_condition())
    assert [output.shape[1] for output in outputs] == [10, 12]
    assert native.transformer._pro_execution_context.get() is None
    with pytest.raises(ValueError, match="task_condition"):
        native.transformer(**inputs())


def test_pipeline_to_preserves_native_exception():
    failure = RuntimeError("native device conversion failed")

    class FailingPipeline(FakePipeline):
        def to(self, *args, **kwargs):
            self.to_arguments = (args, kwargs)
            raise failure

    native = FailingPipeline(installed())
    pipe = pipeline_api().ProPipeline(native)
    with pytest.raises(RuntimeError, match="native device conversion") as raised:
        pipe.to(device="unsupported", dtype=torch.bfloat16)
    assert raised.value is failure
    assert native.to_arguments == ((), {"device": "unsupported", "dtype": torch.bfloat16})

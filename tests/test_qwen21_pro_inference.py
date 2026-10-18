from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from test_qwen21_pro_contracts import make_condition


def grid_pack(hw):
    pack = make_condition(people=1)
    n = hw[0] * hw[1]

    def resize(v, channels=False):
        shape = (*v.shape[:-2], *hw) if channels else (*v.shape[:-1], n)
        return v.flatten()[:1].expand(shape).clone()

    return replace(
        pack,
        target_hw=hw,
        spatial={k: resize(v, True) for k, v in pack.spatial.items()},
        **{
            k: resize(getattr(pack, k))
            for k in ("person_mask", "owner", "face_mask", "hand_mask")
        }
    )


def condition(hw, negative=True):
    n = hw[0] * hw[1]

    def prompt(length):
        slots = torch.zeros(1, length + n // 4, dtype=torch.bool)
        slots[:, :4] = True
        slots[:, length:] = True
        return dict(
            prompt_embeds=torch.ones(1, length, 8),
            prompt_mask=torch.ones(1, length, dtype=torch.bool),
            img_mask=slots,
        )

    value = dict(
        reference_latents=torch.ones(1, 16, 64),
        control_latents=torch.ones(1, n, 64),
        img_shapes=[[(1, 4, 4), (1, *hw)]],
        target_hw=hw,
        canvas_hw=(hw[0] * 16, hw[1] * 16),
        **prompt(7)
    )
    if negative:
        value["negative"] = prompt(11)
    return value


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

        self.task_adapter = SimpleNamespace(config=ProAdapterConfig())
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return (
            kwargs["hidden_states"] * 0.2
            + kwargs["encoder_hidden_states"].shape[1] / 10,
        )


@pytest.mark.parametrize("hw", [(32, 32), (48, 48), (32, 48)])
def test_cached_sampler_actual_grid_native_euler_cfg_masks_same_binding(hw):
    from src.pose_control.qwen21_pro.inference import sample_cached

    transformer = Transformer()
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipe = SimpleNamespace(
        transformer=transformer, scheduler=scheduler, _execution_device="cpu"
    )
    pack = grid_pack(hw)
    result = sample_cached(
        pipe,
        condition(hw),
        task_condition=pack,
        num_inference_steps=2,
        seed=5,
        true_cfg_scale=2.0,
    )
    assert result.latents.shape == (1, hw[0] * hw[1], 64)
    assert result.audit["target_hw"] == list(hw)
    assert result.audit["cfg_norm_rescale"] is True
    assert len(transformer.calls) == 4
    for positive, negative in zip(transformer.calls[::2], transformer.calls[1::2]):
        assert positive["task_condition"] is negative["task_condition"] is pack
        assert positive["encoder_hidden_states"].shape[1] == 7
        assert negative["encoder_hidden_states"].shape[1] == 11
        assert torch.equal(positive["control_context"], negative["control_context"])
        assert positive["control_context"].shape[-1] == 129
        assert positive["kv_cache"] is None and negative["kv_cache_mode"] is None
    again = sample_cached(
        pipe,
        condition(hw),
        task_condition=pack,
        num_inference_steps=2,
        seed=5,
        true_cfg_scale=2.0,
    )
    assert torch.equal(result.latents, again.latents)


def test_decode_uses_actual_grid_and_vae_statistics():
    from src.pose_control.qwen21_pro.inference import decode_latents

    class VAE:
        dtype = torch.float32
        config = SimpleNamespace(
            z_dim=64, latents_mean=[2.0] * 64, latents_std=[3.0] * 64
        )

        def decode(self, value, return_dict):
            assert value.shape == (1, 64, 1, 32, 48)
            assert (value == 5).all()
            return (value[:, :3],)

    processor = SimpleNamespace(postprocess=lambda v, output_type: v)
    pipe = SimpleNamespace(vae=VAE(), image_processor=processor)
    assert decode_latents(
        pipe, torch.ones(1, 1536, 64), (512, 768), output_type="pt"
    ).shape == (1, 3, 32, 48)


def test_cached_sampler_rejects_grid_and_missing_negative_and_kv():
    from src.pose_control.qwen21_pro.inference import sample_cached

    pipe = SimpleNamespace(
        transformer=Transformer(),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        _execution_device="cpu",
    )
    for c, kw in [
        (condition((32, 32)), dict(use_kv_cache=True)),
        (condition((32, 32), False), dict(true_cfg_scale=2.0)),
    ]:
        with pytest.raises(ValueError):
            sample_cached(pipe, c, task_condition=grid_pack((32, 32)), **kw)


def test_sampler_preserves_readonly_legacy_attestation_in_artifact_audit():
    from src.pose_control.qwen21_pro.inference import sample_cached
    from types import MappingProxyType

    class ReadonlyLegacy(dict):
        attestation = MappingProxyType(
            {"sample_id": "legacy", "condition_fingerprint": "locked-sha"}
        )

    native = ReadonlyLegacy(condition((32, 32)))
    pipe = SimpleNamespace(
        transformer=Transformer(),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        _execution_device="cpu",
    )
    result = sample_cached(
        pipe, native, task_condition=grid_pack((32, 32)), num_inference_steps=1
    )
    assert result.audit["native_condition_binding"] == dict(native.attestation)


@pytest.mark.parametrize("hw", [(32, 48), (48, 48)])
def test_coarse_noise_uses_official_channel_major_seed_layout(hw):
    from src.pose_control.qwen21_pro.inference import sample_cached

    transformer = Transformer()
    pipe = SimpleNamespace(
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        _execution_device="cpu",
    )
    result = sample_cached(
        pipe,
        condition(hw),
        task_condition=grid_pack(hw),
        num_inference_steps=1,
        seed=19,
    )
    expected = (
        torch.randn((1, 64, 1, *hw), generator=torch.Generator().manual_seed(19))
        .flatten(2)
        .transpose(1, 2)
    )
    assert torch.equal(
        transformer.calls[0]["hidden_states"][:, -hw[0] * hw[1] :], expected
    )
    assert result.audit["noise_layout"] == "channel_major_1x64x1xHxW_then_unpatched64"

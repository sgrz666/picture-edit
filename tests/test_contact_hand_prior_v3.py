from dataclasses import replace
import hashlib
import json

import pytest
import torch

from tests.test_contact_hand_adapter import condition, fixture_model


@pytest.fixture(autouse=True, scope="module")
def small_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def tiny_prior():
    from src.pose_control.contact_hand_adapter.prior import HandPrior

    return HandPrior(width=32, layers=3, rank=4, sha256="4" * 64).requires_grad_(False)


def test_prior_updates_only_packed_hand_query_rows_and_preserves_native_processor():
    from src.pose_control.contact_hand_adapter import bind_hand_condition, install_hand_adapter
    from src.pose_control.contact_hand_adapter.geometry import pixel_geometry

    model, adapter, inputs = fixture_model()
    sample = condition()
    attention = model.transformer_blocks[0].attn
    processor = attention.processor
    install_hand_adapter(model, adapter, hand_prior=tiny_prior())
    observed = []
    hook = attention.register_forward_hook(lambda _, inputs, output: observed.append(output.detach()))
    with torch.no_grad(), bind_hand_condition(model, sample, strength=0):
        expected = model(**inputs)[0]
    with torch.no_grad(), bind_hand_condition(model, sample, strength=0, prior_strength=1.):
        actual = model(**inputs)[0]
    hook.remove()
    _, support = pixel_geometry(sample)
    roi = torch.nn.functional.avg_pool2d(support, sample.canvas.stride).flatten(1).any(0)
    target_rows = torch.cat((torch.zeros(observed[0].shape[1] - len(roi), dtype=torch.bool), roi))
    assert torch.equal(observed[0][:, ~target_rows], observed[1][:, ~target_rows])
    assert not torch.equal(observed[0][:, target_rows], observed[1][:, target_rows])
    assert not torch.equal(expected, actual)
    assert attention.processor is processor


def test_prior_off_empty_or_at_initial_noise_is_exact_native_identity():
    from src.pose_control.contact_hand_adapter import bind_hand_condition, install_hand_adapter

    model, adapter, inputs = fixture_model()
    install_hand_adapter(model, adapter, hand_prior=tiny_prior())
    empty = condition()
    empty.visible.zero_()
    initial = {**inputs, "timestep": torch.ones(1)}
    with torch.no_grad():
        for sample, model_inputs, prior_strength in ((condition(), inputs, 0.), (empty, inputs, 1.),
                                                     (condition(), initial, 1.)):
            baseline = model(**model_inputs)[0]
            with bind_hand_condition(model, sample, strength=0, prior_strength=prior_strength):
                assert torch.equal(model(**model_inputs)[0], baseline)


def test_combined_frozen_prior_keeps_adapter_gradients_with_checkpointing():
    from src.pose_control.contact_hand_adapter import bind_hand_condition, install_hand_adapter

    model, adapter, inputs = fixture_model()
    prior = tiny_prior()
    install_hand_adapter(model, adapter, hand_prior=prior)
    sample = condition()

    def run():
        with bind_hand_condition(model, sample, strength=.8, prior_strength=.5):
            output = model(**inputs)[0]
        output.square().mean().backward()
        gradient = adapter.output.shared.weight.grad.detach().clone()
        adapter.zero_grad(set_to_none=True)
        return output.detach(), gradient

    baseline, expected_grad = run()
    model.enable_gradient_checkpointing()
    actual, actual_grad = run()
    torch.testing.assert_close(actual, baseline)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in prior.parameters())


def test_official_prior_loader_pins_safetensors_and_freezes_weights(tmp_path):
    from safetensors.torch import save_file
    from src.pose_control.contact_hand_adapter.prior import load_hand_prior

    path = tmp_path / "pytorch_lora_weights.safetensors"
    keys = [f"transformer.transformer_blocks.{i}.attn.{projection}"
            for i in range(32) for projection in ("to_q", "to_k", "to_v", "to_out.0")]
    tensors = {f"{key}.lora_{side}.weight": torch.zeros(shape, dtype=torch.bfloat16)
               for key in keys for side, shape in (("A", (16, 4096)), ("B", (4096, 16)))}
    tensors[f"{keys[0]}.lora_B.weight"][0, 0] = .5
    metadata = {"transformer.r": 16, "transformer.lora_alpha": 16,
                "transformer.target_modules": ["to_q", "to_k", "to_v", "to_out.0"]}
    save_file(tensors, path, metadata={"format": "pt", "lora_adapter_metadata": json.dumps(metadata)})
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA256"):
        load_hand_prior(path, sha256="4" * 64)
    rng = torch.random.get_rng_state()
    prior = load_hand_prior(path, sha256=sha)
    assert torch.equal(torch.random.get_rng_state(), rng)
    info = prior.provenance()
    assert info["sha256"] == sha and info["rank"] == 16 and info["alpha"] == 16 and info["layers"] == 32
    assert sum(p.numel() for p in prior.parameters()) == 16_777_216
    assert all(not p.requires_grad for p in prior.parameters())


def test_v3_checkpoint_uses_its_own_schema(tmp_path):
    from src.pose_control.contact_hand_adapter import ContactHandAdapter
    from src.pose_control.contact_hand_adapter.runtime import load_adapter, save_adapter

    _, adapter, _ = fixture_model()
    v3 = ContactHandAdapter(replace(adapter.config, architecture_version=3))
    path = tmp_path / "hand-v3.pt"
    save_adapter(path, v3, base_sha="a" * 64, step=0)
    assert torch.load(path, weights_only=True)["schema"] == "contact_hand_adapter_v3"
    loaded, _ = load_adapter(path, base_sha="a" * 64)
    assert loaded.config == v3.config


def test_runner_exposes_hand_prior_options(capsys):
    from scripts.run.run_contact_hand_adapter import main

    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    assert {"--hand-prior", "--hand-prior-sha", "--hand-prior-strength"} <= set(capsys.readouterr().out.split())


def test_local_plan_preserves_source_descriptors_and_masks_unselected_validity():
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan

    source = condition(hw=(64, 256))
    source.joints[3, :, 0] += 180
    source.support[3] = torch.roll(source.support[3], 180, dims=1)
    valid = source.visible.any(1)
    descriptors = torch.ones(4, 2, 256)
    source = replace(source, pairs=(), appearance=torch.rand(4, 3, 24, 24), appearance_valid=valid,
                     appearance_sources=(None, "a" * 64, None, "b" * 64), appearance_dino=descriptors,
                     appearance_mask=torch.ones(4, 24, 24, dtype=torch.bool),
                     appearance_dino_valid=valid, appearance_dino_model_sha="c" * 64,
                     appearance_dino_crop_sha=(None, "d" * 64, None, "e" * 64))
    cropped = make_local_plan(source, size=64, slots=(1,)).condition
    assert cropped.appearance_dino_valid.tolist() == [False, True, False, False]
    assert torch.equal(cropped.appearance_dino, descriptors)
    assert cropped.appearance_dino_crop_sha == (None, "d" * 64, None, None)

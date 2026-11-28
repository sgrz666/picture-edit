from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from PIL import Image

from src.data.canvas_contract import CanvasSpec


@pytest.fixture(autouse=True, scope="module")
def small_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def condition(hw=(64, 96)):
    from src.pose_control.contact_hand_adapter import HandCondition, HandPair

    canvas = CanvasSpec(hw, hw, hw)
    joints = torch.zeros(4, 21, 3)
    visible = torch.zeros(4, 21, dtype=torch.bool)
    support = torch.zeros(4, *hw)
    for slot, offset in ((1, 10), (3, 28)):
        joints[slot, :, 0] = offset + torch.arange(21) % 5
        joints[slot, :, 1] = 12 + torch.arange(21) // 5
        joints[slot, :, 2] = 0.8
        visible[slot] = True
        support[slot, 8:24, offset:offset + 12] = 1
    contact = torch.full((16, 16), -1, dtype=torch.int8)
    contact[1, 4] = 1
    contact[6, 7] = 0
    return HandCondition(canvas, ("actor-a", "actor-b"), joints, visible, support=support,
                         pairs=(HandPair((1, 3), contact),))


def fixture_model():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter

    torch.manual_seed(102)
    base = QwenImage21Transformer2DModel(num_layers=3, attention_head_dim=16, num_attention_heads=2,
                                         context_in_dim=32, axes_dims_rope=(2, 6, 8)).eval()
    config = AdapterConfig(width=32, heads=4, base_width=32, base_heads=2, layers=(0, 2),
                           intra_layers=1, inter_layers=1, roi_size=32, rank=4, architecture_version=21)
    adapter = ContactHandAdapter(config)
    latents = torch.randn(1, 4 + 24, 64)
    image_mask = torch.tensor([[False, True, False, True, True, True, True, True, True]])
    inputs = dict(hidden_states=latents, timestep=torch.tensor([0.4]),
                  encoder_hidden_states=torch.randn(1, 3, 32),
                  encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.bool),
                  img_shapes=[[(1, 2, 2), (1, 4, 6)]], img_mask=image_mask, return_dict=False)
    return base, adapter, inputs


def test_hierarchy_and_three_state_contact_preserve_both_right_hands():
    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter

    model = ContactHandAdapter(AdapterConfig(width=32, heads=4, intra_layers=1, inter_layers=1,
                                            roi_size=32, surface_tokens=64, architecture_version=21))
    sample = condition()
    memory = model.prepare(sample)
    assert memory.hierarchy_counts == (1, 6, 21, 64, 16, 1)
    assert memory.tokens.shape == (56, 32)
    assert memory.slots.unique().tolist() == [1, 3]
    assert memory.graph.part_state[1, 3, 1, 4] == 1
    assert memory.graph.part_state[3, 1, 4, 1] == 1
    assert memory.graph.part_state[1, 3, 6, 7] == 0
    assert memory.graph.part_state[1, 3, 6, 6] == -1
    assert memory.graph.surface_edges.numel() == 0
    assert torch.isfinite(memory.tokens).all()
    assert memory.support.shape == (4, 24)
    assert (memory.support.sum(0) <= 1).all()


def test_spatial_padding_and_unknown_inputs_are_safe():
    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter

    sample = condition()
    canvas = CanvasSpec((32, 64), (32, 64), (64, 96), (0, 0))
    sample.joints[0] = float("nan")
    sample.support[:, :, 80:] = 1
    model = ContactHandAdapter(AdapterConfig(width=32, heads=4, intra_layers=1, inter_layers=1, roi_size=32))
    memory = model.prepare(replace(sample, canvas=canvas))
    assert memory.support[:, 5::6].count_nonzero() == 0
    assert torch.isfinite(memory.tokens).all()


def test_qwen_attention_identity_and_roi_gradient():
    from src.pose_control.contact_hand_adapter import install_hand_adapter, bind_hand_condition

    base, adapter, inputs = fixture_model()
    frozen = deepcopy(base)
    install_hand_adapter(base, adapter)
    expected = frozen(**inputs)[0]
    with bind_hand_condition(base, condition()):
        observed = base(**inputs)[0]
    assert torch.equal(expected, observed)
    observed.square().mean().backward()
    assert adapter.output.shared.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters() if not p.requires_grad)
    with torch.no_grad():
        adapter.output.shared.weight.normal_(std=.03)
        empty = condition()
        empty.visible.zero_()
        with bind_hand_condition(base, empty):
            assert torch.equal(base(**inputs)[0], expected)
        with bind_hand_condition(base, condition(), strength=0):
            assert torch.equal(base(**inputs)[0], expected)
        with bind_hand_condition(base, condition()):
            assert not torch.equal(base(**inputs)[0], expected)
        assert torch.equal(base(**inputs)[0], expected)


def test_install_preserves_native_flex_processor_identity():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor
    from src.pose_control.contact_hand_adapter import ContactHandAdapter, bind_hand_condition, install_hand_adapter

    base, adapter, inputs = fixture_model()
    adapter = ContactHandAdapter(replace(adapter.config, layers=(0, 1, 2)))
    for block in base.transformer_blocks:
        block.attn.set_processor(QwenImage21FlexAttnProcessor())
    native = [block.attn.processor for block in base.transformer_blocks]
    with torch.no_grad():
        expected = base(**inputs)[0]
        install_hand_adapter(base, adapter)
        assert all(block.attn.processor is original for block, original in zip(base.transformer_blocks, native))
        with bind_hand_condition(base, condition()):
            assert torch.equal(base(**inputs)[0], expected)


def test_qwen_dynamic_state_checkpoint_and_forward_reset():
    from src.pose_control.contact_hand_adapter import install_hand_adapter, bind_hand_condition

    base, adapter, inputs = fixture_model()
    install_hand_adapter(base, adapter)
    with torch.no_grad():
        adapter.output.shared.weight.normal_(std=.03)
    sample = condition()
    seen = []
    hook = adapter.feedback.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    with bind_hand_condition(base, sample):
        first = base(**inputs)[0]
        second = base(**inputs)[0]
    hook.remove()
    assert len(seen) == 4
    assert not torch.equal(seen[0], seen[1])
    assert torch.equal(seen[0], seen[2])
    assert torch.equal(first, second)
    first.square().mean().backward()
    expected = adapter.feedback.image_query.weight.grad.clone()
    adapter.zero_grad(set_to_none=True)
    base.enable_gradient_checkpointing()
    with bind_hand_condition(base, sample):
        checked = base(**inputs)[0]
    checked.square().mean().backward()
    torch.testing.assert_close(checked, first)
    torch.testing.assert_close(adapter.feedback.image_query.weight.grad, expected)


def test_mesh_appearance_and_sparse_contact_require_a_shared_camera():
    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter, HandMesh

    sample = condition()
    mesh = HandMesh(torch.rand(4, 778, 3) * .1, torch.tensor([[0, 1], [1, 2]]),
                    torch.arange(128).reshape(64, 2), torch.full((4, 64, 2), 16.),
                    torch.tensor([False, True, False, True]), (None, "common", None, "common"))
    pair = replace(sample.pairs[0], surface_indices=torch.tensor([[2, 4], [5, 6]]),
                   surface_features=torch.tensor([[.002, -.9, 1., .9], [.024, -.4, 0., 1.]]))
    sample = replace(sample, mesh=mesh, pairs=(pair,), appearance=torch.rand(4, 3, 16, 16),
                     appearance_valid=mesh.valid, appearance_sources=(None, "a" * 64, None, "b" * 64))
    cfg = AdapterConfig(width=32, heads=4, intra_layers=1, inter_layers=1, roi_size=32, surface_tokens=32,
                        architecture_version=21)
    memory = ContactHandAdapter(cfg).prepare(sample)
    assert memory.tokens.shape == (2 * 76, 32)
    assert memory.graph.surface_edges.shape == (4, 4)
    assert memory.graph.surface_edges.tolist()[2] == [3, 1, 4, 2]
    with pytest.raises(ValueError, match="verified common camera"):
        replace(sample, mesh=replace(mesh, camera_ids=(None, "first", None, "second"))).validate()


def test_joint_local_generation_expands_contact_and_preserves_outside_mask():
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan

    sample = condition()
    plan = make_local_plan(sample, size=128, slots=(1,))
    assert plan.slots == (1, 3)
    assert plan.condition.canvas.canvas_hw == (128, 128)
    assert plan.condition.canvas.original_hw[0] != plan.condition.canvas.original_hw[1]
    x0, y0, x1, y1 = plan.box
    base = Image.new("RGB", (96, 64), "black")
    assert plan.image(base).size == (128, 128)
    patched = plan.composite(base, Image.new("RGB", (128, 128), "white"))
    assert patched.getpixel((12, 15)) == (255, 255, 255)
    assert patched.getpixel((80, 45)) == (0, 0, 0)
    assert patched.crop((x1, 0, 96, 64)).tobytes() == base.crop((x1, 0, 96, 64)).tobytes()


def test_unknown_distant_pairs_do_not_force_local_connectivity():
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan

    sample = condition(hw=(64, 256))
    sample.joints[3, :, 0] += 180
    sample.support[3] = torch.roll(sample.support[3], 180, dims=1)
    pair = replace(sample.pairs[0], part_state=torch.full((16, 16), -1, dtype=torch.int8))
    plan = make_local_plan(replace(sample, pairs=(pair,)), size=128, slots=(1,))
    assert plan.slots == (1,) and plan.box[2] < 64


def test_hand_level_contact_and_low_confidence_geometry_have_distinct_signals():
    from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter

    sample = condition()
    cfg = AdapterConfig(width=32, heads=4, intra_layers=1, inter_layers=1, roi_size=32, contact_level="hand")
    model = ContactHandAdapter(cfg)
    touched = model.prepare(sample)
    state = sample.pairs[0].part_state.clone()
    state[state == 1] = 0
    separated = model.prepare(replace(sample, pairs=(replace(sample.pairs[0], part_state=state),)))
    assert not torch.equal(touched.graph.hand_features, separated.graph.hand_features)
    assert touched.graph.part_state.eq(-1).all()
    joints = sample.joints.clone()
    joints[..., 2] *= .1
    weak = model.prepare(replace(sample, joints=joints, support=None))
    assert weak.roi.numel() and weak.support.max() < .1


def test_strict_checkpoint_and_condition_roundtrip(tmp_path):
    from src.pose_control.contact_hand_adapter.runtime import (
        condition_record, load_adapter, load_condition, save_adapter,
    )

    base, adapter, _ = fixture_model()
    state_path, condition_path = tmp_path / "hand.pt", tmp_path / "condition.pt"
    signature = "4" * 64
    sha = save_adapter(state_path, adapter, base_sha=signature, step=7)
    restored, state = load_adapter(state_path, base_sha=signature)
    assert len(sha) == 64 and state["step"] == 7 and restored.config == adapter.config
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in adapter.state_dict().items())
    with pytest.raises(ValueError, match="base"):
        load_adapter(state_path, base_sha="5" * 64)
    with pytest.raises(FileExistsError):
        save_adapter(state_path, adapter, base_sha=signature, step=8)
    sample = condition()
    torch.save(condition_record(sample), condition_path)
    loaded = load_condition(condition_path)
    assert torch.equal(loaded.pairs[0].part_state, sample.pairs[0].part_state)


def test_real_qwen_training_loss_reaches_adapter_through_frozen_rgb_heads():
    from types import SimpleNamespace
    from diffusers import AutoencoderKLQwenImage21
    from src.pose_control.contact_hand_adapter import install_hand_adapter
    from src.pose_control.contact_hand_adapter.training import (
        FlowBatch, RegionSupervision, check_output_gradients, decode_clean_latent,
        output_supervision, predict_flow, region_flow_objective,
    )

    base, adapter, model_kwargs = fixture_model()
    install_hand_adapter(base, adapter)
    with torch.no_grad():
        adapter.output.shared.weight.normal_(std=.03)
    sample = condition()
    target = model_kwargs["hidden_states"][:, -24:].clone()
    batch = FlowBatch(sample, target, model_kwargs["hidden_states"][:, :-24],
                      model_kwargs["encoder_hidden_states"], model_kwargs["encoder_hidden_states_mask"],
                      model_kwargs["img_mask"], tuple(model_kwargs["img_shapes"][0]))
    sigma = torch.tensor([.3])
    noise = torch.randn_like(target)
    predicted, expected, clean = predict_flow(base, batch, sigma, noise)
    region = RegionSupervision(sample.support)
    loss, terms = region_flow_objective(predicted, expected, sample.canvas, region)
    assert loss > 0 and set(terms) == {"contact", "hand", "wrist", "other"}
    head = torch.nn.Conv2d(3, 3, 1).requires_grad_(False)
    vae = AutoencoderKLQwenImage21(base_dim=4, decoder_base_dim=4, dim_mult=[1, 2, 2, 2, 2],
                                   num_res_blocks=1).requires_grad_(False)
    pipe = SimpleNamespace(vae=vae)
    rgb = decode_clean_latent(pipe, clean, sample.canvas)
    assert rgb.shape == (1, 3, 64, 96)
    out = torch.nn.functional.adaptive_avg_pool2d(head(rgb), (4, 6))
    predictions = {"joint": out[:, 0], "part": out, "contact": out[:, 1], "anatomy": out[:, 2]}
    labels = {name: (torch.zeros((1, 4, 6), dtype=torch.long), torch.ones((1, 4, 6), dtype=torch.bool))
              for name in predictions}
    labels["contact"][1][:] = False
    auxiliary = output_supervision(predictions, labels)
    assert auxiliary["contact"].item() == 0
    assert all(auxiliary[name] > 0 for name in ("joint", "part", "anatomy"))
    check_output_gradients(clean, {name: auxiliary[name] for name in ("joint", "part", "anatomy")})
    with pytest.raises(ValueError, match="anatomy"):
        check_output_gradients(clean, {"anatomy": auxiliary["anatomy"].detach()})
    sum(auxiliary.values()).backward()
    assert adapter.feedback.image_query.weight.grad.abs().sum() > 0
    assert head.weight.grad is None


def test_training_manifest_loads_local_output_and_rejects_same_frame(tmp_path, capsys):
    import json
    from scripts.train.train_contact_hand_adapter import main
    from src.pose_control.contact_hand_adapter.runtime import condition_record
    from src.pose_control.contact_hand_adapter.training import load_manifest, load_training_pair
    from src.pose_edit.common import sha256_file

    source, target, geometry = (tmp_path / filename for filename in ("a.png", "b.png", "geometry.pt"))
    Image.new("RGB", (96, 64), "black").save(source)
    Image.new("RGB", (96, 64), "grey").save(target)
    torch.save(condition_record(condition()), geometry)
    ref = lambda path: {"path": path.name, "sha256": sha256_file(path)}
    row = {"id": "pair-1", "group": "clip-1", "split": "train", "mode": "local", "prompt": "Join hands.",
           "source_refs": [ref(source)], "context": ref(source), "target": ref(target), "condition": ref(geometry)}
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    samples = load_manifest(manifest)
    pair = load_training_pair(samples[0], local_size=128)
    assert pair.target.size == (128, 128)
    assert pair.condition.canvas.canvas_hw == (128, 128)
    assert pair.regions.hands.shape == (4, 128, 128)
    assert len(pair.references) == 2 and pair.references[-1].size == (128, 128)
    main(["--manifest", str(manifest), "--check-data", "--local-size", "128"])
    assert json.loads(capsys.readouterr().out) == {"train/local": 1}
    without_context = {key: value for key, value in row.items() if key != "context"}
    manifest.write_text(json.dumps(without_context) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="context"):
        load_manifest(manifest)
    row["target"] = row["source_refs"][0]
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target RGB"):
        load_manifest(manifest)


def test_source_appearance_rejects_unbound_and_wrong_actor_references():
    from src.pose_control.contact_hand_adapter.conditions import verify_appearance_sources

    sample = replace(condition(), appearance=torch.rand(4, 3, 16, 16),
                     appearance_valid=torch.tensor([False, True, False, True]),
                     appearance_sources=(None, "a" * 64, None, "b" * 64))
    verify_appearance_sources(sample, ("a" * 64, "b" * 64))
    with pytest.raises(ValueError, match="appearance"):
        verify_appearance_sources(sample, ("b" * 64, "a" * 64))
    with pytest.raises(ValueError, match="appearance"):
        verify_appearance_sources(sample, ())
    both = replace(sample, appearance_sources=(None, "a" * 64, None, "a" * 64))
    with pytest.raises(ValueError, match="appearance"):
        verify_appearance_sources(both, ("a" * 64,))
    verify_appearance_sources(replace(both, shared_appearance_source=True), ("a" * 64,))


def test_output_label_eligibility_checks_mask_and_unknown_labels():
    from src.pose_control.contact_hand_adapter.training import eligible_output_labels

    no_labels = torch.zeros(2, dtype=torch.bool)
    assert eligible_output_labels({"contact": (torch.tensor([-1, -1]), no_labels)}) == ()
    assert eligible_output_labels({"anatomy": (torch.tensor([0, 1]), ~no_labels)}) == ("anatomy",)
    with pytest.raises(ValueError, match="unknown"):
        eligible_output_labels({"contact": (torch.tensor([0, -1]), ~no_labels)})


def test_optional_lora_is_roi_local_and_warm_starts_from_pure_adapter(tmp_path):
    from src.pose_control.contact_hand_adapter import ContactHandAdapter, bind_hand_condition, install_hand_adapter
    from src.pose_control.contact_hand_adapter.runtime import initialize_adapter, save_adapter

    base, adapter, inputs = fixture_model()
    signature = "4" * 64
    before = base(**inputs)[0]
    path = tmp_path / "pure.pt"
    save_adapter(path, adapter, base_sha=signature, step=100)
    cfg = replace(adapter.config, hand_lora_rank=4)
    lora = ContactHandAdapter(cfg)
    assert initialize_adapter(lora, path, base_sha=signature) == 100
    install_hand_adapter(base, lora)
    with bind_hand_condition(base, condition()):
        after = base(**inputs)[0]
    assert torch.equal(before, after)
    after.square().mean().backward()
    assert lora.native_lora[0].up.weight.grad.abs().sum() > 0
    with torch.no_grad():
        lora.native_lora[-1].up.weight.normal_(std=.02)
        with bind_hand_condition(base, condition()):
            edited = base(**inputs)[0]
    assert torch.equal(edited[:, :-24], before[:, :-24])
    assert not torch.equal(edited[:, -24:], before[:, -24:])

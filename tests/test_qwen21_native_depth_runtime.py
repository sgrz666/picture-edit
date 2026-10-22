from __future__ import annotations

import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def api():
    from src.pose_control.qwen21_full import native_depth_runtime
    return native_depth_runtime


def sample():
    return {"target_canvas": {"original_hw": [32, 16], "resized_hw": [32, 16],
             "canvas_hw": [32, 32], "offset_xy": [0, 0]},
            "valid_image_mask": torch.cat((torch.ones(1, 32, 16), torch.zeros(1, 32, 16)), dim=-1),
            "pair_metadata": {"pair_id": "pair-1", "target_stem": "frame-1", "cross_person": False},
            "target_token_hw": (2, 2)}


def manifest(tmp_path):
    Image.fromarray(np.zeros((32, 16), np.uint8)).save(tmp_path / "depth.png")
    Image.fromarray(np.full((32, 16), 255, np.uint8)).save(tmp_path / "valid.png")
    row = {"pair_id": "pair-1", "target_stem": "frame-1", "kind": "relative_png",
           "depth_path": "depth.png", "valid_path": "valid.png", "polarity": "near_one",
           "camera_id": "cam-1", "units": "unknown", "scale_status": "unknown"}
    path = tmp_path / "depth.jsonl"
    path.write_text(json.dumps(row)+"\n", encoding="utf8")
    return path


def test_depth_manifest_binds_pair_target_canvas_and_keeps_zero_gray_valid(tmp_path):
    records = api().NativeDepthRecords(manifest(tmp_path), mode="relative_png")
    c = records.condition(sample())
    assert c.available.tolist() == [True] and c.relative.eq(0).all()
    assert c.metadata[0]["source_id"] == "pair-1"
    assert c.canvas_valid.sum() == 512


@pytest.mark.parametrize("change", ["missing_pair", "wrong_frame", "wrong_canvas", "wrong_mode"])
def test_manifest_mismatches_fail_before_model_load(tmp_path, change):
    path = manifest(tmp_path)
    s = sample()
    if change == "missing_pair":
        s["pair_metadata"]["pair_id"] = "absent"
    elif change == "wrong_frame":
        s["pair_metadata"]["target_stem"] = "another-frame"
    elif change == "wrong_canvas":
        s["valid_image_mask"].fill_(1)
    with pytest.raises(ValueError):
        api().NativeDepthRecords(path, mode="camera_z" if change == "wrong_mode" else "relative_png").condition(s)


def runtime(depth=True):
    from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import NativeBodyConfig, Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth import NativeDepthConfig, Qwen21NativeDepth
    full = Qwen21FullAdapter(UnifiedSMPLXAdapterV6.build_tiny(), qwen_hidden_size=16,
                             qwen_text_dim=24, pooled_input_dim=20)
    geom = Qwen21NativeGeometry.from_full_adapter(full)
    width = geom.reasoner.config.hidden_dim
    body = Qwen21NativeBody(NativeBodyConfig(hidden_dim=16, geometry_dim=width, width=8))
    d = Qwen21NativeDepth(NativeDepthConfig(geometry_dim=width)) if depth else None
    return api().NativeDepthRuntime(geom, body, d)


def test_native_checkpoint_roundtrip_and_explicit_nodepth_warm_start():
    original = runtime(depth=False)
    saved = original.checkpoint_payload(metadata={"step": 3})
    target = runtime(depth=True)
    target.load_native_weights(saved, include_depth=False)
    assert all(torch.equal(a,b) for a,b in zip(original.body.parameters(),target.body.parameters()))
    assert target.depth.projection.weight.eq(0).all()
    with pytest.raises(ValueError):
        target.load_native_weights(saved, include_depth=True)
    full = target.checkpoint_payload(metadata={"step": 0})
    restored = runtime(depth=True)
    restored.load_native_weights(copy.deepcopy(full), include_depth=True)
    assert restored.depth.get_extra_state() == target.depth.get_extra_state()


def test_native_checkpoint_can_rebuild_without_original_full_weights(tmp_path):
    system = runtime()
    path = tmp_path / "native.pt"
    torch.save(system.checkpoint_payload(metadata={"step":3}), path)
    saved = torch.load(path,map_location="cpu",weights_only=True)
    rebuilt = api().NativeDepthRuntime.from_native_checkpoint(saved)
    assert rebuilt.body.config == system.body.config
    assert rebuilt.depth.config == system.depth.config
    assert all(torch.equal(a,b) for a,b in zip(system.parameters(),rebuilt.parameters()))


def test_depth_asset_drift_is_rejected_after_first_read(tmp_path):
    records = api().NativeDepthRecords(manifest(tmp_path), mode="relative_png")
    records.condition(sample())
    replacement = tmp_path / "another_depth.png"
    Image.fromarray(np.full((32,16),127,np.uint8)).save(replacement)
    records.records["pair-1"]["depth_path"] = str(replacement)
    with pytest.raises(ValueError,match="changed"):
        records.condition(sample())


def test_flow_loss_masks_padding_without_erasing_partial_tokens():
    pred = torch.tensor([[[2., 2.], [10., 10.], [4., 4.], [20., 20.]]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = api().native_velocity_loss(pred,target,torch.tensor([[1.,0.,.5,0.]]))
    assert loss.item() == pytest.approx(8.)
    loss.backward()
    assert pred.grad[:,[1,3]].eq(0).all() and pred.grad[:,2].abs().sum()>0


def test_depth_and_backbone_dimensions_are_checked():
    r = runtime()
    assert not any(p.requires_grad for p in r.geometry.parameters())
    assert any(p.requires_grad for p in r.depth.parameters())
    with pytest.raises(ValueError):
        api().NativeDepthRuntime(r.geometry, r.body,
              __import__('src.pose_control.qwen21_full.native_depth',fromlist=['Qwen21NativeDepth']).Qwen21NativeDepth())


def test_partial_canvas_coverage_scales_the_body_residual():
    from src.pose_control.qwen21_full.native_geometry import NativeGeometryFeatures
    system = runtime()
    body = system.body
    for head in body.output_heads:
        torch.nn.init.normal_(head.weight,std=.03)
    dim = body.config.geometry_dim
    features = NativeGeometryFeatures(torch.randn(1,4,dim),torch.zeros(1,4,dim),(2,2),torch.tensor([[True,False]]))
    prepared = body.prepare_geometry(features)
    hidden = torch.randn(1,4,16)
    kwargs = {"prepared":prepared,"layer":0,"sigma":torch.tensor([.5]),"target_mask":torch.ones(4,dtype=torch.bool)}
    full = body(hidden,**kwargs)
    coverage = torch.tensor([[1.,.5,0.,1.]])
    partial = body(hidden,**kwargs,valid_mask=coverage)
    torch.testing.assert_close(partial-hidden,(full-hidden)*coverage[...,None],rtol=1e-5,atol=2e-7)


def test_training_entry_refuses_ambiguous_depth_configuration():
    from scripts.train.train_qwen21_native_depth import arguments, validate_arguments
    argv = ["--pairs", "pairs.jsonl", "--output", "run", "--full-checkpoint", "full.pt"]
    args = arguments(argv + ["--depth-mode", "relative_png"])
    with pytest.raises(ValueError, match="manifest"):
        validate_arguments(args)
    args = arguments(argv + ["--depth-mode", "off", "--champ-checkpoint", "champ.pth"])
    with pytest.raises(ValueError, match="off"):
        validate_arguments(args)


def test_training_preflight_checks_depth_and_rejects_unpaired_gt(tmp_path):
    from scripts.train.train_qwen21_native_depth import preflight_samples
    records = api().NativeDepthRecords(manifest(tmp_path), mode="relative_png")
    report = preflight_samples([sample()], records, count=8)
    assert report["checked_pairs"] == 1 and report["depth_kind"] == "relative_png"
    s = sample()
    s["pair_metadata"]["cross_person"] = True
    with pytest.raises(ValueError, match="ground truth"):
        preflight_samples([s], records, count=1)


def test_real_qwen_checkpoint_backward_reaches_depth_without_changing_prefix():
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.depth_condition import relative_depth_condition
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    system = runtime()
    model = QwenImage21Transformer2DModel(in_channels=64, out_channels=64, num_layers=32,
            num_attention_heads=2, attention_head_dim=8, context_in_dim=8,
            axes_dims_rope=(2,2,4), mlp_ratio=2).eval()
    original = list(model.parameters())
    install_native_body(model, system.body)
    for head in system.body.output_heads:
        torch.nn.init.normal_(head.weight, std=.03)
    torch.nn.init.normal_(system.depth.projection.weight, std=.02)
    batch = {"normal": torch.rand(1,3,32,64)*2-1, "pose_heatmap": torch.rand(1,25,32,64),
             "part_onehot": torch.zeros(1,14,32,64), "human_mask": torch.ones(1,1,32,64),
             "smplx_global": torch.randn(1,26), "task_id": torch.tensor([0])}
    batch["part_onehot"][:,1] = 1
    d = torch.rand(1,1,32,64)
    depth = relative_depth_condition(d,d.bool(),d.bool(),
        [{"source_id":"pair", "canvas_id":"canvas", "camera_id":"camera", "units":"unknown", "scale_status":"unknown"}],
        polarity="near_one")
    f = system.encode(batch,target_token_hw=(2,4),depth=depth)
    img_mask = torch.tensor([[True,False,False,True,True]])
    values = {"hidden_states":torch.randn(1,12,64), "encoder_hidden_states":torch.randn(1,3,8),
              "encoder_hidden_states_mask":torch.ones(1,3,dtype=torch.bool), "timestep":torch.tensor([.5]),
              "img_shapes":[[(1,2,2),(1,2,4)]], "img_mask":img_mask,"return_dict":False}
    with torch.no_grad():
        before = model(**values,native_enabled=False)[0]
    model.enable_gradient_checkpointing()
    with bind_native_body(model,f,valid_mask=torch.ones(1,8)):
        after = model(**values)[0]
    assert torch.equal(after[:,:4],before[:,:4])
    assert not torch.equal(after[:,4:],before[:,4:])
    after[:,4:].square().mean().backward()
    assert system.depth.encoder.conv_in.weight.grad.norm()>0
    assert system.depth.encoder.attention.proj_in.weight.grad.norm()>0
    assert all(head.weight.grad.norm()>0 for head in system.body.output_heads)
    assert all(not p.requires_grad and p.grad is None for p in original)

from types import SimpleNamespace
from dataclasses import replace
import json

import pytest
import torch
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from src.pose_control.contact_hand_adapter.conditions import HandCondition


def _pose_condition():
    canvas = CanvasSpec((32, 32), (32, 32), (32, 32))
    joints = torch.zeros(4, 21, 3)
    joints[1, :2] = torch.tensor([[8., 16., 1.], [12., 16., 1.]])
    valid_2d = joints[..., 2].bool()
    visible = valid_2d.clone()
    visible[1, 1] = False
    joints_3d = torch.zeros(4, 21, 3)
    joints_3d[1, :2] = torch.tensor([[.1, .2, 1.], [.15, .2, 1.]])
    return HandCondition(canvas, ("alice", None), joints, visible, joints_3d=joints_3d,
                         joint_3d_valid=valid_2d, joint_2d_valid=valid_2d,
                         joint_occluded=valid_2d & ~visible).validate()


def test_pose_supervision_retains_occluded_3d_and_excludes_its_2d_label():
    from src.pose_control.contact_hand_adapter.objectives import pose_output_labels

    labels = pose_output_labels(_pose_condition())
    pose_2d, seen = labels["pose_2d"]
    pose_3d, known = labels["pose_3d"]
    torch.testing.assert_close(pose_2d[0, 1, 0], torch.tensor([.25, .5]))
    assert seen[0, 1, 0].all() and not seen[0, 1, 1].any()
    assert known[0, 1, 1].all() and not known[:, (0, 2, 3)].any() and not known[:, :, 0].any()
    torch.testing.assert_close(pose_3d[0, 1, 1], torch.tensor([.05, 0., 0.]))
    assert set(labels) == {"pose_2d", "pose_3d"}


def test_pose_3d_output_labels_skip_joints_outside_the_rendered_canvas():
    from src.pose_control.contact_hand_adapter.objectives import pose_output_labels

    sample = _pose_condition()
    sample.joints[1, 1, 0] = 80.
    cropped = replace(sample, joint_occluded=torch.zeros_like(sample.visible)).validate()
    assert not pose_output_labels(cropped)["pose_3d"][1].any()


def test_unreachable_sigma_gate_rejected_during_preflight(capsys):
    from scripts.train.train_contact_hand_adapter import main

    with pytest.raises(SystemExit, match="2"):
        main(["--manifest", "unused.jsonl", "--check-data", "--objective-weight", "appearance=.1",
              "--rgb-sigma-max", ".00001"])
    assert "minimum sampled sigma" in capsys.readouterr().err


def test_unknown_relation_labels_have_zero_gradient_and_detached_heads_fail():
    from src.pose_control.contact_hand_adapter.objectives import (
        check_output_gradients, eligible_output_labels, output_supervision,
    )

    parameter = torch.nn.Parameter(torch.tensor([.3, -.5]))
    clean = parameter * 2
    labels = {"occlusion": (torch.tensor([1., float("nan")]), torch.tensor([True, False])),
              "collision": (torch.tensor([0., -1.]), torch.tensor([True, False]))}
    assert set(eligible_output_labels(labels)) == {"occlusion", "collision"}
    losses = output_supervision({"occlusion": clean, "collision": clean}, labels)
    gradients = check_output_gradients(clean, losses, adapter_parameters=(parameter,))
    assert all(values["adapter"] > 0 and values["clean"] > 0 for values in gradients.values())
    sum(losses.values()).backward()
    assert parameter.grad[0] != 0 and parameter.grad[1] == 0
    with pytest.raises(ValueError, match="collision"):
        check_output_gradients(clean, {"collision": losses["collision"].detach()}, adapter_parameters=(parameter,))


def test_appearance_and_perceptual_use_only_qualified_hand_pixels():
    from src.pose_control.contact_hand_adapter.objectives import ImageObjectiveData, image_output_supervision

    mask = torch.zeros((32, 32), dtype=torch.bool)
    mask[4:12, 4:12] = True
    reference = Image.new("RGB", (32, 32), "white")
    data = ImageObjectiveData(appearance=((mask, torch.tensor([.2, .3, .4])),), perceptual=(mask,))
    rgb = torch.full((1, 3, 32, 32), .4, requires_grad=True)
    extractor = lambda x: torch.cat((x.mean((2, 3)), x.amax((2, 3))), dim=-1)
    terms = image_output_supervision(rgb, reference, data, ("appearance", "perceptual"), extractor)
    assert terms["appearance"] > 0 and terms["perceptual"] > 0
    sum(terms.values()).backward()
    assert rgb.grad[..., mask].any() and not rgb.grad[..., ~mask].any()


def test_source_appearance_matches_semantic_part_and_rejects_unknown_source_mask():
    from src.pose_control.contact_hand_adapter.objectives import image_objective_data

    base = _pose_condition()
    dense = torch.zeros(4, 5, 32, 32)
    known = torch.zeros_like(dense, dtype=torch.bool)
    dense[1, 4, 4:10, 4:10] = 5 / 15
    known[1, 4, 4:10, 4:10] = True
    support = known[:, 4].float()
    source = torch.zeros(4, 3, 8, 8)
    source[1, :, 1:7, 1:7] = .6
    source_parts = torch.full((4, 8, 8), -1)
    source_parts[1, 1:7, 1:7] = 5
    condition = SimpleNamespace(canvas=base.canvas, geometry_maps=dense, map_valid=known, appearance=source,
                                appearance_valid=torch.tensor([False, True, False, False]),
                                appearance_parts=source_parts, appearance_mask=source_parts >= 0)
    pair = SimpleNamespace(condition=condition, regions=SimpleNamespace(hands=support))
    data = image_objective_data(pair)
    assert len(data.appearance) == 1 and len(data.perceptual) == 1
    torch.testing.assert_close(data.appearance[0][1], torch.full((3,), .6))
    condition.appearance_mask.zero_()
    assert not image_objective_data(pair).appearance


def test_objective_weight_parser_rejects_invalid_duplicate_and_zero_flow():
    from src.pose_control.contact_hand_adapter.objectives import objective_weights

    assert objective_weights(["pose_3d=.2", "perceptual=.1"]) == {"flow": 1., "pose_3d": .2, "perceptual": .1}
    for options in (["bad=.1"], ["flow=0"], ["appearance=nan"], ["contact=1", "contact=0"]):
        with pytest.raises(ValueError):
            objective_weights(options)


def _manifest_row(root, name, *, appearance_sha=None):
    from src.pose_control.contact_hand_adapter.conditions import dino_crop_fingerprint
    from src.pose_control.contact_hand_adapter.runtime import condition_record
    from src.pose_edit.common import sha256_file

    source, target, geometry = (root / f"{name}-{suffix}" for suffix in ("source.png", "target.png", "geometry.pt"))
    Image.new("RGB", (32, 32), "black").save(source)
    Image.new("RGB", (32, 32), "white").save(target)

    def asset(path):
        return {"path": path.name, "sha256": sha256_file(path)}

    source_info = asset(source)
    condition = _pose_condition()
    if appearance_sha is not None:
        crop, mask = torch.zeros(4, 3, 8, 8), torch.zeros(4, 8, 8, dtype=torch.bool)
        valid, features = torch.tensor([False, True, False, False]), torch.zeros(4, 2, 256)
        crop[1] = .5
        mask[1] = True
        features[1] = 1.
        digest = dino_crop_fingerprint(source_info["sha256"], appearance_sha, crop[1], mask[1], features[1])
        condition = replace(condition, appearance=crop, appearance_mask=mask, appearance_valid=valid,
                            appearance_sources=(None, source_info["sha256"], None, None), appearance_dino=features,
                            appearance_dino_valid=valid, appearance_dino_model_sha=appearance_sha,
                            appearance_dino_crop_sha=(None, digest, None, None))
    torch.save(condition_record(condition), geometry)
    return {"id": name, "group": "clip", "split": "train", "mode": "global", "prompt": "Edit the hand.",
            "source_refs": [source_info], "target": asset(target), "condition": asset(geometry)}


def test_cli_preflight_detects_missing_independent_collision_labels(tmp_path, capsys):
    from scripts.train.train_contact_hand_adapter import main

    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps(_manifest_row(tmp_path, "pair")) + "\n", encoding="utf-8")
    evaluator = tmp_path / "head.ts"
    evaluator.write_bytes(b"Preflight only hashes the evaluator; training loads its TorchScript.")
    args = ["--manifest", str(manifest), "--check-data", "--rgb-evaluator", str(evaluator)]
    main([*args, "--objective-weight", "pose_2d=.1"])
    assert "'pose_2d': 1" in capsys.readouterr().err
    with pytest.raises(SystemExit) as status:
        main([*args, "--objective-weight", "collision=.1"])
    assert status.value.code == 2 and "No qualified reachable training rows for collision" in capsys.readouterr().err


def test_cli_preflight_locks_one_cached_appearance_model_sha(tmp_path, capsys):
    from scripts.train.train_contact_hand_adapter import main

    manifest = tmp_path / "pairs.jsonl"
    records = [_manifest_row(tmp_path, "first", appearance_sha="a" * 64),
               _manifest_row(tmp_path, "second", appearance_sha="b" * 64)]
    manifest.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    args = ["--manifest", str(manifest), "--check-data"]
    with pytest.raises(SystemExit) as status:
        main(args)
    assert status.value.code == 2 and "single frozen appearance DINO model SHA256" in capsys.readouterr().err
    main([*args, "--no-appearance-dino"])
    assert "appearance_dino_model_sha=None" in capsys.readouterr().err


def test_local_dino_loader_keeps_input_gradients(tmp_path):
    pytest.importorskip("transformers")
    from transformers import Dinov2Config, Dinov2Model
    from src.pose_control.contact_hand_adapter.objectives import load_perceptual_features

    config = Dinov2Config(image_size=224, patch_size=14, hidden_size=16,
                          num_hidden_layers=1, num_attention_heads=4, intermediate_size=32)
    Dinov2Model(config).save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [.485, .456, .406], "image_std": [.229, .224, .225]}), encoding="utf-8")
    extractor = load_perceptual_features(tmp_path, device="cpu")
    rgb = torch.rand(1, 3, 18, 18, requires_grad=True)
    features = extractor(rgb)
    features[0, 1, 0].backward()
    assert features.shape == (1, 257, 16)
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all() and rgb.grad.any()

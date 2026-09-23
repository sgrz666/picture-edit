import torch

from src.pose_control.v6.hand.geometry import axis_angle45_to_6d96, hand_heatmaps, split_iper_134_hand_points
from src.pose_control.v6.hand.conditions import HandFineCondition, HandReferenceFeatures
from src.pose_control.v6.hand.preparer import HandConditioningPreparer
from src.pose_control.v6.hand.adapter import HandControlAdapter
from src.pose_control.v6.hand.adapter import scatter_hand_roi_residuals
from src.pose_control.v6.checkpoint import build_v65_checkpoint, build_v66_checkpoint, load_v65_checkpoint, load_v66_checkpoint, freeze_for_hand_training
from src.pose_control.v6.deepgen_adapter import UnifiedSMPLXAdapterV6
from src.pose_control.v6.control_core import BranchControlResiduals
from src.pose_control.v6.interface.strength import ControlStrengthController
from src.pose_control.v6.conditions import AdapterIdentityCondition
from test_deepgen_control_interface_v1 import _control_state
from src.pose_control.v6.hand.quality import hand_quality_gate, merge_vitpose_keypoints
from src.pose_control.v6.hand.refiner_export import hand_refiner_payload
from scripts.precompute_iper_hand_features_v66 import build_cache


def test_hand_pose_and_confident_heatmap():
    pose = torch.zeros(1, 2, 2, 45)
    six = axis_angle45_to_6d96(pose)
    assert six.shape == (1, 2, 2, 96)
    assert torch.allclose(six[..., :6], torch.tensor([1., 0., 0., 0., 1., 0.]))
    points = torch.zeros(1, 2, 2, 21, 3)
    points[..., 0, :] = torch.tensor([.5, .5, 1.])
    heatmaps = hand_heatmaps(points, size=16)
    assert heatmaps.shape == (1, 2, 2, 21, 16, 16)
    assert torch.count_nonzero(heatmaps[..., 1:, :, :]) == 0
    whole = torch.arange(134).float()[None, :, None].expand(1, 134, 3)
    sliced = split_iper_134_hand_points(whole)
    assert sliced.shape == (1, 2, 21, 3)
    assert sliced[0, 0, 0, 0] == 92 and sliced[0, 1, 0, 0] == 113


def _inputs():
    keypoints = torch.zeros(1, 2, 2, 21, 3)
    keypoints[0, 0, 0, :, :2] = .5
    keypoints[0, 0, 0, :, 2] = 1
    boxes = torch.zeros(1, 2, 2, 4)
    boxes[0, 0, 0] = torch.tensor([.25, .25, .75, .75])
    valid = torch.zeros(1, 2, 2, dtype=torch.bool)
    valid[0, 0, 0] = True
    condition = HandFineCondition(
        hand_keypoints=keypoints, hand_pose=torch.zeros(1, 2, 2, 45),
        source_boxes=boxes, target_boxes=boxes, region_valid=valid,
        source_indices=torch.tensor([[3, 0]]),
        mesh_bps=torch.zeros(1, 2, 2, 1024),
        mesh_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
        depth=torch.zeros(1, 2, 2, 1, 16, 16),
        visibility=torch.ones(1, 2, 2),
    )
    patches = torch.randn(1, 2, 2, 1, 16, 1536)
    references = HandReferenceFeatures(
        dino_patches=patches,
        reference_valid=torch.tensor([[[[True], [False]], [[False], [False]]]]),
    )
    return condition, references


def test_hand_branch_zero_init_and_invalid_gradients():
    condition, references = _inputs()
    references.dino_patches.requires_grad_(True)
    preparer = HandConditioningPreparer(dim=32, resampler_depth=1)
    prepared = preparer(condition, references)
    assert prepared.spatial_features.shape == (1, 2, 2, 256, 32)
    assert prepared.appearance_tokens.shape == (1, 2, 2, 8, 32)
    assert torch.count_nonzero(prepared.spatial_features[~condition.region_valid]) == 0
    adapter = HandControlAdapter(dim=32, deepgen_dim=48, heads=4)
    zero = adapter(prepared, target_token_hw=(8, 8), timestep=torch.tensor([500.]))
    assert len(zero) == 6
    assert all(x.shape == (1, 64, 48) and torch.count_nonzero(x) == 0 for x in zero)
    with torch.no_grad():
        adapter.zero_heads[0].weight.fill_(.01)
    out = adapter(prepared, target_token_hw=(8, 8), timestep=torch.tensor([500.]))
    out[0].sum().backward()
    assert torch.count_nonzero(references.dino_patches.grad[~references.reference_valid[..., None, None].expand_as(references.dino_patches)]) == 0


def test_mode_fusion_is_exclusive_and_preserves_global():
    zeros = tuple(torch.zeros(1, 4, 3) for _ in range(6))
    geometry = tuple(torch.ones(1, 4, 3) for _ in range(6))
    legacy = tuple(torch.full((1, 4, 3), 2.) for _ in range(6))
    hand = tuple(torch.full((1, 4, 3), 3.) for _ in range(6))
    raw = BranchControlResiduals(geometry, zeros, (2, 2), detail=legacy, hand=hand)
    controller = ControlStrengthController(enable_hand=True)
    masks = torch.zeros(1, 2, 3, 2, 2)
    masks[:, :, 1:] = 1
    off, _ = controller(raw, denoise_progress=1., hand_mode="off", detail_region_masks=masks)
    old, _ = controller(raw, denoise_progress=1., hand_mode="legacy", detail_region_masks=masks)
    new, _ = controller(raw, denoise_progress=1., hand_mode="v66", detail_region_masks=masks)
    assert torch.equal(off[0], geometry[0])
    assert torch.all(old[0] > off[0])
    assert torch.all(new[0] > off[0])
    assert torch.all(new[0] < old[0])  # first-group prior 0.1: 3*.1 < 2


def test_v65_migration_only_new_hand_keys_and_v66_roundtrip():
    old = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4, enable_face_adapter=True, face_dim=16)
    new = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4, enable_face_adapter=True, enable_hand_adapter=True, face_dim=16)
    checkpoint = build_v65_checkpoint(old)
    load_v65_checkpoint(new, checkpoint)
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert all(torch.count_nonzero(head.weight) == 0 for head in new.control_interface.hand_adapter.zero_heads)
    selected = freeze_for_hand_training(new, warmup=True)
    assert selected and all(p.requires_grad for p in selected)
    assert all(not p.requires_grad for p in new.control_interface.control_core.parameters())
    saved = build_v66_checkpoint(new)
    reloaded = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=16, geometry_channels=4, enable_face_adapter=True, enable_hand_adapter=True, face_dim=16)
    load_v66_checkpoint(reloaded, saved)
    for key, value in new.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key]), key


def test_unified_v66_preparation_and_mode_switch_preserve_body_raw_residuals():
    model = UnifiedSMPLXAdapterV6.build_tiny(
        hidden_dim=32, geometry_channels=16, context_input_dim=24,
        pooled_input_dim=20, num_heads=4, enable_hand_adapter=True, face_dim=32,
    ).eval()
    condition, references = _inputs()
    prepared = model.prepare_control(
        state=_control_state(dual=True),
        identity_condition=AdapterIdentityCondition(
            source_person_latents=torch.randn(1, 2, 16, 8, 12),
            source_indices=torch.tensor([[3, 0]]),
        ),
        source_scene_latents=torch.randn(1, 16, 8, 12),
        target_latent_hw=(8, 12), hand_condition=condition, hand_features=references,
    )
    assert prepared.hand is not None and prepared.detail is not None
    kwargs = dict(target_latents=torch.randn(1, 16, 8, 12), prepared=prepared,
                  encoder_hidden_states=torch.randn(1, 7, 24),
                  pooled_projections=torch.randn(1, 20), timestep=torch.tensor([500]))
    off = model.control_interface.build_raw_residuals(**kwargs, hand_mode="off")
    new = model.control_interface.build_raw_residuals(**kwargs, hand_mode="v66")
    assert new.hand is not None and off.hand is None
    for branch in ("geometry", "interaction"):
        for old_value, new_value in zip(getattr(off, branch), getattr(new, branch)):
            torch.testing.assert_close(old_value, new_value)
    with torch.no_grad():
        model.control_interface.hand_adapter.zero_heads[0].weight.fill_(.01)
    output = model(
        **kwargs, cond_hidden_states=None, denoise_progress=1.0,
        hand_mode="v66", face_strength=0,
    )
    assert len(output.block_controlnet_hidden_states) == 6
    assert torch.count_nonzero(output.block_controlnet_hidden_states[0]) > 0


def test_quality_gate_and_precompute_use_only_source_crops():
    points = torch.zeros(2, 2, 21, 3)
    points[..., :2] = .5
    points[..., 2] = 1
    boxes = torch.tensor([.2, .2, .8, .8]).expand(2, 3, 4).clone()
    valid = torch.ones(2, 3, dtype=torch.bool)
    boxes[1, 1] = torch.tensor([.5, .5, .51, .51])
    assert hand_quality_gate(points, boxes[:, 1:])[1, 0]
    refined = points.clone()
    refined[..., :2] += .01
    selected = merge_vitpose_keypoints(points, refined, hand_quality_gate(points, boxes[:, 1:]))
    assert torch.equal(selected[0], points[0])
    class FakeExtractor:
        path = "fake-dino"
        def extract(self, crop):
            return torch.ones(256, 1536, dtype=torch.float16) * crop.mean()
    detail = {"stem_to_idx": {"source": 0, "target": 1}, "hand_keypoints": points,
              "boxes": boxes, "region_valid": valid,
              "reference_crops": torch.ones(2, 3, 3, 224, 224)}
    cache = build_cache(detail, FakeExtractor(), vitpose={"keypoints": refined})
    assert cache["dino_patches"].shape == (2, 2, 256, 1536)
    assert cache["metadata"]["quality_flagged"] == 1
    assert torch.equal(cache["hand_keypoints"][0], points[0])
    assert not torch.equal(cache["hand_keypoints"][1, 0], points[1, 0])


def test_optional_refiner_export_never_modifies_image():
    condition, _ = _inputs()
    image = torch.randn(1, 3, 32, 32)
    payload = hand_refiner_payload(image, condition)
    assert payload["image"] is image
    assert payload["hand_masks"].shape == (1, 2, 2, 32, 32)
    assert torch.count_nonzero(payload["hand_masks"][~condition.region_valid]) == 0


def test_overlap_blends_by_visibility_and_depth_without_double_strength():
    local = torch.zeros(1, 2, 2, 4, 4, 1)
    local[0, 0, 0] = 1
    local[0, 0, 1] = 3
    masks = torch.ones(1, 2, 2, 4, 4)
    boxes = torch.tensor([0., 0., 1., 1.]).expand(1, 2, 2, 4).clone()
    valid = torch.tensor([[[True, True], [False, False]]])
    visibility = torch.ones(1, 2, 2)
    depth = torch.zeros(1, 2, 2, 1, 4, 4)
    equal = scatter_hand_roi_residuals(local, masks, boxes, valid, visibility, (4, 4), depth)
    assert torch.allclose(equal, torch.full_like(equal, 2.))
    depth[0, 0, 1] = 1
    front = scatter_hand_roi_residuals(local, masks, boxes, valid, visibility, (4, 4), depth)
    assert torch.all(front < equal)
    assert torch.all(front >= 1.)

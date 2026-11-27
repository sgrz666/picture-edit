from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from src.pose_control.contact_hand_adapter import AdapterConfig, ContactHandAdapter, HandCondition, install_hand_adapter
from src.pose_control.contact_hand_adapter.runtime import (
    condition_record, generate, load_adapter, load_condition, make_local_plan, save_adapter,
)


@pytest.fixture(autouse=True, scope="module")
def small_cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def scene():
    canvas = CanvasSpec((56, 224), (56, 224), (64, 256), (16, 4))
    joints = torch.zeros(4, 21, 3)
    visible = torch.zeros(4, 21, dtype=torch.bool)
    support = torch.zeros(4, *canvas.canvas_hw)
    for slot, x in ((0, 28), (1, 130), (3, 206)):
        joints[slot, :, :] = torch.tensor([x + 2, 32, 1.])
        visible[slot, :] = True
        support[slot, 28:36, x:x + 8] = 1.
    return HandCondition(canvas, ("person-a", "person-b"), joints, visible, support=support)


class PaintCrop:
    _execution_device = "cpu"

    def __init__(self):
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

        self.transformer = QwenImage21Transformer2DModel(num_layers=1, attention_head_dim=16,
                                                        num_attention_heads=2, context_in_dim=32,
                                                        axes_dims_rope=(2, 6, 8)).eval()
        adapter = ContactHandAdapter(AdapterConfig(width=16, heads=4, base_width=32, base_heads=2, layers=(0,),
                                                   intra_layers=1, inter_layers=1, roi_size=32, rank=4))
        install_hand_adapter(self.transformer, adapter)
        self.slots = []

    def __call__(self, *, width, height, image, **kwargs):
        binding = self.transformer._contact_hand_binding.get()
        self.slots.append(tuple(binding.condition.visible.any(1).nonzero().flatten().tolist()))
        assert len(image) == 2 and image[-1].size == (width, height)
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (80 * len(self.slots), 0, 0))])


def test_default_local_pass_visits_every_target_component():
    sample = scene()
    pipe = PaintCrop()
    base = Image.new("RGB", (256, 64), "black")
    result = generate(pipe, sample, prompt="Repair the hands", references=[base], seed=5, strength=0.,
                      local_context=base, local_size=64)
    assert pipe.slots == [(0,), (1,), (3,)]
    assert [result.getpixel((x, 32)) for x in (29, 131, 207)] == [(80, 0, 0), (160, 0, 0), (240, 0, 0)]


def test_current_and_target_masks_are_dilated_merged_and_strictly_composited():
    sample = scene()
    pipe = PaintCrop()
    base = Image.new("RGB", (256, 64), "black")
    current = torch.zeros_like(sample.support)
    current[0, 28:36, 121:126] = 1.
    current[1, 28:36, 197:202] = 1.
    current[3, 28:36, :18] = 1.
    generate(pipe, sample, prompt="Repair the hands", references=[base], seed=5, strength=0.,
             local_context=base, local_size=64, current_support=current, local_dilation=2)
    assert len(pipe.slots) == 3
    # Three empty pixel columns between an old and target support merge after radius-two dilation.
    current[0, 28:36, 121:127] = 1.
    current[1, 28:36, 197:203] = 1.
    pipe.slots.clear()
    result = generate(pipe, sample, prompt="Repair the hands", references=[base], seed=5, strength=0.,
                      local_context=base, local_size=64, current_support=current, local_dilation=2)
    assert pipe.slots == [(0, 1, 3)]
    for x in (16, 28, 119, 131, 195, 207):
        assert result.getpixel((x, 32)) == (80, 0, 0)
    for x in (15, 50, 118, 145, 194, 216):
        assert result.getpixel((x, 32)) == (0, 0, 0)


@pytest.mark.parametrize("current", [torch.zeros(4, 32, 64), torch.full((4, 64, 256), float("nan"))])
def test_explicit_current_support_must_match_the_full_canvas(current):
    with pytest.raises(ValueError, match="Current hand support"):
        make_local_plan(scene(), current_support=current)


def test_current_only_group_can_be_removed_without_target_hand_geometry():
    sample = scene()
    pipe = PaintCrop()
    base = Image.new("RGB", (256, 64), "black")
    current = torch.zeros_like(sample.support, dtype=torch.bool)
    current[2, 30:35, 76:80] = True
    result = generate(pipe, sample, prompt="Clear the old hand", references=[base], seed=5, strength=0.,
                      local_context=base, local_size=64, local_slots=(2,), current_support=current)
    assert pipe.slots == [()]
    assert result.getpixel((78, 32)) == (80, 0, 0)
    assert result.getpixel((132, 32)) == (0, 0, 0)


def test_cli_exposes_current_support_in_target_canvas_coordinates(capsys):
    from scripts.run.run_contact_hand_adapter import main

    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    output = capsys.readouterr().out
    assert "--current-support" in output and "--local-dilation" in output


def test_local_plan_keeps_occluded_geometry_for_selected_slots():
    sample = scene()
    valid_3d = sample.visible.clone()
    visible = sample.visible.clone()
    visible[0, 3:5] = False
    joints = sample.joints.clone()
    joints[0, 4, 0] = 100
    sample = replace(sample, joints=joints, visible=visible, joint_2d_valid=sample.visible,
                     joint_occluded=sample.visible & ~visible, joint_3d_valid=valid_3d,
                     joints_3d=torch.rand(4, 21, 3), hand_present=valid_3d.any(-1))
    crop = make_local_plan(sample, size=64, slots=(0,), dilation=0).condition
    assert crop.projected_joints()[0, 3] and crop.joint_occluded[0, 3]
    assert not crop.active_joints()[0, 3]
    assert not crop.projected_joints()[0, 4] and not crop.joint_occluded[0, 4]
    assert crop.joint_3d_valid[0, 4]
    assert torch.equal(crop.joints_3d[0], sample.joints_3d[0])
    assert not crop.joint_3d_valid[1:].any() and not crop.hand_present[1:].any()


def test_loads_legacy_condition_and_writes_v21_condition(tmp_path):
    sample = scene()
    assert condition_record(sample)["schema"] == "contact_hand_condition_v2_1"
    record = {key: value for key, value in asdict(sample).items() if value is not None}
    record.update(schema="contact_hand_condition_v2", canvas=sample.canvas.metadata(), mesh=None)
    path = tmp_path / "legacy.pt"
    torch.save(record, path)
    assert torch.equal(load_condition(path).active_joints(), sample.visible)


def test_checkpoint_schemas_pin_the_architecture(tmp_path):
    signature = "4" * 64
    for version, schema in ((2, "contact_hand_adapter_v2"), (21, "contact_hand_adapter_v2_1")):
        config = AdapterConfig(width=16, heads=4, base_width=32, base_heads=2, layers=(0,),
                               intra_layers=1, inter_layers=1, roi_size=32, rank=4, architecture_version=version)
        adapter = ContactHandAdapter(config)
        path = tmp_path / f"hand-v{version}.pt"
        save_adapter(path, adapter, base_sha=signature, step=3)
        record = torch.load(path, weights_only=True)
        assert record["schema"] == schema
        if version == 2:
            record["config"].pop("architecture_version", None)
            torch.save(record, path)
        loaded, metadata = load_adapter(path, base_sha=signature)
        assert loaded.config == config and metadata["step"] == 3
        assert all(torch.equal(weight, loaded.state_dict()[name]) for name, weight in adapter.state_dict().items())

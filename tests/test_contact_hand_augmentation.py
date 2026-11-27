from dataclasses import replace
import json
import math

import pytest
import torch
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from src.pose_control.contact_hand_adapter.conditions import HandCondition, HandMesh, HandPair, verify_appearance_sources
from src.pose_control.contact_hand_adapter.runtime import condition_record
from src.pose_edit.common import sha256_file


SHA_A, SHA_B = "a" * 64, "b" * 64


def _sample(*, two_people):
    canvas = CanvasSpec((32, 64), (32, 64), (32, 64))
    joints = torch.zeros(4, 21, 3)
    visible = torch.zeros(4, 21, dtype=torch.bool)
    support = torch.zeros(4, 32, 64)
    for slot, x in ((1, 17), (3, 35)) if two_people else ((1, 17),):
        joints[slot, :, 0] = x + torch.arange(21) % 5
        joints[slot, :, 1] = 12 + torch.arange(21) // 5
        joints[slot, :, 2] = .9
        visible[slot] = True
        support[slot, 7:22, x - 3:x + 7] = 1.
    return HandCondition(canvas, ("alice", "bob" if two_people else None), joints, visible, support=support)


def _training_pair(condition, *, dual):
    from src.pose_control.contact_hand_adapter.training import RegionSupervision, TrainingPair

    refs = tuple(Image.new("RGB", (64, 32), color) for color in ("red", "blue", "black"))
    if not dual:
        refs = (refs[0], refs[2])
    target = Image.new("RGB", (64, 32), "white")
    return TrainingPair(condition, refs, target, "Edit the hand pose.",
                        RegionSupervision(condition.support, condition.support * .5),
                        None, (SHA_A, SHA_B) if dual else (SHA_A,))


def test_single_actor_moves_to_b_without_mirroring_or_moving_reference():
    from src.pose_control.contact_hand_adapter.training import swap_training_pair

    condition = replace(_sample(two_people=False), appearance=torch.zeros(4, 3, 2, 2),
                        appearance_valid=torch.tensor([False, True, False, False]),
                        appearance_sources=(None, SHA_A, None, None))
    original = _training_pair(condition, dual=False)
    changed = swap_training_pair(original)
    assert changed.person_swap_status == "swapped"
    assert changed.condition.person_ids == (None, "alice")
    assert changed.condition.shared_appearance_source
    verify_appearance_sources(changed.condition, changed.source_hashes)
    assert changed.condition.appearance_sources == (None, None, None, SHA_A)
    assert changed.references == original.references
    assert changed.target is original.target
    torch.testing.assert_close(changed.condition.joints[3], condition.joints[1])
    torch.testing.assert_close(changed.regions.hands, original.regions.hands[[2, 3, 0, 1]])
    torch.testing.assert_close(changed.regions.wrists, original.regions.wrists[[2, 3, 0, 1]])


def test_dual_actor_reorders_refs_and_inverts_relation_in_the_other_hand_frame():
    from src.pose_control.contact_hand_adapter.training import swap_training_pair

    base = _sample(two_people=True)
    state = torch.full((16, 16), -1, dtype=torch.int8)
    state[1, 2] = 1
    confidence = torch.zeros(16, 16)
    confidence[1, 2] = .9
    occlusion = torch.zeros(16, 16, dtype=torch.int8)
    occlusion[1, 2] = 1
    pose = torch.tensor([1., 0., 0., 0., 0., math.pi / 2])
    mesh = HandMesh(torch.zeros(4, 778, 3), torch.tensor([[0], [1]]), torch.zeros(64, 1, dtype=torch.long),
                    torch.ones(4, 64, 2), torch.tensor([False, True, False, True]), (None, "cam", None, "cam"))
    mesh.vertices[1, 0, 0] = 2.
    relation = HandPair((1, 3), state, confidence, occlusion, pose,
                        torch.tensor([[1, 6]]), torch.tensor([[.01, -.5, 1., .9]]),
                        torch.tensor([[.9, .8, .7, .6]]))
    within_person = HandPair((0, 1), state)
    original = _training_pair(replace(base, mesh=mesh, pairs=(relation, within_person),
                                      appearance=torch.zeros(4, 3, 2, 2), appearance_valid=mesh.valid,
                                      appearance_sources=(None, SHA_A, None, SHA_B)), dual=True)
    assert swap_training_pair(original).person_swap_status == "ordered_prompt"
    changed = swap_training_pair(original, swapped_prompt="Use reference 2 first.")
    assert changed.prompt == "Use reference 2 first."
    assert changed.condition.person_ids == ("bob", "alice")
    assert changed.references == (original.references[1], original.references[0], original.references[2])
    assert changed.source_hashes == (SHA_B, SHA_A)
    verify_appearance_sources(changed.condition, changed.source_hashes)
    cross, within = changed.condition.pairs
    assert cross.slots == (1, 3) and within.slots == (2, 3)
    torch.testing.assert_close(cross.relative_pose, torch.tensor([0., 1., 0., 0., 0., -math.pi / 2]), atol=1e-6, rtol=0)
    torch.testing.assert_close(cross.part_state, state.T)
    torch.testing.assert_close(cross.part_confidence, confidence.T)
    torch.testing.assert_close(cross.occlusion, -occlusion.T)
    torch.testing.assert_close(cross.surface_indices, torch.tensor([[6, 1]]))
    torch.testing.assert_close(cross.surface_features, relation.surface_features)
    torch.testing.assert_close(cross.surface_quality, relation.surface_quality)
    torch.testing.assert_close(within.part_state, state)
    assert changed.condition.mesh.vertices[3, 0, 0] == 2
    recovered = swap_training_pair(changed, swapped_prompt=original.prompt)
    torch.testing.assert_close(recovered.condition.pairs[0].relative_pose, pose)


def test_shared_source_and_all_multimodal_fields_follow_the_actors():
    from src.pose_control.contact_hand_adapter.training import swap_training_pair

    base = _sample(two_people=True)
    visible = base.visible.clone()
    visible[1, 2] = False
    occluded = base.visible & ~visible
    rotation_valid = base.visible.clone()
    rotation_valid[:, 4::4] = False
    identity = torch.tensor([1., 0., 0., 0., 1., 0.])
    anatomy = dict(joints_3d=base.joints * .01, joint_3d_valid=base.visible,
                   joint_2d_valid=base.visible, joint_occluded=occluded, hand_present=base.visible.any(1),
                   joint_rotation_6d=identity.repeat(4, 21, 1), joint_rotation_valid=rotation_valid,
                   mano_shape=torch.arange(40.).reshape(4, 10), mano_wrist_rotation_6d=identity.repeat(4, 1),
                   mano_palm_scale=torch.tensor([0., .1, 0., .15]), mano_valid=base.visible.any(1))
    pair = _training_pair(replace(base, **anatomy, visible=visible, shared_appearance_source=True,
                                  appearance=torch.rand(4, 3, 2, 2), appearance_valid=base.visible.any(1),
                                  appearance_sources=(None, SHA_A, None, SHA_A)), dual=True)
    swapped = swap_training_pair(pair)
    assert swapped.references == pair.references and swapped.source_hashes == pair.source_hashes
    verify_appearance_sources(swapped.condition, swapped.source_hashes)
    for name, value in anatomy.items():
        torch.testing.assert_close(getattr(swapped.condition, name), value[[2, 3, 0, 1]])


def test_swap_keeps_ambiguous_rgb_supervision_attached():
    from src.pose_control.contact_hand_adapter.training import swap_training_pair

    original = _training_pair(_sample(two_people=False), dual=False)
    labels = {"joint": (torch.ones(4, 21), torch.ones(4, 21, dtype=torch.bool))}
    original = replace(original, rgb_labels=labels)
    changed = swap_training_pair(original)
    assert changed.person_swap_status == "rgb_labels"
    assert changed.condition is original.condition and changed.rgb_labels is labels


def _manifest(tmp_path, *, kind="synthetic", include_support=True, synthetic_from_source=True):
    condition = _sample(two_people=False)
    source, target, coarse, geometry, support = (
        tmp_path / name for name in ("source.png", "target.png", "coarse.png", "geometry.pt", "support.pt"))
    # Use a textured source so local resampling changes observable pixels.
    rgb = Image.new("RGB", (64, 32))
    rgb.putdata([(x * 3, y * 7, (x + y) % 2 * 255) for y in range(32) for x in range(64)])
    rgb.save(source)
    Image.new("RGB", (64, 32), "white").save(target)
    Image.new("RGB", (64, 32), "gray").save(coarse)
    torch.save(condition_record(condition), geometry)
    torch.save(dict(canvas_sha=condition.canvas.fingerprint(), support=condition.support), support)

    def ref(path):
        return {"path": path.name, "sha256": sha256_file(path)}

    record = {"id": "pair", "group": "clip", "split": "train", "mode": "local", "prompt": "Edit the hand pose.",
              "source_refs": [ref(source)], "context": ref(source if synthetic_from_source else coarse),
              "target": ref(target), "condition": ref(geometry), "context_kind": kind}
    if include_support:
        record["current_support"] = ref(support)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_synthetic_context_uses_only_source_pixels_and_source_support(tmp_path):
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan
    from src.pose_control.contact_hand_adapter.training import load_manifest, load_training_pair

    record, = load_manifest(_manifest(tmp_path))
    synthetic = load_training_pair(record, local_size=64, rng=torch.Generator().manual_seed(7))
    replay = load_training_pair(record, local_size=64, rng=torch.Generator().manual_seed(7), swap_people=True)
    source = load_training_pair({**record, "context_kind": "source"}, local_size=64)
    assert synthetic.context_kind == "synthetic" and source.context_kind == "source"
    assert replay.person_swap_status == "swapped"
    assert synthetic.references[-1].tobytes() == replay.references[-1].tobytes()
    assert synthetic.references[0].tobytes() == source.references[0].tobytes()
    assert synthetic.target.tobytes() == source.target.tobytes()
    affected = torch.tensor(bytearray(synthetic.references[-1].tobytes())).reshape(64, 64, 3)
    clean = torch.tensor(bytearray(source.references[-1].tobytes())).reshape(64, 64, 3)
    changed = affected.ne(clean).any(-1)
    condition = _sample(two_people=False)
    plan = make_local_plan(condition, size=64, current_support=condition.support)
    x0, y0, x1, y1 = plan.box
    allowed = plan.condition.canvas.stack(condition.support[1:2, y0:y1, x0:x1], nearest=True)[0].bool()
    assert changed.any() and not changed[~allowed].any()


@pytest.mark.parametrize(("support", "from_source"), [(False, True), (True, False)])
def test_synthetic_context_requires_bound_source_and_support(tmp_path, support, from_source):
    from src.pose_control.contact_hand_adapter.training import load_manifest

    with pytest.raises(ValueError, match="Synthetic local context"):
        load_manifest(_manifest(tmp_path, include_support=support, synthetic_from_source=from_source))


def test_explicit_mixture_is_reproducible_and_requires_weighted_kinds():
    from src.pose_control.contact_hand_adapter.training import LocalContextMixture

    records = [{"context_kind": kind, "id": f"{kind}:{idx}"}
               for kind in ("source", "coarse", "synthetic") for idx in range(3)]
    mixture = LocalContextMixture(records, (.25, .25, .5))
    rng = torch.Generator().manual_seed(7)
    prefix = [mixture.sample(rng)["context_kind"] for _ in range(20)]
    replay = torch.Generator().set_state(rng.get_state())
    assert set(prefix) == {"source", "coarse", "synthetic"}
    assert [mixture.sample(rng) for _ in range(10)] == [mixture.sample(replay) for _ in range(10)]
    with pytest.raises(ValueError, match="synthetic"):
        LocalContextMixture(records[:6], (.25, .25, .5))


def test_legacy_authored_local_regions_keep_their_bound_crop(tmp_path):
    from src.pose_control.contact_hand_adapter.runtime import make_local_plan
    from src.pose_control.contact_hand_adapter.training import load_manifest, load_training_pair

    manifest = _manifest(tmp_path, kind="source", include_support=False)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row.pop("context_kind")
    old_plan = make_local_plan(_sample(two_people=False), size=64, dilation=0)
    regions = tmp_path / "regions.pt"
    torch.save({"canvas_sha": old_plan.condition.canvas.fingerprint(), "hands": old_plan.condition.support}, regions)
    row["regions"] = {"path": regions.name, "sha256": sha256_file(regions)}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    record, = load_manifest(manifest)
    pair = load_training_pair(record, local_size=64, swap_people=True)
    assert pair.condition.canvas == old_plan.condition.canvas
    assert pair.person_swap_status == "swapped"

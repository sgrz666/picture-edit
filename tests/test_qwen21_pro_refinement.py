from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from test_qwen21_pro_inference import grid_pack
from test_qwen21_pro_encoded import Encoder
from src.pose_control.qwen21_pro.contracts import ProAdapterConfig


def local_pack(coarse, roi, pack=None, source_crops=None):
    from src.pose_control.qwen21_pro.refinement import _crop_signature

    return replace(
        grid_pack((48, 48)) if pack is None else pack,
        provenance={
            "source": (
                {"source_crops": [_crop_signature(crop) for crop in source_crops]}
                if source_crops is not None
                else {} if pack is None else pack.provenance.get("source", {})
            ),
            "geometry": {
                "local_transform": roi.transform.to_dict(),
                "parent_geometry": coarse.manifest["geometry_signature"],
                "person_ids": list(roi.person_ids),
            },
        },
    )


def coarse(tmp_path):
    from src.pose_control.qwen21_pro.inference import SamplingResult
    from src.pose_control.qwen21_pro.refinement import write_coarse_artifact

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text('{"metadata":{"scope":"joint"}}')
    (checkpoint / "state.pt").write_bytes(b"locked weights")
    pixels = np.arange(64 * 96 * 3, dtype=np.uint8).reshape(64, 96, 3)
    from src.pose_control.qwen21_pro.cache import file_sha256

    result = SamplingResult(
        torch.ones(1, 24, 64),
        [Image.fromarray(pixels)],
        {
            "sampling": "native_dynamic_shift_euler_unpatched64_v1",
            "seed": 5,
            "steps": 2,
            "source_fingerprint": "source",
            "geometry_signature": {"mode": "driver_exact", "current": "mesh"},
            "trained_model_signature": {
                "state_sha256": file_sha256(checkpoint / "state.pt"),
                "metadata": {"scope": "joint"},
            },
        },
    )
    return write_coarse_artifact(
        result,
        output_dir=tmp_path / "coarse",
        output_root=tmp_path,
        stage1_checkpoint=checkpoint,
        stage1_signatures={"scope": "joint"},
        source_fingerprint="source",
        geometry_signature={"mode": "driver_exact", "current": "mesh"},
        selection_role="engineering_probe",
    )


def request(tmp_path):
    from src.pose_control.qwen21_pro.refinement import (
        ROITransform,
        prepare_refinement_request,
    )

    base = coarse(tmp_path)
    control = tmp_path / "control.png"
    Image.new("RGB", (96, 64), "blue").save(control)
    source = tmp_path / "source.png"
    Image.new("RGB", (768, 768), "green").save(source)
    mask = tmp_path / "source_mask.png"
    Image.new("L", (768, 768), 255).save(mask)
    from src.pose_control.qwen21_pro.refinement import SourceCrop

    crop = SourceCrop.bind(0, source, mask, original_assets=[source, mask])
    roi = ROITransform.from_box("face", (0,), (10, 8, 42, 40), (64, 96))
    req = prepare_refinement_request(
        base,
        roi=roi,
        source_crops=(crop,),
        geometry_image=control,
        output_dir=tmp_path / "request",
        output_root=tmp_path,
        seed=7,
        model_signature={"trained": "sha"},
        source_bank_fingerprint=grid_pack((48, 48)).source_fingerprint,
        task_condition=local_pack(base, roi, source_crops=(crop,)),
        config=ProAdapterConfig(),
    )
    return req


def test_roi_roundtrip_reflection_clipping_and_local_task_ownership():
    from src.pose_control.qwen21_pro.refinement import (
        ROITransform,
        localize_task_condition,
    )
    from src.pose_control.qwen21_pro.features import RawSourceFeatures
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    roi = ROITransform.from_box(
        "face", (0,), (-12, 8, 42, 40), (64, 96), reflect_x=True
    )
    points = torch.tensor([[0.0, 8.0], [41.0, 39.0]])
    assert torch.allclose(roi.to_canvas(roi.to_local(points)), points, atol=1e-5)
    assert roi.box_xyxy[0] == 0
    pack = grid_pack((4, 6))
    features = RawSourceFeatures(
        (0,),
        torch.ones(1, 1, 256, 1600),
        torch.ones(1, 1, 256, dtype=torch.bool),
        torch.zeros(1, 1, 512),
        torch.zeros(1, 1, dtype=torch.bool),
        "new-source-crop-sha",
        {"source_crops": [{"person_id": 0, "part_id": 0}]},
    )
    localized = localize_task_condition(
        pack, roi, source_features=features, config=ProAdapterConfig(source_dim=1600)
    )
    assert localized.target_hw == (48, 48)
    assert localized.source_fingerprint == "new-source-crop-sha"
    assert localized.owner.shape == (1, 1, 2304)
    localized.validate(ProAdapterConfig(source_dim=1600))


def test_face_and_paired_hand_rois_include_both_stable_contact_people():
    from src.pose_control.qwen21_pro.refinement import derive_refinement_rois
    from test_qwen21_pro_contracts import make_condition
    from src.pose_control.qwen21_pro.contracts import ContactRelations

    pack = replace(make_condition(), person_ids=(41, 7))
    relation = ContactRelations(
        torch.tensor([[0]]),
        torch.tensor([[4]]),
        torch.tensor([[1]]),
        torch.tensor([[7]]),
        torch.zeros(1, 1, 6),
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    pack = replace(pack, relations=relation)
    boxes = (
        {"face": [1, 1, 10, 12], "left_hand": [8, 20, 16, 28]},
        {"face": [30, 1, 40, 12], "right_hand": [20, 24, 30, 34]},
    )
    rois = derive_refinement_rois(pack, boxes, canvas_hw=(48, 48), margin_px=0)
    paired = next(r for r in rois if r.kind == "paired_hands")
    assert paired.person_ids == (41, 7)
    assert paired.box_xyxy == (8, 20, 30, 34)
    switched = replace(pack, person_ids=(7, 41))
    swapped = derive_refinement_rois(switched, boxes, canvas_hw=(48, 48), margin_px=0)
    assert next(r for r in swapped if r.kind == "paired_hands").person_ids == (7, 41)


@pytest.mark.parametrize("native_mode", ["RGB", "RGBA"])
def test_full_native_refiner_uses_fixed_coarse_and_exact_outside_mask(
    tmp_path, native_mode
):
    from src.pose_control.qwen21_pro.refinement import run_refinement

    req = request(tmp_path)

    class Pipeline:
        text_encoder = object()
        _execution_device = "cpu"
        trained_model_signature = {"trained": "sha"}

        def __call__(self, **kwargs):
            assert kwargs["image"].size == (768, 768)
            assert (
                kwargs["height"]
                == kwargs["width"]
                == kwargs["output_resolution"]
                == 768
            )
            assert kwargs["use_kv_cache"] is False
            assert kwargs["cfg_norm_rescale"] is True
            assert kwargs["reference_images"][0].getpixel((0, 0)) == (0, 128, 0)
            return SimpleNamespace(
                images=[
                    Image.new(
                        native_mode,
                        (768, 768),
                        (255, 0, 0, 31) if native_mode == "RGBA" else (255, 0, 0),
                    )
                ]
            )

    result = run_refinement(
        Pipeline(),
        req,
        task_condition=req.task_condition,
        output_dir=tmp_path / "result",
        output_root=tmp_path,
        prompt="refine",
    )
    before = np.array(Image.open(req.coarse.image_path))
    after = np.array(Image.open(result.image_path))
    outside = np.ones(before.shape[:2], bool)
    outside[8:40, 10:42] = False
    assert np.array_equal(before[outside], after[outside])
    assert (after[10:38, 12:40] == [255, 0, 0]).all()
    assert result.audit["outside_edit_mask_bitwise_equal"] is True
    native = Image.open(tmp_path / "result/generated_roi.png")
    assert native.mode == native_mode
    if native_mode == "RGBA":
        assert native.getpixel((0, 0))[3] == 31


def test_refinement_rejects_foreign_loaded_model_or_source_bank(tmp_path):
    from src.pose_control.qwen21_pro.refinement import run_refinement

    req = request(tmp_path)
    pipe = SimpleNamespace(
        text_encoder=object(), trained_model_signature={"trained": "foreign"}
    )
    with pytest.raises(ValueError, match="model signature"):
        run_refinement(
            pipe,
            req,
            task_condition=req.task_condition,
            output_dir=tmp_path / "wrong_model",
            output_root=tmp_path,
        )
    pipe.trained_model_signature = {"trained": "sha"}
    with pytest.raises(ValueError, match="source bank"):
        run_refinement(
            pipe,
            req,
            task_condition=replace(req.task_condition, source_fingerprint="foreign"),
            output_dir=tmp_path / "wrong_source",
            output_root=tmp_path,
        )


def test_roi_source_bank_encodes_each_actual_source_crop(tmp_path):
    from src.pose_control.qwen21_pro.refinement import encode_roi_source_bank

    req = request(tmp_path)

    class Dino(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 257, 1536))

    extractor = SimpleNamespace(
        processor=lambda **kw: {"pixel_values": torch.zeros(1, 3, 224, 224)},
        dino=Dino(),
        device=torch.device("cpu"),
        model_signature={
            "dino": "current-weights",
            "source_vae": "current-vae-weights",
        },
    )
    bank = encode_roi_source_bank(
        Encoder(), extractor, req.source_crops, person_ids=(0,)
    )
    assert bank.tokens.shape == (1, 1, 256, 1600)
    assert bank.valid.all()
    assert (bank.tokens[..., :64] == 1).all()
    assert (
        bank.provenance["source_crops"][0]["files"]["crop"]
        == req.source_crops[0].hashes["crop"]
    )
    extractor.model_signature = {"dino": "current-weights"}
    with pytest.raises(ValueError, match="VAE producer"):
        encode_roi_source_bank(Encoder(), extractor, req.source_crops, person_ids=(0,))


def test_coarse_manifest_cannot_be_gt_or_noisy_gt_and_tiny_roi_falls_back(tmp_path):
    from src.pose_control.qwen21_pro.refinement import (
        CoarseArtifact,
        run_refinement,
        ROITransform,
    )
    import json

    req = request(tmp_path)
    manifest = json.loads(req.coarse.manifest_path.read_text())
    manifest["artifact_kind"] = "noisy_gt"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generated coarse"):
        CoarseArtifact.load(bad)
    tiny = replace(req, roi=ROITransform.from_box("face", (0,), (1, 1, 2, 2), (64, 96)))
    # An immutable request rejects mutation; derive a fresh request for fallback.
    from src.pose_control.qwen21_pro.refinement import prepare_refinement_request

    tiny = prepare_refinement_request(
        req.coarse,
        roi=tiny.roi,
        source_crops=req.source_crops,
        geometry_image=req.geometry_original_path,
        output_dir=tmp_path / "tiny",
        output_root=tmp_path,
        seed=7,
        model_signature={"trained": "sha"},
        source_bank_fingerprint=req.source_bank_fingerprint,
    )
    result = run_refinement(
        object(),
        tiny,
        task_condition=None,
        output_dir=tmp_path / "fallback",
        output_root=tmp_path,
    )
    assert result.audit["fallback_reason"] == "tiny_roi"
    assert np.array_equal(
        np.array(Image.open(result.image_path)),
        np.array(Image.open(req.coarse.image_path)),
    )


def test_refiner_training_factory_attests_coarse_keep_inpaint_and_gt_only_supervision(
    tmp_path,
):
    from src.pose_control.qwen21_pro.refinement import make_refiner_training_sample
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro.encoded import (
        EncodedConditionCache,
        sample_validator,
    )

    req = request(tmp_path)
    gt = tmp_path / "supervision.png"
    Image.new("RGB", (96, 64), "orange").save(gt)
    pack = req.task_condition
    result = make_refiner_training_sample(
        Encoder(),
        req,
        task_condition=pack,
        supervision_image=gt,
        config=ProAdapterConfig(),
        cache=EncodedConditionCache(tmp_path / "cache"),
        sample_id="refiner",
        producer_signature={"current": "encoder"},
        output_dir=tmp_path / "training",
        output_root=tmp_path,
        prompt="refine",
        engineering_probe=True,
    )
    condition, supervision, bound = result
    assert condition["keep_mask"].shape == (1, 2304, 1)
    assert condition["inpaint_latents"].shape == (1, 2304, 64)
    assert condition["keep_mask"].sum() == 0
    assert supervision["target_latent"].shape == (1, 2304, 64)
    assert str(gt) not in str(condition.binding)
    sample_validator(condition, bound, "refiner")
    with pytest.raises(ValueError, match="selected|locked"):
        make_refiner_training_sample(
            Encoder(),
            req,
            task_condition=pack,
            supervision_image=gt,
            config=ProAdapterConfig(),
            cache=EncodedConditionCache(tmp_path / "formal"),
            sample_id="formal",
            producer_signature={"current": "encoder"},
            output_dir=tmp_path / "formal_training",
            output_root=tmp_path,
            prompt="refine",
            engineering_probe=False,
        )


def test_training_inpaint_rgb_zero_after_preprocess_and_loss_only_white_edit(tmp_path):
    from src.pose_control.qwen21_pro.refinement import (
        prepare_refinement_request,
        make_refiner_training_sample,
    )
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro.encoded import EncodedConditionCache

    req = request(tmp_path)
    edit = tmp_path / "edit.png"
    pixels = np.zeros((768, 768), np.uint8)
    pixels[:384] = 255
    Image.fromarray(pixels).save(edit)
    req = prepare_refinement_request(
        req.coarse,
        roi=req.roi,
        source_crops=req.source_crops,
        geometry_image=req.geometry_original_path,
        edit_mask=edit,
        output_dir=tmp_path / "partial",
        output_root=tmp_path,
        seed=7,
        model_signature={"trained": "sha"},
        source_bank_fingerprint=req.source_bank_fingerprint,
        task_condition=req.task_condition,
        config=req.config,
    )
    encoder = Encoder()
    captured = []
    encoder.image_processor.preprocess = lambda image, height, width: torch.full(
        (1, 4, height, width), 0.75
    )
    original = encoder.vae.encode

    def encode(pixels):
        captured.append(pixels.clone())
        return original(pixels)

    encoder.vae.encode = encode
    gt = tmp_path / "gt.png"
    Image.new("RGB", (96, 64), "orange").save(gt)
    condition, supervision, _ = make_refiner_training_sample(
        encoder,
        req,
        task_condition=req.task_condition,
        supervision_image=gt,
        config=ProAdapterConfig(),
        cache=EncodedConditionCache(tmp_path / "partial_cache"),
        sample_id="partial",
        producer_signature={"actual": "frozen"},
        output_dir=tmp_path / "partial_train",
        output_root=tmp_path,
        prompt="refine",
        engineering_probe=True,
    )
    inpaint = captured[-2]
    assert (inpaint[:, :3, :, :384] == 0).all()
    assert (inpaint[:, :3, :, 384:] == 0.75).all()
    assert (inpaint[:, 3:] == 1).all()
    coverage = supervision["valid_target_tokens"].reshape(48, 48)
    assert (coverage[:24] == 1).all() and (coverage[24:] == 0).all()
    assert condition["keep_mask"].sum() == 1152


def test_refiner_edit_mask_respects_original_physical_fov_and_source_order(tmp_path):
    import json
    from src.pose_control.qwen21_pro.geometry import PixelCanvasTransform
    from src.pose_control.qwen21_pro.refinement import (
        CoarseArtifact,
        prepare_refinement_request,
        ROITransform,
    )

    req = request(tmp_path)
    manifest = json.loads(req.coarse.manifest_path.read_text())
    manifest["geometry_signature"]["canvas"] = PixelCanvasTransform.letterbox(
        (64, 32), (64, 96)
    ).to_dict()
    req.coarse.manifest_path.write_text(json.dumps(manifest))
    base = CoarseArtifact.load(req.coarse.manifest_path)
    roi = ROITransform.from_box("face", (0,), (0, 0, 96, 64), (64, 96))
    localized = prepare_refinement_request(
        base,
        roi=roi,
        source_crops=req.source_crops,
        geometry_image=req.geometry_original_path,
        output_dir=tmp_path / "physical",
        output_root=tmp_path,
        seed=5,
        model_signature=req.model_signature,
        source_bank_fingerprint=req.source_bank_fingerprint,
        task_condition=local_pack(base, roi, req.task_condition),
        config=req.config,
    )
    mask = np.array(Image.open(localized.edit_mask_path))
    assert not mask[:, :256].any() and not mask[:, 512:].any()
    assert (mask[:, 256:512] == 255).all()


def test_coarse_artifact_rejects_checkpoint_not_used_by_sampler(tmp_path):
    from src.pose_control.qwen21_pro.refinement import write_coarse_artifact
    from src.pose_control.qwen21_pro.inference import SamplingResult

    coarse(tmp_path)
    result = SamplingResult(
        torch.ones(1, 24, 64),
        [Image.new("RGB", (96, 64))],
        {
            "sampling": "native_dynamic_shift_euler_unpatched64_v1",
            "source_fingerprint": "source",
            "geometry_signature": {"mode": "driver_exact", "current": "mesh"},
            "trained_model_signature": {
                "state_sha256": "wrong-or-unloaded-weights",
                "metadata": {"scope": "joint"},
            },
        },
    )
    with pytest.raises(
        ValueError, match="actually used|model.*checkpoint|checkpoint.*model"
    ):
        write_coarse_artifact(
            result,
            output_dir=tmp_path / "mislabeled",
            output_root=tmp_path,
            stage1_checkpoint=tmp_path / "checkpoint",
            stage1_signatures={"scope": "joint"},
            source_fingerprint="source",
            geometry_signature={"mode": "driver_exact", "current": "mesh"},
            selection_role="engineering_probe",
        )


@pytest.mark.parametrize("entry", ["inference", "training"])
@pytest.mark.parametrize("mutation", ["foreign_affine", "changed_spatial"])
def test_refiner_rejects_foreign_local_geometry_pack_before_native_call(
    tmp_path, entry, mutation
):
    from src.pose_control.qwen21_pro.refinement import (
        run_refinement,
        make_refiner_training_sample,
    )
    from src.pose_control.qwen21_pro.encoded import EncodedConditionCache
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    req = request(tmp_path)
    pack = (
        replace(
            req.task_condition,
            provenance={
                **req.task_condition.provenance,
                "geometry": {"local_transform": {"arbitrary": "wrong"}},
            },
        )
        if mutation == "foreign_affine"
        else replace(
            req.task_condition,
            spatial={
                **req.task_condition.spatial,
                "normal": req.task_condition.spatial["normal"] + 0.123,
            },
        )
    )

    class Pipe:
        text_encoder = object()
        trained_model_signature = {"trained": "sha"}
        _execution_device = "cpu"

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (768, 768), "red")])

    with pytest.raises(ValueError, match="local|geometry|pack|binding"):
        if entry == "inference":
            run_refinement(
                Pipe(),
                req,
                task_condition=pack,
                output_dir=tmp_path / "foreign",
                output_root=tmp_path,
            )
        else:
            gt = tmp_path / "gt.png"
            Image.new("RGB", (96, 64), "blue").save(gt)
            make_refiner_training_sample(
                Encoder(),
                req,
                task_condition=pack,
                supervision_image=gt,
                config=ProAdapterConfig(),
                cache=EncodedConditionCache(tmp_path / "foreign-cache"),
                sample_id="foreign",
                producer_signature={"actual": "frozen"},
                output_dir=tmp_path / "foreign",
                output_root=tmp_path,
                prompt="refine",
                engineering_probe=True,
            )


def test_paired_hand_roi_rejects_face_crops_for_both_people(tmp_path):
    from src.pose_control.qwen21_pro.refinement import (
        ROITransform,
        PartEndpoint,
        SourceCrop,
        prepare_refinement_request,
    )

    req = request(tmp_path)
    crop = req.source_crops[0]
    other = SourceCrop.bind(
        1,
        crop.image_path,
        crop.mask_path,
        original_assets=[crop.image_path, crop.mask_path],
    )
    roi = ROITransform.from_box(
        "paired_hands",
        (0, 1),
        (10, 8, 42, 40),
        (64, 96),
        endpoints=(PartEndpoint(0, 4), PartEndpoint(1, 7)),
    )
    with pytest.raises(ValueError, match="endpoint|part|hand"):
        prepare_refinement_request(
            req.coarse,
            roi=roi,
            source_crops=(crop, other),
            geometry_image=req.geometry_original_path,
            output_dir=tmp_path / "face-substitution",
            output_root=tmp_path,
            seed=5,
            model_signature=req.model_signature,
            source_bank_fingerprint=req.source_bank_fingerprint,
        )


def hand_source_crops(tmp_path):
    from src.pose_control.qwen21_pro.refinement import write_source_crop
    from src.pose_control.qwen21_pro.cache import file_sha256, asset_hashes

    source = tmp_path / "hand_source.png"
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[:, :16] = [220, 20, 10]
    rgb[:, 16:] = [10, 180, 30]
    Image.fromarray(rgb).save(source)
    mask_paths = []
    for slot in (0, 1):
        path = tmp_path / f"hand_mask_{slot}.png"
        mask = np.zeros((32, 32), np.uint8)
        mask[:, slot * 16 : (slot + 1) * 16] = 255
        Image.fromarray(mask).save(path)
        mask_paths.append(path)
    files = {f"source_mask_{slot}": str(path) for slot, path in enumerate(mask_paths)}
    bundle = {
        "person_ids": [0, 1],
        "source_rgb_sha256": file_sha256(source),
        "source_image_wh": [32, 32],
        "asset_files": files,
        "asset_hashes": asset_hashes(files),
        "code_signature": {"unit_fixture": "explicit"},
        "policy": {"min_support_pixels": 8},
        "rois": [
            {
                "face": [2, 1, 10, 3],
                "left_hand": [1, 4, 9, 12],
                "right_hand": [3, 18, 11, 26],
            },
            {
                "face": [22, 1, 30, 3],
                "left_hand": [20, 4, 28, 12],
                "right_hand": [22, 18, 30, 26],
            },
        ],
    }
    return tuple(
        write_source_crop(
            person_id=slot,
            part_id=part,
            source_image=source,
            source_mask=mask_paths[slot],
            box=bundle["rois"][slot][name],
            source_geometry_provenance=bundle,
            output_dir=tmp_path / f"crop_{slot}",
            output_root=tmp_path,
        )
        for slot, part, name in ((0, 4, "left_hand"), (1, 7, "right_hand"))
    )


@pytest.mark.parametrize(
    "mutation", ["correct", "opposite_hand", "face", "other_person", "cropped_pixels"]
)
def test_source_crop_bank_requires_exact_hand_endpoints_and_native_geometry(
    tmp_path, mutation
):
    from src.pose_control.qwen21_pro.refinement import (
        PartEndpoint,
        encode_roi_source_bank,
    )

    crops = hand_source_crops(tmp_path)
    endpoints = (PartEndpoint(0, 4), PartEndpoint(1, 7))

    class Dino(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 257, 1536))

    extractor = SimpleNamespace(
        processor=lambda **kw: {"pixel_values": torch.zeros(1, 3, 224, 224)},
        dino=Dino(),
        device=torch.device("cpu"),
        model_signature={"dino": "current", "source_vae": "current"},
    )
    if mutation == "correct":
        bank = encode_roi_source_bank(
            Encoder(), extractor, crops, person_ids=(1, 0), endpoints=endpoints
        )
        assert bank.person_ids == (1, 0)
        assert [
            (v["person_id"], v["part_id"]) for v in bank.provenance["source_crops"]
        ] == [(1, 7), (0, 4)]
        from src.pose_control.qwen21_pro.refinement import (
            ROITransform,
            prepare_refinement_request,
            localize_task_condition,
            make_refiner_training_sample,
        )
        from src.pose_control.qwen21_pro.encoded import (
            EncodedConditionCache,
            sample_validator,
        )
        from test_qwen21_pro_contracts import make_condition

        base = coarse(tmp_path)
        roi = ROITransform.from_box(
            "paired_hands", (1, 0), (10, 8, 42, 40), (64, 96), endpoints=endpoints
        )
        source_pack = replace(
            make_condition(),
            provenance={"geometry": base.manifest["geometry_signature"]},
        )
        cfg = ProAdapterConfig(source_dim=1600)
        pack = localize_task_condition(
            source_pack, roi, source_features=bank, config=cfg
        )
        control = tmp_path / "geometry.png"
        Image.new("RGB", (96, 64), "blue").save(control)
        req = prepare_refinement_request(
            base,
            roi=roi,
            source_crops=crops,
            geometry_image=control,
            output_dir=tmp_path / "paired_request",
            output_root=tmp_path,
            seed=5,
            model_signature={"trained": "sha"},
            source_bank_fingerprint=bank.fingerprint,
            task_condition=pack,
            config=cfg,
        )
        req.verify_task_condition(pack)
        assert [(crop.person_id, crop.part_id) for crop in req.source_crops] == [
            (1, 7),
            (0, 4),
        ]
        gt = tmp_path / "paired_gt.png"
        Image.new("RGB", (96, 64), "orange").save(gt)
        condition, _, bound = make_refiner_training_sample(
            Encoder(),
            req,
            task_condition=pack,
            supervision_image=gt,
            config=cfg,
            cache=EncodedConditionCache(tmp_path / "paired_cache"),
            sample_id="paired",
            producer_signature={"actual": "frozen"},
            output_dir=tmp_path / "paired_train",
            output_root=tmp_path,
            prompt="paired hands",
            engineering_probe=True,
        )
        sample_validator(condition, bound, "paired")
        assert condition.binding["source_reference_person_ids"] == [1, 0]
        return
    first = crops[0]
    if mutation == "opposite_hand":
        first = replace(first, part_id=7)
    elif mutation == "face":
        first = replace(first, part_id=0)
    elif mutation == "other_person":
        first = replace(first, person_id=1)
    else:
        foreign = tmp_path / "foreign.png"
        Image.new("RGB", (768, 768), "blue").save(foreign)
        from src.pose_control.qwen21_pro.cache import asset_hashes

        files = {**first.files, "crop": str(foreign.resolve())}
        first = replace(
            first, image_path=foreign, files=files, hashes=asset_hashes(files)
        )
    with pytest.raises(ValueError, match="endpoint|part|person|pixels|geometry"):
        encode_roi_source_bank(
            Encoder(),
            extractor,
            (first, crops[1]),
            person_ids=(0, 1),
            endpoints=endpoints,
        )


def test_face_crop_path_cannot_change_without_its_bound_file_role(tmp_path):
    req = request(tmp_path)
    foreign = tmp_path / "foreign_face.png"
    Image.new("RGB", (768, 768), "red").save(foreign)
    with pytest.raises(ValueError, match="role|path|binding"):
        replace(req.source_crops[0], image_path=foreign).verify()


def roi_extractor_fixture():
    class Dino(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 257, 1536))

    return SimpleNamespace(
        processor=lambda **kw: {"pixel_values": torch.zeros(1, 3, 224, 224)},
        dino=Dino(),
        device=torch.device("cpu"),
        model_signature={"dino": "current", "source_vae": "current"},
    )


@pytest.mark.parametrize("parts", [(0, 0), (7, 4)])
def test_localize_paired_hands_rejects_real_face_or_opposite_hand_bank(tmp_path, parts):
    from src.pose_control.qwen21_pro.refinement import (
        PartEndpoint,
        ROITransform,
        write_source_crop,
        encode_roi_source_bank,
        localize_task_condition,
    )
    from test_qwen21_pro_contracts import make_condition

    hands = hand_source_crops(tmp_path)
    bundle = hands[0].source_geometry_provenance
    crops = []
    for slot, part in enumerate(parts):
        prototype = hands[slot]
        name = {0: "face", 4: "left_hand", 7: "right_hand"}[part]
        crops.append(
            write_source_crop(
                person_id=slot,
                part_id=part,
                source_image=prototype.files["source_original_0"],
                source_mask=prototype.files["source_original_1"],
                box=bundle["rois"][slot][name],
                source_geometry_provenance=bundle,
                output_dir=tmp_path / f"foreign_{slot}",
                output_root=tmp_path,
            )
        )
    endpoints = tuple(PartEndpoint(pid, part) for pid, part in enumerate(parts))
    bank = encode_roi_source_bank(
        Encoder(),
        roi_extractor_fixture(),
        crops,
        person_ids=(1, 0),
        endpoints=endpoints,
    )
    roi = ROITransform.from_box(
        "paired_hands",
        (1, 0),
        (10, 8, 42, 40),
        (64, 96),
        endpoints=(PartEndpoint(0, 4), PartEndpoint(1, 7)),
    )
    with pytest.raises(ValueError, match="bank.*endpoint|endpoint.*bank"):
        localize_task_condition(
            make_condition(),
            roi,
            source_features=bank,
            config=ProAdapterConfig(source_dim=1600),
        )


def test_request_rejects_same_hand_endpoint_bank_from_other_native_crop(tmp_path):
    from src.pose_control.qwen21_pro.refinement import (
        PartEndpoint,
        ROITransform,
        write_source_crop,
        encode_roi_source_bank,
        localize_task_condition,
        prepare_refinement_request,
    )
    from test_qwen21_pro_contracts import make_condition

    hands = hand_source_crops(tmp_path)
    first = hands[0]
    altered = write_source_crop(
        person_id=first.person_id,
        part_id=first.part_id,
        source_image=first.files["source_original_0"],
        source_mask=first.files["source_original_1"],
        box=first.original_box,
        source_geometry_provenance=first.source_geometry_provenance,
        crop_margin=0.12,
        output_dir=tmp_path / "same_endpoint_other_crop",
        output_root=tmp_path,
    )
    endpoints = (PartEndpoint(0, 4), PartEndpoint(1, 7))
    bank = encode_roi_source_bank(
        Encoder(),
        roi_extractor_fixture(),
        (altered, hands[1]),
        person_ids=(1, 0),
        endpoints=endpoints,
    )
    base = coarse(tmp_path)
    roi = ROITransform.from_box(
        "paired_hands", (1, 0), (10, 8, 42, 40), (64, 96), endpoints=endpoints
    )
    parent = replace(
        make_condition(), provenance={"geometry": base.manifest["geometry_signature"]}
    )
    cfg = ProAdapterConfig(source_dim=1600)
    local = localize_task_condition(parent, roi, source_features=bank, config=cfg)
    control = tmp_path / "geometry.png"
    Image.new("RGB", (96, 64), "blue").save(control)
    with pytest.raises(ValueError, match="bank.*crop|crop.*bank"):
        prepare_refinement_request(
            base,
            roi=roi,
            source_crops=hands,
            geometry_image=control,
            output_dir=tmp_path / "wrong_bank_request",
            output_root=tmp_path,
            seed=5,
            model_signature={"trained": "sha"},
            source_bank_fingerprint=bank.fingerprint,
            task_condition=local,
            config=cfg,
        )

from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from test_qwen21_pro_geometry import RodBody, request


class Renderer:
    def __init__(self):
        self.states = []

    def render(self, state, camera, body_model, *, canvas, contacts):
        from src.pose_control.qwen21_pro.geometry import compose_dense_geometry

        self.states.append(state)
        h, w = canvas.output_hw
        p = len(state.parameters.person_ids)
        mask = canvas.valid_fov_mask()[None].expand(p, h, w).clone()
        normal = torch.zeros(p, h, w, 3)
        normal[..., 2] = -mask.float()
        depth = mask.float() * state.parameters.betas[:, 0, None, None].add(3)
        return compose_dense_geometry(
            normal,
            depth,
            mask,
            mask.long(),
            torch.zeros_like(depth),
            person_ids=state.parameters.person_ids,
            camera=camera,
            canvas=canvas,
        )


def test_current_source_shape_regenerates_native_control_pose_and_fov(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import prepare_geometry

    renderer = Renderer()
    changed = request(beta_delta=1)
    prepared = prepare_geometry(
        changed,
        body_model=RodBody(),
        renderer=renderer,
        output_dir=tmp_path / "current",
        output_root=tmp_path,
        canvas_hw=(64, 96),
        mode="source_shape",
    )
    assert renderer.states == [changed.solved]
    assert prepared.dense.person_ids == (41, 7)
    assert prepared.manifest["paired_supervision"] is False
    assert prepared.manifest["native_control_style"] == "current_depth_body24"
    assert len(prepared.pose_images) == 2
    assert all(
        Image.open(p).size == (96, 64)
        for p in [prepared.control_image, *prepared.pose_images]
    )
    assert prepared.manifest["geometry_signature"]
    assert not np.asarray(Image.open(prepared.control_image))[:, :5].any()
    with pytest.raises(FileExistsError):
        prepare_geometry(
            changed,
            body_model=RodBody(),
            renderer=renderer,
            output_dir=tmp_path / "current",
            output_root=tmp_path,
            canvas_hw=(64, 96),
            mode="source_shape",
        )


def test_training_exact_pair_guard_rejects_beta_or_solved_changes(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import prepare_geometry

    for mode, training in [("driver_exact", False), ("source_shape", True)]:
        with pytest.raises(ValueError, match="exact|unpaired"):
            prepare_geometry(
                request(beta_delta=1),
                body_model=RodBody(),
                renderer=Renderer(),
                output_dir=tmp_path / mode,
                output_root=tmp_path,
                canvas_hw=(64, 96),
                mode=mode,
                for_training=training,
            )


def test_geometry_typed_pack_replaces_spatial_and_roi_in_current_canvas(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import (
        prepare_geometry,
        geometry_task_condition,
    )
    from test_qwen21_pro_contracts import make_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    prepared = prepare_geometry(
        request(),
        body_model=RodBody(),
        renderer=Renderer(),
        output_dir=tmp_path / "exact",
        output_root=tmp_path,
        canvas_hw=(64, 96),
        mode="driver_exact",
        for_training=True,
    )
    source = replace(make_condition(), person_ids=(41, 7))
    pack = geometry_task_condition(source, prepared, config=ProAdapterConfig())
    assert pack.target_hw == (4, 6)
    assert pack.spatial_valid["normal"].all()
    assert pack.source_fingerprint == source.source_fingerprint
    assert torch.equal(pack.source_tokens, source.source_tokens)
    pack.validate(ProAdapterConfig())


def test_separate_geometry_worker_handoff_loads_current_typed_pack(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import (
        prepare_geometry,
        load_geometry_task_condition,
    )
    from test_qwen21_pro_contracts import make_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig

    prepared = prepare_geometry(
        request(),
        body_model=RodBody(),
        renderer=Renderer(),
        output_dir=tmp_path / "worker",
        output_root=tmp_path,
        canvas_hw=(64, 96),
        mode="driver_exact",
    )
    pack = load_geometry_task_condition(
        replace(make_condition(), person_ids=(41, 7)),
        tmp_path / "worker/manifest.json",
        config=ProAdapterConfig(),
    )
    assert pack.target_hw == (4, 6)
    assert pack.spatial_valid["normal"].all()
    assert pack.provenance["geometry"]["mode"] == "driver_exact"
    Image.new("RGB", (96, 64), "red").save(prepared.control_image)
    with pytest.raises(ValueError, match="geometry|SHA|asset"):
        load_geometry_task_condition(
            replace(make_condition(), person_ids=(41, 7)),
            tmp_path / "worker/manifest.json",
            config=ProAdapterConfig(),
        )


def test_source_only_geometry_cli_never_requires_or_reads_driver_job(
    monkeypatch, tmp_path
):
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api

    req = request()
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 60)).save(source)
    masks = []
    for slot in range(2):
        path = tmp_path / f"mask_{slot}.png"
        Image.new("L", (80, 60), 255).save(path)
        masks.append(str(path))
    params = tmp_path / "source.npz"
    np.savez(params, verts=req.driver.body.vertices.numpy())
    cam = tmp_path / "source_cam.npz"
    cam.write_bytes(b"own source camera")
    calls = []

    def load(source_path, driver_path, **kwargs):
        assert source_path == driver_path == str(params)
        calls.append("source")
        return None, replace(req.driver.parameters, person_ids=(0, 1)), None, {}

    monkeypatch.setattr(geometry, "load_hi4d_parameters", load)
    monkeypatch.setattr(geometry, "load_hi4d_smpl_backend", lambda path: RodBody())
    monkeypatch.setattr(api, "load_hi4d_camera", lambda *a, **kw: req.camera)
    monkeypatch.setattr(api, "P_ROOT", tmp_path)
    api._job_main(
        dict(
            dataset="hi4d",
            source_geometry_only=True,
            source_params=str(params),
            source_camera_path=str(cam),
            source_camera_id=4,
            source_image_hw=[60, 80],
            person_order=[0, 1],
            source_rgb=str(source),
            source_masks=masks,
            body_model_path="body model",
            output_root=str(tmp_path),
            output_dir=str(tmp_path / "source_only"),
        )
    )
    assert calls == ["source"]
    assert (tmp_path / "source_only/source_geometry.json").is_file()


def test_current_geometry_encode_handoff_regenerates_native_reference_inputs(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import prepare_geometry
    from src.pose_control.qwen21_pro.encoded import encode_prepared_condition
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition
    from test_qwen21_pro_encoded import Encoder

    current = prepare_geometry(
        request(beta_delta=1),
        body_model=RodBody(),
        renderer=Renderer(),
        output_dir=tmp_path / "source_shape",
        output_root=tmp_path,
        canvas_hw=(64, 96),
        mode="source_shape",
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 96), "green").save(source)
    value, pack = encode_prepared_condition(
        Encoder(),
        replace(make_condition(), person_ids=(41, 7)),
        manifest_path=tmp_path / "source_shape/manifest.json",
        config=ProAdapterConfig(),
        sample_id="current",
        source_images=[source],
        prompt="requested pose",
        producer_signature={"actual": "encoder"},
    )
    assert pack.target_hw == (4, 6)
    assert value["control_latents"].shape == (1, 24, 64)
    assert value["reference_latents"].shape == (1, 72, 64)
    assert value.binding["geometry"] == current.manifest["geometry_signature"]
    assert value.binding["geometry_files"][0]["path"] == str(
        current.pose_images[0].resolve()
    )


def test_source_mesh_roi_rejects_other_action_camera_before_projection(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import build_source_geometry

    directory = tmp_path / "sources/pair16/cam4/source_action/000022"
    directory.mkdir(parents=True)
    source = directory / "rgb.jpg"
    Image.new("RGB", (80, 60)).save(source)
    mask = directory / "mask_A.png"
    Image.new("L", (80, 60), 255).save(mask)
    wrong = tmp_path / "raw/pair16/target_action/cameras/rgb_cameras.npz"
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"wrong action")
    params = tmp_path / "raw/pair16/source_action/smpl/000022.npz"
    params.parent.mkdir(parents=True)
    params.write_bytes(b"own source")
    req = request(people=1)
    with pytest.raises(ValueError, match="own source action"):
        build_source_geometry(
            {"person_order": [41]},
            {"source_rgb": source, "source_masks": [mask]},
            body_model=RodBody(),
            state=req.driver,
            camera=req.camera,
            asset_files={"source_raw_parameters": params, "source_camera": wrong},
        )


def test_current_contact_relation_preserves_extent_codec_camera_and_visible_owner(
    tmp_path,
):
    from src.pose_control.qwen21_pro.geometry_preparation import (
        prepare_geometry,
        geometry_task_condition,
    )
    from src.pose_control.qwen21_pro.geometry import (
        SourceShape,
        retarget_request,
        VertexContact,
        VertexEndpoint,
    )
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition

    original = request()
    params = replace(original.driver.parameters, frame="world")
    extrinsic = torch.tensor([[0.0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    cam = replace(original.camera, world_to_camera=extrinsic)
    req = retarget_request(
        SourceShape("smpl", params.person_ids, params.betas), params, cam, RodBody()
    )
    contact = VertexContact(VertexEndpoint(41, 1), VertexEndpoint(7, 1))
    prepared = prepare_geometry(
        req,
        body_model=RodBody(),
        renderer=Renderer(),
        output_dir=tmp_path / "relation",
        output_root=tmp_path,
        canvas_hw=(64, 96),
        mode="driver_exact",
        contacts=[contact],
    )
    pack = geometry_task_condition(
        replace(make_condition(), person_ids=(41, 7)),
        prepared,
        config=ProAdapterConfig(),
    )
    expected = torch.tensor([0.0, 0.04, 0.0, 0.04, 1.0, 0.0])
    assert torch.allclose(pack.relations.relative[0, 0], expected, atol=1e-5)
    assert (
        pack.provenance["geometry_relations"]["normalization"]
        == "mean_current_parameter_frame_mesh_max_axis_extent_m"
    )
    assert (
        pack.provenance["geometry_relations"]["endpoint_visibility_available"] is True
    )


def test_hi4d_matched_male_worker_checks_real_source_and_driver_forward(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import verify_hi4d_forward

    req = request()
    raw = tmp_path / "raw.npz"
    np.savez(
        raw, verts=req.driver.body.vertices.numpy(), genders=np.array(["male", "male"])
    )
    audit = verify_hi4d_forward(raw, req.driver, person_order=(0, 1))
    assert audit["max_abs_error_m"] == 0
    np.savez(
        raw,
        verts=req.driver.body.vertices.numpy() + 0.002,
        genders=np.array(["male", "male"]),
    )
    with pytest.raises(ValueError, match="forward mismatch"):
        verify_hi4d_forward(raw, req.driver, person_order=(0, 1))
    np.savez(
        raw,
        verts=req.driver.body.vertices.numpy(),
        genders=np.array(["male", "female"]),
    )
    with pytest.raises(ValueError, match="male|gender"):
        verify_hi4d_forward(raw, req.driver, person_order=(0, 1))

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
    assert prepared.vertex_parts.ndim == 1  # shared legacy backend keeps its [V] ABI
    assert torch.load(tmp_path / "current/geometry.pt", weights_only=True)["vertex_parts"].ndim == 1
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


def test_prepared_parts_are_in_state_order_and_contact_endpoints_use_person_slot(tmp_path):
    from src.pose_control.qwen21_pro.geometry_preparation import prepare_geometry, geometry_task_condition, load_geometry_task_condition
    from src.pose_control.qwen21_pro.geometry import VertexContact, VertexEndpoint
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition
    class Parts(RodBody):
        person_ids = (7, 41)
        vertex_parts = torch.tensor([[1, 7, 1, 1], [1, 4, 1, 1]])
    prepared = prepare_geometry(request(), body_model=Parts(), renderer=Renderer(),
        output_dir=tmp_path / "parts", output_root=tmp_path, canvas_hw=(64, 96),
        mode="driver_exact", contacts=[VertexContact(VertexEndpoint(41, 1), VertexEndpoint(7, 1))])
    assert prepared.vertex_parts[:, 1].tolist() == [4, 7]
    source = replace(make_condition(), person_ids=(41, 7))
    for pack in (geometry_task_condition(source, prepared, config=ProAdapterConfig()),
                 load_geometry_task_condition(source, tmp_path / "parts/manifest.json", config=ProAdapterConfig())):
        assert pack.relations.src_person.tolist() == [[0]]
        assert pack.relations.dst_person.tolist() == [[1]]
        assert pack.relations.src_part.tolist() == [[4]]
        assert pack.relations.dst_part.tolist() == [[7]]


def bound_gender_backend(tmp_path, monkeypatch, *, order=(0, 1)):
    """Actual files and actual Tensor layers; external licensed assets replaced."""
    import src.pose_control.qwen21_pro.geometry as geometry
    from test_qwen21_pro_geometry import mixed_sentinel_backend, parameters
    from src.pose_control.qwen21_pro.cache import file_sha256, tensor_sha256
    models = mixed_sentinel_backend().models
    model_root = tmp_path / "body"
    (model_root / "smpl").mkdir(parents=True)
    loader = tmp_path / "smplx_loader.py"
    loader.write_text("# actual bound external loader")
    for gender in ("male", "female"):
        model = model_root / "smpl" / f"SMPL_{gender.upper()}.pkl"
        model.write_bytes(f"{gender} weights".encode())
        monkeypatch.setattr(geometry, f"HI4D_{gender.upper()}_MODEL_SHA256", file_sha256(model))
        # Parts pins use the same tensor ABI as the production original-weight audit.
        monkeypatch.setitem(geometry.HI4D_PARTS_SHA256, gender, tensor_sha256(models[gender].vertex_parts))
        models[gender].provenance = {
            "model_path": str(model.resolve()), "model_sha256": file_sha256(model),
            "loader_path": str(geometry.Path(geometry.__file__).resolve()), "loader_sha256": file_sha256(geometry.__file__),
            "smplx_loader_path": str(loader.resolve()), "smplx_loader_sha256": file_sha256(loader),
            "smplx_body_loader_path": str(loader.resolve()), "smplx_body_loader_sha256": file_sha256(loader),
            "joint_mapping": "raw_smpl24_only"}
    monkeypatch.setattr(geometry, "_load_hi4d_smpl_gender_backend", lambda path, gender: models[gender])
    action = tmp_path / "raw/pair28/dance28"
    (action / "smpl").mkdir(parents=True)
    params_path = action / "smpl/000006.npz"
    params = replace(parameters(), person_ids=(0, 1), frame="world")
    body = geometry.GenderedBodyModelBackend(models, {0: "male", 1: "female"}).forward(params)
    np.savez(params_path, **{key: getattr(params, key).numpy() for key in
             ("betas", "global_orient", "body_pose", "transl")},
             verts=body.vertices.numpy(), joints_3d=body.joints.numpy(), contact=np.zeros((2, 6890)))
    metadata = action / "meta.npz"
    np.savez(metadata, genders=np.array(["male", "female"]))
    backend = geometry.load_hi4d_gendered_backend(model_root, metadata, parameters_path=params_path, person_order=order)
    return backend, params_path, metadata, model_root


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_gendered_source_only_job_binds_models_metadata_and_never_reads_driver(tmp_path, monkeypatch, order):
    import src.pose_control.qwen21_pro.geometry_preparation as api
    backend, params, metadata, root = bound_gender_backend(tmp_path, monkeypatch, order=order)
    from src.pose_control.qwen21_pro.geometry import Camera
    source = tmp_path / "source.png"
    Image.new("RGB", (80, 60)).save(source)
    masks = []
    for slot in range(2):
        path = tmp_path / f"source_mask_{slot}.png"
        Image.new("L", (80, 60), 255).save(path)
        masks.append(str(path))
    camera = tmp_path / "source_cam.npz"
    camera.write_bytes(b"own camera")
    monkeypatch.setattr(api, "P_ROOT", tmp_path)
    monkeypatch.setattr(api, "load_hi4d_camera", lambda *a: Camera(request().camera.K, (60, 80), torch.eye(4)[:3]))
    api._job_main(dict(dataset="hi4d", source_geometry_only=True, body_model_root=str(root),
        gender_metadata=str(metadata), source_params=str(params), person_order=list(order),
        source_camera_path=str(camera), source_camera_id=4, source_image_hw=[60, 80],
        source_rgb=str(source), source_masks=masks, output_root=str(tmp_path), output_dir=str(tmp_path / "source_only"),
        driver_params="DRIVER_MUST_NEVER_BE_READ", driver_gender_metadata="DRIVER_METADATA_MUST_NEVER_BE_READ"))
    import json
    bundle = json.loads((tmp_path / "source_only/source_geometry.json").read_text())
    assert bundle["forward_audit"]["max_abs_error_m"] == 0
    assert bundle["forward_audit"]["genders"] == ["male", "female"]
    assert str(metadata.resolve()) in bundle["asset_files"].values()
    assert len([p for p in bundle["asset_files"].values() if p.endswith(".pkl")]) == 2
    api.validate_source_geometry(bundle, {"person_order": list(order)}, {"source_rgb": source})
    np.savez(metadata, genders=np.array(["female", "male"]))
    with pytest.raises(ValueError, match="SHA|gender|binding"):
        api.validate_source_geometry(bundle, {"person_order": list(order)}, {"source_rgb": source})


def test_gendered_handoff_binds_actual_assets_and_parts_without_smplx(tmp_path, monkeypatch):
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition
    from src.pose_control.qwen21_pro.cache import asset_hashes, tensor_sha256
    import builtins
    import json
    backend, params, metadata, _ = bound_gender_backend(tmp_path, monkeypatch, order=(1, 0))
    shape, driver, _, _ = geometry.load_hi4d_parameters(params, params, person_order=(1, 0))
    camera = replace(request().camera, world_to_camera=torch.eye(4)[:3])
    req = geometry.retarget_request(shape, driver, camera, backend)
    prepared = api.prepare_geometry(req, body_model=backend, renderer=Renderer(),
        output_dir=tmp_path / "current", output_root=tmp_path, canvas_hw=(64, 96), mode="driver_exact")
    signature = prepared.manifest["geometry_signature"]
    assert str(metadata.resolve()) in signature["input_asset_files"].values()
    assert "female_smplx_body_loader_path" in signature["input_asset_files"]
    assert signature["vertex_parts_sha256"] == tensor_sha256(prepared.vertex_parts)
    importer = builtins.__import__
    def no_smplx(name, *args, **kwargs):
        if name == "smplx" or name.startswith("smplx."):
            raise AssertionError("Qwen handoff must not import smplx")
        return importer(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_smplx)
    source = replace(make_condition(), person_ids=(1, 0))
    manifest_path = tmp_path / "current/manifest.json"
    api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())
    changed = json.loads(manifest_path.read_text())
    changed["geometry_signature"]["body_model"]["backend"] = "unbound_backend"
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="gender|binding|model"):
        api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())
    manifest_path.write_text(json.dumps(prepared.manifest))
    # Changing a model signature while raw file hashes stay unchanged must fail.
    changed = json.loads(manifest_path.read_text())
    changed["geometry_signature"]["body_model"]["person_genders"][0]["gender"] = "female"
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="gender|binding|model"):
        api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())
    manifest_path.write_text(json.dumps(prepared.manifest))
    # Even updating the aggregate asset hashes cannot approve swapped gender metadata.
    np.savez(metadata, genders=np.array(["female", "male"]))
    changed = json.loads(manifest_path.read_text())
    changed["geometry_signature"]["input_asset_hashes"] = asset_hashes(signature["input_asset_files"])
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SHA|gender|binding"):
        api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())


def test_gendered_source_and_driver_metadata_must_agree_for_each_raw_person(tmp_path, monkeypatch):
    import src.pose_control.qwen21_pro.geometry as geometry
    backend, params, metadata, root = bound_gender_backend(tmp_path, monkeypatch)
    action = tmp_path / "raw/pair28/hug28"
    (action / "smpl").mkdir(parents=True)
    driver = action / "smpl/000045.npz"
    driver.write_bytes(params.read_bytes())
    driver_meta = action / "meta.npz"
    np.savez(driver_meta, genders=np.array(["female", "male"]))
    with pytest.raises(ValueError, match="source/driver|gender"):
        geometry.load_hi4d_gendered_backend(root, metadata, parameters_path=params,
            driver_gender_metadata=driver_meta, driver_parameters_path=driver)
    np.savez(driver_meta, genders=np.array(["male", "female"]))
    combined = geometry.load_hi4d_gendered_backend(root, metadata, parameters_path=params,
        driver_gender_metadata=driver_meta, driver_parameters_path=driver, person_order=(1, 0))
    assert combined.person_genders == {1: "female", 0: "male"}
    assert combined.provenance["gender_metadata"]["driver"]["path"] == str(driver_meta.resolve())


def test_gendered_raw_forward_checks_own_order_joints_and_model_binding(tmp_path, monkeypatch):
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api
    backend, params, _, _ = bound_gender_backend(tmp_path, monkeypatch, order=(1, 0))
    _, parameters, _, _ = geometry.load_hi4d_parameters(params, params, person_order=(1, 0))
    state = geometry.GeometryState(parameters, backend.forward(parameters))
    audit = api.verify_hi4d_forward(params, state, person_order=(1, 0), body_model=backend)
    assert audit["max_abs_joint_error_m"] == 0
    with pytest.raises(ValueError, match="order|binding"):
        api.verify_hi4d_forward(params, state, person_order=(0, 1), body_model=backend)
    with pytest.raises(ValueError, match="joint.*mismatch"):
        api.verify_hi4d_forward(params, replace(state, body=geometry.BodyOutput(state.body.vertices, state.body.joints + .002)),
                                person_order=(1, 0), body_model=backend)


def test_gendered_handoff_rejects_resigned_wrong_parts_and_missing_model_assets(tmp_path, monkeypatch):
    import json
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition
    from src.pose_control.qwen21_pro.cache import file_sha256, tensor_sha256
    backend, params, _, _ = bound_gender_backend(tmp_path, monkeypatch)
    shape, driver, _, _ = geometry.load_hi4d_parameters(params, params)
    camera = replace(request().camera, world_to_camera=torch.eye(4)[:3])
    req = geometry.retarget_request(shape, driver, camera, backend)
    api.prepare_geometry(req, body_model=backend, renderer=Renderer(), output_dir=tmp_path / "current",
                         output_root=tmp_path, canvas_hw=(64, 96), mode="driver_exact")
    manifest_path, payload_path = tmp_path / "current/manifest.json", tmp_path / "current/geometry.pt"
    manifest = json.loads(manifest_path.read_text())
    signature = manifest["geometry_signature"]
    payload = torch.load(payload_path, weights_only=True)
    payload["vertex_parts"] = payload["vertex_parts"].flip(0)
    torch.save(payload, payload_path)
    signature["dense_sha256"] = file_sha256(payload_path)
    signature["vertex_parts_sha256"] = tensor_sha256(payload["vertex_parts"])
    manifest_path.write_text(json.dumps(manifest))
    source = replace(make_condition(), person_ids=(0, 1))
    with pytest.raises(ValueError, match="parts.*binding"):
        api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())
    # Keeping valid hashes while omitting the bound model file is also invalid.
    payload["vertex_parts"] = payload["vertex_parts"].flip(0)
    torch.save(payload, payload_path)
    signature["dense_sha256"] = file_sha256(payload_path)
    signature["vertex_parts_sha256"] = tensor_sha256(payload["vertex_parts"])
    signature["input_asset_files"].pop("female_model_path")
    signature["input_asset_hashes"].pop("female_model_path")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model input file binding"):
        api.load_geometry_task_condition(source, manifest_path, config=ProAdapterConfig())


def test_current_handoff_rejects_complete_gender_proof_downgrade_with_resigned_parts(tmp_path, monkeypatch):
    import copy
    import json
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api
    from src.pose_control.qwen21_pro.cache import file_sha256, tensor_sha256
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from test_qwen21_pro_contracts import make_condition

    backend, params, _, _ = bound_gender_backend(tmp_path, monkeypatch, order=(1, 0))
    shape, driver, _, _ = geometry.load_hi4d_parameters(params, params, person_order=(1, 0))
    camera = replace(request().camera, world_to_camera=torch.eye(4)[:3])
    req = geometry.retarget_request(shape, driver, camera, backend)
    api.prepare_geometry(req, body_model=backend, renderer=Renderer(),
        output_dir=tmp_path / "current", output_root=tmp_path, canvas_hw=(64, 96), mode="driver_exact")
    manifest_path, payload_path = tmp_path / "current/manifest.json", tmp_path / "current/geometry.pt"
    manifest = json.loads(manifest_path.read_text())
    signature = manifest["geometry_signature"]
    legacy = copy.deepcopy(signature["body_model"])
    legacy["backend"] = "hi4d_matched_male_cpu"
    for field in ("models", "person_genders", "gender_metadata"):
        legacy.pop(field)
    signature["body_model"] = legacy
    signature["request_provenance"]["body_model"] = copy.deepcopy(legacy)
    data = torch.load(payload_path, weights_only=True)
    assert data["vertex_parts"][:, 0].tolist() == [7, 4]
    data["vertex_parts"][0] = data["vertex_parts"][1]
    torch.save(data, payload_path)
    signature["dense_sha256"] = file_sha256(payload_path)
    signature["vertex_parts_sha256"] = tensor_sha256(data["vertex_parts"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gender.*proof|gender.*binding"):
        api.load_geometry_task_condition(replace(make_condition(), person_ids=(1, 0)),
            manifest_path, config=ProAdapterConfig())


def test_source_bundle_rejects_complete_gender_proof_downgrade_after_metadata_resign(tmp_path, monkeypatch):
    import copy
    import src.pose_control.qwen21_pro.geometry as geometry
    import src.pose_control.qwen21_pro.geometry_preparation as api
    from src.pose_control.qwen21_pro.cache import asset_hashes

    backend, params, metadata, _ = bound_gender_backend(tmp_path, monkeypatch)
    _, parameters, _, _ = geometry.load_hi4d_parameters(params, params)
    state = geometry.GeometryState(parameters, backend.forward(parameters))
    camera = replace(request().camera, world_to_camera=torch.eye(4)[:3])
    source, mask, calibration = tmp_path / "source.png", tmp_path / "mask.png", tmp_path / "camera.npz"
    Image.new("RGB", (80, 60)).save(source)
    Image.new("L", (80, 60), 255).save(mask)
    calibration.write_bytes(b"own calibration")
    record = {"person_order": [0, 1]}
    assets = {"source_rgb": source, "source_masks": [mask, mask]}
    bundle = api.build_source_geometry(record, assets, body_model=backend, state=state, camera=camera,
        asset_files={"source_raw_parameters": params, "source_camera": calibration})
    legacy = copy.deepcopy(bundle["body_model"])
    legacy["backend"] = "hi4d_matched_male_cpu"
    for field in ("models", "person_genders", "gender_metadata"):
        legacy.pop(field)
    bundle["body_model"] = legacy
    bundle.pop("vertex_part_bindings")
    bundle.pop("vertex_parts_shape", None)  # actual files must enforce proof even without this declaration
    np.savez(metadata, genders=np.array(["female", "male"]))
    bundle["asset_hashes"] = asset_hashes(bundle["asset_files"])
    with pytest.raises(ValueError, match="gender.*proof|gender.*binding"):
        api.validate_source_geometry(bundle, record, assets)

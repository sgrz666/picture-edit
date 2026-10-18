"""Source shape retargeting, geometric constraints and explicit raster conventions."""
from dataclasses import replace
import importlib
import importlib.util
import io
import json
import os
import pickle
import sys
import types

import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec("src.pose_control.qwen21_pro.geometry") is not None, "Pro geometry is missing"
    return importlib.import_module("src.pose_control.qwen21_pro.geometry")


class RodBody:
    """An analytic differentiable body: two metre rod with one articulated endpoint."""
    family = "smpl"
    vertex_count = 4
    provenance = {"backend": "analytic_test_rod"}
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    vertex_parts = torch.tensor([10, 4, 1, 1])

    def forward(self, params):
        m = api()
        height = 2 + params.betas[:, 0] * .1
        theta = params.body_pose[:, 2]
        zero = torch.zeros_like(theta)
        bottom = torch.stack([zero, -height / 2, zero], -1)
        tip = torch.stack([torch.sin(theta), height / 2 * torch.cos(theta), zero], -1)
        side = torch.stack([zero + .1, zero, zero], -1)
        vertices = torch.stack([bottom, tip, side, side * -1], 1) + params.transl[:, None]
        return m.BodyOutput(vertices, vertices[:, :2])


def parameters(people=2, family="smpl"):
    m = api()
    kwargs = dict(family=family, person_ids=(41, 7)[:people], betas=torch.zeros(people, 10),
                  global_orient=torch.zeros(people, 3), body_pose=torch.zeros(people, 69 if family == "smpl" else 63),
                  transl=torch.tensor([[-.04, 0, 3], [.04, 0, 3]])[:people], frame="camera_cv")
    if family == "smplx":
        kwargs.update(left_hand_pose=torch.zeros(people, 45), right_hand_pose=torch.zeros(people, 45),
                      jaw_pose=torch.zeros(people, 3), expression=torch.zeros(people, 10))
    return m.BodyParameters(**kwargs)


def camera():
    return api().Camera(torch.tensor([[100., 0, 40], [0, 120, 30], [0, 0, 1]]), (60, 80))


def request(people=2, beta_delta=0):
    m = api()
    driver = parameters(people)
    source = m.SourceShape("smpl", driver.person_ids, driver.betas + beta_delta)
    return m.retarget_request(source, driver, camera(), RodBody())


def test_source_shape_changes_mesh_preserves_driver_pose_camera_and_inputs():
    m = api()
    driver = parameters()
    original = {k: v.clone() for k, v in vars(driver).items() if isinstance(v, torch.Tensor)}
    source = m.SourceShape("smpl", driver.person_ids, driver.betas + 1)
    cam = camera()
    result = m.retarget_request(source, driver, cam, RodBody())
    assert torch.equal(result.driver.parameters.betas, driver.betas)
    assert torch.equal(result.requested.parameters.betas, source.betas)
    assert torch.equal(result.requested.parameters.body_pose, driver.body_pose)
    assert torch.equal(result.camera.K, cam.K)
    assert not torch.equal(result.driver.body.vertices, result.requested.body.vertices)
    assert result.person_ids == (41, 7)
    for key, before in original.items():
        assert torch.equal(getattr(driver, key), before)
    result.requested.parameters.betas[0, 0] = 99
    assert torch.equal(source.betas, torch.ones_like(source.betas))
    assert torch.equal(result.driver.parameters.betas, driver.betas)


def test_height_is_source_zero_pose_height_not_driver_pose_bbox():
    m = api()
    driver = parameters(1)
    driver = replace(driver, body_pose=driver.body_pose + torch.nn.functional.one_hot(torch.tensor([2]), 69) * 1.5)
    result = m.retarget_request(m.SourceShape("smpl", (41,), driver.betas + 2), driver, camera(), RodBody())
    assert result.source_rest_heights_m.item() == pytest.approx(2.2)
    posed_y = result.requested.body.vertices[0, :, 1]
    assert result.source_rest_heights_m.item() > (posed_y.max() - posed_y.min()).item()


@pytest.mark.parametrize("change", ["source", "solve"])
def test_changed_geometry_cannot_use_original_exact_gt(change):
    m = api()
    result = request(beta_delta=1 if change == "source" else 0)
    if change == "solve":
        result = m.solve_contacts(result, [m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1), tolerance_m=.002)], RodBody())
        assert result.solver_log["success"]
    with pytest.raises(ValueError, match="exact.*pair|pair.*exact"):
        result.require_exact_gt_pairing()


def test_e0_bitwise_identity_even_with_supplied_ground_and_anchor():
    m = api()
    result = request()
    anchor = m.VertexAnchor(m.VertexEndpoint(41, 0), torch.tensor([99., 99, 99]))
    solved = m.solve_contacts(result, [], RodBody(), anchors=[anchor], ground_plane=m.GroundPlane((0., 1., 0.), 0., (m.VertexEndpoint(7, 0),)))
    assert solved.solver_log["success"] and solved.solver_log["reason"] == "no_contacts"
    for name in ("body_pose", "transl", "betas"):
        assert torch.equal(getattr(result.requested.parameters, name), getattr(solved.solved.parameters, name))
    assert torch.equal(result.requested.body.vertices, solved.solved.body.vertices)
    solved.require_exact_gt_pairing()


def test_contact_solver_uses_body_pose_with_fixed_foot_anchors_and_bounded_roots():
    m = api()
    original = request()
    before = original.requested.body.vertices.clone()
    anchors = [m.VertexAnchor(m.VertexEndpoint(pid, 0), before[i, 0], tolerance_m=.0003) for i, pid in enumerate(original.person_ids)]
    relation = m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1), tolerance_m=.002)
    result = m.solve_contacts(original, [relation], RodBody(), anchors=anchors, max_iterations=400)
    assert result.solver_log["success"], result.solver_log
    assert (result.solved.body.vertices[0, 1] - result.solved.body.vertices[1, 1]).norm() <= .002
    assert not torch.equal(result.solved.parameters.body_pose, original.requested.parameters.body_pose)
    assert torch.equal(original.requested.body.vertices, before)
    assert (result.solved.body.vertices[:, 0] - before[:, 0]).norm(dim=-1).max() <= .0003
    assert max(result.solver_log["root_displacement_m"]) <= .05 * 2
    assert max(result.solver_log["joint_rotation_delta_deg"]) <= 5.0001
    assert result.solver_log["contact_residual_m"][0] <= .002
    assert result.solver_log["objective_history"]
    assert torch.equal(result.camera.K, original.camera.K)


def test_impossible_contacts_fall_back_without_mutating_requested_or_driver():
    m = api()
    original = request()
    far = replace(original.requested.parameters, transl=original.requested.parameters.transl + torch.tensor([[0., 0, 0], [5., 0, 0]]))
    original = m.retarget_request(m.SourceShape("smpl", (41, 7), far.betas), far, camera(), RodBody())
    contact = m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1), tolerance_m=.001)
    result = m.solve_contacts(original, [contact], RodBody(), max_iterations=30)
    assert not result.solver_log["success"]
    assert result.solver_log["reason"]
    assert torch.equal(result.solved.body.vertices, original.requested.body.vertices)
    assert torch.equal(result.solved.parameters.body_pose, original.requested.parameters.body_pose)
    assert torch.equal(result.solved.parameters.transl, original.requested.parameters.transl)
    result.require_exact_gt_pairing()


def test_explicit_ground_plane_constrains_only_declared_feet():
    m = api()
    original = request()
    plane = m.GroundPlane((0., 1., 0.), 1., tuple(m.VertexEndpoint(pid, 0) for pid in original.person_ids), tolerance_m=.0005)
    contact = m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1), tolerance_m=.003)
    result = m.solve_contacts(original, [contact], RodBody(), ground_plane=plane, max_iterations=300)
    assert result.solver_log["success"], result.solver_log
    assert max(result.solver_log["ground_residual_m"]) <= .0005
    assert result.solver_log["collision_guarantee"] == "none"


@pytest.mark.parametrize("bad", ["family", "shape", "missing_hands", "fake_smpl_hands", "ids"])
def test_family_dimensions_and_person_binding_fail_closed(bad):
    m = api()
    with pytest.raises(ValueError):
        if bad == "family":
            m.retarget_request(m.SourceShape("smplx", (41,), torch.zeros(1, 10)), parameters(1), camera(), RodBody())
        elif bad == "shape":
            replace(parameters(1), body_pose=torch.zeros(1, 63))
        elif bad == "missing_hands":
            replace(parameters(1, "smplx"), left_hand_pose=None)
        elif bad == "fake_smpl_hands":
            replace(parameters(1), left_hand_pose=torch.zeros(1, 45))
        else:
            m.retarget_request(m.SourceShape("smpl", (7, 41), torch.zeros(2, 10)), parameters(), camera(), RodBody())


def test_endpoints_are_stable_person_ids_and_vertex_indices_are_checked():
    m = api()
    for endpoint in (m.VertexEndpoint(0, 1), m.VertexEndpoint(41, 55)):
        with pytest.raises(ValueError, match="person|vertex"):
            m.solve_contacts(request(), [m.VertexContact(endpoint, m.VertexEndpoint(7, 1))], RodBody())


def test_positive_raw_annotations_become_flagged_nearest_vertex_pairs_not_partner_ids():
    m = api()
    vertices = request().requested.body.vertices
    raw = torch.zeros(2, 4)
    raw[0, 1] = 6335
    raw[1, 1] = 29
    pairs = m.contacts_from_raw_flags(vertices, raw, person_ids=(41, 7))
    assert len(pairs) == 1
    assert pairs[0].a == m.VertexEndpoint(41, 1) and pairs[0].b == m.VertexEndpoint(7, 1)
    assert m.contacts_from_raw_flags(vertices, raw * 0, person_ids=(41, 7)) == ()


def test_world_camera_projection_canvas_and_clip_share_physical_camera():
    m = api()
    extr = torch.cat([torch.eye(3), torch.tensor([[1.], [2.], [3.]])], 1)
    cam = m.Camera(camera().K, (60, 80), extr)
    xyz = torch.tensor([[0., 0, 1.]])
    cv = cam.to_camera(xyz, frame="world")
    assert torch.equal(cv, torch.tensor([[1., 2, 4.]]))
    canvas = m.PixelCanvasTransform.letterbox((60, 80), (120, 120))
    xy = cam.project(xyz, frame="world", canvas=canvas)
    assert torch.allclose(xy, torch.tensor([[97.5, 150.]]))
    clip = m.camera_to_clip(cv, cam, canvas, near=.1, far=10.)
    ndc = clip[:, :2] / clip[:, 3:4]
    back = torch.stack([(ndc[:, 0] + 1) * 60, (1 - ndc[:, 1]) * 60], -1)
    assert torch.allclose(xy, back)
    assert torch.equal(cam.K, camera().K)
    with pytest.raises(ValueError, match="extrinsic"):
        camera().to_camera(xyz, frame="world")


def test_pixel_canvas_valid_fov_preserves_native_tiktok_image_boundary():
    m = api()
    canvas = m.PixelCanvasTransform.letterbox((1080, 604), (512, 512))
    before = canvas.matrix.clone()
    assert hasattr(canvas, "valid_fov_mask"), "canvas is missing original physical FoV support"
    valid = canvas.valid_fov_mask()
    assert valid.dtype == torch.bool and valid.shape == (512, 512)
    assert valid.sum().item() == 512 * 286
    assert valid[:, 113:399].all()
    assert not valid[:, :113].any() and not valid[:, 399:].any()
    assert torch.equal(canvas.matrix, before)


@pytest.mark.parametrize("matrix,input_hw,output_hw,expected", [
    ([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]], (2, 2), (2, 4),
     [[True, True, False, False], [False, True, True, False]]),
    ([[-1., 0., 2.5], [0., 1., 0.], [0., 0., 1.]], (1, 2), (1, 4),
     [[False, True, True, False]]),
    ([[1., 0., .5], [0., 1., 0.], [0., 0., 1.]], (1, 2), (1, 4),
     [[True, True, False, False]]),
])
def test_pixel_canvas_valid_fov_uses_inverse_affine_pixel_centres(matrix, input_hw, output_hw, expected):
    m = api()
    canvas = m.PixelCanvasTransform(torch.tensor(matrix), input_hw, output_hw)
    assert hasattr(canvas, "valid_fov_mask"), "canvas is missing inverse-affine physical FoV support"
    assert canvas.valid_fov_mask().tolist() == expected


def test_compose_rejects_geometry_outside_original_physical_camera_fov():
    m = api()
    cam = m.Camera(torch.tensor([[1., 0, 1], [0, 1, 2], [0, 0, 1.]]), (4, 2))
    canvas = m.PixelCanvasTransform.letterbox(cam.image_hw, (4, 4))
    depth = torch.ones(1, 4, 4)
    mask = torch.ones_like(depth, dtype=torch.bool)
    normal = torch.zeros(1, 4, 4, 3)
    normal[..., 2] = 1
    with pytest.raises(ValueError, match="FoV|field.of.view|physical"):
        m.compose_dense_geometry(normal, depth, mask, mask.long(), torch.ones_like(depth),
                                 person_ids=(41,), camera=cam, canvas=canvas)
    valid = torch.zeros_like(mask)
    valid[:, :, 1:3] = True
    result = m.compose_dense_geometry(normal * valid[..., None], depth * valid, valid, valid.long(), valid.float(),
                                      person_ids=(41,), camera=cam, canvas=canvas)
    assert torch.equal(result.mask, valid)
    assert not result.visible_owner[:, (0, 3)].any()
    assert torch.equal(result.visible_owner[:, 1:3], torch.ones(4, 2, dtype=torch.long))


def test_smplestx_frame_map_uses_tensor_index_and_raw_camera(tmp_path):
    m = api()
    count = 3
    p = {name: torch.zeros(count, size) for name, size in {"betas": 10, "global_orient": 3, "body_pose": 63,
        "left_hand_pose": 45, "right_hand_pose": 45, "jaw_pose": 3, "expression": 10, "transl": 3, "transl_raw": 3,
        "focal_length_xy": 2, "focal_length_xy_raw": 2, "princpt": 2, "princpt_raw": 2}.items()}
    p.update(width=torch.ones(count, 1) * 1024, height=torch.ones(count, 1) * 1024, retargeted_cam=torch.ones(1, dtype=torch.uint8))
    p["transl_raw"][2] = torch.tensor([1., 2, 3])
    p["focal_length_xy_raw"][:] = 100
    p["princpt_raw"][:] = 512
    p["body_pose"][2, 0] = .2
    p["betas"][1, 0] = .4
    path = tmp_path / "params.pt"
    torch.save(p, path)
    frames = {"frames": [{"smplestx_index": 1, "stem": "source_f1253", "original_frame_index0": 1253},
                         {"smplestx_index": 2, "stem": "target_f219", "original_frame_index0": 219}]}
    frame_path = tmp_path / "frame_map.json"
    frame_path.write_text(json.dumps(frames))
    source, driver, cam, provenance = m.load_smplestx_parameters(path, frame_path, source_frame="source_f1253", driver_frame="target_f219", dataset="iper", person_id=41)
    assert source.betas[0, 0] == .4
    assert driver.body_pose[0, 0] == .2
    assert torch.equal(driver.transl[0], p["transl_raw"][2])
    assert cam.K[0, 0] == 100
    assert provenance["driver_tensor_index"] == 2 and provenance["source_tensor_index"] == 1
    assert provenance["params_sha256"] and provenance["frame_map_sha256"]
    del p["transl_raw"]
    torch.save(p, path)
    with pytest.raises(ValueError, match="raw|transl"):
        m.load_smplestx_parameters(path, frame_path, source_frame="source_f1253", driver_frame="target_f219", dataset="iper")


def test_joint_mapping_and_frozen_layer_keep_input_gradient():
    m = api()
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def forward(self, **kwargs):
            class Result:
                pass
            result = Result()
            x = kwargs["body_pose"].sum(1) * self.weight
            result.vertices = x[:, None, None].expand(-1, 10475, 3)
            result.joints = torch.arange(144.)[None, :, None].expand(len(x), -1, 3) + x[:, None, None]
            return result
    # Actual SMPLest-X joint_idx: 57 and 56 each appear twice by design.
    order = (0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63,
             64, 65, 59, 58, 57, 56, 55, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29,
             30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 52, 53, 54, 71, 40, 41, 42,
             72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75, 22, 15, 57, 56,
             76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
             93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
             108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
             136, 137, 138, 139, 140, 141, 142, 143)
    layer = Layer()
    backend = m.BodyModelBackend("smplx", layer, np.array([[0, 1, 2]]), joint_indices=order, vertex_parts=torch.zeros(10475, dtype=torch.long))
    params = parameters(1, "smplx")
    params = replace(params, body_pose=params.body_pose.requires_grad_())
    output = backend.forward(params)
    assert output.joints.shape == (1, 137, 3) and output.joints[0, 0, 0] == 0
    assert output.joints[0, 22, 0] == output.joints[0, 67, 0] == 57
    output.vertices.sum().backward()
    assert params.body_pose.grad.abs().sum() > 0
    assert not layer.weight.requires_grad and layer.weight.grad is None
    assert m.dwpose134(output, family="smplx").shape == (1, 134, 3)
    with pytest.raises(ValueError, match="SMPL|smpl"):
        m.dwpose134(m.BodyOutput(torch.zeros(1, 6890, 3), torch.zeros(1, 24, 3)), family="smpl")


def test_hi4d_parameter_order_and_absent_hands(tmp_path):
    m = api()
    path = tmp_path / "raw.npz"
    flags = np.zeros((2, 6890))
    flags[0, 1], flags[1, 0] = 6335, 2
    np.savez(path, betas=np.zeros((2, 10)), body_pose=np.zeros((2, 69)), global_orient=np.zeros((2, 3)),
             transl=np.array([[1., 2, 3], [4., 5, 6]]), contact=flags)
    shape, params, flags, provenance = m.load_hi4d_parameters(path, path, person_ids=(41, 7), person_order=(1, 0))
    assert params.transl[0, 0] == 4 and flags[0, 0] == 2
    assert params.frame == "world" and params.left_hand_pose is None
    assert provenance["person_order"] == [1, 0]
    assert shape.person_ids == params.person_ids
    raw_shape, raw_params, _, _ = m.load_hi4d_parameters(path, path, person_order=(1, 0))
    assert raw_shape.person_ids == raw_params.person_ids == (1, 0)


def test_hi4d_missing_contact_topology_and_wrong_model_fail_closed(tmp_path):
    m = api()
    path = tmp_path / "raw.npz"
    np.savez(path, betas=np.zeros((2, 10)), body_pose=np.zeros((2, 69)), global_orient=np.zeros((2, 3)),
             transl=np.zeros((2, 3)), contact=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="contact"):
        m.load_hi4d_parameters(path, path)
    model = tmp_path / "SMPL_MALE.pkl"
    model.write_bytes(b"different model")
    with pytest.raises(ValueError, match="fingerprint"):
        m.load_hi4d_smpl_backend(model)


def test_smplestx_record_dimensions_do_not_require_redundant_param_dimensions(tmp_path):
    m = api()
    p = {name: torch.zeros(2, size) for name, size in {"betas": 10, "global_orient": 3, "body_pose": 63,
        "left_hand_pose": 45, "right_hand_pose": 45, "jaw_pose": 3, "expression": 10, "transl": 3,
        "focal_length_xy": 2, "princpt": 2}.items()}
    p["focal_length_xy"][:] = 100
    path = tmp_path / "params.pt"
    torch.save(p, path)
    frame_path = tmp_path / "frames.json"
    frame_path.write_text(json.dumps({"frames": [{"stem": "source", "index": 0},
        {"stem": "driver", "index": 1, "width": 604, "height": 1080}]}))
    _, _, cam, _ = m.load_smplestx_parameters(path, frame_path, source_frame="source", driver_frame="driver", dataset="tiktok")
    assert cam.image_hw == (1080, 604)


def test_solver_rejects_a_different_backend_instead_of_reinterpreting_the_mesh():
    m = api()
    class DifferentBody(RodBody):
        provenance = {"backend": "different_test_body"}
    contact = m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1))
    with pytest.raises(ValueError, match="backend|model"):
        m.solve_contacts(request(), [contact], DifferentBody())


def test_literal_pickle_reader_rejects_unapproved_globals_and_chumpy_expressions():
    m = api()
    with pytest.raises(pickle.UnpicklingError, match="global"):
        m.load_literal_smpl_model(io.BytesIO(pickle.dumps(os.system)))
    literal = m.LiteralChumpyArray()
    literal.__setstate__({"x": np.ones((2, 3)), "_dirty_vars": set()})
    assert np.array_equal(literal.array(), np.ones((2, 3)))
    with pytest.raises(pickle.UnpicklingError, match="literal|expression"):
        m.LiteralChumpyArray().__setstate__({"x": np.ones(3), "a": object()})


def test_audited_male_chumpy_state_is_literal_and_unknown_dirty_operands_rejected():
    m = api()
    state = {"x": np.ones((2, 3)), "_dirty_vars": {"x"}, "_itr": None, "_depends_on_deps": {}}
    literal = m.LiteralChumpyArray()
    literal.__setstate__(state)
    assert np.array_equal(literal.array(), state["x"])
    with pytest.raises(pickle.UnpicklingError):
        m.LiteralChumpyArray().__setstate__({**state, "_dirty_vars": {"operand"}})


def test_legacy_sparse_reconstruction_is_allowlisted_without_arbitrary_classes():
    m = api()
    from scipy.sparse import csc_matrix
    sparse = m._LiteralSMPLUnpickler._reconstructor(csc_matrix, object, None)
    assert type(sparse) is csc_matrix
    with pytest.raises(pickle.UnpicklingError):
        m._LiteralSMPLUnpickler._reconstructor(dict, object, None)


def test_contact_solver_under_no_grad_keeps_input_gradients_and_owns_tensors():
    m = api()
    with torch.inference_mode():
        original = request()
        relation = m.VertexContact(m.VertexEndpoint(41, 1), m.VertexEndpoint(7, 1), tolerance_m=.003)
        result = m.solve_contacts(original, [relation], RodBody())
    assert result.solver_log["success"], result.solver_log
    assert torch.equal(original.requested.parameters.body_pose, torch.zeros(2, 69))


def test_projected_part_rois_describe_mesh_bounds_without_fabricating_smpl_fingers():
    m = api()
    result = request()
    rois = m.projected_part_rois(result.requested, result.camera, RodBody())
    assert len(rois) == 2
    assert rois[0]["landmark_availability"] == "SMPL24_body_only_no_fingers"
    assert rois[0]["mesh_part_bounds_xyxy"]["left_hand"] is not None
    assert rois[0]["mesh_part_bounds_xyxy"]["face"] is None


@pytest.mark.skipif(not os.environ.get("QWEN21_REAL_SMPL_MALE"), reason="optional original male SMPL assets are external")
def test_real_hi4d_cpu_backend_contract():
    m = api()
    backend = m.load_hi4d_smpl_backend(os.environ["QWEN21_REAL_SMPL_MALE"])
    assert backend.vertex_parts.shape == (6890,)
    params = parameters()
    body = backend.forward(params)
    assert body.vertices.shape == (2, 6890, 3) and body.joints.shape == (2, 24, 3)
    assert backend.provenance["model_sha256"] == "ed4d55bb3041fefc6f73b70694d6c8edc1020c0d07340be5cc651cae2c6a6ae3"


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("nvdiffrast") is None, reason="real dense nvdiffrast requires external CUDA runtime")
def test_real_dense_triangles_metric_normals_row_order_and_visible_owner():
    m = api()
    class PlaneBody:
        family = "smpl"
        vertex_count = 3
        faces = torch.tensor([[0, 1, 2]])
        vertex_parts = torch.tensor([1, 1, 1])
        provenance = {"backend": "known_CV_triangles"}
    params = parameters()
    # Top-of-image asymmetric triangles; first is behind the second at identical pixels.
    v = torch.tensor([[[-.5, -.5, 3.], [.5, -.5, 3.], [0., 0., 3.]],
                      [[-1/3, -1/3, 2.], [1/3, -1/3, 2.], [0., 0., 2.]]])
    state = m.GeometryState(params, m.BodyOutput(v, v))
    cam = camera()
    rendered = m.NVDiffDenseRenderer().render(state, cam, PlaneBody())
    mask = rendered.mask[1]
    assert mask.sum() > 100  # Filled triangle interiors, not a few splatted vertices.
    assert torch.allclose(rendered.depth_m[1][mask], torch.ones_like(rendered.depth_m[1][mask]) * 2, atol=1e-5)
    assert torch.all(rendered.visible_owner[mask] == 2)
    assert torch.all(rendered.normal_cv[1][mask][:, 2] > .999)
    assert mask.nonzero()[:, 0].float().mean() < cam.image_hw[0] / 2


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("nvdiffrast") is None, reason="real dense nvdiffrast requires external CUDA runtime")
def test_real_dense_triangle_crossing_native_fov_never_renders_into_letterbox_padding():
    m = api()
    class PlaneBody:
        family = "smpl"
        vertex_count = 3
        faces = torch.tensor([[0, 1, 2]])
        vertex_parts = torch.tensor([1, 1, 1])
        provenance = {"backend": "known_CV_triangle_crossing_physical_fov"}
    # Physical image XY: (-2,1), (3,1), (1,7). Its left tip lies outside the
    # original 4x8 image but inside the wider 16x16 letterboxed output canvas.
    vertices = torch.tensor([[[-4., -3., 2.], [1., -3., 2.], [-1., 3., 2.]]])
    state = m.GeometryState(parameters(1), m.BodyOutput(vertices, vertices))
    cam = m.Camera(torch.tensor([[2., 0., 2.], [0., 2., 4.], [0., 0., 1.]]), (8, 4))
    canvas = m.PixelCanvasTransform.letterbox(cam.image_hw, (16, 16))
    contacts = [m.VertexContact(m.VertexEndpoint(41, 0), m.VertexEndpoint(41, 1))]
    result = m.NVDiffDenseRenderer().render(state, cam, PlaneBody(), canvas=canvas, contacts=contacts)
    # The physical FoV is x=[4,12). Triangle interiors remain visible at x=4.
    valid = torch.zeros(16, 16, device=result.mask.device, dtype=torch.bool)
    valid[:, 4:12] = True
    assert result.mask.sum() > 20 and result.mask[0, :, 4].any()
    assert result.contact.sum() > 0
    for value in (result.mask, result.depth_m, result.part, result.contact, result.visibility):
        assert not value[:, ~valid].any()
    assert not result.normal_cv[:, ~valid].any()
    assert not result.visible_owner[~valid].any()


def test_dense_geometry_composite_owner_is_slot_plus_one_and_metric_nearest_depth():
    m = api()
    depth = torch.tensor([[[3., 0], [3., 3]], [[2., 2], [0., 3]]])
    mask = depth > 0
    normal = torch.zeros(2, 2, 2, 3)
    normal[..., 2] = mask
    parts = mask.long() * torch.tensor([3, 8])[:, None, None]
    result = m.compose_dense_geometry(normal, depth, mask, parts, torch.zeros_like(depth), person_ids=(41, 7),
                                      camera=camera(), canvas=m.PixelCanvasTransform.letterbox((60, 80), (2, 2)))
    assert result.visible_owner.tolist() == [[2, 2], [1, 1]]
    assert result.person_ids == (41, 7)
    assert result.visibility[0, 0, 0] == 0 and result.visibility[1, 0, 0] == 1
    assert result.metadata["normal_basis"] == "camera_cv_xyz"
    assert result.metadata["depth_units"] == "metre"
    assert result.metadata["owner_encoding"] == "0=background; slot+1=visible_person"
    typed = result.typed_spatial()
    assert set(typed) == {"normal", "depth", "mask", "visibility", "part", "contact"}
    assert typed["normal"].shape == (1, 2, 3, 2, 2)
    assert typed["part"].dtype == torch.long
    assert typed["part"][0, 0, 0].tolist() == [[2, -1], [2, 2]]
    # Joint scene range [2,3] applies to both people, never per-person minmax.
    assert typed["depth"][0, :, 0].tolist() == [[[1., 0], [1., 1.]], [[0., 0.], [0., 1.]]]
    smaller = result.typed_spatial(target_hw=(1, 1))
    assert smaller["part"].shape == (1, 2, 1, 1, 1)


def test_renderer_and_backend_load_are_lazy():
    m = api()
    assert m.NVDiffDenseRenderer is not None
    # Import itself must not require nvdiffrast, smplx, chumpy or model assets.
    assert m.load_smplestx_backend and m.load_hi4d_smpl_backend


def test_smplestx_loader_rejects_an_existing_parent_package_from_another_root(tmp_path, monkeypatch):
    m = api()
    requested = tmp_path / "requested" / "human_models"
    requested.mkdir(parents=True)
    (requested / "human_models.py").write_text("raise AssertionError('REQUESTED_LOADER_IMPORTED')")
    wrong = tmp_path / "wrong" / "human_models"
    wrong.mkdir(parents=True)
    (wrong / "human_models.py").write_text("raise AssertionError('WRONG_LOADER_IMPORTED')")
    parent = types.ModuleType("human_models")
    parent.__path__ = [str(wrong)]
    monkeypatch.setitem(sys.modules, "human_models", parent)
    monkeypatch.delitem(sys.modules, "human_models.human_models", raising=False)
    with pytest.raises(RuntimeError, match="package|different|binding"):
        m.load_smplestx_backend(requested.parent)


def test_smplestx_loader_checks_imported_child_path_before_constructing_body(tmp_path, monkeypatch):
    m = api()
    requested = tmp_path / "requested" / "human_models"
    requested.mkdir(parents=True)
    (requested / "human_models.py").write_text("# requested loader")
    parent = types.ModuleType("human_models")
    parent.__path__ = [str(requested)]
    child = types.ModuleType("human_models.human_models")
    child.__file__ = str(tmp_path / "different" / "human_models.py")
    def wrong_constructor(*args):
        raise AssertionError("WRONG_BODY_CONSTRUCTED")
    child.SMPLX = wrong_constructor
    monkeypatch.setitem(sys.modules, "human_models", parent)
    monkeypatch.delitem(sys.modules, "human_models.human_models", raising=False)
    monkeypatch.setattr(m.importlib, "import_module", lambda name: child)
    with pytest.raises(RuntimeError, match="module|loader|binding"):
        m.load_smplestx_backend(requested.parent)


def test_camera_projection_epsilon_stabilizes_depth_without_scaling_principal_point():
    m = api()
    cam = m.Camera(torch.tensor([[100., 3, 40], [0, 120, 30], [0, 0, 1.]]), (60, 80))
    xyz = torch.tensor([[.4, .2, 2.], [0, 0, 3.]])
    xy = cam.project(xyz, frame="camera_cv", z_epsilon=.25)
    expected = torch.tensor([[(40 + .6) / 2.25 + 40, 24 / 2.25 + 30], [40, 30]])
    assert torch.allclose(xy, expected)


@pytest.mark.skipif(not os.environ.get("QWEN21_REAL_SMPL_ROOT"), reason="optional real CPU model assets are external")
def test_real_smplestx_cpu_backend_contract():
    m = api()
    backend = m.load_smplestx_backend(os.environ["QWEN21_REAL_SMPL_ROOT"])
    params = parameters(1, "smplx")
    result = m.retarget_request(m.SourceShape("smplx", (41,), params.betas), params, camera(), backend)
    assert result.requested.body.vertices.shape == (1, 10475, 3)
    assert result.requested.body.joints.shape == (1, 137, 3)
    assert result.source_rest_heights_m.item() > 1
    assert backend.provenance["model_sha256"]
    result.require_exact_gt_pairing()

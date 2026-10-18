"""共享入口、受保护路径与实际数据流路由；大模型在CPU契约中用小backend。"""
import importlib
import json
from pathlib import Path
import sys

import pytest


def api():
    try:
        return importlib.import_module("src.pose_control.qwen21_pro.runner")
    except ModuleNotFoundError:
        pytest.fail("Shared Pro CLI runner is missing")


def test_every_write_command_requires_new_name_and_existing_run_is_not_overwritten(tmp_path):
    module = api()
    root, legacy = tmp_path / "pro", tmp_path / "old"
    run = module.allocate_run(root, "pilot_20261018", legacy_workspace=legacy)
    (run / "sentinel.json").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.allocate_run(root, "pilot_20261018", legacy_workspace=legacy)
    assert (run / "sentinel.json").read_text() == "old"
    for name in ("../old", "a/b", "a\\b", "", ".."):
        with pytest.raises(ValueError):
            module.allocate_run(root, name, legacy_workspace=legacy)
    for protected in (legacy, legacy / "child", tmp_path / "models", tmp_path / "datasets"):
        with pytest.raises(ValueError):
            module.allocate_run(protected, "blocked", legacy_workspace=legacy)


def test_status_is_read_only_lightweight_and_does_not_create_root(tmp_path, capsys):
    module = api()
    absent = tmp_path / "absent"
    assert module.main(["status", "--output-root", str(absent)]) == 0
    assert not absent.exists()
    data = json.loads(capsys.readouterr().out)
    assert data["runs"] == []
    root = tmp_path / "pro"
    run = root / "runs/already"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"command": "smoke"}))
    (run / "gate.json").write_text(json.dumps({"status": "incomplete", "passed": False}))
    before = sorted((p.relative_to(root), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())
    assert module.main(["status", "--output-root", str(root)]) == 0
    assert sorted((p.relative_to(root), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()) == before
    assert json.loads(capsys.readouterr().out)["runs"][0]["gate"]["status"] == "incomplete"


def test_script_status_does_not_import_torch(tmp_path):
    module = api()
    import subprocess
    script = Path(module.CODE_ROOT) / "scripts/run/qwen21_pro.py"
    code = """import importlib.abc, runpy, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch':
            raise RuntimeError('status imported torch')
sys.meta_path.insert(0, BlockTorch())
sys.argv = [sys.argv[1], 'status', '--output-root', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(script), str(tmp_path / "absent")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runs"] == []


def test_parser_exposes_real_commands_and_fixed_protocol_defaults():
    module = api()
    parser = module.build_parser()
    for command in ("prepare", "cache", "audit", "smoke", "train", "overfit", "ablate", "infer", "refine"):
        args = parser.parse_args([command, "--name", "contract"])
        assert args.command == command
        assert args.memory_mode == "resident"
        assert args.arm == "full4"
    assert parser.parse_args(["smoke", "--name", "exact", "--exact-resume"]).exact_resume


def test_import_code_signature_changes_when_actual_code_changes(tmp_path):
    module = api()
    root = tmp_path / "code"
    p = root / "src/pose_control/qwen21_pro/runner.py"
    p.parent.mkdir(parents=True)
    p.write_text("x=1")
    first = module.source_snapshot(root)
    p.write_text("x=2")
    assert module.source_snapshot(root)["signature"] != first["signature"]
    stable = module.source_snapshot(root)["signature"]
    unrelated = root / "src/unused_model.py"
    unrelated.write_text("unused = 1")
    assert module.source_snapshot(root)["signature"] == stable
    dependency = root / "src/pose_edit/geometry.py"
    dependency.parent.mkdir()
    dependency.write_text("coordinate = 1")
    assert module.source_snapshot(root)["signature"] != stable


def test_cli_dispatch_is_real_and_records_actual_import_root(tmp_path, monkeypatch):
    module = api()
    observed = []
    def execute(args, run):
        observed.append((args.command, run))
        return {"samples": 2, "passed": False}
    monkeypatch.setitem(module.COMMANDS, "prepare", execute)
    assert module.main(["prepare", "--name", "route", "--output-root", str(tmp_path / "pro")]) == 0
    manifest = json.loads((observed[0][1] / "run.json").read_text())
    assert manifest["code_root"] == str(Path(module.__file__).resolve().parents[3])
    assert manifest["command"] == "prepare" and manifest["code_signature"]
    assert Path(manifest["run_dir"]).is_absolute()
    assert (observed[0][1] / "code_snapshot/src/pose_control/qwen21_pro/runner.py").is_file()


def test_geometry_jobs_use_own_source_and_driver_metadata_and_retarget_unpaired(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from PIL import Image
    image = tmp_path / "raw/pair01/cam01/source/0001/rgb.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (94, 128)).save(image)
    record = {"sample_id": "hi4d_val_one", "dataset": "hi4d", "split": "val", "group_id": "pair01",
              "person_order": [1, 0], "source_rgb": str(image), "metadata": {}}
    assets = {"source_rgb": image, "source_masks": [image, image],
              "contact_mesh": tmp_path / "raw/pair01/driver/smpl/0002.npz", "camera": image,
              "camera_id": 1, "target_keypoints": []}
    monkeypatch.setattr(module, "resolve_condition_assets", lambda *a: assets)
    monkeypatch.setattr(module, "source_geometry_paths", lambda *a: {"source_raw_parameters": tmp_path / "raw/pair01/source/smpl/0001.npz",
        "source_camera": image, "source_camera_id": 1})
    jobs = module.geometry_jobs([record], project_root=tmp_path, run_dir=tmp_path / "run", mode="all", canvas_hw=(512, 768))
    source, current, retarget = [value["job"] for value in jobs]
    assert source["source_geometry_only"]
    assert Path(source["gender_metadata"]).parts[-2:] == ("source", "meta.npz")
    assert Path(current["driver_gender_metadata"]).parts[-2:] == ("driver", "meta.npz")
    assert source["person_order"] == current["person_order"] == [1, 0]
    assert current["mode"] == "driver_exact" and current["for_training"]
    assert retarget["mode"] == "source_shape" and not retarget["for_training"]
    assert retarget["canvas_hw"] == [512, 768]


def test_native_mask_is_new_full_fov_nearest_transform_and_original_is_bound(tmp_path):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    import numpy as np
    from PIL import Image
    source, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (20, 40)).save(source)
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(mask)
    original = mask.read_bytes()
    prepared, binding = module.native_source_mask(source, mask, tmp_path / "new_native_mask.png")
    assert Image.open(prepared).size == (20, 40)
    assert np.asarray(Image.open(prepared))[:20, :10].min() == 255
    assert binding["original_mask_sha256"]
    assert mask.read_bytes() == original


def test_infer_routes_actual_variable_grid_sampling_and_writes_bound_coarse(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro import inference, refinement
    seen = []
    pack = SimpleNamespace(target_hw=(32, 48), source_fingerprint="source", provenance={"geometry": {"canvas": "real"}})
    args = SimpleNamespace(sample_id=["one"], checkpoint="actual_checkpoint", steps=3, seed=43,
                           true_cfg_scale=1., output_root=str(tmp_path), device="cpu", engineering_probe=True)
    monkeypatch.setattr(module, "records", lambda _: [{"sample_id": "one"}])
    monkeypatch.setattr(module, "_inference_pipeline", lambda *a, **k: (SimpleNamespace(), "cfg", {"actual": "metadata"}))
    monkeypatch.setattr(module, "_encoded_samples", lambda *a, **k: ({"one": ("sealed_condition", None, pack)}, "source"))
    def sample(pipe, condition, **kwargs):
        seen.append(("sample", condition, kwargs["task_condition"].target_hw, kwargs["num_inference_steps"]))
        return SimpleNamespace(audit={"actual_sample": True})
    def coarse(result, **kwargs):
        seen.append(("coarse", kwargs["stage1_checkpoint"], kwargs["geometry_signature"]))
        return SimpleNamespace(manifest_path=tmp_path / "coarse.json")
    monkeypatch.setattr(inference, "sample_cached", sample)
    monkeypatch.setattr(refinement, "write_coarse_artifact", coarse)
    result = module.infer(args, tmp_path)
    assert seen == [("sample", "sealed_condition", (32, 48), 3), ("coarse", "actual_checkpoint", {"canvas": "real"})]
    assert result["coarse_images"] == 1


def test_current_cache_tamper_is_rejected_before_torch_payload_load(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    cfg = ProAdapterConfig(source_dim=1600)
    (tmp_path / "encoded_index.json").write_text("{}")
    (tmp_path / "report.json").write_text(json.dumps({"kind": "current", "config": cfg.to_dict(),
        "passed": True, "index_sha256": "0" * 64}))
    args = SimpleNamespace(cache_run=str(tmp_path))
    with pytest.raises(ValueError, match="index changed"):
        module._encoded_samples(args, [], cfg)


def test_refine_executes_source_crop_bank_localization_request_and_real_refiner_route(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from src.pose_control.qwen21_pro import refinement
    from src.pose_control.qwen21_pro.features import RawSourceFeatures
    from src.pose_control.qwen21_pro.cache import ProCache
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    run, cache_root = tmp_path / "run", tmp_path / "cache_run"
    run.mkdir()
    rgb, mask = tmp_path / "source.png", tmp_path / "mask.png"
    Image.new("RGB", (16, 16), "red").save(rgb)
    Image.new("L", (8, 8), 255).save(mask)
    raw = RawSourceFeatures((0,), torch.zeros(1, 1, 1, 1600), torch.ones(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 512), torch.zeros(1, 1, dtype=torch.bool), "a" * 64,
        {"coverage": [{"crops": [{"kind": "face", "box": [0, 0, 16, 16]}]}]})
    ProCache(cache_root / "cache", namespace="raw_source").put("one", raw, raw.fingerprint, signature="fixture")
    geometry = {"canvas": {"output_hw": [64, 64]}}
    manifest = tmp_path / "geometry.json"
    manifest.write_text(json.dumps({"geometry_signature": geometry, "rois": [], "control_image": str(rgb)}))
    index = tmp_path / "geometry_index.json"
    index.write_text(json.dumps({"one": str(manifest)}))
    pack = SimpleNamespace(source_fingerprint=raw.fingerprint, provenance={"geometry": geometry}, person_ids=(0,))
    coarse = SimpleNamespace(manifest={"source_fingerprint": raw.fingerprint, "geometry_signature": geometry}, image_path=rgb)
    args = SimpleNamespace(coarse="actual_coarse", checkpoint="actual_checkpoint", current_geometry_index=str(index),
        cache_run=str(cache_root), roi_kind="face", roi_person=0, hand_endpoint=[], output_root=str(tmp_path),
        seed=43, steps=2, device="cpu", true_cfg_scale=4.)
    pipe, cfg = SimpleNamespace(trained_model_signature={"actual": "checkpoint"}), ProAdapterConfig(source_dim=1600)
    monkeypatch.setattr(module, "records", lambda _: [{"sample_id": "one"}])
    monkeypatch.setattr(module, "_inference_pipeline", lambda *a, **k: (pipe, cfg, {}))
    monkeypatch.setattr(module, "_encoded_samples", lambda *a, **k: ({"one": (None, None, pack)}, "source"))
    monkeypatch.setattr(module, "validated_refinement_geometry", lambda *a, **k: json.loads(manifest.read_text()), raising=False)
    monkeypatch.setattr(module, "_assets", lambda *a: {"source_rgb": rgb, "source_masks": [mask]})
    monkeypatch.setattr(refinement.CoarseArtifact, "load", lambda *a: coarse)
    roi = refinement.ROITransform.from_box("face", (0,), [0, 0, 64, 64], (64, 64))
    monkeypatch.setattr(refinement, "derive_refinement_rois", lambda *a, **k: [roi])
    monkeypatch.setattr(module, "_extractor", lambda _: "actual_extractor")
    seen = []
    def bank(pipe, extractor, crops, **kwargs):
        seen.append(("source_bank", extractor, len(crops), crops[0].verify().part_id))
        return SimpleNamespace(fingerprint="roi_bank")
    def local(*a, **kwargs):
        seen.append(("localize", kwargs["source_features"].fingerprint))
        return SimpleNamespace(target_hw=(48, 48))
    def request(*a, **kwargs):
        seen.append(("request", kwargs["task_condition"].target_hw, kwargs["seed"]))
        return "real_request"
    def execute(pipe, request, **kwargs):
        seen.append(("refine", request, kwargs["num_inference_steps"], kwargs["true_cfg_scale"]))
        return SimpleNamespace(image_path=rgb, audit={"fallback_reason": None, "outside_edit_mask_bitwise_equal": True})
    monkeypatch.setattr(refinement, "encode_roi_source_bank", bank)
    monkeypatch.setattr(refinement, "localize_task_condition", local)
    monkeypatch.setattr(refinement, "prepare_refinement_request", request)
    monkeypatch.setattr(refinement, "run_refinement", execute)
    report = module.refine(args, run)
    assert seen == [("source_bank", "actual_extractor", 1, 0), ("localize", "roi_bank"),
                    ("request", (48, 48), 43), ("refine", "real_request", 2, 4.)]
    assert report["refined_images"] == 1
    assert Image.open(mask).size == (8, 8)


def test_refine_rejects_substituted_manifest_even_when_geometry_signature_is_copied(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro import refinement
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    geometry = {"canvas": {"output_hw": [64, 64]}}
    original, substituted = tmp_path / "audited.json", tmp_path / "substituted.json"
    original.write_text(json.dumps({"geometry_signature": geometry, "rois": [], "control_image": "audited_control.png"}))
    substituted.write_text(json.dumps({"geometry_signature": geometry, "rois": [], "control_image": "different_control.png"}))
    index = tmp_path / "requested_index.json"
    index.write_text(json.dumps({"one": str(substituted)}))
    cfg = ProAdapterConfig(source_dim=1600)
    pack = SimpleNamespace(source_fingerprint="bound_source", provenance={"geometry": geometry})
    coarse = SimpleNamespace(manifest={"source_fingerprint": "bound_source", "geometry_signature": geometry},
                             image_path=tmp_path / "coarse.png")
    args = SimpleNamespace(coarse="coarse.json", current_geometry_index=str(index), roi_kind="face", roi_person=0)
    monkeypatch.setattr(module, "records", lambda _: [{"sample_id": "one"}])
    monkeypatch.setattr(module, "_inference_pipeline", lambda *a, **k: (None, cfg, {}))
    monkeypatch.setattr(module, "_encoded_samples", lambda *a, **k: ({"one": (None, None, pack)}, "source"))
    monkeypatch.setattr(module, "encoded_cache_layout", lambda *a: (tmp_path / "native", {
        "one": {"geometry_manifest": str(original), "native_audit_adapter": True}}, "source"))
    monkeypatch.setattr(refinement.CoarseArtifact, "load", lambda _: coarse)
    monkeypatch.setattr(refinement, "derive_refinement_rois", lambda *a, **k: [])
    with pytest.raises(ValueError, match="cache.bound.*manifest|manifest.*cache.bound"):
        module.refine(args, tmp_path)


def real_geometry_handoff(tmp_path, *, mode="driver_exact", beta_delta=0):
    from dataclasses import replace
    from test_qwen21_pro_contracts import make_condition
    from test_qwen21_pro_geometry import RodBody, request
    from test_qwen21_pro_geometry_preparation import Renderer
    from test_qwen21_pro_encoded import Encoder
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    from src.pose_control.qwen21_pro.geometry_preparation import prepare_geometry
    from src.pose_control.qwen21_pro.encoded import encode_prepared_condition
    from PIL import Image
    cfg = ProAdapterConfig()
    prepared = prepare_geometry(request(beta_delta=beta_delta), body_model=RodBody(), renderer=Renderer(),
        output_dir=tmp_path / "current", output_root=tmp_path, canvas_hw=(64, 96), mode=mode)
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 96), "green").save(source)
    manifest = tmp_path / "current/manifest.json"
    condition, pack = encode_prepared_condition(Encoder(), replace(make_condition(), person_ids=(41, 7)),
        manifest_path=manifest, config=cfg, sample_id="one", source_images=[source], prompt="positive",
        producer_signature={"weights": "test_encoder"})
    return condition, pack, cfg, manifest


@pytest.mark.parametrize("mutation", [None, "control_path", "pose_path", "rois", "dense_bytes", "helper_code"])
@pytest.mark.parametrize("audited_native", [True, False])
def test_native_refiner_revalidates_actual_control_pose_dense_helper_and_pack(tmp_path, monkeypatch, mutation, audited_native):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro.cache import file_sha256
    from PIL import Image
    condition, pack, cfg, manifest_path = real_geometry_handoff(tmp_path)
    index_path = tmp_path / "requested.json"
    index_path.write_text(json.dumps({"one": str(manifest_path)}))
    item = {"geometry_manifest": str(manifest_path), "native_audit_adapter": audited_native,
            "geometry_manifest_sha256": file_sha256(manifest_path), "binding": condition.binding}
    monkeypatch.setattr(module, "encoded_cache_layout", lambda *a: (tmp_path / "native", {"one": item}, "source"))
    manifest = json.loads(manifest_path.read_text())
    if mutation in ("control_path", "pose_path"):
        different = tmp_path / "different.png"
        Image.new("RGB", (96, 64), "red").save(different)
        if mutation == "control_path":
            manifest["control_image"] = str(different)
            manifest["asset_hashes"]["control"] = file_sha256(different)
        else:
            manifest["pose_images"][0] = str(different)
            manifest["asset_hashes"]["pose_0"] = file_sha256(different)
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "rois":
        manifest["rois"] = []
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "dense_bytes":
        with (manifest_path.parent / "geometry.pt").open("ab") as stream:
            stream.write(b"tampered")
    elif mutation == "helper_code":
        import src.pose_control.qwen21_pro.geometry_preparation as geometry
        monkeypatch.setattr(geometry, "_geometry_code", lambda: {"changed": "helper"})
    args = SimpleNamespace(current_geometry_index=str(index_path))
    if mutation is None:
        assert module.validated_refinement_geometry(args, {"sample_id": "one"}, cfg, condition, pack) == manifest
    else:
        with pytest.raises(ValueError, match="geometry|manifest|SHA|Encoded|handoff|ROI"):
            module.validated_refinement_geometry(args, {"sample_id": "one"}, cfg, condition, pack)


def test_unpaired_current_supervision_rejected_before_target_is_opened(tmp_path):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    with pytest.raises(ValueError, match="unpaired"):
        module._save_supervision(None, None, {}, {"target_rgb": "/unreadable.png"}, {"paired_supervision": False}, tmp_path)


def test_forged_paired_flag_does_not_read_any_gt_bytes(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    gt = tmp_path / "forbidden_gt.png"
    original = module.file_sha256
    def guard(path):
        if Path(path) == gt:
            pytest.fail("Unpaired supervision read GT before verifying geometry")
        return original(path)
    monkeypatch.setattr(module, "file_sha256", guard)
    manifest = {"schema": "current_geometry_preparation_v1", "paired_supervision": True,
                "geometry_signature": {"mode": "source_shape"}}
    with pytest.raises(ValueError, match="unpaired|driver_exact"):
        module._save_supervision(None, None, {}, {"target_rgb": str(gt), "metadata": {"target_sha256": "0" * 64}},
                                 manifest, tmp_path)


@pytest.mark.parametrize("mutation", [None, "paired_bool", "mode_and_bool", "state", "dense_bytes", "schema"])
def test_supervision_requires_actual_unchanged_exact_driver_geometry(tmp_path, monkeypatch, mutation):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from test_qwen21_pro_encoded import Encoder
    from src.pose_control.qwen21_pro.cache import file_sha256
    from PIL import Image
    import torch
    source_shape = mutation in ("paired_bool", "mode_and_bool")
    condition, pack, cfg, path = real_geometry_handoff(tmp_path,
        mode="source_shape" if source_shape else "driver_exact", beta_delta=1 if source_shape else 0)
    manifest = json.loads(path.read_text())
    if mutation == "paired_bool":
        manifest["paired_supervision"] = True
    elif mutation == "mode_and_bool":
        manifest["paired_supervision"] = True
        manifest["geometry_signature"]["mode"] = "driver_exact"
    elif mutation == "state":
        data = torch.load(path.parent / "geometry.pt", map_location="cpu", weights_only=True)
        data["state"]["parameters"]["betas"].add_(1)
        with (tmp_path / "changed_state.pt").open("xb") as stream:
            torch.save(data, stream)
        # Point the immutable manifest proof at changed dense bytes by writing a
        # separate malformed handoff, never changing the real prepared images.
        changed = tmp_path / "malformed"
        changed.mkdir()
        import shutil
        shutil.copyfile(tmp_path / "changed_state.pt", changed / "geometry.pt")
        path = changed / "manifest.json"
        manifest["geometry_signature"]["dense_sha256"] = file_sha256(changed / "geometry.pt")
    elif mutation == "dense_bytes":
        with (path.parent / "geometry.pt").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "schema":
        manifest["schema"] = "unverified_geometry"
    if mutation is not None:
        path.write_text(json.dumps(manifest))
    gt = tmp_path / "target.png"
    Image.new("RGB", (80, 60), "blue").save(gt)
    truth = {"target_rgb": str(gt), "metadata": {"target_sha256": file_sha256(gt)}}
    run = tmp_path / "supervision_run"
    run.mkdir()
    original = module.file_sha256
    if mutation is not None:
        def guard(file):
            if Path(file) == gt:
                pytest.fail("Malformed geometry read GT before its exact-pair proof")
            return original(file)
        monkeypatch.setattr(module, "file_sha256", guard)
        with pytest.raises(ValueError, match="geometry|driver|paired|schema|SHA|state"):
            module._save_supervision(None, Encoder(), {"sample_id": "one"}, truth, manifest, run,
                manifest_path=path, task_condition=pack, config=cfg)
    else:
        info = module._save_supervision(None, Encoder(), {"sample_id": "one"}, truth, manifest, run,
            manifest_path=path, task_condition=pack, config=cfg)
        payload = torch.load(info["path"], weights_only=True)
        assert tuple(payload["target_latent"].shape) == (1, 24, 64)
        assert payload["valid_target_tokens"].sum() > 0


def test_cpu_checkpoint_rejects_cuda_execution_metadata_before_any_mutation(tmp_path, monkeypatch, request):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    import copy
    import torch
    from test_qwen21_pro_training import TinyPro, run
    from test_qwen21_pro_checkpoint import metadata, equal_tree
    from src.pose_control.qwen21_pro import checkpoint, training
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    model = TinyPro()
    result, optimizer, lr = run(model, stop_after_updates=2)
    cpu = module.training_execution_protocol("cpu", frozen_dtype=torch.float32, noise_dtype=torch.float32)
    cuda = module.training_execution_protocol("cuda", frozen_dtype=torch.float32, noise_dtype=torch.float32)
    assert cpu["training_device"] == "cpu" and cuda["training_device"] == "cuda"
    assert cpu["precision"] != cuda["precision"] and cpu["noise_backend"] != cuda["noise_backend"]
    order = (("iper", "one"), ("tiktok", "two"), ("hi4d", "three"))
    cpu_trajectory = training.trajectory_signature(order, precision=cpu["precision"])
    cuda_trajectory = training.trajectory_signature(order, precision=cuda["precision"])
    assert cpu_trajectory != cuda_trajectory
    saved_meta = {**metadata(model), **cpu, "trajectory_signature": cpu_trajectory}
    path = tmp_path / "cpu_checkpoint"
    checkpoint.save_checkpoint(path, model, optimizer, lr, metadata=saved_meta, cursor=result["cursor"])
    before = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr.state_dict(),
                            "rng": checkpoint.capture_rng_state()})
    monkeypatch.setattr(checkpoint, "_apply_incremental", lambda *a: pytest.fail("Checkpoint mutated before device rejection"))
    with pytest.raises(ValueError, match="metadata"):
        checkpoint.load_checkpoint(path, model, optimizer, lr,
            expected_metadata={**saved_meta, **cuda, "trajectory_signature": cuda_trajectory})
    equal_tree(before, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "lr": lr.state_dict(),
                        "rng": checkpoint.capture_rng_state()})


def selection_fixture(tmp_path, step=200):
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.checkpoint import TrainingCursor
    from dataclasses import asdict
    path = tmp_path / f"checkpoint_{step:06d}"
    path.mkdir()
    (path / "state.pt").write_bytes(b"unit fixture for immutable weight file identity")
    cursor = asdict(TrainingCursor(optimizer_step=step, microstep=step, next_sample_position=step,
        noise_draw_cursor=step, timestep_draw_cursor=step, loss_log_cursor=step, gradient_log_cursor=step,
        accumulation_steps=1))
    metadata = {"arm": "full4", "profile": "overfit", "selection_role": "controlled_pilot"}
    manifest = {"cursor": cursor, "metadata": metadata, "state_sha256": file_sha256(path / "state.pt")}
    (path / "checkpoint.json").write_text(json.dumps(manifest))
    attestation = {"path": str(path.resolve()), "cursor": cursor, "metadata": metadata,
        "state_sha256": manifest["state_sha256"], "manifest_sha256": file_sha256(path / "checkpoint.json"),
        "selection_role": "selected_stage1"}
    gate = {"passed": True, "engineering_passed": True, "overfit_metric_gate": {"passed": True},
            "profile": "overfit", "status": "complete", "cursor": cursor, "final_checkpoint": attestation}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    return path, manifest


@pytest.mark.parametrize("step,mutation", [(200, None), (0, None), (25, None), (200, "state"), (200, "metadata")])
def test_formal_selection_is_bound_to_the_passed_final200_checkpoint(tmp_path, step, mutation):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    path, manifest = selection_fixture(tmp_path, step)
    if mutation == "state":
        with (path / "state.pt").open("ab") as stream:
            stream.write(b"modified")
    elif mutation == "metadata":
        manifest["metadata"]["profile"] = "smoke"
        (path / "checkpoint.json").write_text(json.dumps(manifest))
    if step == 200 and mutation is None:
        assert module.validate_quality_checkpoint(path) == manifest
    else:
        with pytest.raises(ValueError, match="quality|final|checkpoint|200|SHA|metadata"):
            module.validate_quality_checkpoint(path)


@pytest.mark.parametrize("step", [0, 25])
def test_formal_inference_rejects_early_checkpoint_despite_parent_gate_pass(tmp_path, monkeypatch, step):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro import pipeline, runner
    from src.pose_control.qwen21_pro.cache import stable_hash
    from src.pose_control.qwen21_pro.losses import RegionalLossConfig
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    path = tmp_path / f"checkpoint_{step:06d}"
    path.mkdir()
    metadata = {"config": ProAdapterConfig(source_dim=1600).to_dict(), "scope": "joint", "arm": "full4",
                "task_signature": stable_hash(RegionalLossConfig().to_dict()), "source_signature": stable_hash({}),
                "code_signature": "actual_source"}
    (path / "checkpoint.json").write_text(json.dumps({"metadata": metadata}))
    (tmp_path / "gate.json").write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"signature": "actual_source"})
    monkeypatch.setattr(module, "source_model_signature", lambda _: {})
    monkeypatch.setattr(module, "_cuda_budget", lambda _: {"device": "cpu"})
    monkeypatch.setattr(pipeline, "build_pro_pipeline", lambda *a, **k: (SimpleNamespace(), {}))
    args = SimpleNamespace(checkpoint=str(path), steps=40, true_cfg_scale=1., arm="full4", engineering_probe=False,
        device="cpu", offload="none", base_model="base", project_root="project", source_root=None, union_checkpoint="union")
    run = tmp_path / "inference"
    run.mkdir()
    with pytest.raises(ValueError, match="quality|final|checkpoint|200"):
        module._inference_pipeline(args, run, cached=True)


def test_same_person_paired_hands_can_prepare_two_native_masks(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from dataclasses import dataclass
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from src.pose_control.qwen21_pro import refinement
    from src.pose_control.qwen21_pro.cache import ProCache, asset_hashes
    from src.pose_control.qwen21_pro.features import RawSourceFeatures
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    run, cache_root = tmp_path / "run", tmp_path / "cache_run"
    run.mkdir()
    rgb, mask = tmp_path / "source.png", tmp_path / "small_mask.png"
    Image.new("RGB", (16, 16), "red").save(rgb)
    Image.new("L", (8, 8), 255).save(mask)
    raw = RawSourceFeatures((0,), torch.zeros(1, 1, 1, 1600), torch.ones(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 512), torch.zeros(1, 1, dtype=torch.bool), "b" * 64,
        {"coverage": [{"crops": [{"kind": part, "box": [0, 0, 16, 16]} for part in ("left_hand", "right_hand")]}]})
    ProCache(cache_root / "cache", namespace="raw_source").put("one", raw, raw.fingerprint, signature="fixture")
    geometry = {"canvas": {"output_hw": [64, 64]}}
    manifest = {"geometry_signature": geometry, "rois": [], "control_image": str(rgb)}
    pack = SimpleNamespace(source_fingerprint=raw.fingerprint, provenance={"geometry": geometry}, person_ids=(0,))
    coarse = SimpleNamespace(manifest={"source_fingerprint": raw.fingerprint, "geometry_signature": geometry}, image_path=rgb)
    roi = refinement.ROITransform.from_box("paired_hands", (0,), [0, 0, 64, 64], (64, 64),
        endpoints=(refinement.PartEndpoint(0, 4), refinement.PartEndpoint(0, 7)))
    args = SimpleNamespace(coarse="coarse.json", cache_run=str(cache_root), roi_kind="paired_hands", roi_person=None,
        hand_endpoint=["0:4", "0:7"], output_root=str(tmp_path), seed=43, steps=2, device="cpu", true_cfg_scale=1.)
    pipe, cfg = SimpleNamespace(trained_model_signature={"weights": "actual"}), ProAdapterConfig(source_dim=1600)
    monkeypatch.setattr(module, "records", lambda _: [{"sample_id": "one"}])
    monkeypatch.setattr(module, "_inference_pipeline", lambda *a, **k: (pipe, cfg, {}))
    monkeypatch.setattr(module, "_encoded_samples", lambda *a, **k: ({"one": (None, None, pack)}, "source"))
    monkeypatch.setattr(module, "validated_refinement_geometry", lambda *a, **k: manifest)
    monkeypatch.setattr(module, "_assets", lambda *a: {"source_rgb": rgb, "source_masks": [mask]})
    monkeypatch.setattr(refinement.CoarseArtifact, "load", lambda _: coarse)
    monkeypatch.setattr(refinement, "derive_refinement_rois", lambda *a, **k: [roi])
    monkeypatch.setattr(module, "_extractor", lambda _: "extractor")
    @dataclass
    class Crop:
        person_id: int
        part_id: int
        files: dict
        hashes: dict
        def verify(self):
            assert asset_hashes(self.files) == self.hashes
            return self
    def crop(**kwargs):
        files = {"source_rgb": str(kwargs["source_image"]), "native_mask": str(kwargs["source_mask"])}
        return Crop(kwargs["person_id"], kwargs["part_id"], files, asset_hashes(files))
    seen = []
    def bank(pipe, extractor, crops, **kwargs):
        seen.extend((c.person_id, c.part_id) for c in crops)
        return SimpleNamespace(fingerprint="both_hands")
    monkeypatch.setattr(refinement, "write_source_crop", crop)
    monkeypatch.setattr(refinement, "encode_roi_source_bank", bank)
    monkeypatch.setattr(refinement, "localize_task_condition", lambda *a, **k: SimpleNamespace(target_hw=(48, 48)))
    monkeypatch.setattr(refinement, "prepare_refinement_request", lambda *a, **k: "paired_request")
    monkeypatch.setattr(refinement, "run_refinement", lambda *a, **k: SimpleNamespace(image_path=rgb,
        audit={"fallback_reason": None, "outside_edit_mask_bitwise_equal": True}))
    report = module.refine(args, run)
    assert seen == [(0, 4), (0, 7)]
    assert report["refined_images"] == 1
    assert len({value["native_mask"] for value in report["source_mask_transforms"]}) == 2
    assert all(Image.open(value["native_mask"]).size == (16, 16) for value in report["source_mask_transforms"])
    assert Image.open(mask).size == (8, 8)


def test_budget_is_fixed_once_from_available_memory_and_explicit32gib_is_checked(monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    import torch
    from types import SimpleNamespace
    free, total = 42_869_784_576, 89_171_099_648
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free, total))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", calls.append)
    args = SimpleNamespace(device="cuda", memory_budget_gib=None)
    default = module._cuda_budget(args)
    assert default["budget_bytes"] == free - 1024**3
    args.memory_budget_gib = 32
    assert module._cuda_budget(args)["budget_bytes"] == 32 * 1024**3
    args.memory_budget_gib = 40
    with pytest.raises(RuntimeError, match="Fixed CUDA budget"):
        module._cuda_budget(args)
    assert len(calls) == 2


def test_preregistered_native_cache_adapter_is_read_only_and_checks_real_producer(tmp_path, monkeypatch):
    module = importlib.import_module("src.pose_control.qwen21_pro.dataflow") if api() else None
    from types import SimpleNamespace
    from src.pose_control.qwen21_pro.cache import stable_hash
    from src.pose_control.qwen21_pro.contracts import ProAdapterConfig
    cfg = ProAdapterConfig(source_dim=1600)
    producer = {"assets": {"actual": "weights"}, "runtime": {"torch": "2.8"}}
    report = {"passed": True, "config": cfg.to_dict(), "samples": [{"sample_id": "one",
        "condition_binding": {"producer": {**producer, "conditioning_offload": "sync_group_offload"}},
        "geometry_manifest": "real_geometry.json", "source_geometry": "real_source.json", "raw_cache": {}}],
        "native_cache": str(tmp_path / "native"), "source_signature": stable_hash({"real": "extractors"}),
        "producer_signature": {**producer, "conditioning_offload": "sync_group_offload"},
        "target_rgb_read": False, "sampled_or_trained": False}
    (tmp_path / "report.json").write_text(json.dumps(report))
    before = (tmp_path / "report.json").read_bytes()
    monkeypatch.setattr(module, "source_model_signature", lambda _: {"real": "extractors"})
    monkeypatch.setattr(module, "_current_producer", lambda _: producer)
    args = SimpleNamespace(cache_run=str(tmp_path))
    root, index, signature = module.encoded_cache_layout(args, cfg)
    assert root == tmp_path / "native"
    assert index["one"]["binding"] == report["samples"][0]["condition_binding"]
    assert index["one"]["native_audit_adapter"]
    assert before == (tmp_path / "report.json").read_bytes()
    report["producer_signature"]["assets"] = {"actual": "changed"}
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="producer"):
        module.encoded_cache_layout(args, cfg)

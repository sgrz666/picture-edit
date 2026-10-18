"""共享CLI的真实数据接线；条件、监督、身体worker和模型环境分别校验。"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import torch

from .cache import (ProCache, asset_hashes, file_sha256, load_bound_condition, ordinary_cpu,
                    stable_hash, typed_pack_fingerprint)
from .contracts import ProAdapterConfig, TaskCondition
from .features import FrozenSourceFeatureExtractor, RawSourceFeatures
from .geometry_preparation import load_source_geometry, source_geometry_paths
from .preparation import (geometry_asset_roles, prepare_source_task_condition, prepare_task_condition,
                          resolve_condition_assets)
from .gates import write_json

PRODUCER_SHA256 = "ffb3077938662e16277d5e4862af8c3466e316de4f0c3b64ec107981d23ad0ae"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(args):
    path = Path(args.conditions or Path(args.legacy_workspace) / "data/overfit6_conditions.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Condition manifest requires nonempty unique sample IDs")
    if any(r.get("split") == "test" for r in rows):
        raise ValueError("Formal test is gated separately; pilot CLI does not access test records")
    if args.sample_id:
        selected = set(args.sample_id)
        if selected - {r["sample_id"] for r in rows}:
            raise ValueError("Requested sample ID absent from condition manifest")
        rows = [r for r in rows if r["sample_id"] in selected]
    return rows


def configuration(args):
    from .ablations import arm_stages
    return ProAdapterConfig(source_dim=1600, stage_indices=arm_stages(args.arm))


def _assets(args, row):
    assets = resolve_condition_assets(row, args.project_root)
    index = _json(args.source_geometry_index) if args.source_geometry_index else {}
    if row["sample_id"] in index:
        assets["source_geometry"] = load_source_geometry(index[row["sample_id"]], row, assets)
    if args.contact_vertex_parts:
        assets["contact_vertex_parts"] = Path(args.contact_vertex_parts)
    return assets


def geometry_jobs(rows, *, project_root, run_dir, mode="all", canvas_hw=(512, 512)):
    """own source camera/metadata与driver metadata分别构job，不在Qwen进程import身体库。"""
    from PIL import Image
    root, run = Path(project_root), Path(run_dir)
    if mode not in ("source", "current", "retarget", "all") or any(v <= 0 or v % 32 for v in canvas_hw):
        raise ValueError("Invalid geometry mode/canvas; native sizes must be divisible by32")
    jobs = []
    for row in rows:
        sid, dataset = row["sample_id"], row["dataset"]
        assets = resolve_condition_assets(row, root)
        base = {"dataset": dataset, "group_id": row.get("group_id"), "person_order": row["person_order"]}
        if dataset == "hi4d":
            source = source_geometry_paths(row, root)
            with Image.open(assets["source_rgb"]) as image:
                native_hw = [image.height, image.width]
            source_params = Path(source["source_raw_parameters"])
            driver_params = Path(assets["contact_mesh"])
            base.update(body_model_root=str(root / "add/tools/SMPLest-X/human_models/human_model_files"),
                source_params=str(source_params), gender_metadata=str(source_params.parent.parent / "meta.npz"))
            if mode in ("source", "all"):
                source_job = {**base, "source_geometry_only": True, "source_camera_path": str(source["source_camera"]),
                    "source_camera_id": source["source_camera_id"], "source_image_hw": native_hw,
                    "source_rgb": str(assets["source_rgb"]), "source_masks": list(map(str, assets["source_masks"])),
                    "output_dir": str(run / sid / "source")}
                jobs.append({"sample_id": sid, "kind": "source", "job": source_job})
            base.update(driver_params=str(driver_params), driver_gender_metadata=str(driver_params.parent.parent / "meta.npz"),
                camera_path=str(assets["camera"]), camera_id=assets["camera_id"], image_hw=native_hw)
        else:
            seq = row["metadata"]["sequence_id"]
            folder = (root / "datasets/iPER/iper_sampled_6src64tgt" / seq / "smplestx" if dataset == "iper" else
                      root / "datasets/TikTokDataset/TikTok_3d_assets_native" / seq / "smplestx")
            base.update(smplest_root=str(root / "add/tools/SMPLest-X"), params=str(folder / "params.pt"),
                frame_map=str(folder / "frame_map.json"), source_frame=Path(assets["source_rgb"]).stem,
                driver_frame=Path(assets["target_keypoints"][0]).stem, person_id=row["person_order"][0])
        for kind, worker_mode in (("current", "driver_exact"), ("retarget", "source_shape")):
            if mode not in (kind, "all"):
                continue
            job = {**base, "mode": worker_mode, "for_training": kind == "current",
                   "canvas_hw": list(canvas_hw), "output_dir": str(run / sid / worker_mode)}
            jobs.append({"sample_id": sid, "kind": kind, "job": job})
    if not jobs:
        raise ValueError("Source-only mesh bundles are needed only for Hi4D; select current/retarget for single-person data")
    return jobs


def prepare(args, run):
    from .runner import CODE_ROOT
    if not args.geometry_python or not Path(args.geometry_python).is_file():
        raise ValueError("prepare requires explicit existing --geometry-python from the body model environment")
    jobs = geometry_jobs(records(args), project_root=args.project_root, run_dir=run,
                         mode=args.geometry_mode, canvas_hw=tuple(args.canvas_hw))
    indexes = {kind: {} for kind in ("source", "current", "retarget")}
    job_dir = run / "jobs"
    job_dir.mkdir()
    for i, value in enumerate(jobs):
        path = job_dir / f"{i:03d}.json"
        write_json(path, value["job"])
        result = subprocess.run([args.geometry_python, "-m", "src.pose_control.qwen21_pro.geometry_preparation", str(path)],
                                cwd=CODE_ROOT, text=True, capture_output=True)
        with (job_dir / f"{i:03d}.log").open("x", encoding="utf-8") as stream:
            stream.write(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Geometry worker failed for {value['sample_id']} {value['kind']}; see {path.with_suffix('.log')}")
        artifact = Path(value["job"]["output_dir"]) / ("source_geometry.json" if value["kind"] == "source" else "manifest.json")
        if not artifact.is_file():
            raise ValueError("Geometry worker did not create its declared artifact")
        indexes[value["kind"]][value["sample_id"]] = str(artifact)
    for kind, index in indexes.items():
        write_json(run / f"{kind}_geometry_index.json", index)
    write_json(run / "jobs.json", jobs)
    return {"worker_jobs": len(jobs), "source_samples": len(indexes["source"]),
            "current_samples": len(indexes["current"]), "retarget_unpaired_samples": len(indexes["retarget"]),
            "completed": True, "quality_claim": False}


def _legacy_context(args):
    from ..qwen21_union.integrity import verify_workspace_inputs
    from ..qwen21_union.runner import model_signature
    root = Path(args.legacy_workspace)
    verify_workspace_inputs(root)
    event = root / "cache/completed_20261016T184014.json"
    if file_sha256(event) != PRODUCER_SHA256:
        raise ValueError("Legacy producer event SHA256 differs from pinned audit")
    producer = _json(event)["model_signature"]
    model_signature.cache_clear()
    current = model_signature()
    if Path(args.base_model).resolve() != Path(current["base_model"]).resolve():
        raise ValueError("base-model differs from actual encoder asset verifier")
    for key in ("assets", "configs", "preprocessing_code", "runtime", "videox_commit", "union_revision"):
        if current[key] != producer[key]:
            raise ValueError(f"Current encoder differs from frozen legacy producer: {key}")
    return producer, _json(Path(args.base_model) / "vae/config.json")


def source_model_signature(args):
    """只读实际extractor权重字节；训练和cached推理不必为验SHA加载DINO。"""
    root = Path(args.project_root) / "models/antelopev2"
    dino = root / "dinov2-giant"
    vae = Path(args.base_model) / "vae"
    vae_files = [vae / "config.json", *sorted(vae.glob("*.safetensors"))]
    if len(vae_files) == 1:
        raise FileNotFoundError("Source VAE weight shards are missing")
    if (vae / "diffusion_pytorch_model.safetensors.index.json").is_file():
        vae_files.append(vae / "diffusion_pytorch_model.safetensors.index.json")
    return {"dino": {name: file_sha256(dino / name) for name in ("config.json", "preprocessor_config.json", "model.safetensors")},
            "source_vae": {p.name: file_sha256(p) for p in vae_files},
            "face": {name: file_sha256(root / name) for name in ("scrfd_10g_bnkps.onnx", "glintr100.onnx")}}


def _extractor(args):
    root = Path(args.project_root) / "models/antelopev2"
    return FrozenSourceFeatureExtractor.from_local(root / "dinov2-giant", vae_root=Path(args.base_model) / "vae",
                                                  device=args.device, face_root=root)


def _legacy_samples(args, rows, cfg, *, destination=None, with_supervision=False):
    producer, vae = _legacy_context(args)
    samples, source_signatures, results = {}, [], []
    feature = Path(args.feature_run or args.cache_run) if args.feature_run or args.cache_run else None
    raw_input = ProCache(feature / "cache", namespace="raw_source") if feature else None
    extractor = None
    raw_output = ProCache(destination / "cache", namespace="raw_source") if destination else None
    typed_output = ProCache(destination / "cache", namespace="typed_condition") if destination else None
    truths = _truth_records(args) if with_supervision else {}
    for row in rows:
        sid = row["sample_id"]
        condition = load_bound_condition(row, Path(args.legacy_workspace) / "cache", model_signature=producer, vae_config=vae)
        assets = _assets(args, row)
        if row["dataset"] == "hi4d" and not args.contact_vertex_parts:
            raise ValueError("Legacy Hi4D conditions require explicit --contact-vertex-parts")
        if raw_input:
            meta = raw_input.metadata(sid)
            raw = RawSourceFeatures.from_dict(raw_input.get(sid, expected_hash=meta["hash"], signature=meta["signature"]))
        else:
            if destination is None:
                raise ValueError("Legacy training requires --feature-run from cache legacy")
            extractor = extractor or _extractor(args)
            raw = extractor.extract(row, condition, project_root=args.project_root, assets=assets)
        pack = prepare_task_condition(row, condition, project_root=args.project_root, config=cfg,
                                      raw_source_features=raw, assets=assets)
        source_signatures.append(raw.provenance["model_signature"]["extractors"])
        if destination:
            raw_output.put(sid, raw, raw.fingerprint, signature=stable_hash(raw.provenance["model_signature"]["extractors"]))
            fingerprint = typed_pack_fingerprint(source_fingerprint=pack.source_fingerprint,
                geometry_assets=geometry_asset_roles(assets), condition=row, target_hw=pack.target_hw,
                config=cfg, geometry_codec=pack.provenance["geometry"]["codec"])
            typed_output.put(sid, pack, fingerprint, signature=cfg.signature(), config=cfg)
            restored = TaskCondition.from_dict(typed_output.get(sid, expected_hash=fingerprint,
                                                signature=cfg.signature(), config=cfg))
            restored.validate(cfg)
        truth = None
        if with_supervision:
            from ..qwen21_union.cache import expected_supervision_fingerprint, load_supervision
            expected = expected_supervision_fingerprint({**truths[sid], "split": row["split"]}, producer, vae)
            truth = load_supervision(Path(args.legacy_workspace) / "cache", sid, expected_hash=expected)
        samples[sid] = condition, truth, pack
        results.append({"sample_id": sid, "dataset": row["dataset"], "record_sha256": stable_hash(row),
            "target_hw": list(pack.target_hw), "source_fingerprint": pack.source_fingerprint,
            "source_tokens": list(pack.source_tokens.shape), "person_ids": list(pack.person_ids),
            "source_identity_valid": pack.face_identity_valid.tolist(), "condition_kind": "BoundUnionCondition"})
    actual_source = source_model_signature(args)
    if any(value != actual_source for value in source_signatures):
        raise ValueError("Source cache extractors differ from actual current model bytes")
    return samples, results, stable_hash(actual_source)


def _truth_records(args):
    path = Path(args.supervision or Path(args.legacy_workspace) / "data/overfit6_supervision.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["sample_id"] for r in rows}) != len(rows) or any(r.get("split") == "test" for r in rows):
        raise ValueError("Independent pilot supervision must be unique and outside test")
    return {r["sample_id"]: r for r in rows}


def _current_producer(args):
    from ..qwen21_union.runner import model_signature
    model_signature.cache_clear()
    signature = model_signature()
    if Path(signature["base_model"]).resolve() != Path(args.base_model).resolve():
        raise ValueError("Current encoder model path differs from verified actual model")
    return {k: signature[k] for k in ("assets", "configs", "runtime", "videox_commit", "union_revision", "preprocessing_code")}


def _pipeline_kwargs(args):
    return {"base_model": args.base_model, "source_root": args.source_root,
        "union_checkpoint": args.union_checkpoint or str(Path(args.project_root) /
            "union-adapter/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")}


def _cuda_budget(args):
    if args.device != "cuda":
        return {"device": "cpu", "cuda_verified": False, "budget_bytes": None}
    if not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA request unavailable; no CPU fallback")
    free, total = torch.cuda.mem_get_info()
    budget = (int(args.memory_budget_gib * 1024**3) if args.memory_budget_gib is not None else
              min(int(total * .9), free - 1024**3))
    if budget <= 0 or budget > total * .9 or budget > free - 1024**3:
        raise RuntimeError(f"Fixed CUDA budget {budget} cannot fit currently free {free}, total {total}; variables unchanged")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    return {"device": "cuda", "cuda_verified": True, "free_bytes": free, "total_bytes": total, "budget_bytes": budget,
            "budget_selection": "explicit_gib" if args.memory_budget_gib is not None else "startup_min90pct_total_free_minus1gib"}


def _verify_exact_supervision_geometry(manifest, *, manifest_path, task_condition, config):
    """GT消费者先证明实际driver配对；source_shape推理的合法性不受影响。"""
    from .cache import _payload_digest
    from .geometry_preparation import load_geometry_task_condition
    if manifest.get("paired_supervision") is not True:
        raise ValueError("source_shape geometry is unpaired; original GT supervision is forbidden")
    if manifest.get("schema") != "current_geometry_preparation_v1":
        raise ValueError("Supervision geometry schema is not a verified current preparation")
    signature = manifest.get("geometry_signature", {})
    if signature.get("mode") != "driver_exact":
        raise ValueError("Unpaired geometry requires driver_exact before GT supervision")
    states = signature.get("states", {})
    driver = states.get("driver")
    if (not isinstance(driver, str) or len(driver) != 64
            or states.get("requested") != driver or states.get("solved") != driver):
        raise ValueError("Supervision driver/requested/solved geometry states are not identical")
    if manifest_path is None or task_condition is None or config is None:
        raise ValueError("Supervision requires the actual geometry manifest and sealed typed pack")
    path = Path(manifest_path)
    manifest_sha = file_sha256(path)
    if stable_hash(_json(path)) != stable_hash(manifest):
        raise ValueError("Supervision geometry manifest differs from actual file")
    dense_path = path.parent / "geometry.pt"
    if file_sha256(dense_path) != signature.get("dense_sha256"):
        raise ValueError("Supervision geometry dense SHA256 changed")
    data = torch.load(dense_path, map_location="cpu", weights_only=True)
    if _payload_digest(data["state"]) != driver:
        raise ValueError("Supervision actual geometry state differs from exact driver")
    rebuilt = load_geometry_task_condition(task_condition, path, config=config)
    if _payload_digest(rebuilt.to_dict()) != _payload_digest(task_condition.to_dict()):
        raise ValueError("Supervision geometry proof differs from sealed typed pack")
    if file_sha256(path) != manifest_sha:
        raise ValueError("Supervision geometry manifest changed during validation")


def _save_supervision(args, pipe, row, truth, manifest, run, *, manifest_path=None, task_condition=None, config=None):
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F
    from .encoded import encode_image
    from .geometry import PixelCanvasTransform
    _verify_exact_supervision_geometry(manifest, manifest_path=manifest_path, task_condition=task_condition, config=config)
    target = Path(truth["target_rgb"])
    expected = truth.get("metadata", {}).get("target_sha256")
    if not expected or file_sha256(target) != expected:
        raise ValueError("Independent GT SHA256 differs from locked supervision")
    canvas = manifest["geometry_signature"]["canvas"]
    transform = PixelCanvasTransform(torch.tensor(canvas["matrix"]), tuple(canvas["input_hw"]), tuple(canvas["output_hw"]))
    with Image.open(target) as image:
        rgb = np.asarray(image.convert("RGB"))
    if tuple(rgb.shape[:2]) != transform.input_hw:
        raise ValueError("GT native dimensions differ from exact driver geometry")
    inverse = torch.linalg.inv(transform.matrix).flatten()[:6].tolist()
    image = Image.fromarray(rgb).transform(transform.output_hw[::-1], Image.Transform.AFFINE,
        inverse, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
    latent = encode_image(pipe, image, canvas_hw=transform.output_hw)
    valid = F.interpolate(transform.valid_fov_mask().float()[None, None], size=tuple(v // 16 for v in transform.output_hw),
                          mode="area").flatten(1)
    payload = {"target_latent": latent, "valid_target_tokens": ordinary_cpu(valid)}
    directory = run / "supervision"
    directory.mkdir(exist_ok=True)
    path = directory / f"{stable_hash(row['sample_id'])}.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return {"path": str(path), "sha256": file_sha256(path), "target_rgb": str(target),
            "target_sha256": expected, "geometry_signature": manifest["geometry_signature"],
            "record_sha256": stable_hash(truth)}


def cache(args, run):
    from . import training
    training.configure_determinism()
    torch.set_num_threads(4)
    cfg, rows = configuration(args), records(args)
    resources = _cuda_budget(args)
    write_json(run / "resources.json", resources)
    if args.kind == "legacy":
        _, results, source_signature = _legacy_samples(args, rows, cfg, destination=run)
        return {"schema": "qwen21_pro_cache_v1", "kind": "legacy", "config": cfg.to_dict(),
            "profile": "effect1600", "samples": results, "source_signature": source_signature,
            "passed": True, "quality_claim": False}
    from .encoded import EncodedConditionCache, encode_condition
    from .geometry_preparation import load_geometry_task_condition
    from .pipeline import build_pro_pipeline
    if not args.current_geometry_index:
        raise ValueError("Current cache requires --current-geometry-index from prepare")
    geometry_index = _json(args.current_geometry_index)
    pipe, backend = build_pro_pipeline("joint", args.device, args.offload, torch.bfloat16,
        config=cfg, cached=False, **_pipeline_kwargs(args))
    write_json(run / "backend.json", backend)
    extractor = _extractor(args)
    raw_cache = ProCache(run / "cache", namespace="raw_source")
    encoded = EncodedConditionCache(run / "encoded")
    producer = _current_producer(args)
    truths = _truth_records(args) if args.encode_supervision else {}
    index = {}
    for row in rows:
        sid = row["sample_id"]
        manifest_path = geometry_index[sid]
        manifest = _json(manifest_path)
        assets = _assets(args, row)
        canvas_hw = tuple(manifest["geometry_signature"]["canvas"]["output_hw"])
        bootstrap = encode_condition(pipe, sample_id=sid, source_images=[row["source_rgb"]],
            pose_images=manifest["pose_images"], control_image=manifest["control_image"],
            canvas_hw=canvas_hw, person_ids=tuple(row["person_order"]), prompt=row["prompt"],
            negative_prompt="", producer_signature=producer, geometry_signature=manifest["geometry_signature"],
            source_fingerprint="unsealed_source_extraction_only")
        raw = extractor.extract(row, bootstrap, project_root=args.project_root, assets=assets)
        raw_cache.put(sid, raw, raw.fingerprint, signature=stable_hash(extractor.model_signature))
        source_pack = prepare_source_task_condition(raw, config=cfg, target_hw=tuple(v // 16 for v in canvas_hw))
        pack = load_geometry_task_condition(source_pack, manifest_path, config=cfg)
        condition = bootstrap.with_source_binding(pack, config=cfg)
        encoded.put(condition, pack, config=cfg)
        encoded.load(sid, expected_binding=condition.binding, config=cfg)
        item = {"binding": condition.binding, "record_sha256": stable_hash(row), "geometry_manifest": manifest_path,
                "geometry_manifest_sha256": file_sha256(manifest_path), "target_hw": list(pack.target_hw)}
        if args.encode_supervision:
            item["supervision"] = _save_supervision(args, pipe, row, truths[sid], manifest, run,
                manifest_path=manifest_path, task_condition=pack, config=cfg)
        index[sid] = item
    write_json(run / "encoded_index.json", index)
    return {"schema": "qwen21_pro_cache_v1", "kind": "current", "config": cfg.to_dict(),
        "profile": "effect1600", "samples": len(index), "source_signature": stable_hash(extractor.model_signature),
        "index_sha256": file_sha256(run / "encoded_index.json"), "passed": True, "quality_claim": False}


def encoded_cache_layout(args, cfg):
    """只读兼容预登记val12 native缓存；逐字段验证真实producer，保留offload来源。"""
    if not args.cache_run:
        raise ValueError("Encoded operation requires --cache-run from cache current")
    root = Path(args.cache_run)
    report = _json(root / "report.json")
    if report["config"] != cfg.to_dict() or not report.get("passed"):
        raise ValueError("Encoded cache run config/producer completion differs")
    if report.get("kind") == "current":
        index = _json(root / "encoded_index.json")
        if report["index_sha256"] != file_sha256(root / "encoded_index.json"):
            raise ValueError("Encoded cache run index changed")
        native_root = root / "encoded"
    elif isinstance(report.get("samples"), list) and report.get("native_cache"):
        if report.get("target_rgb_read") is not False or report.get("sampled_or_trained") is not False:
            raise ValueError("Native cache producer did not declare source-only encoding")
        native_root = Path(report["native_cache"])
        if native_root.resolve() != (root / "native").resolve():
            raise ValueError("Native cache root differs from audited run")
        producer = report["producer_signature"]
        actual = _current_producer(args)
        if any(producer.get(key) != value for key, value in actual.items()):
            raise ValueError("Native cache actual producer assets/config/runtime/code differs")
        if producer.get("conditioning_offload") not in ("none", "sync_group_offload"):
            raise ValueError("Native cache requires explicit audited conditioning offload")
        if "base_model" in producer and Path(producer["base_model"]).resolve() != Path(args.base_model).resolve():
            raise ValueError("Native cache actual producer base model differs")
        index = {}
        for entry in report["samples"]:
            sid = entry["sample_id"]
            if sid in index or entry["condition_binding"]["producer"] != producer:
                raise ValueError("Native cache sample/producer binding differs")
            index[sid] = {"binding": entry["condition_binding"], "geometry_manifest": entry["geometry_manifest"],
                "source_geometry": entry.get("source_geometry"), "raw_cache": entry["raw_cache"],
                "native_audit_adapter": True, "conditioning_offload": producer["conditioning_offload"]}
    else:
        raise ValueError("Unknown current encoded cache report schema")
    source_signature = stable_hash(source_model_signature(args))
    if source_signature != report["source_signature"]:
        raise ValueError("Encoded source model signature differs from actual current extractor weights")
    return native_root, index, source_signature


def _encoded_samples(args, rows, cfg, *, with_supervision=False):
    from .cache import _payload_digest
    from .encoded import EncodedConditionCache, sample_validator
    from .geometry_preparation import load_geometry_task_condition
    native_root, index, source_signature = encoded_cache_layout(args, cfg)
    root = Path(args.cache_run)
    producer = _current_producer(args)
    cache = EncodedConditionCache(native_root)
    samples = {}
    truths = _truth_records(args) if with_supervision else {}
    for row in rows:
        sid, item = row["sample_id"], index[row["sample_id"]]
        audited = item.get("native_audit_adapter", False)
        if ((not audited and item["record_sha256"] != stable_hash(row))
                or any(item["binding"]["producer"].get(k) != v for k, v in producer.items())):
            raise ValueError("Encoded record or actual model producer binding differs")
        if not audited and file_sha256(item["geometry_manifest"]) != item["geometry_manifest_sha256"]:
            raise ValueError("Encoded current geometry manifest changed")
        condition, pack = cache.load(sid, expected_binding=item["binding"], config=cfg)
        sample_validator(condition, pack, sid).validate(pack, sid)
        if audited:
            if condition.binding["prompt"] != row["prompt"]:
                raise ValueError("Native cache prompt differs from requested record")
            _, _, _, source_latent = condition.validate_source_binding(row)
            assets = resolve_condition_assets(row, args.project_root)
            if item["source_geometry"]:
                assets["source_geometry"] = load_source_geometry(item["source_geometry"], row, assets)
            raw_cache = ProCache(root / "cache", namespace="raw_source")
            meta = raw_cache.metadata(sid)
            if meta != item["raw_cache"]:
                raise ValueError("Native raw source cache metadata differs from audited report")
            raw = RawSourceFeatures.from_dict(raw_cache.get(sid, expected_hash=meta["hash"], signature=meta["signature"]))
            raw.validate_binding(row, source_latent, assets, condition.producer_signature)
            rebuilt = load_geometry_task_condition(prepare_source_task_condition(raw, config=cfg,
                target_hw=pack.target_hw), item["geometry_manifest"], config=cfg)
            if _payload_digest(rebuilt.to_dict()) != _payload_digest(pack.to_dict()):
                raise ValueError("Native encoded typed pack differs from current source/geometry reconstruction")
        supervision = None
        if with_supervision:
            info = item.get("supervision")
            if not info or stable_hash(truths[sid]) != info["record_sha256"]:
                raise ValueError("Missing or different independently encoded GT supervision")
            if file_sha256(info["path"]) != info["sha256"] or file_sha256(info["target_rgb"]) != info["target_sha256"]:
                raise ValueError("Independent supervision bytes changed")
            supervision = ordinary_cpu(torch.load(info["path"], map_location="cpu", weights_only=True))
        samples[sid] = condition, supervision, pack
    return samples, source_signature


def audit(args, run):
    cfg, rows = configuration(args), records(args)
    if args.kind == "current":
        samples, source_signature = _encoded_samples(args, rows, cfg)
        kind = "EncodedCondition"
    else:
        samples, _, source_signature = _legacy_samples(args, rows, cfg)
        kind = "BoundUnionCondition"
    return {"passed": True, "quality_claim": False, "samples": len(samples), "condition_kind": kind,
            "source_signature": source_signature, "config": cfg.to_dict()}


def training_execution_protocol(device, *, frozen_dtype=torch.bfloat16, noise_dtype=torch.bfloat16):
    """描述实际训练执行路径；仅构造metadata，不探测或执行CUDA kernel。"""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or not frozen_dtype.is_floating_point or not noise_dtype.is_floating_point:
        raise ValueError("Unsupported training device or precision/noise dtype")
    autocast = "bfloat16_autocast" if device.type == "cuda" else "no_autocast"
    return {"training_device": str(device), "frozen_dtype": str(frozen_dtype), "noise_dtype": str(noise_dtype),
        "precision": f"{device.type}_{frozen_dtype}_frozen_{autocast}_fp32_trainables",
        "noise_backend": f"torch.randn_{device.type}_{noise_dtype}_generator"}


def train(args, run):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from . import training
    from .ablations import configure_ablation
    from .encoded import sample_validator
    from .gates import run_pilot
    from .losses import RegionalLossConfig
    from .offload import place_training_model
    from .pipeline import build_pro_transformer
    from .runner import source_snapshot
    training.configure_determinism()
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    rows, cfg = records(args), configuration(args)
    if len(rows) != 6 or any(r.get("split") != "train" for r in rows):
        raise ValueError("Controlled pilot requires exactly six locked train records")
    order = training.balanced_sample_order({dataset: [r["sample_id"] for r in rows if r["dataset"] == dataset]
                                            for dataset in training.DATASET_ORDER}, sampler_seed=20261015)
    if any(sum(r["dataset"] == d for r in rows) != 2 for d in training.DATASET_ORDER):
        raise ValueError("Controlled pilot requires two records per dataset")
    # The main gate and retrained arms preserve the same locked native Union inputs.
    if args.kind != "legacy" and args.command != "smoke":
        raise ValueError("Controlled overfit/ablation protocol locks legacy native inputs; use separate profile for current training")
    if args.kind == "legacy":
        samples, _, source_signature = _legacy_samples(args, rows, cfg, with_supervision=True)
    else:
        samples, source_signature = _encoded_samples(args, rows, cfg, with_supervision=True)
    budget = _cuda_budget(args)
    write_json(run / "resources.json", budget)
    profile = "ablate" if args.command == "ablate" else (
        "smoke_exact" if args.command == "smoke" and args.exact_resume else "smoke" if args.command == "smoke" else "overfit")
    accumulation = 3 if profile.startswith("smoke") else 1
    if profile == "overfit" and args.arm != "full4":
        raise ValueError("Main200 is full4; retrained variants use ablate25")
    if profile == "ablate" and args.arm == "full4":
        raise ValueError("full4 ablation uses main200 checkpoint_000025; do not retrain a second full4")
    model, backend = build_pro_transformer("joint", "cpu", torch.bfloat16, config=cfg,
        transformer_setup=lambda m: configure_ablation(m, arm=args.arm, seed=3407), **_pipeline_kwargs(args))
    write_json(run / "backend.json", backend)
    placement = place_training_model(model, args.device, memory_mode=args.memory_mode)
    model.gradient_checkpointing = True
    write_json(run / "memory_policy.json", placement)
    data_paths = {"conditions": args.conditions or str(Path(args.legacy_workspace) / "data/overfit6_conditions.jsonl"),
                  "supervision": args.supervision or str(Path(args.legacy_workspace) / "data/overfit6_supervision.jsonl")}
    feature = Path(args.cache_run if args.kind == "current" else args.feature_run or args.cache_run)
    noise_dtypes = {sample[1]["target_latent"].dtype for sample in samples.values()}
    if len(noise_dtypes) != 1:
        raise ValueError("Controlled training requires one explicit noise dtype across samples")
    frozen_dtype = next(p.dtype for p in model.parameters() if not p.requires_grad and p.is_floating_point())
    execution = training_execution_protocol(args.device, frozen_dtype=frozen_dtype, noise_dtype=next(iter(noise_dtypes)))
    metadata = {"config": cfg.to_dict(), "scope": "joint", "task_signature": stable_hash(RegionalLossConfig().to_dict()),
        "source_signature": source_signature, "cache_signature": file_sha256(feature / "report.json"),
        "code_signature": source_snapshot()["signature"], "data_signature": stable_hash(asset_hashes(data_paths)),
        "trajectory_signature": training.trajectory_signature(order, train_seed=3407, accumulation_steps=accumulation,
                                                              memory_mode=args.memory_mode, precision=execution["precision"]),
        "memory_mode": args.memory_mode, "control_lora_targets": [], "arm": args.arm,
        "profile": profile, "train_seed": 3407, "sampler_seed": 20261015, "warmup_updates": 32,
        "initialization_seed": 3407, "condition_kind": args.kind, **execution,
        "selection_role": "engineering_probe" if profile.startswith("smoke") else "controlled_pilot"}
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model, subfolder="scheduler", local_files_only=True)
    write_json(run / "training_protocol.json", {"metadata": metadata, "order": order, "updates":
        {"overfit": 200, "ablate": 25, "smoke": 2, "smoke_exact": 4}[profile], "accumulation_steps": accumulation,
        "optimizer": {"kind": "AdamW", "lr": 1e-4, "betas": [.9, .999], "eps": 1e-8, "weight_decay": .01},
        "gradient_clip": 1., "native_inputs": "locked_legacy_union" if args.kind == "legacy" else "current_geometry_encoded",
        "memory_budget": budget})
    return run_pilot(model, scheduler=scheduler, load_sample=samples.__getitem__, sample_order=order,
        metadata=metadata, run_dir=run, profile=profile, device=args.device, resume=args.resume,
        max_memory_bytes=budget["budget_bytes"], sample_validator=sample_validator if args.kind == "current" else None)


def native_source_mask(source_rgb, source_mask, output):
    """旧低分辨率mask按完整FoV最近邻还原；源文件不动，绑定原mask字节。"""
    from PIL import Image
    with Image.open(source_rgb) as image:
        native_wh = image.size
    with Image.open(source_mask) as image:
        original_wh = image.size
        if original_wh == native_wh:
            path = Path(source_mask)
        else:
            path = Path(output)
            with path.open("xb") as stream:
                image.convert("L").resize(native_wh, Image.Resampling.NEAREST).save(stream, format="PNG")
    return path, {"source_rgb": str(Path(source_rgb).resolve()), "source_rgb_sha256": file_sha256(source_rgb),
        "original_mask": str(Path(source_mask).resolve()), "original_mask_sha256": file_sha256(source_mask),
        "native_mask": str(path.resolve()), "native_mask_sha256": file_sha256(path),
        "original_wh": list(original_wh), "native_wh": list(native_wh), "transform": "full_fov_nearest"}


def validate_quality_checkpoint(checkpoint_path):
    """正式Stage1仅选择通过完整门禁的最终200步实际权重文件。"""
    from .checkpoint import TrainingCursor
    path = Path(checkpoint_path).resolve()
    gate_path = path.parent / "gate.json"
    if not gate_path.is_file():
        raise ValueError("Stage1 quality selection requires a passed final200 checkpoint gate")
    gate = _json(gate_path)
    if (gate.get("passed") is not True or gate.get("engineering_passed") is not True
            or gate.get("overfit_metric_gate", {}).get("passed") is not True
            or gate.get("profile") != "overfit" or gate.get("status") != "complete"):
        raise ValueError("Stage1 quality selection requires complete engineering and overfit200 gates")
    selected = gate.get("final_checkpoint", {})
    if selected.get("selection_role") != "selected_stage1" or Path(selected.get("path", "")).resolve() != path:
        raise ValueError("Requested checkpoint is not the gate-selected final200 checkpoint")
    manifest_path = path / "checkpoint.json"
    if file_sha256(manifest_path) != selected.get("manifest_sha256"):
        raise ValueError("Selected final checkpoint manifest SHA256 changed")
    manifest = _json(manifest_path)
    cursor = TrainingCursor(**manifest["cursor"]).validate()
    if cursor.optimizer_step != 200 or cursor.accumulation_steps != 1:
        raise ValueError("Stage1 quality checkpoint must be the complete final200 cursor")
    if manifest["cursor"] != selected.get("cursor") or manifest["cursor"] != gate.get("cursor"):
        raise ValueError("Selected final checkpoint cursor differs from quality gate")
    metadata = manifest["metadata"]
    if (metadata != selected.get("metadata") or metadata.get("profile") != "overfit"
            or metadata.get("arm") != "full4" or metadata.get("selection_role") != "controlled_pilot"):
        raise ValueError("Selected final checkpoint metadata differs from main200 quality role")
    if (manifest["state_sha256"] != selected.get("state_sha256")
            or file_sha256(path / "state.pt") != selected.get("state_sha256")):
        raise ValueError("Selected final checkpoint state SHA256 differs from actual weights")
    return manifest


def _inference_pipeline(args, run, *, cached):
    from . import training
    from .ablations import configure_ablation
    from .losses import RegionalLossConfig
    from .pipeline import build_pro_pipeline
    from .runner import source_snapshot
    if not args.checkpoint:
        raise ValueError("Inference requires explicit complete --checkpoint")
    if args.steps < 1 or args.true_cfg_scale < 1:
        raise ValueError("Inference steps must be positive and CFG at least1")
    training.configure_determinism()
    torch.set_num_threads(4)
    metadata = _json(Path(args.checkpoint) / "checkpoint.json")["metadata"]
    cfg = configuration(args)
    expected = {"config": cfg.to_dict(), "scope": "joint", "task_signature": stable_hash(RegionalLossConfig().to_dict()),
        "code_signature": source_snapshot()["signature"], "source_signature": stable_hash(source_model_signature(args)),
        "arm": args.arm}
    if metadata.get("control_lora_targets"):
        raise ValueError("Pilot inference defaults to task-only; explicit LoRA inventory not configured in this CLI")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Inference actual code/model/config/arm binding differs: {key}")
    if not args.engineering_probe:
        selected = validate_quality_checkpoint(args.checkpoint)
        if selected["metadata"] != metadata:
            raise ValueError("Stage1 quality checkpoint metadata changed during selection")
    write_json(run / "resources.json", _cuda_budget(args))
    pipe, backend = build_pro_pipeline("joint", args.device, args.offload, torch.bfloat16,
        config=cfg, cached=cached, checkpoint_path=args.checkpoint, expected_metadata=expected,
        transformer_setup=lambda model: configure_ablation(model, arm=args.arm, seed=3407), **_pipeline_kwargs(args))
    write_json(run / "backend.json", backend)
    return pipe, cfg, metadata


def infer(args, run):
    from .inference import sample_cached
    from .refinement import write_coarse_artifact
    rows = records(args)
    pipe, cfg, metadata = _inference_pipeline(args, run, cached=True)
    samples, _ = _encoded_samples(args, rows, cfg)
    index = {}
    for row in rows:
        sid = row["sample_id"]
        condition, _, pack = samples[sid]
        output = run / sid
        output.mkdir()
        with torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else torch.no_grad():
            result = sample_cached(pipe, condition, task_condition=pack, num_inference_steps=args.steps,
                seed=args.seed, true_cfg_scale=args.true_cfg_scale, output_type="pil", device=args.device)
        write_json(output / "sampling.json", result.audit)
        coarse = write_coarse_artifact(result, output_dir=output / "coarse", stage1_checkpoint=args.checkpoint,
            stage1_signatures=metadata, source_fingerprint=pack.source_fingerprint,
            geometry_signature=pack.provenance["geometry"], output_root=Path(args.output_root),
            selection_role="engineering_probe" if args.engineering_probe else "selected_stage1")
        index[sid] = str(coarse.manifest_path)
    write_json(run / "coarse_index.json", index)
    return {"coarse_images": len(index), "samples": list(index), "steps": args.steps, "seed": args.seed,
            "true_cfg_scale": args.true_cfg_scale, "engineering_only": args.engineering_probe, "quality_claim": False}


def validated_refinement_geometry(args, row, cfg, condition, pack):
    """精修只消费缓存绑定的同一manifest及其原生文件/typed几何证明。"""
    from .cache import _payload_digest
    from .encoded import geometry_input_files
    from .geometry_preparation import load_geometry_task_condition
    if not args.current_geometry_index:
        raise ValueError("refine requires --current-geometry-index matching coarse cache")
    sid = row["sample_id"]
    _, index, _ = encoded_cache_layout(args, cfg)
    item = index[sid]
    path = Path(item["geometry_manifest"]).resolve()
    requested = Path(_json(args.current_geometry_index)[sid]).resolve()
    if requested != path:
        raise ValueError("Refiner manifest differs from cache-bound geometry manifest path")
    manifest_sha = file_sha256(path)
    if not item.get("native_audit_adapter", False) and manifest_sha != item["geometry_manifest_sha256"]:
        raise ValueError("Refiner cache-bound geometry manifest SHA256 changed")
    condition.verify(pack)
    if condition.binding != item["binding"]:
        raise ValueError("Refiner encoded geometry binding differs from audited cache")
    manifest = _json(path)
    signature = manifest["geometry_signature"]
    if signature != pack.provenance["geometry"] or signature != condition.binding["geometry"]:
        raise ValueError("Refiner native geometry differs from sealed encoded condition")
    native_files = [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)}
                    for p in [*manifest["pose_images"], manifest["control_image"]]]
    if native_files + geometry_input_files(signature) != condition.binding["geometry_files"]:
        raise ValueError("Refiner native geometry files differ from cache-bound control/pose proof")
    if stable_hash(manifest["rois"]) != stable_hash(pack.provenance["geometry_rois"]):
        raise ValueError("Refiner geometry ROI manifest differs from sealed typed pack")
    # Reuses the production validator for control/pose asset hashes, dense bytes,
    # current helper code, body model inputs and the complete reconstructed pack.
    rebuilt = load_geometry_task_condition(pack, path, config=cfg)
    if _payload_digest(rebuilt.to_dict()) != _payload_digest(pack.to_dict()):
        raise ValueError("Refiner current geometry reconstruction differs from sealed typed pack")
    if file_sha256(path) != manifest_sha:
        raise ValueError("Refiner geometry manifest changed during validation")
    return manifest


def refine(args, run):
    from .refinement import (CoarseArtifact, PartEndpoint, derive_refinement_rois, encode_roi_source_bank,
        localize_task_condition, prepare_refinement_request, run_refinement, write_source_crop)
    if not args.coarse:
        raise ValueError("refine requires a verified generated --coarse manifest")
    rows = records(args)
    if len(rows) != 1:
        raise ValueError("refine requires exactly one --sample-id")
    row, sid = rows[0], rows[0]["sample_id"]
    pipe, cfg, _ = _inference_pipeline(args, run, cached=False)
    samples, _ = _encoded_samples(args, rows, cfg)
    condition, _, pack = samples[sid]
    coarse = CoarseArtifact.load(args.coarse)
    if coarse.manifest["source_fingerprint"] != pack.source_fingerprint or coarse.manifest["geometry_signature"] != pack.provenance["geometry"]:
        raise ValueError("Generated coarse differs from requested source/current geometry")
    manifest = validated_refinement_geometry(args, row, cfg, condition, pack)
    canvas_hw = tuple(manifest["geometry_signature"]["canvas"]["output_hw"])
    candidates = derive_refinement_rois(pack, manifest["rois"], canvas_hw=canvas_hw)
    endpoints = None
    if args.roi_kind == "paired_hands":
        try:
            endpoints = frozenset(PartEndpoint(*map(int, value.split(":"))) for value in args.hand_endpoint)
        except (ValueError, TypeError) as error:
            raise ValueError("--hand-endpoint expects stable person_id:part_id") from error
        if len(endpoints) != 2 or any(e.part_id not in (4, 7) for e in endpoints):
            raise ValueError("Paired hands require two explicit endpoints with part4/7")
    elif args.roi_person is None:
        raise ValueError("Face refinement requires explicit --roi-person stable ID")
    selected = [roi for roi in candidates if roi.kind == args.roi_kind and (
        frozenset(roi.endpoints) == endpoints if endpoints is not None else roi.person_ids == (args.roi_person,))]
    if not selected or selected[0].invalid_reason:
        reason = selected[0].invalid_reason if selected else "requested_geometry_roi_unavailable"
        write_json(run / "fallback.json", {"fallback_reason": reason, "coarse": str(coarse.image_path)})
        return {"refined_images": 0, "fallback_reason": reason, "image_path": str(coarse.image_path), "quality_claim": False}
    roi = selected[0]
    assets = _assets(args, row)
    bundle = assets.get("source_geometry")
    raw_cache = ProCache(Path(args.cache_run) / "cache", namespace="raw_source")
    meta = raw_cache.metadata(sid)
    raw = RawSourceFeatures.from_dict(raw_cache.get(sid, expected_hash=meta["hash"], signature=meta["signature"]))
    if raw.fingerprint != pack.source_fingerprint:
        raise ValueError("Refiner raw source features differ from coarse source")
    crops, masks = [], []
    for endpoint in roi.endpoints:
        slot = pack.person_ids.index(endpoint.person_id)
        part = {0: "face", 4: "left_hand", 7: "right_hand"}[endpoint.part_id]
        if bundle is not None:
            box = bundle["rois"][slot].get(part)
        else:
            box = next((item["box"] for item in raw.provenance["coverage"][slot]["crops"] if item["kind"] == part), None)
        if box is None:
            reason = f"source_pid{endpoint.person_id}_part{endpoint.part_id}_roi_unavailable"
            write_json(run / "fallback.json", {"fallback_reason": reason, "coarse": str(coarse.image_path)})
            return {"refined_images": 0, "fallback_reason": reason, "image_path": str(coarse.image_path), "quality_claim": False}
        mask, binding = native_source_mask(assets["source_rgb"], assets["source_masks"][slot],
                                           run / f"native_mask_pid{endpoint.person_id}_part{endpoint.part_id}.png")
        binding_path = run / f"native_mask_pid{endpoint.person_id}_part{endpoint.part_id}.json"
        write_json(binding_path, binding)
        crop = write_source_crop(person_id=endpoint.person_id, part_id=endpoint.part_id,
            source_image=assets["source_rgb"], source_mask=mask, box=box,
            source_geometry_provenance=bundle, output_root=Path(args.output_root),
            output_dir=run / f"source_pid{endpoint.person_id}_part{endpoint.part_id}")
        extra = {"source_unresized_mask": str(assets["source_masks"][slot]), "mask_transform_manifest": str(binding_path)}
        crop = replace(crop, files={**crop.files, **extra}, hashes={**crop.hashes, **asset_hashes(extra)}).verify()
        crops.append(crop)
        masks.append(binding)
    extractor = _extractor(args)
    bank = encode_roi_source_bank(pipe, extractor, crops, person_ids=roi.person_ids, endpoints=roi.endpoints)
    local = localize_task_condition(pack, roi, source_features=bank, config=cfg)
    request = prepare_refinement_request(coarse, roi=roi, source_crops=crops, geometry_image=manifest["control_image"],
        output_dir=run / "request", output_root=Path(args.output_root), seed=args.seed,
        model_signature=copy.deepcopy(pipe.trained_model_signature), source_bank_fingerprint=bank.fingerprint,
        task_condition=local, config=cfg)
    with torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else torch.no_grad():
        result = run_refinement(pipe, request, task_condition=local, output_dir=run / "refined",
            output_root=Path(args.output_root), num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale, negative_prompt="" if args.true_cfg_scale > 1 else None)
    return {"refined_images": int(result.audit["fallback_reason"] is None), "image_path": str(result.image_path),
        "source_mask_transforms": masks, "roi_kind": args.roi_kind, "local_grid": list(local.target_hw),
        "fallback_reason": result.audit["fallback_reason"], "audit": result.audit, "quality_claim": False}

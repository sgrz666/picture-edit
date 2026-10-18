"""用锁定 overfit6 实读几何、冻结源特征与独立缓存；不读取目标 RGB。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRODUCER_SHA256 = "ffb3077938662e16277d5e4862af8c3466e316de4f0c3b64ec107981d23ad0ae"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True, help="新建绝对运行目录")
    parser.add_argument("--profile", choices=("vae64", "effect1600"), default="effect1600")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    parser.add_argument("--producer-event")
    parser.add_argument("--producer-sha256", default=PRODUCER_SHA256)
    parser.add_argument("--contact-vertex-parts")
    parser.add_argument("--source-geometry-index", help="sample_id → 已验证 source_geometry.json 的 JSON；Hi4D ROI bank 必需")
    args = parser.parse_args(argv)
    legacy = Path(args.legacy_workspace).resolve()
    run = Path(args.run_dir)
    if not run.is_absolute():
        parser.error("run-dir 必须是绝对路径")
    run = run.resolve()
    if run.is_relative_to(legacy) or any(p.lower() in {"models", "datasets"} for p in run.parts):
        parser.error("新运行不能写入旧工作区、models 或 datasets")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import torch
    from src.pose_control.qwen21_union.training import configure_determinism
    from src.pose_control.qwen21_union.integrity import verify_workspace_inputs
    from src.pose_control.qwen21_union.runner import model_signature
    from src.pose_control.qwen21_pro import ProAdapterConfig, TaskCondition
    from src.pose_control.qwen21_pro.cache import (ProCache, file_sha256, load_bound_condition,
                                                  stable_hash, typed_pack_fingerprint)
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from src.pose_control.qwen21_pro.preparation import (geometry_asset_roles, prepare_task_condition,
                                                        resolve_condition_assets)
    policy = configure_determinism()
    torch.set_num_threads(4)
    write_json(run / "run.json", {"stage": "condition_audit", "argv": vars(args),
               "script_sha256": file_sha256(__file__), "determinism": policy,
               "code_root": str(ROOT), "training_or_quality_claim": False})
    try:
        verify_workspace_inputs(legacy)
        event_path = Path(args.producer_event) if args.producer_event else legacy / "cache/completed_20261016T184014.json"
        if file_sha256(event_path) != args.producer_sha256:
            raise ValueError("encoder producer event SHA256 differs from the pinned audit")
        producer = json.loads(event_path.read_text(encoding="utf-8"))["model_signature"]
        model_signature.cache_clear()
        current = model_signature()
        if Path(args.base_model).resolve() != Path(current["base_model"]).resolve():
            raise ValueError("audit base-model differs from the model path used by the encoder asset verifier")
        for key in ("assets", "configs", "preprocessing_code", "runtime", "videox_commit", "union_revision"):
            if current[key] != producer[key]:
                raise ValueError(f"current encoder assets/code/runtime differs from frozen producer: {key}")
        base = Path(args.base_model)
        vae_config = json.loads((base / "vae/config.json").read_text(encoding="utf-8"))
        manifest = legacy / "data/overfit6_conditions.jsonl"
        manifest_sha256 = file_sha256(manifest)
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 6 or {name: sum(r["dataset"] == name for r in records)
                                 for name in ("iper", "tiktok", "hi4d")} != {"iper": 2, "tiktok": 2, "hi4d": 2}:
            raise ValueError("expected exactly two locked training records per dataset")
        config = ProAdapterConfig(source_dim=1600 if args.profile == "effect1600" else 64)
        typed = ProCache(run / "cache", namespace="typed_condition")
        raw_cache = ProCache(run / "cache", namespace="raw_source")
        extractor, results = None, []
        geometry_index = (json.loads(Path(args.source_geometry_index).read_text(encoding="utf-8"))
                          if args.source_geometry_index else {})
        for original in records:
            record = copy.deepcopy(original)
            record["target_rgb"] = "/unreadable/target_rgb_must_not_be_used.png"
            cached = load_bound_condition(record, legacy / "cache", model_signature=producer, vae_config=vae_config)
            assets = resolve_condition_assets(record, args.project_root)
            if record["sample_id"] in geometry_index:
                from src.pose_control.qwen21_pro.geometry_preparation import load_source_geometry
                assets["source_geometry"] = load_source_geometry(
                    geometry_index[record["sample_id"]], record, assets)
            if record["dataset"] == "hi4d" and args.contact_vertex_parts:
                assets["contact_vertex_parts"] = Path(args.contact_vertex_parts)
            raw = None
            if config.source_dim == 1600:
                if extractor is None:
                    models = Path(args.project_root) / "models/antelopev2"
                    extractor = FrozenSourceFeatureExtractor.from_local(models / "dinov2-giant",
                            vae_root=base / "vae", device=args.device, face_root=models)
                raw = extractor.extract(record, cached, project_root=args.project_root, assets=assets)
                raw_meta = raw_cache.put(record["sample_id"], raw, raw.fingerprint,
                                         signature=stable_hash(extractor.model_signature))
            pack = prepare_task_condition(record, cached, project_root=args.project_root, config=config,
                                          raw_source_features=raw, assets=assets)
            fingerprint = typed_pack_fingerprint(source_fingerprint=pack.source_fingerprint,
                    geometry_assets=geometry_asset_roles(assets), condition=record, target_hw=pack.target_hw,
                    config=config, geometry_codec=pack.provenance["geometry"]["codec"])
            meta = typed.put(record["sample_id"], pack, fingerprint, signature=config.signature(), config=config)
            restored = TaskCondition.from_dict(typed.get(record["sample_id"], expected_hash=fingerprint,
                        signature=config.signature(), config=config)).validate(config)
            if restored.layout_signature() != pack.layout_signature():
                raise ValueError("typed cache changed the condition layout")
            result = {"sample_id": record["sample_id"], "dataset": record["dataset"],
                      "person_ids": pack.person_ids, "target_hw": pack.target_hw,
                      "source_tokens": list(pack.source_tokens.shape),
                      "source_valid_tokens": pack.source_token_valid.sum(-1).tolist(),
                      "face_identity_valid": pack.face_identity_valid.tolist(),
                      "spatial_valid": {k: v.tolist() for k, v in pack.spatial_valid.items()},
                      "roi_mass": {"face": pack.face_mask.sum(-1).tolist(), "hands": pack.hand_mask.sum(-1).tolist()},
                      "owner_sum_max": float(pack.owner.sum(1).max()),
                      "contact_edges": int(pack.relations.valid.sum()), "typed_cache": meta,
                      "provenance": pack.provenance, "unreadable_target_rgb_ignored": True}
            if raw is not None:
                result["raw_cache"] = raw_meta
            results.append(result)
            print(json.dumps({"sample_id": result["sample_id"], "dataset": result["dataset"],
                              "source_valid_tokens": result["source_valid_tokens"],
                              "contact_edges": result["contact_edges"]}), flush=True)
        if manifest_sha256 != file_sha256(manifest):
            raise ValueError("legacy manifest changed during audit")
        report = {"schema": "qwen21_pro_condition_audit_v1", "profile": args.profile,
                  "producer_event": str(event_path), "producer_sha256": args.producer_sha256,
                  "manifest_sha256": manifest_sha256, "config": config.to_dict(),
                  "samples": results, "seconds": time.monotonic() - started, "passed": True}
        write_json(run / "report.json", report)
        print(json.dumps({"passed": True, "samples": len(results), "run_dir": str(run),
                          "seconds": report["seconds"]}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

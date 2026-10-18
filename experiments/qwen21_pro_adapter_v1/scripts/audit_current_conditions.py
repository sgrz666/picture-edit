"""为预登记val12编码当前几何和source-only特征；不读取监督manifest或GT像素。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--legacy-workspace", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--geometry-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--offload", choices=("none", "sync_group_offload"), default="none")
    args = parser.parse_args()
    import torch
    from src.pose_control.qwen21_pro import ProAdapterConfig
    from src.pose_control.qwen21_pro.cache import ProCache, file_sha256, stable_hash
    from src.pose_control.qwen21_pro.encoded import EncodedConditionCache, encode_condition
    from src.pose_control.qwen21_pro.features import FrozenSourceFeatureExtractor
    from src.pose_control.qwen21_pro.geometry_preparation import P_ROOT, load_source_geometry, load_geometry_task_condition
    from src.pose_control.qwen21_pro.pipeline import build_pro_pipeline
    from src.pose_control.qwen21_pro.preparation import prepare_source_task_condition, resolve_condition_assets
    from src.pose_control.qwen21_union.runner import model_signature
    from src.pose_control.qwen21_union.training import configure_determinism
    policy = configure_determinism()
    torch.set_num_threads(4)
    run, project, legacy, geometry = map(Path, (args.run_dir, args.project_root, args.legacy_workspace, args.geometry_run))
    if not run.is_absolute() or not run.resolve().is_relative_to(P_ROOT.resolve()):
        parser.error("run-dir必须是P内新建绝对目录")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(run / "run.json", {"argv": vars(args), "code_root": str(ROOT), "determinism": policy,
                                 "script_sha256": file_sha256(__file__), "target_rgb_read": False})
    try:
        original_path = legacy / "data/val_conditions.jsonl"
        lock = json.loads((legacy / "data/lock.json").read_text())
        if file_sha256(original_path) != lock["selected_manifest_sha256"][original_path.name]:
            raise ValueError("Locked val manifest changed")
        original = {r["sample_id"]: r for r in map(json.loads, original_path.read_text().splitlines())}
        records = list(map(json.loads, Path(args.conditions).read_text().splitlines()))
        if len(records) != 12 or any(original.get(r["sample_id"]) != r for r in records):
            raise ValueError("Expected unchanged preregistered val12")
        current_index = json.loads((geometry / "current_geometry_index.json").read_text())
        source_index = json.loads((geometry / "source_geometry_index.json").read_text())
        if not json.loads((geometry / "report.json").read_text())["passed"]:
            raise ValueError("Current geometry preparation did not pass")
        free, total = torch.cuda.mem_get_info()
        budget = min(int(.9 * total), free - 1024**3)
        minimum = 40 if args.offload == "none" else 16
        if budget < minimum * 1024**3:
            raise RuntimeError(f"This {args.offload} encoder audit requires{minimum}GiB process budget; no silent fallback")
        torch.cuda.set_per_process_memory_fraction(budget / total)
        model_signature.cache_clear()
        producer = {**model_signature(), "conditioning_offload": args.offload}
        cfg = ProAdapterConfig(source_dim=1600)
        pipe, backend = build_pro_pipeline("joint", "cuda", args.offload, torch.bfloat16, config=cfg, cached=False,
                union_checkpoint=project / "union-adapter/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
        write_json(run / "backend.json", backend)
        models = project / "models/antelopev2"
        extractor = FrozenSourceFeatureExtractor.from_local(models / "dinov2-giant", vae_root=Path(producer["base_model"]) / "vae",
                                                           device="cuda", face_root=models)
        cache = EncodedConditionCache(run / "native")
        raw_cache = ProCache(run / "cache", namespace="raw_source")
        rows = []
        for record in records:
            sid = record["sample_id"]
            manifest_path = current_index[sid]
            manifest = json.loads(Path(manifest_path).read_text())
            assets = resolve_condition_assets(record, project)
            if sid in source_index:
                assets["source_geometry"] = load_source_geometry(source_index[sid], record, assets)
            initial = encode_condition(pipe, sample_id=sid, source_images=[record["source_rgb"]],
                pose_images=manifest["pose_images"], control_image=manifest["control_image"], canvas_hw=(512, 512),
                person_ids=tuple(record["person_order"]), prompt=record["prompt"], producer_signature=producer,
                geometry_signature=manifest["geometry_signature"], source_fingerprint="unsealed_extraction_only")
            raw = extractor.extract(record, initial, project_root=project, assets=assets)
            raw_meta = raw_cache.put(sid, raw, raw.fingerprint, signature=stable_hash(extractor.model_signature))
            source = prepare_source_task_condition(raw, config=cfg)
            pack = load_geometry_task_condition(source, manifest_path, config=cfg)
            condition = initial.with_source_binding(pack, config=cfg)
            cache.put(condition, pack, config=cfg)
            restored, restored_pack = cache.load(sid, expected_binding=condition.binding, config=cfg)
            restored.verify(restored_pack)
            rows.append({"sample_id": sid, "dataset": record["dataset"], "condition_binding": condition.binding,
                         "geometry_manifest": manifest_path, "source_geometry": source_index.get(sid),
                         "raw_cache": raw_meta, "source_tokens": list(pack.source_tokens.shape),
                         "source_valid_tokens": pack.source_token_valid.sum(-1).tolist(),
                         "face_identity_valid": pack.face_identity_valid.tolist(),
                         "contact_vertex_pairs": int(pack.relations.valid.sum())})
            print(json.dumps({k: rows[-1][k] for k in ("sample_id", "dataset", "source_tokens", "source_valid_tokens", "contact_vertex_pairs")}), flush=True)
        write_json(run / "report.json", {"passed": True, "config": cfg.to_dict(), "samples": rows,
            "native_cache": str(run / "native"), "source_signature": stable_hash(extractor.model_signature),
            "producer_signature": producer, "target_rgb_read": False, "sampled_or_trained": False,
            "max_memory_allocated": torch.cuda.max_memory_allocated(), "max_memory_reserved": torch.cuda.max_memory_reserved(),
            "initial_free_memory_bytes": free, "seconds": time.monotonic() - started})
        print(json.dumps({"passed": True, "samples": len(rows), "seconds": time.monotonic() - started}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

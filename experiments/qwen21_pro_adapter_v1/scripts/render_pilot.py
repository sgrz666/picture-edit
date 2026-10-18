"""冻结V4上的六例legacy B诊断采样；step0/200各独立新run，40步seed43 CFG1。"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

V4_SIGNATURE = "1be3083d0447b44b55e393586404190a98ccbac9fcca52daa4989fcf2bb2f05e"
P_ROOT = "/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1"
B_ROOT = "/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_frozen_code(code_root):
    root = Path(code_root).resolve()
    if not Path(code_root).is_absolute() or not (root / "src/pose_control/qwen21_pro/runner.py").is_file():
        raise ValueError("code-root must identify the existing frozen V4 source tree")
    sys.path.insert(0, str(root))
    packages = {name: importlib.import_module(f"src.pose_control.qwen21_pro.{name}")
                for name in ("runner", "dataflow", "inference", "refinement", "gates", "training", "cache")}
    packages["union_runner"] = importlib.import_module("src.pose_control.qwen21_union.runner")
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "src" or name.startswith("src.")) and path and not Path(path).resolve().is_relative_to(root / "src"):
            raise ValueError(f"Imported src module is outside frozen code-root: {name}: {path}")
    snapshot = packages["runner"].source_snapshot()
    if Path(snapshot["code_root"]).resolve() != root or snapshot["signature"] != V4_SIGNATURE:
        raise ValueError("Actual imported source bytes differ from frozen V4 signature")
    return SimpleNamespace(**packages, code_root=root, signature=snapshot["signature"],
                           torch=importlib.import_module("torch"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-run", required=True)
    parser.add_argument("--source-geometry-index", required=True)
    parser.add_argument("--contact-vertex-parts", required=True)
    parser.add_argument("--output-root", default=P_ROOT)
    parser.add_argument("--legacy-workspace", default=B_ROOT)
    parser.add_argument("--project-root", default="/home/shangguanrz/project/pic-edit")
    parser.add_argument("--base-model", default="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
    parser.add_argument("--source-root")
    parser.add_argument("--union-checkpoint")
    parser.add_argument("--offload", choices=("none", "sync_group_offload"), default="sync_group_offload")
    parser.add_argument("--memory-budget-gib", type=float, default=32.)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def render_checkpoint(args):
    frozen = load_frozen_code(args.code_root)
    for key in ("code_root", "checkpoint", "feature_run", "source_geometry_index", "contact_vertex_parts",
                "output_root", "legacy_workspace", "project_root", "base_model", "source_root", "union_checkpoint"):
        value = getattr(args, key)
        if value:
            setattr(args, key, str(Path(value).resolve()))
    args.kind, args.arm, args.cache_run = "legacy", "full4", None
    args.conditions = str(Path(args.legacy_workspace) / "data/overfit6_conditions.jsonl")
    args.supervision, args.sample_id = None, []
    args.steps, args.seed, args.true_cfg_scale, args.engineering_probe = 40, 43, 1., True
    manifest = json.loads((Path(args.checkpoint) / "checkpoint.json").read_text(encoding="utf-8"))
    step, metadata = manifest["cursor"]["optimizer_step"], manifest["metadata"]
    if (step not in (0, 200) or metadata.get("profile") != "overfit" or metadata.get("arm") != "full4"
            or metadata.get("code_signature") != frozen.signature):
        raise ValueError("Diagnostic render accepts only actual V4 full4 main checkpoints at step0 or200")
    if file_sha256(Path(args.checkpoint) / "state.pt") != manifest["state_sha256"]:
        raise ValueError("Diagnostic checkpoint weight SHA256 changed")
    run = frozen.runner.allocate_run(args.output_root, args.name, legacy_workspace=args.legacy_workspace)
    started = time.monotonic()
    write_json(run / "run.json", {"schema": "pro_private_legacy_render_v1", "argv": vars(args),
        "script_path": str(Path(__file__).resolve()), "script_sha256": file_sha256(__file__),
        "actual_src_root": str(frozen.code_root), "source_snapshot_signature": frozen.signature,
        "checkpoint_manifest_sha256": file_sha256(Path(args.checkpoint) / "checkpoint.json"),
        "checkpoint_state_sha256": manifest["state_sha256"], "checkpoint_metadata": metadata,
        "source_inputs": "locked_legacy_B", "engineering_only": True, "quality_claim": False,
        "gt_rgb_read": False})
    try:
        rows = frozen.dataflow.records(args)
        if len(rows) != 6 or any(r.get("split") != "train" for r in rows):
            raise ValueError("Diagnostic render requires the exact six locked legacy train records")
        cfg = frozen.dataflow.configuration(args)
        samples, _, source_signature = frozen.dataflow._legacy_samples(args, rows, cfg, with_supervision=False)
        pipe, loaded_cfg, loaded_metadata = frozen.dataflow._inference_pipeline(args, run, cached=True)
        if loaded_cfg != cfg or loaded_metadata != metadata:
            raise ValueError("Loaded diagnostic model differs from the requested checkpoint")
        zero_equivalence = None
        if step == 0:
            condition, _, pack = samples[rows[0]["sample_id"]]
            frozen.training._verify_sample(condition, pack, rows[0]["sample_id"], required=True)
            # The existing zero probe consumes only target.shape, never its pixel
            # values. This empty carrier is explicitly not GT supervision.
            shape_only = frozen.torch.empty((1, pack.target_hw[0] * pack.target_hw[1], 64), dtype=frozen.torch.bfloat16)
            frozen.gates.probe_zero_equivalence(pipe.transformer, (condition, {"target_latent": shape_only}, pack), device=args.device)
            zero_equivalence = True
        results = []
        for row in rows:
            sid = row["sample_id"]
            condition, _, pack = samples[sid]
            frozen.training._verify_sample(condition, pack, sid, required=True)
            output = run / sid
            output.mkdir()
            with frozen.torch.autocast("cuda", dtype=frozen.torch.bfloat16):
                sampled = frozen.inference.sample_cached(pipe, condition, task_condition=pack, num_inference_steps=40,
                    seed=43, true_cfg_scale=1., use_kv_cache=False, output_type="pil", device=args.device)
            write_json(output / "sampling.json", sampled.audit)
            coarse = frozen.refinement.write_coarse_artifact(sampled, output_dir=output / "coarse",
                stage1_checkpoint=args.checkpoint, stage1_signatures=metadata,
                source_fingerprint=pack.source_fingerprint, geometry_signature=pack.provenance["geometry"],
                selection_role="engineering_probe", output_root=Path(args.output_root))
            results.append({"sample_id": sid, "dataset": row["dataset"], "coarse_manifest": str(coarse.manifest_path),
                "sampling_audit": str(output / "sampling.json"), "source_fingerprint": pack.source_fingerprint})
        result = {"status": "complete", "coarse_images": len(results), "checkpoint_step": step,
            "source_inputs": "locked_legacy_B", "source_signature": source_signature,
            "seed": 43, "steps": 40, "true_cfg_scale": 1., "use_kv_cache": False,
            "zero_equivalence": zero_equivalence, "quality_claim": False, "engineering_only": True,
            "samples": results, "seconds": time.monotonic() - started}
        write_json(run / "report.json", result)
        return result
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error), "seconds": time.monotonic() - started})
        raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = render_checkpoint(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

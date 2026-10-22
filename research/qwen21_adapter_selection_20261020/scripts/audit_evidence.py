"""只读重算本次报告的参数规模、固定评测与历史结果；不加载模型权重。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_drop(before, after):
    return (before - after) / before


def summarize():
    before = read(STUDY / "evidence/fixed_eval_000000.json")
    after = read(STUDY / "evidence/fixed_eval_000200.json")
    gate = read(STUDY / "evidence/gate.json")
    sample_report = read(STUDY / "evidence/pro_results_20261018.json")
    key = lambda row: (row["dataset"], row["sample_id"], row["index"])
    assert len(before["entries"]) == len(after["entries"]) == 18
    assert {key(x) for x in before["entries"]} == {key(x) for x in after["entries"]}
    metrics = {}
    for name in ("global", "face", "contact", "hand", "body"):
        values = [statistics.mean(row["losses"][name] for row in data["entries"])
                  for data in (before, after)]
        metrics[name] = {"initial": values[0], "final": values[1],
                         "relative_drop": relative_drop(*values)}
    for data in (before, after):
        assert math.isclose(statistics.mean(r["global_content_mse"] for r in data["entries"]),
                            data["global_content_mse"], abs_tol=1e-12)
    assert math.isclose(metrics["global"]["relative_drop"], gate["relative_global_drop"], abs_tol=1e-12)
    metrics["regional_total"] = {
        "initial": before["regional_total"], "final": after["regional_total"],
        "relative_drop": relative_drop(before["regional_total"], after["regional_total"]),
    }
    header = read(ROOT / "research/qwen21_controlnet_transfer_20261015/source_snapshot/upstream/union_safetensors_header.json")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    total = sum(math.prod(value["shape"]) for value in tensors.values())
    assert len(tensors) == 180 and total == 3_775_479_808
    coverage = sample_report["samples"]
    body = read(ROOT / "research/union_adapter_architecture_20261017/evidence/body20k_50_summary.json")
    body_rows = {arm: {k: v for k, v in row.items() if k in
                      ("identity_cosine", "identity_detected", "pose_pck_002", "psnr")}
                 for arm, row in body["arms"].items() if arm.startswith("qwen")}
    bfs = read(ROOT / "experiments/bfs_compare_30_20261004/results.json")["summary"]
    bfs_rows = {name: {k: row[k] for k in ("identity_A_mean", "identity_A_n", "eligible_pose_pck_002_mean")}
                for name, row in bfs.items()}
    face_rows = {}
    for resolution, file in ((512, "face_four_arm_512.json"), (1024, "face_val_1024.json")):
        data = read(STUDY / "evidence" / file)
        face_rows[str(resolution)] = {
            arm: {metric: data["metrics"][metric][arm]["strict_joint"]
                  for metric in ("identity_cosine", "pose_pck_002")}
            for arm in data["modes"]
        }
    return {
        "schema": "qwen21_adapter_selection_audit_v1", "observed_date": "2026-10-20",
        "union": {"parameter_count": total, "tensor_count": len(tensors),
                  "bf16_weight_bytes": total * 2, "bf16_weight_GiB": total * 2 / 2**30},
        "pro": {"trainable_parameters": gate["configured_trainable"]["trainable_parameters"],
                "fixed_positions": 18, "training_pairs": len(coverage),
                "people": sum(x["persons"] for x in coverage),
                "valid_face_id_people": sum(sum(row) for x in coverage for row in x["face_identity_valid"]),
                "control_lora_targets": gate["configured_trainable"]["control_lora"],
                "metrics": metrics, "engineering_passed": gate["engineering_passed"],
                "quality_passed": gate["passed"],
                "full_sampling_quality_images_at_20261018_report": sample_report["cancelled"]["new_train40step_images"]
                    + sample_report["cancelled"]["new_val40step_images"]},
        "light_body_50_pairs_NOT_full": body_rows,
        "bfs_common_24": bfs_rows,
        "light_body_face_reference_ablation_separate_512_test_and_1024_val": face_rows,
        "interpretation": "flow MSE、ArcFace、DINO 与不同归一化的 PCK 不跨口径排名。",
    }


def count_current_modules():
    # meta device allocates metadata only. Skip value-dependent initialization;
    # constructors and parameter/buffer shapes are otherwise unchanged.
    sys.path.insert(0, str(ROOT))
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter
    from src.pose_control.qwen21_pro import ProTaskAdapter, ProAdapterConfig
    config = SimpleNamespace(**read(STUDY / "source/deepgen_transformer_config.json"))
    with torch.device("meta"), patch.object(Qwen21FullAdapter, "reset_output_bridges", lambda self: None):
        full = Qwen21FullAdapter(UnifiedSMPLXAdapterV6.from_deepgen_config(
            config, enable_face_adapter=True, enable_hand_adapter=True))
        pro = ProTaskAdapter(ProAdapterConfig(source_dim=1600))
    return {
        "full_unique_parameters": sum(p.numel() for p in full.parameters()),
        "full_buffers": sum(p.numel() for p in full.buffers()),
        "full_state_tensor_elements": sum(p.numel() for p in full.state_dict().values() if torch.is_tensor(p)),
        "pro_unique_parameters_source_dim1600": sum(p.numel() for p in pro.parameters()),
        "method": "current source constructors on torch meta; only value-dependent output initialization skipped",
        "torch_version": torch.__version__,
    }


def verify_manifest():
    manifest = read(STUDY / "manifest.json")
    failures = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise SystemExit(json.dumps({"verification": "FAIL", "files": failures}, ensure_ascii=False))
    links = 0
    for document in (STUDY / "README.md", STUDY / "REPORT.zh-CN.md"):
        for link in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if not link.startswith("http"):
                links += 1
                if not (document.parent / link.split("#")[0]).is_file():
                    raise SystemExit(f"Missing report link: {link}")
    for script in (STUDY / "scripts").glob("*.py"):
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    return {"verification": "PASS", "verified_files": len(manifest["files"]), "local_links": links}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", action="store_true", help="额外用 torch meta 重算当前 Full/Pro 参数；无需 GPU")
    parser.add_argument("--verify", action="store_true", help="核对研究材料及引用源码的 SHA256")
    args = parser.parse_args()
    result = summarize()
    if args.parameters:
        result["current_module_counts"] = count_current_modules()
    if args.verify:
        result["manifest"] = verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))

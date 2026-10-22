"""为本次研究首次建立来源与SHA清单；拒绝覆盖已有manifest。"""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import subprocess

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
REFERENCES = [
    "src/pose_control/qwen21_full/qwen_adapter.py",
    "src/pose_control/qwen21_full/control_core.py",
    "src/pose_control/qwen21_full/deepgen_adapter.py",
    "src/pose_control/qwen21_full/joint_training.py",
    "src/pose_control/qwen21_full/execution.py",
    "src/pose_control/qwen21/body_adapter.py",
    "src/pose_control/qwen21_pro/adapter.py",
    "src/pose_control/qwen21_pro/integration.py",
    "src/pose_control/qwen21_pro/contracts.py",
    "src/pose_control/qwen21_pro/banks.py",
    "src/pose_control/qwen21_pro/encoders.py",
    "src/pose_control/qwen21_pro/features.py",
    "src/pose_control/qwen21_pro/losses.py",
    "src/pose_control/qwen21_pro/training.py",
    "src/pose_control/qwen21_pro/refinement.py",
    "research/adapter_value_diagnosis_20261011/evidence/checkpoint_parameter_counts.json",
    "research/adapter_value_diagnosis_20261011/evidence/audit_evidence.json",
    "research/adapter_value_diagnosis_20261011/evidence/fresh_parity_512.json",
    "research/qwen21_controlnet_transfer_20261015/source_snapshot/upstream/union_safetensors_header.json",
    "research/union_adapter_architecture_20261017/evidence/body20k_50_summary.json",
    "research/union_adapter_architecture_20261017/evidence/full_face_step0_summary.json",
    "research/union_adapter_architecture_20261017/evidence/full_face_step3000_summary.json",
    "research/union_adapter_architecture_20261017/evidence/overfit200_gate.json",
    "experiments/bfs_body_face_joint_pilot_20261008/report/summary.md",
    "experiments/bfs_compare_30_20261004/results.json",
    "experiments/qwen21_union_migration_v1/report/METRICS.md",
    "experiments/qwen21_union_migration_v1/output/summary.json",
    "experiments/qwen21_union_migration_v1/output/probe_ref_ablation/summary.json",
    "experiments/qwen21_union_migration_v1/scripts/reference_union_pipeline.py",
    "experiments/qwen21_union_migration_v1/scripts/08_evaluate.py",
    "experiments/qwen21_union_migration_v1/scripts/21_probe_ref_ablation.py",
    "experiments/qwen21_union_migration_v1/output/review/source.png",
    "experiments/qwen21_union_migration_v1/output/review/single__hard_02__union_s05__seed42.png",
    "experiments/qwen21_union_migration_v1/output/review/single__hard_02__union_s10__seed42.png",
]
remote_root = "/data/shangguanrz/pic-edit-experiments/qwen21_pro_adapter_v1"
pilot = remote_root + "/runs/pilot_main200_v4_20261018T161055"
remote_origins = {f"evidence/{name}": f"sg:{pilot}/{name}" for name in (
    "fixed_eval_000000.json", "fixed_eval_000200.json", "gate.json",
    "training_protocol.json", "report.json", "initialization.json")}
remote_origins.update({
    "evidence/pro_val_iper_sampling.json": f"sg:{remote_root}/runs/valpair_main200_20261020T001400/iper_val_e36352d6b4b72ae362cb1ce3/sampling.json",
    "evidence/pro_val_iper_coarse.png": f"sg:{remote_root}/runs/valpair_main200_20261020T001400/iper_val_e36352d6b4b72ae362cb1ce3/coarse/coarse.png",
    "evidence/pro_val_iper_source.png": "sg:/data/shangguanrz/pic-edit-experiments/qwen21_union_dataset_adaptation_v1/data/assets/iper_val_e36352d6b4b72ae362cb1ce3/source.png",
    "evidence/pro_results_20261018.json": "sg:/home/shangguanrz/project/pic-edit/experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1/report/results_20261018.json",
    "evidence/pro_prior_verification.json": "sg:/home/shangguanrz/project/pic-edit/experiments/6.qwen-controlnet union/qwen21_pro_adapter_v1/report/evidence/verification.json",
    "evidence/pro_valpair_started_20261020.json": f"sg:{remote_root}/runs/valpair_main200_20261020T001400/run.json",
    "evidence/face_four_arm_512.json": "sg:/home/shangguanrz/nihaochen/qwen21_face/test_four_arm_metrics/summary.json",
    "evidence/face_val_1024.json": "sg:/home/shangguanrz/nihaochen/iper1024/face_val50_best_loss_metrics/summary.json",
    "source/deepgen_transformer_config.json": "sg:/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/transformer/config.json",
})
for tag in ("10", "00"):
    name = f"single__hard_02__norefs_ctrl{tag}__seed42.png"
    remote_origins[f"evidence/{name}"] = "sg:/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/runs/probe_skeleton_ref_ablation_20261019T222727/" + name
paths = sorted({*(ROOT / ref for ref in REFERENCES),
                *(p for p in STUDY.rglob("*") if p.is_file() and p.name != "manifest.json"
                  and "__pycache__" not in p.parts)})
files = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
result = {
    "schema": "qwen21_adapter_selection_manifest_v1",
    "collected_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "local_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "remote_head_observed": "2ebd026b8568737d2f0e13c4ebea6a0634c54fff",
    "official_source_commit": "13bb3c625e8b1e457d7124d92207b49e00e4a6fe",
    "key_local_remote_source_hashes_equal": [
        "src/pose_control/qwen21_pro/adapter.py", "src/pose_control/qwen21_pro/integration.py",
        "src/pose_control/qwen21_pro/contracts.py", "src/pose_control/qwen21_full/qwen_adapter.py",
        "src/pose_control/qwen21_full/control_core.py"],
    "remote_origins": remote_origins,
    "official_source_snapshot_origin": "research/union_adapter_architecture_20261017/source_snapshot/official",
    "web_sources_checked_20261020": [
        "https://huggingface.co/Qwen/Qwen-Image-2.1",
        "https://github.com/QwenLM/Qwen-Image-2.1",
        "https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
        "https://github.com/aigc-apps/VideoX-Fun/blob/main/videox_fun/models/qwenimage21_transformer2d_control.py",
        "https://arxiv.org/abs/2401.07519", "https://arxiv.org/abs/2404.16022",
        "https://arxiv.org/abs/2506.21416"],
    "limitations": ["No new model training or GPU inference in this study.",
                    "First 40-step iPER validation image appeared during the study; no matched baseline/quality report, dual inference pending.",
                    "5 matching source hashes do not establish whole repository parity.",
                    "Manifest excludes itself; external repository references must remain at these bytes."],
    "files": files,
}
with (STUDY / "manifest.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"Sealed {len(files)} files")

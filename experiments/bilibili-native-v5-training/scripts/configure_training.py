"""Export standard Native V5 stage configs; never construct a model or optimizer."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from split_dataset import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=Path("/home/shangguanrz/project/pic-edit"))
    args = p.parse_args()
    sys.path.insert(0, str(args.repo_root))
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig, stage_preset
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    base = TrainingConfig.from_dict(json.loads((args.native_root / "S0_prepare.json").read_text(encoding="utf-8")))
    if base.experiment_mode != "standard" or base.qwen_condition_profile != "single_2ref":
        raise ValueError("Expected standard, single_2ref Bilibili data")
    if {s.role for s in base.sources} != {"train", "validation"}:
        raise ValueError("Only held-out train/validation sources may enter these configs")
    args.output.mkdir(parents=True, exist_ok=False)
    excluded = ("interaction_heads", "interaction_controller", "relation_raster", "relation_tokens", "strength_interaction")
    configs = {}
    for phase in ("S0", "S1", "S2", "S3"):
        stage = stage_preset(phase)
        if phase in ("S1", "S2"):
            stage = replace(stage, routes=tuple(r for r in stage.routes if r.branch != "interaction"))
        stage = replace(stage, exclude_groups=excluded)
        cfg = replace(base, stage=stage, initialization=base.initialization if phase == "S0" else None,
                      gradient_report=None, body_baseline_report=None)
        result = NativeV5Trainer(cfg).prepare()
        configs[phase] = {"config": write(args.output / (phase + ".json"), cfg.record()),
                          "static_prepare": write(args.output / (phase + "_prepare.json"), result),
                          "missing_bindings": cfg.missing()}
    summary = {"schema": "bilibili_native_v5_stage_templates_v1", "configs": configs,
        "geometry_method": "sapiens_v1", "experiment_mode": "standard", "test_in_training_sources": False,
        "training_started": False, "optimizer_created": False, "model_constructed": False,
        "budget": "explicit --steps is required at launch; not copied from the 300-step pilot",
        "stage_transitions": "S0=train; S1/S2/S3=advance-stage from the exact committed previous checkpoint",
        "required_before_launch": ["Qwen frozen cache index with SHA", "explicit update budget",
            "matching real gradient report", "previous trained checkpoint and applicable stage gates for S1-S3"]}
    write(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

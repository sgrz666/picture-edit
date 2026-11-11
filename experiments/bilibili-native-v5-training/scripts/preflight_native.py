"""Run the real full Native V5 data preflight, printing reader progress only."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path)
    p.add_argument("--repo-root", type=Path, default=Path("/home/shangguanrz/project/pic-edit"))
    args = p.parse_args()
    sys.path.insert(0, str(args.repo_root))
    import torch
    torch.set_num_threads(4)
    from src.pose_control.qwen21_full.native_v5.data import NativeFullDatasetV5
    from src.pose_control.qwen21_full.native_v5.training.config import TrainingConfig
    from src.pose_control.qwen21_full.native_v5.training.trainer import NativeV5Trainer
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    total = sum(sum(bool(line.strip()) for line in Path(s.manifest["path"]).open(encoding="utf-8")) for s in config.sources)
    if args.test_manifest:
        total += sum(bool(line.strip()) for line in args.test_manifest.open(encoding="utf-8"))
    original = NativeFullDatasetV5.__getitem__
    count = 0
    def reader(dataset, index):
        nonlocal count
        result = original(dataset, index)
        count += 1
        if count % 50 == 0:
            print(json.dumps({"event": "native_data_reader", "completed": count, "total": total}), flush=True)
        return result
    NativeFullDatasetV5.__getitem__ = reader
    try:
        result = NativeV5Trainer(config).preflight_data(args.report)
        test_count = 0
        if args.test_manifest:
            from src.pose_control.qwen21_full.native_v5.conditioning import read_condition
            test = NativeFullDatasetV5(args.test_manifest, load_target=True, data_profile="full_smplx")
            for index in range(len(test)):
                sample = test[index]
                read_condition(test.rows[index]["qwen_condition"], sample)
                test_count += 1
    finally:
        NativeFullDatasetV5.__getitem__ = original
    def present(value):
        return any(present(v) for v in value) if isinstance(value, list) else bool(value)
    summary = {"passed": result["passed"], "samples_read": count, "total": total, "test_samples": test_count,
        "preflight_phase": config.stage.phase,
        "active_route_usable_counts_train_val": {name: sum(bool(r["branch_usable"][name]) for r in result["rows"])
                                           for name in ("body", "face", "hand", "detail", "interaction")},
        "available_input_counts_train_val": {name: sum(present(r["present"][name]) for r in result["rows"])
                                             for name in ("face_geometry", "hand_geometry", "arcface", "face_dino", "hand_dino")},
        "all_original_rows_retained": result["all_rows_retained"],
        "model_constructed": False, "optimizer_created": False, "training_started": False,
        "report": str(args.report)}
    with args.report.with_name(args.report.stem + "_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

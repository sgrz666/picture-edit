"""为已登记val12补8条SMPL-X几何，复用已核验4条mixed Hi4D，不读GT像素。"""
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
    parser.add_argument("--mixed-run", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    import torch
    from src.pose_control.qwen21_pro.cache import file_sha256
    from src.pose_control.qwen21_pro.geometry_preparation import P_ROOT, _job_main, load_source_geometry
    from src.pose_control.qwen21_pro.preparation import resolve_condition_assets
    torch.set_num_threads(4)
    run, legacy, project, mixed = map(Path, (args.run_dir, args.legacy_workspace, args.project_root, args.mixed_run))
    if not run.is_absolute() or not run.resolve().is_relative_to(P_ROOT.resolve()):
        parser.error("run-dir必须是P内的新绝对目录")
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(run / "run.json", {"argv": vars(args), "code_root": str(ROOT), "script_sha256": file_sha256(__file__),
                                 "conditions_sha256": file_sha256(args.conditions), "geometry_only": True})
    try:
        original_path = legacy / "data/val_conditions.jsonl"
        lock = json.loads((legacy / "data/lock.json").read_text())
        if file_sha256(original_path) != lock["selected_manifest_sha256"][original_path.name]:
            raise ValueError("Locked val manifest changed")
        original = {r["sample_id"]: r for r in map(json.loads, original_path.read_text().splitlines())}
        records = list(map(json.loads, Path(args.conditions).read_text().splitlines()))
        if (len(records) != 12 or len({r["sample_id"] for r in records}) != 12
                or any(original.get(r["sample_id"]) != r for r in records)
                or any(sum(r["dataset"] == d for r in records) != 4 for d in ("iper", "tiktok", "hi4d"))):
            raise ValueError("Expected unchanged preregistered 4/4/4 val records")
        if not json.loads((mixed / "report.json").read_text())["passed"]:
            raise ValueError("Mixed geometry gate did not pass")
        mixed_index = json.loads((mixed / "geometry_index.json").read_text())
        sources, current, jobs = {}, {}, []
        for record in records:
            sid, dataset = record["sample_id"], record["dataset"]
            assets = resolve_condition_assets(record, project)
            if dataset == "hi4d":
                entry = mixed_index[sid]
                load_source_geometry(entry["source"], record, assets)
                sources[sid], current[sid] = entry["source"], entry["current"]
                continue
            sequence = record["metadata"]["sequence_id"]
            base = (project / "datasets/iPER/iper_sampled_6src64tgt" / sequence / "smplestx" if dataset == "iper"
                    else project / "datasets/TikTokDataset/TikTok_3d_assets_native" / sequence / "smplestx")
            job = {"dataset": dataset, "mode": "driver_exact", "for_training": True,
                   "smplest_root": str(project / "add/tools/SMPLest-X"), "params": str(base / "params.pt"),
                   "frame_map": str(base / "frame_map.json"), "source_frame": Path(assets["source_rgb"]).stem,
                   "driver_frame": Path(assets["target_keypoints"][0]).stem, "person_id": record["person_order"][0],
                   "canvas_hw": [512, 512], "output_dir": str(run / sid / "driver_exact")}
            jobs.append(job)
            _job_main(job)
            current[sid] = str(Path(job["output_dir"]) / "manifest.json")
            print(json.dumps({"sample_id": sid, "dataset": dataset, "geometry": current[sid]}), flush=True)
        write_json(run / "jobs.json", jobs)
        write_json(run / "source_geometry_index.json", sources)
        write_json(run / "current_geometry_index.json", current)
        write_json(run / "report.json", {"passed": True, "samples": 12, "new_single_jobs": len(jobs),
                                         "reused_mixed_samples": len(sources), "seconds": time.monotonic() - started})
        print(json.dumps({"passed": True, "samples": 12, "seconds": time.monotonic() - started}), flush=True)
    except Exception as error:
        write_json(run / "failure.json", {"type": type(error).__name__, "error": str(error),
                                          "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

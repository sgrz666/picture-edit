"""Acceptance for 方案 §4「删除危险默认值」: every legacy door must be locked, and locked loudly.

Each check declares what a *pass* looks like: some doors must refuse (the dangerous ones), two must
accept (the explicit opt-ins, otherwise historical runs are unreproducible). Nothing here loads the
7B backbone, so it is a CPU-cheap gate check.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN = "datasets/TikTokDataset/TikTok_v3/manifests/pairs/train_pairs.jsonl"
VAL = "datasets/TikTokDataset/TikTok_v3/manifests/pairs/val_pairs.jsonl"
ENTRY = "scripts/train/train_qwen21_native_full.py"
LEGACY = "scripts/train/train_iper_qwen21_full.py"


def run_cli(args: list[str], *, env_extra: dict[str, str] | None = None) -> dict:
    import os
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=1800, env=env)
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]
    return {"exit": proc.returncode, "tail": " | ".join(lines[-2:])[:400]}


def call(fn) -> dict:
    try:
        fn()
        return {"exit": 0, "tail": "returned normally"}
    except Exception as exc:  # noqa: BLE001 - the point is which door and what it says
        return {"exit": 1, "tail": f"{type(exc).__name__}: {str(exc)[:320]}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/canvas_lock_check"))
    args = parser.parse_args()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    output = str(args.scratch_dir / "unused")

    results = {
        # must_refuse: the dangerous path is unreachable
        "must_refuse": {
            "new_entry_refuses_resize_square": run_cli(
                [ENTRY, "--train-pairs", TRAIN, "--val-pairs", VAL, "--output", output,
                 "--canvas-policy", "resize-square", "--preflight-only"]),
            "new_entry_has_no_legacy_escape_hatch": run_cli(
                [ENTRY, "--train-pairs", TRAIN, "--val-pairs", VAL, "--output", output,
                 "--canvas-policy", "resize-square", "--legacy-policy-ok", "--preflight-only"]),
            "new_entry_refuses_warmup_swallowing_the_run": run_cli(
                [ENTRY, "--train-pairs", TRAIN, "--val-pairs", VAL, "--output", output,
                 "--canvas-policy", "native-pad", "--steps", "4", "--warmup-steps", "4"]),
            "tiktok_loader_refuses_missing_align_mode": call(lambda: _loader(None)),
            "tiktok_loader_refuses_resize_without_opt_in": call(lambda: _loader("resize")),
            "resize_record_cannot_enter_training": call(_refused_by_canonical),
            "missing_canvas_is_not_defaulted_to_square": call(_refused_when_absent),
            "checkpoint_without_canvas_record_needs_opt_in": call(
                lambda: _gate({"step": 1})),
        },
        # must_accept: the documented reproduction path really works
        "must_accept": {
            "canonical_guard_accepts_real_canvas": call(_canonical_accepts_a_real_canvas),
            "tiktok_loader_accepts_resize_with_opt_in": call(lambda: _loader("resize", legacy=True)),
            "canvas_v2_checkpoint_is_accepted": call(
                lambda: _gate({"canvas_schema_version": "canvas_v2",
                               "canvas_policy": "aspect-preserving"})),
            "legacy_checkpoint_warm_start_with_opt_in": call(
                lambda: _gate({"step": 1}, ok=True)),
            "legacy_entry_help_under_pythonpath": run_cli([LEGACY, "--help"],
                                                           env_extra={"PYTHONPATH": str(ROOT)}),
        },
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    failures = [name for name, res in results["must_refuse"].items() if res["exit"] == 0]
    failures += [name for name, res in results["must_accept"].items() if res["exit"] != 0]
    print(json.dumps({"failed_checks": failures}, ensure_ascii=False))
    return 1 if failures else 0


def _loader(align_mode: str | None, *, legacy: bool = False) -> None:
    from src.data.tiktok_dataset import load_pose_dataset
    kwargs: dict = {"resolution": 512, "dataset_type": "tiktok_v3"}
    if align_mode is not None:
        kwargs["align_mode"] = align_mode
    if legacy:
        kwargs["legacy_policy_ok"] = True
    load_pose_dataset(TRAIN, **kwargs)


def _refused_by_canonical() -> None:
    from src.data.canvas_contract import canvas_for_mode, require_canonical
    require_canonical(canvas_for_mode("resize", (1080, 604), resolution=512), where="probe")


def _refused_when_absent() -> None:
    from src.data.canvas_contract import require_canonical
    require_canonical(None, where="probe")


def _canonical_accepts_a_real_canvas() -> None:
    from src.data.canvas_contract import canvas_for_mode, require_canonical
    spec = canvas_for_mode("aspect_preserving", (1080, 604), max_pixels=65536)
    if require_canonical(spec, where="probe") is not spec:
        raise AssertionError("a canonical aspect-preserving canvas must pass the guard")


def _gate(checkpoint: dict, *, ok: bool = False) -> None:
    from src.pose_control.qwen21_full.canvas_provenance import require_canvas_compatible
    require_canvas_compatible(checkpoint, warm_start_ok=ok)


if __name__ == "__main__":
    raise SystemExit(main())

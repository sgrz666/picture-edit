"""Reproduce the local source audit; does not import or run external models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    snapshot = ROOT / "source_snapshot"
    transformer = json.loads((snapshot / "qwen_model/transformer/config.json").read_text())
    vae = json.loads((snapshot / "qwen_model/vae/config.json").read_text())
    repos = {}
    for tree_path in sorted((ROOT / "repositories").glob("*/tree.json")):
        tree = json.loads(tree_path.read_text(encoding="utf-8-sig"))
        repos[tree_path.parent.name] = {
            "revision": tree["sha"],
            "tree_truncated": tree.get("truncated", False),
            "python_files_in_tree": sum(item["path"].endswith(".py") for item in tree["tree"]),
        }
    repos["PeopleComposer"] = {
        "revision": "efb836371d48f8d42b629084c89d93b30b6cce9c",
        "revision_method": "git ls-remote refs/heads/main; source URLs pinned to this revision",
        "public_loader_missing": "package is not present in this checkout" in (
            ROOT / "repositories/PeopleComposer/train.py"
        ).read_text(encoding="utf-8"),
    }
    groups = [[i for i in range(32) if 6 * i // 32 == group] for group in range(6)]
    intergen = ROOT / "repositories/tr3e__InterGen/models"
    cosh = ROOT / "repositories/AliManjotho__coshmdm-code/models"
    equal_files = [name for name in ("nets.py", "blocks.py", "losses.py")
                   if digest(intergen / name) == digest(cosh / name)]
    files = {}
    for base in (snapshot, ROOT / "repositories"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml"}:
                files[path.relative_to(ROOT).as_posix()] = digest(path)
    checks = {
        "qwen_32_blocks": transformer["num_layers"] == 32,
        "qwen_4096_hidden": transformer["num_attention_heads"] * transformer["attention_head_dim"] == 4096,
        "qwen_native_64_channel_tokens": transformer["in_channels"] == transformer["out_channels"] == 64,
        "qwen_native_patch_size_one": transformer["patch_size"] == 1,
        "vae_64_channels": vae["z_dim"] == 64,
        "qwen_causal_condition": transformer["causal_condition"] is True,
        "full_groups_cover_all_32_blocks": sorted(sum(groups, [])) == list(range(32)),
        "dance_public_tree_has_no_python": repos["yisuanwang__DanceTog"]["python_files_in_tree"] == 0,
        "cosh_core_three_files_equal_intergen": len(equal_files) == 3,
        "peoplecomposer_reports_missing_public_loader": repos["PeopleComposer"]["public_loader_missing"],
    }
    report = {
        "scope": "Static source/configuration evidence only; no upgraded adapter or training was executed.",
        "audit_date": "2026-10-11",
        "remote_project": "/home/shangguanrz/project/pic-edit",
        "remote_git_head_at_snapshot": "3236657b08f0fe51d881a78c114bf57b956e953e",
        "transformer_config": transformer,
        "vae_config": vae,
        "full_injection_groups": groups,
        "legacy_lite_injection_indices": [3, 8, 13, 18, 23, 28],
        "videox_control_indices": list(range(0, 32, 2)),
        "repositories": repos,
        "cosh_files_identical_to_intergen": equal_files,
        "checks": checks,
        "source_sha256": files,
    }
    (ROOT / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": sum(checks.values()), "total": len(checks), "output": str(ROOT / "evidence.json")}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

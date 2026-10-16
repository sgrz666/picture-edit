"""方案 B 的统一入口。"""
from pathlib import Path
import os
import sys

project = Path(os.environ.get("QWEN21_UNION_PROJECT", "/home/shangguanrz/project/pic-edit"))
for parent in Path(__file__).resolve().parents:
    if (parent / "src/pose_control/qwen21_union/runner.py").is_file():
        sys.path.insert(0, str(parent))
        break
import src
if project.is_dir() and str(project / "src") not in src.__path__:
    src.__path__.append(str(project / "src"))
for child in ("data", "pose_control", "pose_edit"):
    import importlib
    package = importlib.import_module("src." + child)
    shared = str(project / "src" / child)
    if project.is_dir() and shared not in package.__path__:
        package.__path__.append(shared)

if len(sys.argv) > 1 and sys.argv[1] in ("cache", "smoke", "overfit", "train", "infer", "evaluate"):
    from src.pose_control.qwen21_union.training import configure_determinism
    configure_determinism()

from src.pose_control.qwen21_union.runner import main

if __name__ == "__main__":
    main()

"""Qwen21 Pro共享准备、缓存、推理和精修入口。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if len(sys.argv) > 1 and sys.argv[1] == "status":
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen21_pro_status", ROOT / "src/pose_control/qwen21_pro/runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.main
else:
    from src.pose_control.qwen21_pro.runner import main

if __name__ == "__main__":
    raise SystemExit(main())

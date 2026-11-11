"""Read existing geometry parameters, producer and model bindings without inference."""
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path("/home/shangguanrz/project/pic-edit")
sys.path.insert(0, str(ROOT))
DATA = Path("/data/shangguanrz/datasets/bilibili")
RUN = Path(__file__).resolve().parents[1]
BASELINE = Path("/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1")

report = {"modules": {}, "parameters": {}, "bindings": {}}
for module in ("smplx", "nvdiffrast", "numpy", "diffusers", "cv2"):
    report["modules"][module] = importlib.util.find_spec(module) is not None
for method in ("smplest", "sapiens_v3", "wilor"):
    path = next((DATA / "geometry/smplx" / method).glob("*/*.pt"))
    raw = torch.load(path, map_location="cpu", weights_only=False)
    report["parameters"][method] = {"file": str(path), "fields": {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} if hasattr(v, "shape") else str(v)[:180]
        for k, v in raw.items()}}
for name, path in {
    "baseline_initialization": BASELINE / "initialization/native_v5_init_v2.pt",
    "qwen_manifest": BASELINE / "source/qwen_model_files.json",
    "base_model": Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"),
    "smplx_model": ROOT / "add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz",
    "hia_tool": ROOT / "experiments/6.qwen-controlnet union/hia_geometry_training_20261029/scripts/hia_training.py",
}.items():
    report["bindings"][name] = {"path": str(path), "exists": path.exists()}
print(json.dumps(report, indent=2))
with (RUN / "report/runtime_bindings.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

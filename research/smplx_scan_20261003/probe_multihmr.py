"""Read-only model smoke test. Run with the existing sg multihmr environment."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--threshold", type=float, default=0.3)
args = parser.parse_args()
image_path = str(Path(args.image).resolve())
output_path = Path(args.output).resolve()
repo = Path("/home/shangguanrz/project/pic-edit/add/tools/multi-hmr")
os.environ["MULTIHMR_DINOV2_REPO"] = str(repo.parent / "dinov2_e1277af")
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.chdir(repo)
sys.path.insert(0, str(repo))
import numpy as np
import torch
from demo_smplx_only import load_model, open_image, get_camera_parameters, forward_model

assert (repo / "models/multiHMR/multiHMR_896_L.pt").is_file()
started = time.perf_counter()
model = load_model("multiHMR_896_L").eval()
load_seconds = time.perf_counter() - started
x, original = open_image(image_path, model.img_size)
K = get_camera_parameters(model.img_size, fov=60)
torch.cuda.synchronize()
started = time.perf_counter()
humans = forward_model(model, x, K, det_thresh=args.threshold, nms_kernel_size=3)
torch.cuda.synchronize()
inference_seconds = time.perf_counter() - started
arrays = {"K_canvas": K.detach().cpu().numpy()}
people = []
for i, person in enumerate(humans):
    info = {"index": i, "fields": {}, "finite": True}
    for key, value in person.items():
        if torch.is_tensor(value):
            arr = value.detach().float().cpu().numpy()
            arrays[f"person_{i}_{key}"] = arr
            info["fields"][key] = list(arr.shape)
            info["finite"] &= bool(np.isfinite(arr).all())
            if key in ("scores", "transl", "transl_pelvis"):
                info[key] = arr.tolist()
    people.append(info)
result = {
    "model": "multiHMR_896_L", "image": image_path,
    "original_size_wh": list(original.size), "canvas_size": model.img_size,
    "camera": {"source": "assumed_fov", "fov_degrees": 60, "K_canvas": arrays["K_canvas"].tolist()},
    "threshold": args.threshold, "nms_kernel_size": 3,
    "person_count": len(people), "people": people,
    "load_seconds": load_seconds, "first_forward_seconds": inference_seconds,
    "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    "torch": torch.__version__,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path.with_suffix(".npz"), **arrays)
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

"""Single-person forward using existing sg weights; outputs only under --output."""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
image_path = str(Path(args.image).resolve())
output_path = Path(args.output).resolve()
repo = Path("/home/shangguanrz/project/pic-edit/add/tools/SMPLest-X")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["YOLO_CONFIG_DIR"] = str(output_path.parent / "ultralytics_settings")
os.chdir(repo)
sys.path.insert(0, str(repo))
import numpy as np
import torch
from torchvision.transforms import ToTensor
from ultralytics import YOLO
from main.config import Config
from human_models.human_models import SMPLX
from models.SMPLest_X import get_model
from utils.data_utils import load_img, process_bbox, generate_patch_image

cfg = Config.load_config(str(repo / "pretrained_models/smplest_x_h/config_base.py"))
checkpoint = repo / "pretrained_models/smplest_x_h/smplest_x_h.pth.tar"
assert checkpoint.is_file()
smpl_x = SMPLX(cfg.model.human_model_path)
started = time.perf_counter()
model = torch.nn.DataParallel(get_model(cfg, "test")).cuda()
ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = OrderedDict()
for key, value in ckpt["network"].items():
    if "module" not in key:
        key = "module." + key
    key = key.replace("backbone", "encoder").replace("body_rotation_net", "body_regressor").replace("hand_rotation_net", "hand_regressor")
    state[key] = value
load_result = model.load_state_dict(state, strict=False)
del state, ckpt
model.eval()
load_seconds = time.perf_counter() - started
original = load_img(image_path)
h, w = original.shape[:2]
detector = YOLO(str(repo / "pretrained_models/yolov8x.pt"))
det = detector.predict(original, device="cuda", classes=0, conf=cfg.inference.detection.conf, save=False, verbose=False)[0]
boxes = det.boxes.xyxy.detach().cpu().numpy()
if len(boxes) == 0:
    raise RuntimeError("No person detected at the configured threshold")
idx = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
xyxy = boxes[idx]
xywh = np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
bbox = process_bbox(xywh, w, h, cfg.model.input_img_shape, ratio=cfg.data.bbox_ratio)
patch, image_to_crop, crop_to_image = generate_patch_image(original, bbox, 1.0, 0.0, False, cfg.model.input_img_shape)
x = (ToTensor()(patch.astype(np.float32)) / 255).cuda().unsqueeze(0)
torch.cuda.synchronize()
started = time.perf_counter()
with torch.no_grad():
    out = model({"img": x}, {}, {}, "test")
torch.cuda.synchronize()
inference_seconds = time.perf_counter() - started
layer = model.module.smplx_layer
zeros = torch.zeros_like(out["smplx_root_pose"])
with torch.no_grad():
    reconstructed = layer(
        global_orient=out["smplx_root_pose"], body_pose=out["smplx_body_pose"],
        left_hand_pose=out["smplx_lhand_pose"], right_hand_pose=out["smplx_rhand_pose"],
        jaw_pose=out["smplx_jaw_pose"], betas=out["smplx_shape"], expression=out["smplx_expr"],
        transl=out["cam_trans"], leye_pose=zeros, reye_pose=zeros,
    )
    mesh_error = float((reconstructed.vertices - out["smplx_mesh_cam"]).abs().max())
    relative = reconstructed.joints[:, smpl_x.joint_idx] - reconstructed.joints[:, :1]
    joint_error = float((relative - out["smplx_joint_cam"]).abs().max())
focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
K = np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1]])
arrays = {key: value.detach().float().cpu().numpy() for key, value in out.items() if torch.is_tensor(value) and key != "img"}
arrays.update(K_image=K, image_to_crop=image_to_crop, crop_to_image=crop_to_image,
              joints_camera=reconstructed.joints.detach().cpu().numpy())
result = {
    "model": "SMPLest-X Huge", "image": image_path, "original_size_wh": [w, h],
    "detected_person_count": len(boxes), "selected_person_index": idx,
    "det_conf": float(det.boxes.conf[idx]), "processed_bbox_xywh": bbox.tolist(),
    "input_crop_hw": list(cfg.model.input_img_shape), "K_image": K.tolist(),
    "fields": {key: list(value.shape) for key, value in arrays.items()},
    "finite": all(bool(np.isfinite(value).all()) for value in arrays.values()),
    "flat_hand_mean": bool(layer.flat_hand_mean),
    "mesh_reconstruction_max_abs_m": mesh_error,
    "root_relative_joint_check_max_abs_m": joint_error,
    "checkpoint_missing_keys": load_result.missing_keys,
    "checkpoint_unexpected_keys": load_result.unexpected_keys,
    "load_seconds": load_seconds, "first_forward_seconds": inference_seconds,
    "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path.with_suffix(".npz"), **arrays)
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

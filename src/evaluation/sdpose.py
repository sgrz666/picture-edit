"""Standardized SDPose-Wholebody + YOLO11x pose estimation backend for evaluation.

All pose-alignment evaluations default to this backend (see ``docs/SDPOSE_POSE_EVAL_SPEC.md``).
Outputs follow the repo's 134-slot DWPose layout (``pose_edit.geometry.cocowholebody133_to_dwpose134``,
where ``14=R_eye, 15=R_ear, 16=L_eye, 17=L_ear``) by default, and can also be converted to
OpenPose-18 order (``14=R_eye, 15=L_eye, 16=R_ear, 17=L_ear``) via ``order="openpose"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from src.pose_edit.geometry import cocowholebody133_to_dwpose134

DEFAULT_SDPOSE_OOD_ROOT = Path("/data/shangguanrz/models/SDpose/SDPose-OOD")
DEFAULT_SDPOSE_CKPT_DIR = Path("/data/shangguanrz/models/SDpose/SDPose-Wholebody")
DEFAULT_YOLO_FILENAME = "yolo11x.pt"

DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.65
DEFAULT_INPUT_SIZE = (768, 1024)
DEFAULT_BBOX_PADDING = 1.25
DEFAULT_TIMESTEP = 999
DEFAULT_SCORE_THRESHOLD = 0.30

NECK_SLOT = 1
NECK_FROM_COCO_SLOTS = (5, 6)  # L_shoulder=5, R_shoulder=6 in COCO-17/133
OPENPOSE_SWAP_SLOTS = (15, 16)


@dataclass
class SDPoseRuntime:
    """Loaded YOLO11x + SDPose-Wholebody pipeline components."""

    yolo: Any
    pipe: Any
    get_cs: Any
    topdown_affine: Any
    flip_indices: list[int]
    device: Any
    dtype: Any
    ckpt_dir: Path
    ood_root: Path


def swap_eye_ear_slots(arr: np.ndarray) -> np.ndarray:
    """Swap slots 15 <-> 16 to toggle between geometry.py and OpenPose/metrics.py eye/ear order."""
    out = np.asarray(arr).copy()
    out[[OPENPOSE_SWAP_SLOTS[0], OPENPOSE_SWAP_SLOTS[1]]] = out[[OPENPOSE_SWAP_SLOTS[1], OPENPOSE_SWAP_SLOTS[0]]]
    return out


def convert_coco133_person(
    points133: np.ndarray,
    scores133: np.ndarray,
    *,
    order: str = "geometry",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw COCO-WholeBody 133 keypoints/scores into this repo's 134-slot layout.

    Args:
        points133: Array of shape ``(133, 2)``.
        scores133: Array of shape ``(133,)``.
        order: ``"geometry"`` keeps ``pose_edit.geometry`` / ``render_poseguide`` order
            (``14=R_eye, 15=R_ear, 16=L_eye, 17=L_ear``); ``"openpose"`` swaps 15<->16
            to match OpenPose / ``src.rl.metrics.DWPOSE134_TO_COCO17``.
    """
    if order not in ("geometry", "openpose"):
        raise ValueError(f"Unsupported 134-slot order: {order!r}")
    pts133 = np.asarray(points133, dtype=float)
    sc133 = np.asarray(scores133, dtype=float)
    if pts133.shape != (133, 2):
        raise ValueError(f"Expected points133 shape (133, 2), got {pts133.shape}")
    if sc133.shape != (133,):
        raise ValueError(f"Expected scores133 shape (133,), got {sc133.shape}")

    pts134 = cocowholebody133_to_dwpose134(pts133)
    scores134 = cocowholebody133_to_dwpose134(sc133[:, None])[:, 0].copy()
    l_sh, r_sh = NECK_FROM_COCO_SLOTS
    if np.isfinite(pts133[l_sh]).all() and np.isfinite(pts133[r_sh]).all():
        scores134[NECK_SLOT] = float(min(sc133[l_sh], sc133[r_sh]))
    else:
        scores134[NECK_SLOT] = 0.0

    if order == "openpose":
        pts134 = swap_eye_ear_slots(pts134)
        scores134 = swap_eye_ear_slots(scores134)
    return pts134, scores134


def load_runtime(
    ckpt_dir: Path | str = DEFAULT_SDPOSE_CKPT_DIR,
    ood_root: Path | str = DEFAULT_SDPOSE_OOD_ROOT,
    yolo_path: Path | str | None = None,
    *,
    device: str | None = None,
) -> SDPoseRuntime:
    """Load YOLO11x detector and SDPose-Wholebody diffusion pose estimator."""
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, CLIPTextModel
    from ultralytics import YOLO

    ood_path = Path(ood_root)
    ckpt_path = Path(ckpt_dir)
    yolo_file = Path(yolo_path) if yolo_path is not None else ckpt_path / DEFAULT_YOLO_FILENAME
    if not ood_path.is_dir():
        raise FileNotFoundError(f"SDPose-OOD repo not found: {ood_path}")
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"SDPose checkpoint directory not found: {ckpt_path}")
    if not yolo_file.is_file():
        raise FileNotFoundError(f"YOLO11x weights not found: {yolo_file}")

    if str(ood_path) not in sys.path:
        sys.path.insert(0, str(ood_path))

    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.datasets.transforms import GetBBoxCenterScale, TopdownAffine
    from models.HeatmapHead import get_heatmap_head
    from models.ModifiedUNet import Modified_forward
    from pipelines.SDPose_D_Pipeline import SDPose_D_Pipeline

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    weight_dtype = torch.float32

    meta = parse_pose_metainfo(dict(from_file=str(ood_path / "configs/_base_/datasets/coco_wholebody.py")))
    flip_indices = list(meta["flip_indices"])

    yolo = YOLO(str(yolo_file))
    unet = UNet2DConditionModel.from_pretrained(str(ckpt_path), subfolder="unet", torch_dtype=weight_dtype)
    unet = Modified_forward(unet, keypoint_scheme="wholebody")
    vae = AutoencoderKL.from_pretrained(str(ckpt_path), subfolder="vae", torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(str(ckpt_path), subfolder="text_encoder", torch_dtype=weight_dtype)
    scheduler = DDPMScheduler.from_pretrained(str(ckpt_path), subfolder="scheduler")
    decoder = get_heatmap_head(mode="wholebody")
    decoder.load_state_dict(load_file(str(ckpt_path / "decoder/decoder.safetensors")))

    pipe = SDPose_D_Pipeline(
        unet=unet.to(torch_device).eval(),
        vae=vae.to(torch_device).eval(),
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(torch_device).eval(),
        decoder=decoder.to(torch_device).eval(),
        scheduler=scheduler,
    )

    return SDPoseRuntime(
        yolo=yolo,
        pipe=pipe,
        get_cs=GetBBoxCenterScale(padding=DEFAULT_BBOX_PADDING),
        topdown_affine=TopdownAffine(input_size=DEFAULT_INPUT_SIZE, use_udp=True),
        flip_indices=flip_indices,
        device=torch_device,
        dtype=weight_dtype,
        ckpt_dir=ckpt_path,
        ood_root=ood_path,
    )


def detect_boxes(
    image_bgr: np.ndarray,
    runtime: SDPoseRuntime,
    *,
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    min_area: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run YOLO11x class-0 person detection and return ``(fg_boxes, fg_scores, raw_count)``."""
    det_res = runtime.yolo.predict(
        image_bgr,
        classes=[0],
        conf=conf,
        iou=iou,
        verbose=False,
        device=str(runtime.device),
    )
    if len(det_res) == 0 or det_res[0].boxes is None or len(det_res[0].boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0

    boxes = det_res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = det_res[0].boxes.conf.cpu().numpy().astype(np.float32)
    raw_count = int(len(boxes))
    if min_area > 0.0:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = areas >= float(min_area)
        boxes = boxes[keep]
        scores = scores[keep]
    return boxes, scores, raw_count


def pose_on_boxes(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray | list[list[float]],
    runtime: SDPoseRuntime,
    *,
    box_scores: np.ndarray | list[float] | None = None,
    order: str = "geometry",
    flip_test: bool = True,
) -> list[dict[str, Any]]:
    """Run SDPose-Wholebody top-down inference on explicit xyxy boxes."""
    import torch
    from mmengine.structures import InstanceData
    from mmpose.structures import PoseDataSample

    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    if box_scores is None:
        bscores = np.ones((len(boxes),), dtype=np.float32)
    else:
        bscores = np.asarray(box_scores, dtype=np.float32).reshape(-1)

    people: list[dict[str, Any]] = []
    for bbox, bscore in zip(boxes, bscores):
        data = runtime.topdown_affine(
            runtime.get_cs(
                {
                    "img": image_bgr,
                    "bbox": bbox[None, :],
                    "bbox_score": np.array([float(bscore)], dtype=np.float32),
                }
            )
        )
        crop_rgb = cv2.cvtColor(data["img"], cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device=runtime.device, dtype=runtime.dtype)

        ds = PoseDataSample()
        ds.gt_instances = InstanceData(
            bboxes=data["bbox"],
            bbox_scores=data["bbox_score"],
            bbox_centers=data["bbox_center"],
            bbox_scales=data["bbox_scale"],
        )
        ds.set_metainfo({"input_size": DEFAULT_INPUT_SIZE, "flip_indices": runtime.flip_indices})

        with torch.inference_mode():
            out_samples = runtime.pipe(
                rgb_tensor,
                show_progress_bar=False,
                timesteps=[DEFAULT_TIMESTEP],
                mode="predict",
                data_samples=[ds],
                test_cfg={"flip_test": bool(flip_test)},
            )

        pred = out_samples[0].pred_instances
        kpts133 = np.asarray(pred.keypoints[0], dtype=float)
        sc133 = np.asarray(pred.keypoint_scores[0], dtype=float)
        pts134, sc134 = convert_coco133_person(kpts133, sc133, order=order)
        people.append(
            {
                "box_xyxy": [float(v) for v in bbox.tolist()],
                "box_score": float(bscore),
                "points": pts134,
                "scores": sc134,
            }
        )
    return people


def detect_people(
    image_bgr: np.ndarray,
    runtime: SDPoseRuntime | None = None,
    *,
    conf: float = DEFAULT_YOLO_CONF,
    iou: float = DEFAULT_YOLO_IOU,
    min_area: float = 0.0,
    sort_by: str = "x_center",
    order: str = "geometry",
    flip_test: bool = True,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], int]:
    """Detect people with YOLO11x and estimate 134-slot keypoints via SDPose-Wholebody.

    Args:
        image_bgr: Input image in BGR ``uint8`` layout.
        runtime: Optional cached :class:`SDPoseRuntime`; loaded on demand when omitted.
        conf: YOLO11x person confidence threshold (default ``0.25``).
        iou: YOLO11x NMS IoU threshold (default ``0.65``).
        min_area: Minimum bounding box area (px^2) to filter tiny background pedestrians.
        sort_by: ``"x_center"`` (left-to-right), ``"score"``, or ``"area"``.
        order: ``"geometry"`` (repo 134-slot convention) or ``"openpose"``.
        flip_test: Horizontal flip TTA in SDPose inference (default ``True``).
        return_stats: When ``True``, returns ``(people, raw_yolo_count)``.
    """
    rt = runtime if runtime is not None else load_runtime()
    boxes, scores, raw_count = detect_boxes(image_bgr, rt, conf=conf, iou=iou, min_area=min_area)
    if sort_by == "x_center" and len(boxes) > 0:
        perm = sorted(range(len(boxes)), key=lambda i: 0.5 * (float(boxes[i, 0]) + float(boxes[i, 2])))
    elif sort_by == "score" and len(boxes) > 0:
        perm = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), float(boxes[i, 0])))
    elif sort_by == "area" and len(boxes) > 0:
        perm = sorted(
            range(len(boxes)),
            key=lambda i: -float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])),
        )
    else:
        perm = list(range(len(boxes)))

    boxes = boxes[perm] if len(boxes) else boxes
    scores = scores[perm] if len(scores) else scores
    people = pose_on_boxes(image_bgr, boxes, rt, box_scores=scores, order=order, flip_test=flip_test)
    if return_stats:
        return people, raw_count
    return people

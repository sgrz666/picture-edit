"""内容区 flow 监督与固定几何区域监督，不经过检测器或 ONNX。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch.nn import functional as F

from .contracts import ProAdapterConfig, TaskCondition


@dataclass(frozen=True)
class RegionalLossConfig:
    lambda_global: float = 1.
    lambda_face: float = 1.
    lambda_contact: float = 1.
    lambda_hand: float = 1.
    lambda_body: float = 1.

    def __post_init__(self):
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Loss weights must be finite nonnegative numbers")

    def to_dict(self):
        return asdict(self)


def _valid_content(valid, shape, device):
    if (not isinstance(valid, torch.Tensor) or valid.shape != shape or not torch.isfinite(valid).all()
            or (valid < 0).any() or (valid > 1).any()):
        raise ValueError("valid_target_tokens must be finite [B,N] values in [0,1]")
    return valid.detach().to(device=device, dtype=torch.float32)


def regional_masks(condition: TaskCondition, valid_target_tokens, *, config: ProAdapterConfig | None = None):
    """按 face/contact/hand/body 分配 support，软边界只计算一次。

    owner 已经受 person_mask/visibility 限制；用 minimum 做集合交集，避免
    重乘 soft support。低优先级只获得尚未分配的面积，四者总和等于可见内容区。
    ROI 只来自 typed geometry，不读取 target RGB 或 target appearance。
    """
    condition.validate(config)
    support = condition.visible_owner().detach().float()
    valid = _valid_content(valid_target_tokens, support.shape[::2], support.device)
    enabled = (condition.person_valid & condition.spatial_valid["mask"] & condition.spatial_valid["visibility"])
    support = torch.minimum(support, valid[:, None]) * enabled[..., None]
    candidates = {
        "face": condition.face_mask.detach().float(),
        "contact": condition.spatial["contact"].detach().float().flatten(2)
            * condition.spatial_valid["contact"][..., None],
        "hand": condition.hand_mask.detach().float().amax(dim=2),
        "body": support,
    }
    remaining, masks = support.clone(), {}
    for name, candidate in candidates.items():
        allocated = torch.minimum(candidate, remaining).clamp_min(0)
        masks[name] = allocated.sum(dim=1)
        remaining = (remaining - allocated).clamp_min(0)
    return masks


def regional_flow_loss(prediction, flow_target, valid_target_tokens, condition, *,
                       config: RegionalLossConfig | None = None, adapter_config=None):
    config = config or RegionalLossConfig()
    if (prediction.ndim != 3 or prediction.shape != flow_target.shape or prediction.shape[-1] < 1
            or not torch.isfinite(prediction).all() or not torch.isfinite(flow_target).all()):
        raise ValueError("Flow predictions and targets must have matching finite [B,N,C] shape")
    valid = _valid_content(valid_target_tokens, prediction.shape[:2], prediction.device)
    if valid.sum() <= 0:
        raise ValueError("Target must contain positive valid content")
    masks = regional_masks(condition, valid, config=adapter_config)
    square = (prediction.float() - flow_target.detach().float()).square()
    channels, area = prediction.shape[-1], valid.sum()
    global_loss = (square * valid[..., None]).sum() / (channels * area)
    total = global_loss * config.lambda_global
    metrics = {"losses": {"global": float(global_loss.detach())}, "area": {"global": float(area)},
               "coverage": {"global": 1.}}
    for name, mask in masks.items():
        mask = mask.to(prediction.device)
        region_area = mask.sum()
        # Keep zero-area loss connected to prediction, but never divide by zero.
        loss = (square * mask[..., None]).sum() / (channels * region_area) if region_area > 0 else square.sum() * 0
        total = total + getattr(config, f"lambda_{name}") * loss
        metrics["losses"][name] = float(loss.detach())
        metrics["area"][name] = float(region_area)
        metrics["coverage"][name] = float(region_area / area)
    return total, metrics


def fixed_geometry_feature_loss(encoder, generated, target, boxes, *, output_size=(112, 112)):
    """boxes 为事先冻结的 [K,5] (batch,x0,y0,x1,y1)，坐标归一化到 [0,1]。

    调用者提供 source/driver geometry 投影框；本函数没有目标检测/选框入口。
    encoder 冻结，生成图 grid_sample 和 encoder 的输入图仍参与 autograd。
    """
    if (generated.ndim != 4 or generated.shape != target.shape or not isinstance(boxes, torch.Tensor)
            or boxes.ndim != 2 or boxes.shape[-1] != 5 or boxes.requires_grad
            or not torch.isfinite(boxes).all()):
        raise ValueError("Expected image pairs and fixed geometry boxes [K,5]")
    if len(output_size) != 2 or any(type(v) is not int or v <= 0 for v in output_size):
        raise ValueError("output_size must be positive (H,W)")
    boxes = boxes.detach().to(generated.device, torch.float32)
    batch = boxes[:, 0]
    if (batch != batch.long()).any() or (batch < 0).any() or (batch >= generated.shape[0]).any():
        raise ValueError("fixed geometry batch indices are invalid")
    coordinates = boxes[:, 1:]
    if ((coordinates < 0) | (coordinates > 1)).any() or (coordinates[:, 2:] <= coordinates[:, :2]).any():
        raise ValueError("fixed geometry boxes must have positive area within [0,1]")
    if any(value.is_floating_point() and (value.dtype != torch.float32 or value.device != generated.device)
           for value in (*encoder.parameters(), *encoder.buffers())):
        raise ValueError("Feature encoder requires FP32 storage on the generated image device")
    encoder.requires_grad_(False).eval()
    if boxes.shape[0] == 0:
        return generated.sum() * 0
    y, x = torch.meshgrid(torch.linspace(0, 1, output_size[0], device=generated.device),
                          torch.linspace(0, 1, output_size[1], device=generated.device), indexing="ij")
    x0, y0, x1, y1 = (boxes[:, index, None, None] for index in range(1, 5))
    grid = torch.stack([x0 + x * (x1 - x0), y0 + y * (y1 - y0)], dim=-1) * 2 - 1
    def crop(value):
        return F.grid_sample(value[batch.long()].float(), grid, mode="bilinear", align_corners=True)
    # FP32 storage and activations are explicit for this optional frozen encoder.
    with torch.autocast(device_type=generated.device.type, enabled=False):
        features = encoder(crop(generated))
        with torch.no_grad():
            reference = encoder(crop(target.detach()))
        if not isinstance(features, torch.Tensor) or features.shape != reference.shape:
            raise ValueError("Feature encoder must return matching tensors")
        return F.mse_loss(features.float(), reference.float())

"""Deterministic geometry in source-camera space, before target canvas sampling."""
from dataclasses import asdict, dataclass, replace
import math

import torch
from torch.nn import functional as F

from .depth_condition_v2 import DETAIL_CHANNELS, GATE_CHANNELS


@dataclass(frozen=True)
class DepthGeometryConfig:
    body_jump_threshold: float = .02
    relative_jump_threshold: float = .05
    epsilon: float = 1e-6

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Geometry thresholds/epsilon must be finite positive')


def full_frame_affine(asset_hw, original_hw, declared):
    """Only explicitly declared full-frame historical maps; not arbitrary crops."""
    ah, aw = asset_hw
    oh, ow = original_hw
    expected = torch.tensor([[ow / aw, 0., 0.], [0., oh / ah, 0.], [0., 0., 1.]], dtype=torch.float64)
    actual = torch.as_tensor(declared, dtype=torch.float64)
    if actual.shape != (3, 3) or not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError('asset_to_original must explicitly match this full-frame source grid')
    return actual.float()


def resize_native(value, hw, *, nearest=True):
    if tuple(value.shape[-2:]) == tuple(hw):
        return value
    return F.interpolate(value.float(), size=hw, mode='nearest' if nearest else 'bilinear',
                         align_corners=None if nearest else False)


def onto_canvas(value, spec, *, edge=False, valid=None):
    """Combined full-frame asset -> canvas; never scale the camera a second time."""
    if edge:
        resized = F.adaptive_max_pool2d(value.float(), spec.resized_hw)
    elif valid is not None:
        support = resize_native(valid.float(), spec.resized_hw, nearest=False)
        resized = resize_native(torch.where(valid, value, 0.), spec.resized_hw, nearest=False)
        resized = torch.where(support > 1e-8, resized / support.clamp_min(1e-8), 0.)
    else:
        resized = resize_native(value, spec.resized_hw)
    x, y = spec.offset_xy
    h, w = spec.resized_hw
    out = resized.new_zeros(*value.shape[:-2], *spec.canvas_hw)
    out[..., y:y+h, x:x+w] = resized
    return out


def camera_rays(k, hw, *, unit=True):
    h, w = hw
    yy, xx = torch.meshgrid(torch.arange(h, device=k.device, dtype=torch.float32) + .5,
                            torch.arange(w, device=k.device, dtype=torch.float32) + .5, indexing='ij')
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), 0).reshape(1, 3, -1).expand(k.shape[0], -1, -1)
    rays = (torch.linalg.inv(k.float()) @ pixels).reshape(k.shape[0], 3, h, w)
    return F.normalize(rays, dim=1, eps=1e-12) if unit else rays


def neighbours(value):
    p = F.pad(value, (1, 1, 1, 1))
    return (p[..., 1:-1, :-2], p[..., 1:-1, 2:], p[..., :-2, 1:-1], p[..., 2:, 1:-1])


def differences(value, valid, *, step_xy):
    left, right, up, down = neighbours(value)
    masks = neighbours(valid.float())
    stencil = valid & torch.stack([m > .5 for m in masks]).all(0)
    dx = (right - left) / (2 * step_xy[0])
    dy = (down - up) / (2 * step_xy[1])
    lap = (left + right - 2 * value) / step_xy[0]**2 + (up + down - 2 * value) / step_xy[1]**2
    jump = torch.stack([torch.where(valid & (m > .5), (n - value).abs(), 0.)
                        for n, m in zip((left, right, up, down), masks)]).amax(0)
    return *(torch.where(stencil, v, 0.) for v in (dx, dy, lap)), jump, stencil


def normal_relation(z, geometry_valid, context, k_asset, body_jump, threshold):
    points = camera_rays(k_asset, z.shape[-2:], unit=False) * z
    left, right, up, down = neighbours(points)
    normal = torch.cross(right - left, down - up, dim=1)
    length = torch.linalg.vector_norm(normal, dim=1, keepdim=True)
    normal = F.normalize(normal, dim=1, eps=1e-12)
    normal = torch.where((normal * points).sum(1, keepdim=True) > 0, -normal, normal)
    rendered = context.normal_cam
    rendered = torch.where((rendered * points).sum(1, keepdim=True) > 0, -rendered, rendered)
    valid = geometry_valid & context.normal_valid & (body_jump <= threshold) & (length > 1e-12)
    valid &= torch.stack([x > .5 for x in neighbours(geometry_valid.float())]).all(0)
    # Where Part is known, do not compare across semantic boundaries.
    labels = context.part_onehot.argmax(1, keepdim=True).float()
    for label, known in zip(neighbours(labels), neighbours(context.part_valid.float())):
        valid &= ~context.part_valid | ((known > .5) & (label == labels))
    cosine = (normal * rendered).sum(1, keepdim=True).clamp(-1., 1.)
    return torch.where(valid, cosine, 0.), valid


def source_features(relative, valid, camera_z, context, confidence, confidence_available, config, *, asset_to_original, original_hw):
    """23 channels on native Depth grid. Unknown physical channels are explicitly zero."""
    b, _, h, w = relative.shape
    device = relative.device
    step = (float(asset_to_original[0, 0]) / original_hw[1], float(asset_to_original[1, 1]) / original_hw[0])
    rdx, rdy, rlap, rjump, _ = differences(relative, valid, step_xy=step)
    sil = F.max_pool2d(valid.float(), 3, 1, 1) - (1 - F.max_pool2d(1 - F.pad(valid.float(), (1, 1, 1, 1)), 3, 1))
    geometry_valid = valid & context.smpl_mask & context.camera_available[:, None, None, None]
    if camera_z is None:
        geometry_valid = torch.zeros_like(valid)
        z = torch.zeros_like(relative)
    else:
        z = torch.where(geometry_valid, camera_z, 0.)
    height = context.body_height.clamp_min(config.epsilon)[:, None, None, None]
    body_z = torch.where(geometry_valid, (z - context.root_z[:, None, None, None]) / height, 0.)
    zdx, zdy, zlap, zjump, _ = differences(body_z, geometry_valid, step_xy=step)
    cosine, relation_valid = normal_relation(z, geometry_valid, context, context.intrinsics, zjump, config.body_jump_threshold)
    rays = torch.where(geometry_valid.expand(-1, 3, -1, -1), camera_rays(context.intrinsics, (h, w)), 0.)
    root = torch.log1p(context.root_z.clamp_min(0) / context.body_height.clamp_min(config.epsilon))[:, None, None, None]
    root = torch.where(geometry_valid, root, 0.)
    metric = geometry_valid & context.metric_available[:, None, None, None]
    absolute_height = torch.where(metric, context.body_height.clamp_min(config.epsilon).log()[:, None, None, None], 0.)
    cp = confidence_available[:, None, None, None].expand_as(valid) & valid
    parts = (torch.asinh(rdx), torch.asinh(rdy), torch.asinh(rlap), torch.asinh(rjump / config.relative_jump_threshold), sil,
             torch.asinh(body_z), torch.asinh(zdx), torch.asinh(zdy), torch.asinh(zlap), torch.asinh(zjump / config.body_jump_threshold),
             rays, cosine, relation_valid.float(), valid.float(), geometry_valid.float(),
             torch.where(valid, confidence, 0.), cp.float(), root, absolute_height, metric.float(), geometry_valid.float())
    result = torch.cat(parts, dim=1).float()
    if result.shape != (b, len(DETAIL_CHANNELS), h, w) or not torch.isfinite(result).all():
        raise ValueError('Invalid geometry feature layout/numerics')
    return result, geometry_valid


def context_to_canvas(context, spec):
    transform = spec.transform.to(context.intrinsics).expand(context.intrinsics.shape[0], -1, -1)
    # context.intrinsics at this point is K_original, not K_asset.
    return replace(context, intrinsics=transform @ context.intrinsics,
                   normal_cam=onto_canvas(context.normal_cam, spec),
                   normal_valid=onto_canvas(context.normal_valid.float(), spec) > .5,
                   part_onehot=onto_canvas(context.part_onehot, spec),
                   part_valid=onto_canvas(context.part_valid.float(), spec) > .5,
                   smpl_mask=onto_canvas(context.smpl_mask.float(), spec) > .5)


def gate_context(condition):
    """29 channels at 1/16: coverage stays separate from vector/score averages."""
    c = condition.context
    def pool(x):
        return F.avg_pool2d(x.float(), 16)
    def average(value, valid):
        return pool(torch.where(valid, value, 0.)) / pool(valid.float()).clamp_min(1e-6)
    normal = average(c.normal_cam, c.normal_valid)
    normal = F.normalize(normal, dim=1, eps=1e-6) * (pool(c.normal_valid) > 0)
    detail = condition.detail_features
    relation_valid = detail[:, 14:15] > .5
    b, _, h, w = condition.relative.shape
    cp = condition.confidence_available[:, None, None, None].expand(b, 1, h, w)
    metric = c.metric_available[:, None, None, None].expand(b, 1, h, w)
    jump = torch.where(condition.geometry_valid, detail[:, 9:10], detail[:, 3:4])
    values = (pool(c.part_onehot), normal, pool(condition.valid), pool(c.smpl_mask),
              pool(condition.geometry_valid), pool(c.normal_valid), pool(c.part_valid),
              average(detail[:, 13:14], relation_valid), pool(relation_valid),
              average(condition.confidence, condition.valid), pool(cp),
              F.max_pool2d(detail[:, 4:5], 16), F.max_pool2d(jump, 16), pool(metric))
    result = torch.cat(values, dim=1)
    if result.shape[1] != len(GATE_CHANNELS) or not torch.isfinite(result).all():
        raise ValueError('Gate context must be finite 29 channels')
    return result

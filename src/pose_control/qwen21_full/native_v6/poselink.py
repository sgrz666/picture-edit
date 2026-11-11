"""V6 外挂 Pose Attention：保持原生 Qwen Attention 和 reference Hidden 不变。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contracts import PoseGeometry, PoseLinkConfig
from .pose_layout import PoseLayout


def _grid(hw: tuple[int, int], batch: int, like: torch.Tensor) -> torch.Tensor:
    h, w = hw
    y = (torch.arange(h, device=like.device, dtype=like.dtype) + .5) / h
    x = (torch.arange(w, device=like.device, dtype=like.dtype) + .5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, h * w, 2).expand(batch, -1, -1)


def _map(value: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return value.transpose(1, 2).reshape(value.shape[0], value.shape[-1], *hw)


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(2).transpose(1, 2)


def _resize_coverage(coverage: torch.Tensor, hw, output_hw) -> torch.Tensor:
    return _flat(F.interpolate(_map(coverage[..., None], hw), size=output_hw,
                               mode="bilinear", align_corners=False))[..., 0]


def _align(value: torch.Tensor, coverage: torch.Tensor, hw, output_hw) -> torch.Tensor:
    """只插值有效区域并归一化，padding 的值和梯度都不进入邻域。"""
    weight = coverage[..., None].to(value.dtype)
    numerator = torch.where(weight > 0, value, 0) * weight
    numerator = F.interpolate(_map(numerator, hw), size=output_hw, mode="bilinear", align_corners=False)
    denominator = F.interpolate(_map(weight, hw), size=output_hw, mode="bilinear", align_corners=False)
    return _flat(numerator / denominator.clamp_min(torch.finfo(value.dtype).eps))


def _budget_grid(hw: tuple[int, int], budget: int) -> tuple[int, int]:
    """保留矩形长宽比，在每图 budget 内做二维池化。"""
    h, w = hw
    if h * w <= budget:
        return hw
    # Search over the shorter dimension; maximize resolution then minimize aspect distortion.
    candidates = [(oh, min(w, budget // oh)) for oh in range(1, min(h, budget) + 1)]
    return min(candidates, key=lambda shape: (abs(math.log((shape[0] / shape[1]) / (h / w))),
                                               -shape[0] * shape[1]))


def _pool(value: torch.Tensor, coverage: torch.Tensor, hw, output_hw) -> torch.Tensor:
    weight = coverage[..., None].to(value.dtype)
    numerator = torch.where(weight > 0, value, 0) * weight
    numerator = F.adaptive_avg_pool2d(_map(numerator, hw), output_hw)
    denominator = F.adaptive_avg_pool2d(_map(weight, hw), output_hw)
    return _flat(numerator / denominator.clamp_min(torch.finfo(value.dtype).eps))


class PoseLinkAttention(nn.Module):
    """Target Q 对 Pose-only K/V 的共享多头 Attention。

    Pose Hidden 可能已包含原生 Attention 的 Source 上下文。本模块保证索引排除
    Source 和文本，不声称从上下文化 Hidden 中消除了外观信息。每次 forward
    重建 K/V；没有跨 layer/step 的可变缓存。
    """
    def __init__(self, config: PoseLinkConfig):
        super().__init__()
        if not isinstance(config, PoseLinkConfig):
            raise TypeError("PoseLinkAttention requires PoseLinkConfig")
        self.config = config
        width = config.width
        self.query_norm = nn.LayerNorm(config.hidden_dim)
        self.query_projection = nn.Linear(config.hidden_dim, width)
        self.pose_norm = nn.LayerNorm(config.hidden_dim)
        self.pose_projection = nn.Linear(config.hidden_dim, width)
        self.geometry_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width))
        self.person_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width))
        self.interaction_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, width))
        self.role_embedding = nn.Embedding(3, width)
        self.layer_embedding = nn.Embedding(max(config.injection_layers) + 1, width)
        self.time_projection = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.log_spatial_scales = nn.Parameter(torch.zeros(config.heads))
        self.log_role_scales = nn.Parameter(torch.full((config.heads,), math.log(2.)))
        self.output_heads = nn.ModuleDict({str(layer): nn.Linear(width, config.hidden_dim)
                                           for layer in config.injection_layers})
        self.layer_log_gains = nn.ParameterDict({str(layer): nn.Parameter(torch.zeros(()))
                                                for layer in config.injection_layers})
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _pose_keys(self, joint: torch.Tensor, layout: PoseLayout, geometry: PoseGeometry):
        b = joint.shape[0]
        hw = geometry.target_token_hw
        coverage = geometry.coverage.to(joint.dtype)
        person_valid = geometry.person_valid.to(joint.dtype)
        sequences, xy, owners, contacts, coverages, validity = [], [], [], [], [], []
        for ref in layout.references:
            ref_hw = ref.token_hw
            pool_hw = _budget_grid(ref_hw, self.config.max_pose_tokens_per_reference)
            ref_coverage = _resize_coverage(coverage, hw, ref_hw)
            pose = joint.index_select(1, ref.indices.to(joint.device))
            pose = _pool(pose, ref_coverage, ref_hw, pool_hw)
            keys = self.pose_projection(self.pose_norm(pose))
            role = 0 if ref.role in ("combined_pose", "target_pose") else (1 if ref.role == "person_0_pose" else 2)
            valid = (person_valid.any(-1) if role == 0 else geometry.person_valid[:, role - 1])
            key_coverage = _flat(F.adaptive_avg_pool2d(_map(ref_coverage[..., None], ref_hw), pool_hw))[..., 0]
            key_valid = (key_coverage > 0) & valid[:, None]
            key_xy = _pool(_grid(ref_hw, b, joint), ref_coverage, ref_hw, pool_hw)

            def aligned(value):
                value = _align(value.to(joint.dtype), coverage, hw, ref_hw)
                return _pool(value, ref_coverage, ref_hw, pool_hw)

            if self.config.use_geometry:
                keys = keys + self.geometry_projection(aligned(geometry.geometry))
                # Projected support deliberately stays independent of visible owner.
                support = aligned(geometry.projected) * person_valid[:, None]
                person_values = []
                for person in range(2):
                    person_values.append(self.person_projection(aligned(geometry.person_tokens[:, person])))
                if role == 0:
                    tokens = torch.stack(person_values, dim=2)
                    keys = keys + (tokens * support[..., None]).sum(2) / support.sum(-1, keepdim=True).clamp_min(1.)
                else:
                    keys = keys + person_values[role - 1] * support[..., role - 1, None]
            if self.config.use_interaction and geometry.interaction is not None:
                keys = keys + self.interaction_projection(aligned(geometry.interaction))
            if self.config.use_role:
                keys = keys + self.role_embedding.weight[role]

            key_owner = aligned(geometry.owner) * person_valid[:, None]
            if role:
                key_owner = torch.zeros_like(key_owner)
                key_owner[..., role - 1] = person_valid[:, None, role - 1]
            key_contact = aligned(geometry.contact) * person_valid[:, None]
            if role:
                role_mask = F.one_hot(torch.tensor(role - 1, device=joint.device), 2).to(joint.dtype)
                key_contact = key_contact * role_mask
            sequences.append(keys)
            xy.append(key_xy)
            owners.append(key_owner)
            contacts.append(key_contact)
            coverages.append(key_coverage)
            validity.append(key_valid)
        return tuple(torch.cat(values, dim=1) for values in (sequences, xy, owners, contacts, coverages, validity))

    def _attention_bias(self, query_xy, key_xy, query_owner, key_owner, query_contact, key_contact):
        b, q = query_xy.shape[:2]
        k = key_xy.shape[1]
        bias = query_xy.new_zeros(b, self.config.heads, q, k)
        if self.config.use_spatial:
            distance = (query_xy[:, :, None] - key_xy[:, None]).square().sum(-1)
            bias = bias - self.log_spatial_scales.exp()[None, :, None, None] * distance[:, None]
        if self.config.use_role:
            cross = (query_owner[..., 0, None] * key_owner[:, None, :, 1]
                     + query_owner[..., 1, None] * key_owner[:, None, :, 0])
            if self.config.use_contact:
                # Both local endpoints require positive evidence; one contact elsewhere cannot relax this pair.
                relaxation = (query_contact[..., 0, None] * key_contact[:, None, :, 1]
                              + query_contact[..., 1, None] * key_contact[:, None, :, 0]).clamp(0, 1)
                cross = cross * (1 - relaxation)
            bias = bias - self.log_role_scales.exp()[None, :, None, None] * cross[:, None]
        return bias

    def _attend_chunk(self, query, key, value, query_xy, key_xy, query_owner, key_owner,
                      query_contact, key_contact, key_coverage, key_valid):
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        logits = logits + self._attention_bias(query_xy, key_xy, query_owner, key_owner,
                                               query_contact, key_contact).to(logits.dtype)
        logits = logits + key_coverage.clamp_min(torch.finfo(logits.dtype).tiny).log()[:, None, None]
        logits = logits.masked_fill(~key_valid[:, None, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        weights = weights * key_valid[:, None, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.)
        return torch.matmul(weights, value).transpose(1, 2).flatten(2)

    def forward(self, target_hidden: torch.Tensor, joint_hidden: torch.Tensor, layout: PoseLayout,
                geometry: PoseGeometry, *, layer: int, sigma: torch.Tensor, strength=1.) -> torch.Tensor:
        geometry.validate()
        if not isinstance(layout, PoseLayout):
            raise TypeError("PoseLink requires a native PoseLayout")
        layout.validate()
        b, n = geometry.geometry.shape[:2]
        if (target_hidden.shape != (b, n, self.config.hidden_dim)
                or joint_hidden.shape != (b, layout.joint_length, self.config.hidden_dim)
                or geometry.target_token_hw != layout.target_token_hw
                or layout.target_indices.numel() != n):
            raise ValueError("Target/Joint Hidden, native layout and geometry grids differ")
        if target_hidden.device != joint_hidden.device or geometry.geometry.device != target_hidden.device:
            raise ValueError("Hidden and geometry must share a device")
        if (not isinstance(sigma, torch.Tensor) or sigma.shape != (b,)
                or not sigma.is_floating_point() or not bool(torch.isfinite(sigma).all())
                or bool(((sigma < 0) | (sigma > 1)).any())):
            raise ValueError("sigma requires floating [B] in [0,1]")
        if isinstance(strength, torch.Tensor):
            if (strength.ndim not in (0, 1) or (strength.ndim == 1 and strength.shape != (b,))
                    or not bool(torch.isfinite(strength).all()) or bool((strength < 0).any())):
                raise ValueError("strength requires a finite nonnegative scalar or [B]")
        elif type(strength) not in (float, int) or not math.isfinite(strength) or strength < 0:
            raise ValueError("strength requires a finite nonnegative scalar or [B]")
        if type(layer) is not int:
            raise ValueError("layer requires an integer index")
        if layer not in self.config.injection_layers:
            return target_hidden.new_zeros(target_hidden.shape)
        dtype = self.query_projection.weight.dtype
        target = target_hidden.to(dtype)
        joint = joint_hidden.to(dtype)
        keys, key_xy, key_owner, key_contact, key_coverage, key_valid = self._pose_keys(joint, layout, geometry)
        conditioning = self.layer_embedding.weight[layer] + self.time_projection(sigma.to(device=target.device, dtype=dtype)[:, None])
        query = self.query_projection(self.query_norm(target)) + conditioning[:, None]
        keys = keys + conditioning[:, None]
        heads, head_dim = self.config.heads, self.config.width // self.config.heads
        query = query.reshape(b, n, heads, head_dim).transpose(1, 2)
        key = self.key_projection(keys).reshape(b, -1, heads, head_dim).transpose(1, 2)
        value = self.value_projection(keys).reshape(b, -1, heads, head_dim).transpose(1, 2)
        query_xy = _grid(geometry.target_token_hw, b, target)
        active = geometry.person_valid[:, None].to(dtype)
        query_owner = geometry.owner.to(dtype) * active
        query_contact = geometry.contact.to(dtype) * active
        chunks = []
        for start in range(0, n, self.config.query_chunk_size):
            stop = start + self.config.query_chunk_size
            args = (query[:, :, start:stop], key, value, query_xy[:, start:stop], key_xy,
                    query_owner[:, start:stop], key_owner, query_contact[:, start:stop], key_contact,
                    key_coverage, key_valid)
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self._attend_chunk, *args, use_reentrant=False))
            else:
                chunks.append(self._attend_chunk(*args))
        residual = self.output_heads[str(layer)](torch.cat(chunks, dim=1))
        gain = self.layer_log_gains[str(layer)].exp() * torch.as_tensor(strength, device=target.device, dtype=dtype)
        if gain.ndim:
            gain = gain[:, None, None]
        return residual * gain * geometry.coverage.to(dtype)[..., None] * key_valid.any(-1)[:, None, None]

"""V6.1 PoseLink keeps native Attention intact and bounds only its own residual."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..native_v6.poselink import PoseLinkAttention, _grid
from .contracts import PoseLinkConfigV61, PoseSpatialConditionV61
from .tokens import sample_pose_keys


class PoseLinkAttentionV61(PoseLinkAttention):
    def __init__(self, config):
        if not isinstance(config, PoseLinkConfigV61):
            raise ValueError('Explicit V6.1 PoseLink configuration required')
        super().__init__(config)
        self.token_type_embedding = nn.Embedding(3, config.width)
        nn.init.zeros_(self.token_type_embedding.weight)

    def build_pose_keys(self, joint_hidden, layout, geometry, spatial):
        spatial.validate(geometry)
        layout.validate()
        return sample_pose_keys(joint_hidden, layout, geometry, spatial, self.config)

    def forward(self, target_hidden, joint_hidden, layout, geometry, *, layer, sigma, strength=1., spatial=None):
        cfg = self.config
        if not (cfg.geometry_key_mask or cfg.edit_region_gate or cfg.multiscale_tokens):
            return super().forward(target_hidden, joint_hidden, layout, geometry, layer=layer, sigma=sigma, strength=strength)
        if not isinstance(spatial, PoseSpatialConditionV61):
            raise ValueError('V6.1 active spatial features require prepared spatial conditions')
        # Reuse V6's complete input validation without changing state or sampling.
        geometry.validate(); spatial.validate(geometry); layout.validate()
        if not (cfg.geometry_key_mask or cfg.multiscale_tokens):
            # Gate-only ablation changes no K/V construction or geometry algebra.
            result = super().forward(target_hidden, joint_hidden, layout, geometry,
                                     layer=layer, sigma=sigma, strength=strength)
            return result * spatial.edit_gate[..., None].to(result.dtype)
        b, n = geometry.geometry.shape[:2]
        if target_hidden.shape != (b, n, cfg.hidden_dim) or joint_hidden.shape != (b, layout.joint_length, cfg.hidden_dim):
            raise ValueError('V6.1 target/joint Hidden shapes differ from layout')
        if geometry.target_token_hw != layout.target_token_hw or layout.target_indices.numel() != n:
            raise ValueError('V6.1 Target grid and geometry differ')
        if target_hidden.device != joint_hidden.device or geometry.geometry.device != target_hidden.device:
            raise ValueError('V6.1 Hidden and geometry must share a device')
        if (not torch.is_tensor(sigma) or sigma.shape != (b,) or not sigma.is_floating_point()
                or not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any()):
            raise ValueError('V6.1 sigma must be finite floating [B] in [0,1]')
        if not torch.is_tensor(strength) and type(strength) not in (int, float):
            raise ValueError('V6.1 strength must be a finite nonnegative scalar or [B]')
        gain = torch.as_tensor(strength, device=target_hidden.device)
        if gain.ndim > 1 or (gain.ndim == 1 and gain.shape != (b,)) or not torch.isfinite(gain).all() or (gain < 0).any():
            raise ValueError('V6.1 strength must be finite nonnegative scalar or [B]')
        if type(layer) is not int: raise ValueError('Layer requires an integer')
        if layer not in cfg.injection_layers: return target_hidden.new_zeros(target_hidden.shape)
        dtype = self.query_projection.weight.dtype
        target, joint = target_hidden.to(dtype), joint_hidden.to(dtype)
        samples = self.build_pose_keys(joint, layout, geometry, spatial)
        keys = self.pose_projection(self.pose_norm(samples.hidden))
        if cfg.use_geometry:
            persons = self.person_projection(samples.person_tokens)
            keys = keys + self.geometry_projection(samples.geometry) + (persons*samples.person_weights[..., None]).sum(2)
        if cfg.use_interaction and samples.interaction is not None:
            keys = keys + self.interaction_projection(samples.interaction)
        if cfg.use_role:
            role = (samples.person_ids + 1).clamp_min(0)
            keys = keys + self.role_embedding(role)
        if cfg.multiscale_tokens:
            keys = keys + self.token_type_embedding(samples.type_ids)
        condition = self.layer_embedding.weight[layer] + self.time_projection(sigma.to(target)[:, None])
        query = self.query_projection(self.query_norm(target)) + condition[:, None]
        keys = keys + condition[:, None]
        heads, dim = cfg.heads, cfg.width // cfg.heads
        query = query.reshape(b, n, heads, dim).transpose(1, 2)
        key = self.key_projection(keys).reshape(b, -1, heads, dim).transpose(1, 2)
        value = self.value_projection(keys).reshape(b, -1, heads, dim).transpose(1, 2)
        xy = _grid(geometry.target_token_hw, b, target)
        active = geometry.person_valid[:, None].to(dtype)
        owner, contact = geometry.owner.to(dtype)*active, geometry.contact.to(dtype)*active
        chunks = []
        for start in range(0, n, cfg.query_chunk_size):
            stop = start + cfg.query_chunk_size
            args = (query[:, :, start:stop], key, value, xy[:, start:stop], samples.xy,
                    owner[:, start:stop], samples.owner, contact[:, start:stop], samples.contact,
                    samples.weight, samples.valid)
            chunks.append(checkpoint(self._attend_chunk, *args, use_reentrant=False)
                if self.training and torch.is_grad_enabled() else self._attend_chunk(*args))
        delta = self.output_heads[str(layer)](torch.cat(chunks, dim=1))
        gain = gain.to(dtype) * self.layer_log_gains[str(layer)].exp()
        if gain.ndim: gain = gain[:, None, None]
        result = delta * gain * geometry.coverage.to(dtype)[..., None] * samples.valid.any(-1)[:, None, None]
        return result * spatial.edit_gate[..., None].to(dtype) if cfg.edit_region_gate else result

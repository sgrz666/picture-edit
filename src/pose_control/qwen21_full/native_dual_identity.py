"""Small native-latent identity binder; no target RGB, new backbone or Qwen tokens."""
import torch
from torch import nn
from torch.nn import functional as F

from .dual_depth_condition import NativeDualIdentityCondition


class ContextDelta(nn.Module):
    """Return an attention update, never a second copy of the query residual."""
    def __init__(self, width=512, heads=8):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*4), nn.SiLU(), nn.Linear(width*4, width))
        self.heads = heads

    def forward(self, query, context, valid, query_valid, bias=None):
        active = valid.any(1) & query_valid.any(1)
        ids = active.nonzero(as_tuple=True)[0]
        out = torch.zeros_like(query)
        if ids.numel():
            q = query[ids]
            c = self.k_norm(context[ids])
            mask = q.new_zeros(q.shape[0], q.shape[1], c.shape[1]) if bias is None else bias[ids].clone()
            mask.masked_fill_(~valid[ids, None], -torch.inf)
            mask = mask[:, None].expand(-1, self.heads, -1, -1).reshape(-1, q.shape[1], c.shape[1])
            update = self.attention(self.q_norm(q), c, c, attn_mask=mask, need_weights=False)[0]
            update = (update + self.ffn(update)) * query_valid[ids, :, None]
            out = out.index_copy(0, ids, update)
        return out


def _roi(latent, mask, size):
    """Masked bilinear ROI with support normalization; no pooled other-person pixels."""
    support = F.interpolate(mask.float(), size=latent.shape[-2:], mode='area')
    values, validity = [], []
    h, w = latent.shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(size[0], device=latent.device)+.5)/size[0],
                            (torch.arange(size[1], device=latent.device)+.5)/size[1], indexing='ij')
    for i in range(latent.shape[0]):
        ys, xs = (support[i, 0] > 0).nonzero(as_tuple=True)
        if not xs.numel():
            values.append(latent.new_zeros(1, 64, *size))
            validity.append(torch.zeros(1, *size, dtype=torch.bool, device=latent.device))
            continue
        x = (xs.min() + xx * (xs.max()+1-xs.min())) * (2/w) - 1
        y = (ys.min() + yy * (ys.max()+1-ys.min())) * (2/h) - 1
        grid = torch.stack((x, y), -1)[None]
        denominator = F.grid_sample(support[i:i+1], grid, align_corners=False)
        numerator = F.grid_sample(latent[i:i+1] * support[i:i+1], grid, align_corners=False)
        values.append(numerator / denominator.clamp_min(1e-6))
        validity.append((denominator[:, 0] > 1e-6))
    return torch.cat(values).flatten(2).transpose(1, 2), torch.cat(validity).flatten(1)


class NativeDualIdentityBinder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))
        self.slot = nn.Embedding(2, 256)
        self.region = nn.Embedding(2, 256)
        self.position = nn.Linear(2, 256, bias=False)
        self.context_projection = nn.Linear(256, 512)
        self.query_binding = nn.Linear(256, 512, bias=False)
        self.cross_attention = ContextDelta()

    def forward(self, identity, person_tokens, person_mask, owner_weights):
        if not isinstance(identity, NativeDualIdentityCondition):
            raise ValueError('Enabled identity requires NativeDualIdentityCondition')
        identity.validate()
        b = person_tokens.shape[0]
        if person_tokens.ndim != 4 or person_tokens.shape[1] != 2 or person_tokens.shape[-1] != 512:
            raise ValueError('Identity query must be [B,2,N,512]')
        latent = identity.source_latent.float()
        updates = []
        for target_slot in range(2):
            source = identity.source_indices[:, target_slot]
            rows = torch.arange(b, device=latent.device)
            body, bv = _roi(latent, identity.person_mask[rows, source], (8, 4))
            head, hv = _roi(latent, identity.head_mask[rows, source], (4, 4))
            regions = []
            for region, tokens, size in ((0, body, (8, 4)), (1, head, (4, 4))):
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, size[0], device=latent.device),
                                        torch.linspace(-1, 1, size[1], device=latent.device), indexing='ij')
                position = torch.stack((xx, yy), -1).reshape(-1, 2)
                regions.append(self.encoder(tokens) + self.region.weight[region] + self.slot.weight[target_slot]
                               + self.position(position)[None])
            context = self.context_projection(torch.cat(regions, 1))
            valid = torch.cat((bv, hv), 1)
            query = person_tokens[:, target_slot] + self.query_binding(self.slot.weight[target_slot])[None, None]
            updates.append(self.cross_attention(query, context, valid, person_mask[:, target_slot]))
        weights = owner_weights.flatten(3).squeeze(2)[..., None]
        return (torch.stack(updates, 1) * weights).sum(1)

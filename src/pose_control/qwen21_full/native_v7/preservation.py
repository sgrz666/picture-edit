"""Explicit Source identity and latent-background objectives, without decoding."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PreservationConfig:
    source_identity: float = 0.
    background: float = 0.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative preservation value required: ' + name)
        if not (self.source_identity > 0 or self.background > 0):
            raise ValueError('At least one Source preservation objective must be enabled')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown Source preservation configuration fields')
        return cls(**raw)


def _finite_float(value, name, *, shape=None, device=None):
    import torch

    if (not torch.is_tensor(value) or not value.is_floating_point()
            or value.device.type == 'meta'
            or (shape is not None and value.shape != shape)
            or (device is not None and value.device != device)
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Invalid finite floating Source preservation tensor: ' + name)


def source_identity_loss(generated, source, valid):
    """Mean ``1-cos`` over valid [B,2,D] face slots; Source is a frozen reference.

    Returns a scalar loss and the integer number of valid faces, including
    singleton scenes. Mapping Source people into target slots is the caller's
    responsibility. Mixed floating dtypes are evaluated in FP32.
    """
    import torch
    from torch.nn import functional as F

    _finite_float(generated, 'generated')
    if generated.ndim != 3 or generated.shape[1] != 2 or any(size <= 0 for size in generated.shape):
        raise ValueError('Source identity requires nonempty [B,2,D] features')
    _finite_float(source, 'source', shape=generated.shape, device=generated.device)
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or valid.shape != generated.shape[:2] or valid.device != generated.device):
        raise ValueError('Source identity requires bool [B,2] validity on the feature device')
    count = int(valid.sum().item())
    if not count:
        return generated.float().sum() * 0., 0
    similarity = F.cosine_similarity(generated[valid].float(), source.detach()[valid].float(), dim=-1)
    return (1. - similarity).mean(), count


def source_background_loss(prediction, noisy, sigma, source_latent, support, coverage):
    """Constrain ``x0=noisy-sigma*prediction`` to the frozen Source background.

    Latents are [B,N,C], sigma is normalized [B], and support/coverage are [B,N]
    token fractions. Support already includes padding coverage: never multiply
    by coverage again. Normalize each supported image by its own support area
    and channels, then average supported images. Returns (loss, detached total
    support area); callers count images with ``support.sum(-1) > 0`` separately.
    """
    import torch

    _finite_float(prediction, 'prediction')
    if prediction.ndim != 3 or any(size <= 0 for size in prediction.shape):
        raise ValueError('Source background requires nonempty [B,N,C] latents')
    for name, value in (('noisy', noisy), ('source_latent', source_latent)):
        _finite_float(value, name, shape=prediction.shape, device=prediction.device)
    _finite_float(sigma, 'sigma', shape=prediction.shape[:1], device=prediction.device)
    if bool(((sigma < 0) | (sigma > 1)).any()):
        raise ValueError('Source background sigma must be normalized to [0,1]')
    for name, value in (('support', support), ('coverage', coverage)):
        _finite_float(value, name, shape=prediction.shape[:2], device=prediction.device)
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError('Source background token fractions must be in [0,1]: ' + name)
    if bool((support.detach() > coverage.detach()).any()):
        raise ValueError('Source background support cannot exceed coverage')
    weights = support.detach().float()
    areas = weights.sum(-1)
    area = areas.sum()
    eligible = areas > 0
    if not bool(eligible.any()):
        return prediction.float().sum() * 0., area
    x0 = noisy.float() - sigma.float()[:, None, None] * prediction.float()
    squared = (x0 - source_latent.detach().float()).square().mean(-1)
    per_image = (squared[eligible] * weights[eligible]).sum(-1) / areas[eligible]
    return per_image.mean(), area

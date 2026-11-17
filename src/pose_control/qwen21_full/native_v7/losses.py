"""Paper-inspired supervision built on the existing FM and GT-aligned face loss.

DanceTogether supplies the per-person weighting idea; WithAnyone supplies the
contrastive objective. GT correspondence/decoding remains the existing V5 path.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PaperLossConfig:
    person_fm: float = .5
    contact_fm: float = .5
    identity_contrastive: float = .1
    temperature: float = .07

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative paper loss value required: ' + name)
        if self.temperature <= 0:
            raise ValueError('Contrastive temperature must be positive')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown paper loss configuration fields')
        return cls(**raw)


def paper_region_losses(prediction, target, coverage, owner, contact):
    """Equal-person FM, intersecting already covered supports exactly once.

    owner/contact are target-token fractions [B,N,2]. Projected (occluded)
    support must never be passed as owner. Contact is intersected with owner.
    """
    import torch
    from ..native_v5.losses import region_velocity
    if prediction.ndim != 3 or target.shape != prediction.shape or coverage.shape != prediction.shape[:2]:
        raise ValueError('Region FM requires matching target tokens and coverage')
    shape = (*prediction.shape[:2], 2)
    for name, value in (('coverage', coverage), ('owner', owner), ('contact', contact)):
        expected = prediction.shape[:2] if name == 'coverage' else shape
        if (not torch.is_tensor(value) or value.shape != expected or value.device != prediction.device
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or ((value < 0) | (value > 1)).any()):
            raise ValueError('Invalid region FM support: ' + name)
    squared = (prediction.float() - target.float()).square()
    visible = torch.minimum(owner.detach().float(), coverage.detach().float()[..., None])
    touching = torch.minimum(contact.detach().float(), visible)
    result = {}
    for name, support in (('person', visible), ('contact', touching)):
        terms = [region_velocity(squared[row:row + 1], support[row:row + 1, :, person])
                 for row in range(prediction.shape[0]) for person in range(2)
                 if bool(support[row, :, person].sum() > 0)]
        result[name + '_fm'] = torch.stack(terms).mean() if terms else squared.sum() * 0
        result[name + '_count'] = len(terms)
        result[name + '_area'] = support.sum()
    return result


def identity_contrastive(generated, references, valid, *, temperature=.07):
    """Known slot correspondence; other valid people in this sample are negatives.

    This is an in-scene adaptation of WithAnyone, not its extended-negative bank.
    Single valid faces provide no negative and contribute differentiable zero.
    """
    import torch
    from torch.nn import functional as F
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Contrastive temperature must be finite and positive')
    if (generated.ndim != 3 or generated.shape != references.shape or generated.shape[1] != 2
            or valid.shape != generated.shape[:2] or valid.dtype != torch.bool
            or valid.device != generated.device or references.device != generated.device
            or not torch.isfinite(generated).all() or not torch.isfinite(references).all()):
        raise ValueError('Identity contrastive requires finite [B,2,D] and bool validity')
    terms, count = [], 0
    for row in range(generated.shape[0]):
        indices = valid[row].nonzero().flatten()
        if indices.numel() < 2:
            continue
        query = F.normalize(generated[row, indices].float(), dim=-1)
        reference = F.normalize(references[row, indices].detach().float(), dim=-1)
        logits = query @ reference.transpose(0, 1) / temperature
        terms.append(F.cross_entropy(logits, torch.arange(len(indices), device=logits.device), reduction='sum'))
        count += len(indices)
    return (torch.stack(terms).sum() / count if count else generated.sum() * 0), count


def aligned_identity_regularizer(temperature):
    """Callback for the existing one-decode GT-aligned auxiliary implementation."""
    def regularize(generated, references, slots):
        import torch
        batches = sorted({row for row, _ in slots})
        positions = {slot: i for i, slot in enumerate(slots)}
        terms, count = [], 0
        for row in batches:
            if (row, 0) not in positions or (row, 1) not in positions:
                continue
            indices = [positions[(row, person)] for person in range(2)]
            loss, eligible = identity_contrastive(generated[indices][None], references[indices][None],
                torch.ones(1, 2, dtype=torch.bool, device=generated.device), temperature=temperature)
            terms.append(loss * eligible)
            count += eligible
        loss = torch.stack(terms).sum() / count if count else generated.sum() * 0
        return {'identity_contrastive': loss, 'contrastive_eligible': count}
    return regularize

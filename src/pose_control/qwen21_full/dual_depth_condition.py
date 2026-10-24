"""Explicit dual-person records. Source appearance never belongs to geometry."""
from dataclasses import dataclass
import torch

from .depth_condition_v2 import DepthConditionV2, _TensorRecord, _tensor

CONDITION_SCHEMA = 'dual_condition_v3'
IDENTITY_SCHEMA = 'native_dual_identity_v3'
RELATION_CHANNELS = ('z_a', 'z_b', 'valid_a', 'valid_b', 'overlap', 'z_b_minus_a', 'front_a', 'front_b')
INTERACTION_CHANNELS = ('projected_a', 'projected_b', 'visible_a', 'visible_b', 'depth_a', 'depth_b',
                        'overlap', 'front_a', 'front_b', 'tie', 'camera_scale', 'distance_available')
FAMILIES = ('smplx', 'smpl')


@dataclass(frozen=True)
class DualContactRelations(_TensorRecord):
    # Directed raw records; 0=official correspondence, 1=inferred, 2=explicit avoid.
    indices: torch.Tensor                 # [B,K,4]: person, vertex, person, vertex
    kind: torch.Tensor                    # [B,K]
    distance: torch.Tensor                # [B,K], normalized by scene height
    valid: torch.Tensor                   # [B,K]
    metadata: tuple[dict, ...]

    def validate(self):
        b, k = self.valid.shape
        d = self.valid.device
        if self.indices.shape != (b, k, 4) or self.indices.dtype != torch.long or self.indices.device != d:
            raise ValueError('Contact indices must be int64 [B,K,4]')
        if self.kind.shape != (b, k) or self.kind.dtype != torch.long or self.kind.device != d:
            raise ValueError('Contact kind must be int64 [B,K]')
        _tensor(self.valid, (b, k), d, 'contact valid', boolean=True)
        _tensor(self.distance, (b, k), d, 'contact distance')
        rows = self.indices[self.valid]
        if (rows < 0).any() or ((rows[:, (0, 2)] > 1).any()) or (rows[:, 0] == rows[:, 2]).any():
            raise ValueError('Contacts must link distinct A/B persons and nonnegative vertices')
        if ((self.kind[self.valid] < 0) | (self.kind[self.valid] > 2)).any() or (self.distance < 0).any():
            raise ValueError('Invalid contact source/distance')
        if len(self.metadata) != b:
            raise ValueError('Contact provenance batch differs')
        return self

    @classmethod
    def stack(cls, records):
        from dataclasses import replace
        from torch.nn import functional as F
        width = max(r.valid.shape[1] for r in records)
        padded = [replace(r, indices=F.pad(r.indices, (0, 0, 0, width-r.valid.shape[1])),
                          kind=F.pad(r.kind, (0, width-r.valid.shape[1])),
                          distance=F.pad(r.distance, (0, width-r.valid.shape[1])),
                          valid=F.pad(r.valid, (0, width-r.valid.shape[1]), value=False)) for r in records]
        return super().stack(padded)


@dataclass(frozen=True)
class DualDepthConditionV3(_TensorRecord):
    person_a: DepthConditionV2
    person_b: DepthConditionV2
    scene_depth: torch.Tensor
    scene_valid: torch.Tensor
    projected: torch.Tensor               # [B,2,1,H,W], independent full surfaces
    visible: torch.Tensor                 # ties may be visible in both slots
    owner_weights: torch.Tensor           # sum=1 on scene support; ties split equally
    relation_raster: torch.Tensor
    overlap: torch.Tensor
    front: torch.Tensor                   # [B,2,1,H,W], exclusive owner only
    tie: torch.Tensor
    part_distance: torch.Tensor
    part_distance_valid: torch.Tensor
    closest_points: torch.Tensor
    closest_points_valid: torch.Tensor
    part_contact: torch.Tensor            # [B,14,14,4]: positive/known/avoid/avoid_known
    contacts: DualContactRelations
    model_family: torch.Tensor            # [B,2], FAMILIES indices
    metadata: tuple[dict, ...]

    @property
    def person_camera_z(self):
        return torch.stack([torch.zeros_like(p.relative) if p.camera_z is None else p.camera_z
                            for p in (self.person_a, self.person_b)], 1)

    @property
    def person_depth_valid(self):
        return torch.stack([p.camera_z_valid for p in (self.person_a, self.person_b)], 1)

    @property
    def person_normal(self):
        return torch.stack([p.context.normal_cam for p in (self.person_a, self.person_b)], 1)

    @property
    def person_part(self):
        return torch.stack([p.context.part_onehot for p in (self.person_a, self.person_b)], 1)

    @property
    def body_height(self):
        return torch.stack([p.context.body_height for p in (self.person_a, self.person_b)], 1)

    @property
    def root_z(self):
        return torch.stack([p.context.root_z for p in (self.person_a, self.person_b)], 1)

    @property
    def camera_available(self):
        return self.person_a.context.camera_available & self.person_b.context.camera_available

    def validate(self, *, target_token_hw=None):
        for p in (self.person_a, self.person_b):
            p.validate(target_token_hw=target_token_hw)
        b, _, h, w = self.person_a.relative.shape
        d = self.scene_depth.device
        if self.person_b.relative.shape != (b, 1, h, w) or self.person_b.relative.device != d or self.person_a.relative.device != d:
            raise ValueError('A/B must share one exact canvas/device')
        for name, shape in (('scene_depth', (b, 1, h, w)), ('owner_weights', (b, 2, 1, h, w)),
                            ('relation_raster', (b, 8, h, w)), ('part_distance', (b, 14, 14)),
                            ('closest_points', (b, 14, 14, 2, 3)), ('part_contact', (b, 14, 14, 4))):
            _tensor(getattr(self, name), shape, d, name)
        for name, shape in (('scene_valid', (b, 1, h, w)), ('projected', (b, 2, 1, h, w)),
                            ('visible', (b, 2, 1, h, w)), ('front', (b, 2, 1, h, w)),
                            ('overlap', (b, 1, h, w)), ('tie', (b, 1, h, w)), ('part_distance_valid', (b, 14, 14)),
                            ('closest_points_valid', (b, 14, 14))):
            _tensor(getattr(self, name), shape, d, name, boolean=True)
        if self.model_family.shape != (b, 2) or self.model_family.dtype != torch.long or self.model_family.device != d:
            raise ValueError('model_family requires int64 [B,2]')
        if ((self.model_family < 0) | (self.model_family >= len(FAMILIES))).any():
            raise ValueError('Unknown body model family')
        if (self.visible & ~self.projected).any() or (self.front & ~self.visible).any():
            raise ValueError('Visible/front must remain on independent projected support')
        expected_projected = torch.stack([p.context.smpl_mask for p in (self.person_a, self.person_b)], 1)
        if not torch.equal(self.projected, expected_projected):
            raise ValueError('Projected support must be the full independent person surfaces')
        if (self.scene_valid & ~self.person_a.canvas_valid).any() or (self.tie & ~self.overlap).any() or self.front.all(1).any():
            raise ValueError('Inconsistent scene/tie/front masks')
        if not torch.allclose(self.owner_weights.sum(1), self.scene_valid.float(), atol=1e-6, rtol=0):
            raise ValueError('Owner weights must partition scene support')
        if ((self.owner_weights < 0) | (self.owner_weights > 1)).any() or (self.owner_weights[~self.visible] != 0).any():
            raise ValueError('Invalid owner weights')
        if ((self.scene_depth < 0) | (self.scene_depth > 1)).any() or (self.scene_depth[~self.scene_valid] != 0).any():
            raise ValueError('Scene depth must be near-one Relative with explicit support')
        if (self.part_distance < 0).any() or ((self.part_contact < 0) | (self.part_contact > 1)).any():
            raise ValueError('Invalid part distance/contact flags')
        if (self.closest_points_valid & ~self.part_distance_valid).any():
            raise ValueError('Surface witnesses require a valid distance')
        if (self.closest_points.masked_select(~self.closest_points_valid[..., None, None].expand_as(self.closest_points)) != 0).any():
            raise ValueError('Unknown surface witnesses must stay zero and explicitly unavailable')
        if ((self.part_contact[..., 0] > self.part_contact[..., 1]) | (self.part_contact[..., 2] > self.part_contact[..., 3])).any():
            raise ValueError('Contact/avoid positives require corresponding annotation availability')
        if not torch.allclose(self.person_a.context.intrinsics, self.person_b.context.intrinsics, atol=1e-5, rtol=0):
            raise ValueError('A/B intrinsics differ')
        if len(self.metadata) != b:
            raise ValueError('Missing dual provenance')
        for i, meta in enumerate(self.metadata):
            if meta.get('schema') != CONDITION_SCHEMA or meta.get('mode') not in ('full_geometry', 'scene_only'):
                raise ValueError('Explicit dual schema/mode required')
            for key in ('pair_id', 'canvas_id', 'camera_id', 'scale_group_id', 'record_sha256'):
                if not isinstance(meta.get(key), str) or not meta[key]:
                    raise ValueError('Missing dual source: ' + key)
            if len(meta.get('person_ids', ())) != 2 or len(set(meta['person_ids'])) != 2:
                raise ValueError('Two stable distinct person IDs required')
            for j, p in enumerate((self.person_a, self.person_b)):
                pm = p.metadata[i]
                if pm['pose_canvas_id'] != meta['canvas_id'] or pm['camera_id'] != meta['camera_id']:
                    raise ValueError('Person/dual camera or canvas provenance differs')
                if pm.get('person_id') != meta['person_ids'][j] or pm.get('scale_group_id') != meta['scale_group_id']:
                    raise ValueError('Person/dual identity or scale provenance differs')
            if meta['mode'] == 'scene_only':
                if (self.camera_available[i] or self.person_depth_valid[i].any() or self.relation_raster[i].any()
                        or self.part_distance_valid[i].any() or self.part_contact[i].any() or self.contacts.valid[i].any()):
                    raise ValueError('scene_only cannot carry physical or contact relations')
            elif not self.camera_available[i]:
                raise ValueError('Full dual geometry requires bound shared camera/scale')
            else:
                contexts = [p.context.metadata[i] for p in (self.person_a, self.person_b)]
                if any(contexts[0].get(k) != contexts[1].get(k) for k in ('units', 'scale_status')):
                    raise ValueError('A/B scale systems differ')
                if not torch.equal(self.relation_raster[i, 2:4], self.person_depth_valid[i, :, 0].float()):
                    raise ValueError('Relation Raster depth masks differ from typed Camera-Z validity')
        self.contacts.validate()
        if self.contacts.valid.shape[0] != b or self.contacts.valid.device != d:
            raise ValueError('Dual contact batch/device differs')
        return self


@dataclass(frozen=True)
class NativeDualIdentityCondition(_TensorRecord):
    source_latent: torch.Tensor           # [B,64,h,w], original Qwen reference latent
    person_mask: torch.Tensor             # [B,2,1,Hs,Ws], visible source masks
    head_mask: torch.Tensor
    source_indices: torch.Tensor         # target slot -> source person index
    source_valid: torch.Tensor
    metadata: tuple[dict, ...]

    def validate(self):
        if self.source_latent.ndim != 4 or self.source_latent.shape[1] != 64 or not self.source_latent.is_floating_point():
            raise ValueError('Identity requires native Qwen 64-channel spatial latents')
        b, _, lh, lw = self.source_latent.shape
        d = self.source_latent.device
        shape = (b, 2, 1, lh*16, lw*16)
        for name in ('person_mask', 'head_mask'):
            _tensor(getattr(self, name), shape, d, name, boolean=True)
        _tensor(self.source_valid, (b, 2), d, 'source_valid', boolean=True)
        if (self.head_mask & ~self.person_mask).any():
            raise ValueError('Source head must belong to its visible source person')
        if (self.person_mask[:, 0] & self.person_mask[:, 1]).any():
            raise ValueError('Visible source instance masks must not assign a pixel to both identities')
        if self.source_indices.shape != (b, 2) or self.source_indices.dtype != torch.long or self.source_indices.device != d:
            raise ValueError('source_indices must be int64 [B,2]')
        if not torch.equal(self.source_indices.sort(1).values, torch.arange(2, device=d).expand(b, -1)):
            raise ValueError('Two target slots must map to distinct source persons')
        if not torch.isfinite(self.source_latent).all() or not self.source_valid.all() or not self.person_mask.flatten(2).any(2).all():
            raise ValueError('Enabled dual identity requires both real source persons')
        if len(self.metadata) != b:
            raise ValueError('Identity provenance batch differs')
        for i, meta in enumerate(self.metadata):
            if meta.get('schema') != IDENTITY_SCHEMA or not meta.get('source_canvas_id') or not meta.get('source_rgb_sha256'):
                raise ValueError('Missing source-only identity provenance')
            src, dst = meta.get('source_person_ids', ()), meta.get('target_person_ids', ())
            if len(src) != 2 or len(set(src)) != 2 or len(dst) != 2:
                raise ValueError('Stable source and target IDs required')
            if [src[j] for j in self.source_indices[i].tolist()] != list(dst):
                raise ValueError('Source mapping disagrees with target person identities')
        return self

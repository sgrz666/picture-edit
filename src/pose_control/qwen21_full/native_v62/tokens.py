"""Person-scoped pooling views; the original Qwen layout is never modified."""
from dataclasses import dataclass, fields, replace

import torch

from ..native_v6.poselink import _align, _budget_grid, _pool
from ..native_v61.tokens import PoseKeySetV61, sample_pose_keys, _pixel_to_tokens, _sample


@dataclass(frozen=True)
class PoseKeySetV62(PoseKeySetV61):
    parts: torch.Tensor                 # [B,M,14], same normalized sampling footprint


@dataclass(frozen=True)
class _PoolingReference:
    """Private support selector, distinct from the immutable native metadata."""
    original: object
    person: int | None = None

    @property
    def role(self):
        return self.original.role if self.person is None else 'person_'+str(self.person)+'_pose'
    @property
    def image_id(self): return self.original.image_id
    @property
    def indices(self): return self.original.indices
    @property
    def token_hw(self): return self.original.token_hw


@dataclass(frozen=True)
class _PoolingView:
    references: tuple


def _with_parts(base, parts):
    return PoseKeySetV62(**{field.name:getattr(base,field.name) for field in fields(PoseKeySetV61)},parts=parts)


def _person_maps(spatial, geometry, person, reference_hw):
    pixels=spatial.key_support[:,person:person+1] * geometry.person_valid[:,person,None,None,None]
    target_weight=_pixel_to_tokens(pixels,geometry.target_token_hw)[...,0]
    reference_weight=_pixel_to_tokens(pixels,reference_hw)[...,0]
    return target_weight,reference_weight


def sample_routed_keys(joint,layout,geometry,spatial,routing,config):
    """Split global quota, never duplicate it; retain V6.1 sparse quotas."""
    pieces=[]
    for ref in layout.references:
        reference_person=0 if ref.role=='person_0_pose' else 1 if ref.role=='person_1_pose' else None
        people=(reference_person,) if reference_person is not None else tuple(
            p for p in range(2) if bool(geometry.person_valid[:,p].any()))
        budget=config.global_token_budget if config.multiscale_tokens else config.max_pose_tokens_per_reference
        for rank,person in enumerate(people):
            quota=budget//len(people)+(rank<budget%len(people))
            if not quota: continue
            scoped=replace(config,global_token_budget=quota,joint_token_budget=0,local_token_budget=0,
                           multiscale_tokens=True,geometry_key_mask=True)
            view=_PoolingView((_PoolingReference(ref,person),))
            base=sample_pose_keys(joint,view,geometry,spatial,scoped)
            target_weight,reference_weight=_person_maps(spatial,geometry,person,ref.token_hw)
            mapped=_align(routing.part_tokens[:,person].to(joint.dtype),target_weight,
                          geometry.target_token_hw,ref.token_hw)
            parts=_pool(mapped,reference_weight,ref.token_hw,_budget_grid(ref.token_hw,quota))
            pieces.append(_with_parts(base,parts))
        if config.multiscale_tokens and config.joint_token_budget+config.local_token_budget:
            scoped=replace(config,global_token_budget=0,geometry_key_mask=True)
            base=sample_pose_keys(joint,_PoolingView((_PoolingReference(ref),)),geometry,spatial,scoped)
            parts=joint.new_zeros(*base.valid.shape,14)
            for person in people:
                target_weight,reference_weight=_person_maps(spatial,geometry,person,ref.token_hw)
                mapped=_align(routing.part_tokens[:,person].to(joint.dtype),target_weight,
                              geometry.target_token_hw,ref.token_hw)
                sampled=_sample(mapped,reference_weight,ref.token_hw,base.xy)[0]
                parts=torch.where((base.person_ids==person)[...,None],sampled,parts)
            pieces.append(_with_parts(base,parts))
    if not pieces:
        base=sample_pose_keys(joint,layout,geometry,spatial,config)
        return _with_parts(base,joint.new_zeros(*base.valid.shape,14))
    return PoseKeySetV62(**{field.name:None if getattr(pieces[0],field.name) is None else
        torch.cat([getattr(piece,field.name) for piece in pieces],1) for field in fields(PoseKeySetV62)})

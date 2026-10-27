"""Inference-only source bank; retain each appearance image on its own Canvas.

This record adds no modules or parameters. The original one-source checkpoint
header remains exact; bank provenance belongs to the prepared inference record.
"""
from dataclasses import dataclass, replace
from copy import deepcopy

import torch

from src.data.canvas_contract import CanvasSpec
from ..native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA
from .contracts import NativeDetailReferencesV5

SOURCE_BANK_SCHEMA = 'native_v5_source_bank_v1'


@dataclass(frozen=True)
class NativeAppearanceSourceV5:
    qwen_image_index: int
    person_ids: tuple
    source_slot_id: int
    canvas: dict
    rgb_sha256: str
    geometry_sha256: str
    person_mask: torch.Tensor             # [B,P,1,Hs,Ws], own source Canvas
    head_mask: torch.Tensor
    references: NativeLocalReferencesV4   # single actual person in physical slot0
    detail: NativeDetailReferencesV5 | None
    latent: torch.Tensor | None = None    # view of frozen Qwen reference [B,N,64]

    @property
    def token_hw(self): return CanvasSpec.from_record(self.canvas).token_hw

    def validate(self, *, require_latent=False):
        canvas=CanvasSpec.from_record(self.canvas);b=self.person_mask.shape[0]
        if self.qwen_image_index not in (0,1): raise ValueError('Appearance source image must be Qwen image0 or image1')
        if len(self.person_ids)!=1 or not self.person_ids[0] or self.person_ids[0]=='__inactive__':
            raise ValueError('Each independent source requires one stable person in physical slot0')
        if type(self.source_slot_id)!=int or not 0<=self.source_slot_id<16:
            raise ValueError('Source logical slot must fit inherited embeddings')
        if not self.rgb_sha256 or not self.geometry_sha256: raise ValueError('Source RGB/geometry fingerprints required')
        for name in ('person_mask','head_mask'):
            mask=getattr(self,name)
            if mask.shape!=(b,1,1,*canvas.canvas_hw) or mask.dtype!=torch.bool:
                raise ValueError('Source masks must retain their own native Canvas [B,1,1,H,W]')
        if (self.head_mask & ~self.person_mask).any(): raise ValueError('Source head support exceeds visible person mask')
        if not self.person_mask.flatten(1).any(1).all():raise ValueError('Each independent appearance source requires a nonempty person mask')
        self.references.validate()
        if self.references.arcface.shape[0]!=b: raise ValueError('Source feature batch differs')
        if (self.references.arcface_valid[:,1].any() or self.references.face_dino_valid[:,1].any()
                or self.references.hand_dino_valid[:,1].any()):
            raise ValueError('Independent source features may only use physical person0')
        for meta in self.references.metadata:
            if (meta.get('source_rgb_sha256')!=self.rgb_sha256 or meta.get('source_canvas_id')!=canvas.fingerprint()
                    or meta.get('source_person_ids')!=[self.person_ids[0],'__inactive__']):
                raise ValueError('Source feature RGB/Canvas/stable person binding differs')
            references=meta.get('reference_sources',())
            if len(references)!=self.references.arcface.shape[2]:
                raise ValueError('Declare all offline reference records')
            if references[0].get('geometry_sha256')!=self.geometry_sha256:
                raise ValueError('Primary offline Source geometry differs')
            for reference in references:
                if reference['person_ids']!=[self.person_ids[0],'__inactive__']:
                    raise ValueError('Offline references stable person binding differs')
        if self.detail is not None:
            self.detail.validate()
            if self.detail.support.shape!=(b,6,*canvas.canvas_hw): raise ValueError('Detail must retain its own source Canvas')
            for meta in self.detail.metadata:
                if (meta['source_rgb_sha256']!=self.rgb_sha256 or meta['source_canvas_id']!=canvas.fingerprint()
                        or meta['source_person_ids']!=[self.person_ids[0],'__inactive__']):
                    raise ValueError('Detail Source provenance differs')
            if self.detail.valid[:,[1,4,5]].any(): raise ValueError('Independent Detail source may only use physical person0')
        if self.latent is None:
            if require_latent: raise ValueError('Bind the actual frozen Qwen source image latent before encoding')
        else:
            h,w=self.token_hw
            if self.latent.shape!=(b,h*w,64) or not torch.isfinite(self.latent).all():
                raise ValueError('Actual Qwen source latent/grid differs')
        return self

    def to(self,device):
        return replace(self,person_mask=self.person_mask.to(device),head_mask=self.head_mask.to(device),
            references=self.references.to(device),detail=None if self.detail is None else self.detail.to(device),
            latent=None if self.latent is None else self.latent.to(device))

    def latent_map(self,row):
        self.validate(require_latent=True);h,w=self.token_hw
        return self.latent[row:row+1].float().transpose(1,2).reshape(1,64,h,w)

    def record(self):
        return dict(qwen_image_index=self.qwen_image_index,person_ids=list(self.person_ids),
            source_slot_id=self.source_slot_id,canvas=deepcopy(self.canvas),
            canvas_id=CanvasSpec.from_record(self.canvas).fingerprint(),rgb_sha256=self.rgb_sha256,
            geometry_sha256=self.geometry_sha256,token_hw=list(self.token_hw))


@dataclass(frozen=True)
class NativeSourceBankV5:
    sources: tuple[NativeAppearanceSourceV5,...]
    target_qwen_image_indices: torch.Tensor  # [B,2], Qwen image index, never person index
    target_person_indices: torch.Tensor      # [B,2], physical person within that image
    target_person_ids: tuple                 # stable identities in target slot order

    def validate(self, *, require_latent=False):
        if len(self.sources)!=2 or tuple(s.qwen_image_index for s in self.sources)!=(0,1):
            raise ValueError('Two ordered independent Qwen source images required')
        for source in self.sources: source.validate(require_latent=require_latent)
        if len({s.source_slot_id for s in self.sources})!=2: raise ValueError('Sources require distinct logical slots')
        if len({s.person_ids[0] for s in self.sources})!=2: raise ValueError('Sources require distinct stable identities')
        b=self.sources[0].person_mask.shape[0]
        if self.sources[1].person_mask.shape[0]!=b or len(self.target_person_ids)!=b: raise ValueError('Source bank batch differs')
        for value in (self.target_qwen_image_indices,self.target_person_indices):
            if value.shape!=(b,2) or value.dtype!=torch.long: raise ValueError('Explicit target source/person int64 [B,2] required')
        if ((self.target_qwen_image_indices<0)|(self.target_qwen_image_indices>1)).any():
            raise ValueError('Target appearance must select a source image, never a skeleton')
        if self.target_person_indices.any(): raise ValueError('Independent sources each have physical person index0')
        if (self.target_qwen_image_indices[:,0]==self.target_qwen_image_indices[:,1]).any():
            raise ValueError('Dual target persons must select distinct source images')
        for row in range(b):
            if len(self.target_person_ids[row])!=2: raise ValueError('Two stable target person IDs required')
            for p in range(2):
                source=self.sources[int(self.target_qwen_image_indices[row,p])]
                if self.target_person_ids[row][p]!=source.person_ids[0]: raise ValueError('Target/source stable identity mapping differs')
        return self

    @property
    def source_indices(self): return self.target_qwen_image_indices

    @property
    def source_valid(self):
        return torch.cat([s.person_mask.flatten(2).any(-1) for s in self.sources],1)

    @property
    def metadata(self):
        record={'schema':SOURCE_BANK_SCHEMA,'source_images_in_qwen':2,
            'sources':[s.record() for s in self.sources]}
        return tuple({**deepcopy(record),'target_qwen_image_indices':self.target_qwen_image_indices[i].tolist(),
            'target_person_indices':self.target_person_indices[i].tolist(),
            'target_person_ids':list(ids)} for i,ids in enumerate(self.target_person_ids))

    def to(self,device):
        return replace(self,sources=tuple(s.to(device) for s in self.sources),
            target_qwen_image_indices=self.target_qwen_image_indices.to(device),target_person_indices=self.target_person_indices.to(device))

    def with_pair(self,pair):
        from .conditioning import validate_pair
        validate_pair(pair)
        if pair['qwen_condition_profile']!='dual_source_5ref': raise ValueError('Two-source bank requires explicit ordered five-reference profile')
        offsets=[0]
        for h,w in pair['reference_token_hws']: offsets.append(offsets[-1]+h*w)
        sources=[]
        for source in self.sources:
            i=source.qwen_image_index
            if tuple(pair['reference_token_hws'][i])!=source.token_hw: raise ValueError('Source bank/Qwen native grid differs')
            if 'reference_sha256s' in pair and pair['reference_sha256s'][i]!=source.rgb_sha256:
                raise ValueError('Source bank/Qwen RGB fingerprint differs')
            sources.append(replace(source,latent=pair['reference_latents'][:,offsets[i]:offsets[i+1]]))
        return replace(self,sources=tuple(sources)).validate(require_latent=True)

    def merged_references(self):
        """Gather frozen features into legacy logical identity slots, never pixels."""
        self.validate();r0,r1=(s.references for s in self.sources)
        count=max(r0.arcface.shape[2],r1.arcface.shape[2])
        names=tuple(n for n in r0.__dataclass_fields__ if n!='metadata')
        values={}
        for name in names:
            rows=[]
            for record in (r0,r1):
                value=getattr(record,name)[:,:1]
                if name not in ('source_face_boxes','source_hand_boxes'):
                    dim=3 if name.startswith('hand_') else 2
                    shape=list(value.shape);shape[dim]=count-value.shape[dim]
                    value=torch.cat((value,value.new_zeros(shape)),dim=dim)
                rows.append(value)
            values[name]=torch.cat(rows,dim=1)
        metadata=tuple({'schema':REFERENCE_SCHEMA,'role':'source_only','source_bank':m,
            'reference_count':count,'source_reference_counts':[r0.arcface.shape[2],r1.arcface.shape[2]]} for m in self.metadata)
        return NativeLocalReferencesV4(**values,metadata=metadata).validate()

    def validate_target(self,batch):
        self.validate(require_latent=True)
        if batch['native_scene_kind']!='dual': raise ValueError('Two-source bank requires a dual target scene')
        c=batch['local_condition_v4'];c.validate()
        if not torch.equal(c.source_indices.cpu(),self.source_indices.cpu()): raise ValueError('Local/Qwen source image mapping differs')
        if not c.person_valid.all(): raise ValueError('Two-source target requires both real people')
        expected=[s.source_slot_id for s in self.sources]
        if c.source_slot_ids.tolist()!=[expected for _ in self.target_person_ids]: raise ValueError('Local logical source slots differ')
        for row,ids in enumerate(self.target_person_ids):
            if c.metadata[row]['source_person_ids']!=[s.person_ids[0] for s in self.sources] or c.metadata[row]['target_person_ids']!=list(ids):
                raise ValueError('Local/bank stable person mapping differs')
            for p in range(2):
                provenance=batch['pose_provenance' if p==0 else 'pose_provenance_b'][row]
                if provenance['person_id']!=ids[p]: raise ValueError('Bank/body stable target person differs')
        return self


def appearance_source_from_sample(sample,*,qwen_image_index,source_slot_id):
    """Bind one verified source-only sample; do not run another VAE or encoder."""
    metadata=sample['identity_metadata'];metadata=metadata[0] if isinstance(metadata,(tuple,list)) else metadata
    def mask(key):
        value=sample[key]
        if value.ndim==4: value=value[None]
        return value[:,:1]
    ids=metadata['source_person_ids']
    if ids[1:]!=['__inactive__']: raise ValueError('Source sample must declare exactly one real person')
    return NativeAppearanceSourceV5(qwen_image_index,(ids[0],),source_slot_id,sample['source_canvas'],
        metadata['source_rgb_sha256'],metadata['source_geometry_sha256'],mask('source_person_mask'),
        mask('source_head_mask'),sample['local_references_v4'],sample.get('detail_references_v5')).validate()

"""V4 source-only references, independent modality validity and prepared local control."""
from dataclasses import dataclass, replace
from copy import deepcopy
import torch

from .depth_condition_v2 import _TensorRecord, _tensor
from .native_geometry import NativeGeometryFeatures

LOCAL_SCHEMA = 'native_local_condition_v4'
REFERENCE_SCHEMA = 'native_local_reference_v4'
REGIONS = ('face_a', 'face_b', 'hand_a_left', 'hand_a_right', 'hand_b_left', 'hand_b_right')
RIGID_FACE_SLOTS = (*range(4), *range(31, 40), 40, 43, 46, 49)


def _bool(value, shape, device, name):
    _tensor(value, shape, device, name, boolean=True)


def _indices(value, shape, device, name):
    if value.shape != shape or value.device != device or value.dtype != torch.long:
        raise ValueError(name+' requires int64 '+str(shape))


@dataclass(frozen=True)
class NativeLocalConditionV4(_TensorRecord):
    person_valid: torch.Tensor
    source_indices: torch.Tensor          # physical source-array positions
    source_slot_ids: torch.Tensor         # stable logical source IDs, source-array order
    face_landmarks: torch.Tensor
    face_point_valid: torch.Tensor
    jaw_pose: torch.Tensor
    jaw_valid: torch.Tensor
    expression: torch.Tensor
    expression_valid: torch.Tensor
    face_boxes: torch.Tensor              # normalized, UNCLIPPED target-canvas xyxy
    face_roi_valid: torch.Tensor
    hand_pose: torch.Tensor               # actual FK rotations after PCA/mean
    hand_pose_valid: torch.Tensor
    hand_keypoints: torch.Tensor
    hand_point_valid: torch.Tensor
    mesh_bps: torch.Tensor
    mesh_valid: torch.Tensor
    hand_boxes: torch.Tensor
    hand_roi_valid: torch.Tensor
    wrist_z: torch.Tensor
    region_masks: torch.Tensor            # [B,6,H,W], geometric soft support
    visibility: torch.Tensor
    local_z: torch.Tensor
    local_z_valid: torch.Tensor
    part_visible: torch.Tensor            # conservative fallback, same person/part
    person_owner: torch.Tensor            # [B,2,H,W], full-scene visible ownership
    scene_z: torch.Tensor
    scene_z_valid: torch.Tensor
    body_height: torch.Tensor
    metadata: tuple[dict, ...]

    @property
    def boxes(self):
        return torch.cat((self.face_boxes, self.hand_boxes.flatten(1, 2)), 1)

    @property
    def region_valid(self):
        return torch.cat((self.face_roi_valid, self.hand_roi_valid.flatten(1)), 1)

    def validate(self):
        b = self.person_valid.shape[0]
        device = self.person_valid.device
        if self.region_masks.ndim != 4:
            raise ValueError('Local masks require [B,6,H,W]')
        h, w = self.region_masks.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError('V4 uses the exact canonical Qwen canvas')
        for name, shape in (('face_landmarks',(b,2,72,3)),('jaw_pose',(b,2,3)),('expression',(b,2,10)),
            ('face_boxes',(b,2,4)),('hand_pose',(b,2,2,45)),('hand_keypoints',(b,2,2,21,3)),
            ('mesh_bps',(b,2,2,1024)),('hand_boxes',(b,2,2,4)),('wrist_z',(b,2,2)),
            ('region_masks',(b,6,h,w)),('visibility',(b,6,h,w)),('local_z',(b,6,h,w)),
            ('person_owner',(b,2,h,w)),('scene_z',(b,1,h,w)),('body_height',(b,2))):
            _tensor(getattr(self,name),shape,device,name)
        for name, shape in (('person_valid',(b,2)),('face_point_valid',(b,2,72)),('jaw_valid',(b,2)),
            ('expression_valid',(b,2)),('face_roi_valid',(b,2)),('hand_pose_valid',(b,2,2)),
            ('hand_point_valid',(b,2,2,21)),('mesh_valid',(b,2,2)),('hand_roi_valid',(b,2,2)),
            ('local_z_valid',(b,6,h,w)),('part_visible',(b,6,h,w)),('scene_z_valid',(b,1,h,w))):
            _bool(getattr(self,name),shape,device,name)
        _indices(self.source_indices,(b,2),device,'source_indices')
        _indices(self.source_slot_ids,(b,2),device,'source_slot_ids')
        if not self.person_valid[:,0].all() or ((self.source_indices[self.person_valid] < 0) | (self.source_indices[self.person_valid] > 1)).any():
            raise ValueError('Person A and active source mappings are required')
        if (self.person_valid.all(1) & (self.source_indices[:,0] == self.source_indices[:,1])).any():
            raise ValueError('Active target persons cannot share a source identity')
        if ((self.source_slot_ids < 0) | (self.source_slot_ids > 15)).any():
            raise ValueError('Logical source slots must fit inherited source embeddings')
        logical=self.source_slot_ids.gather(1,self.source_indices.clamp_min(0))
        if (self.person_valid.all(1)&(logical[:,0]==logical[:,1])).any(): raise ValueError('Active sources require distinct stable logical slots')
        for name in ('face_roi_valid','jaw_valid','expression_valid'):
            if (getattr(self,name) & ~self.person_valid).any():
                raise ValueError('Inactive Person B cannot have valid face conditions')
        for name in ('hand_roi_valid','hand_pose_valid','mesh_valid'):
            if (getattr(self,name) & ~self.person_valid[...,None]).any():
                raise ValueError('Inactive Person B cannot have valid hand conditions')
        if (self.face_point_valid & ~self.person_valid[...,None]).any() or (self.hand_point_valid & ~self.person_valid[...,None,None]).any():
            raise ValueError('Inactive Person B cannot have valid landmarks')
        for points, valid in ((self.face_landmarks,self.face_point_valid),(self.hand_keypoints,self.hand_point_valid)):
            if ((points[...,2] < 0) | (points[...,2] > 1)).any() or (valid & (points[...,2] <= 0)).any():
                raise ValueError('Point availability/confidence disagree')
        for boxes, valid in ((self.face_boxes,self.face_roi_valid),(self.hand_boxes,self.hand_roi_valid)):
            if ((boxes[...,2:] <= boxes[...,:2]).any(-1) & valid).any():
                raise ValueError('Valid unclipped boxes need positive area')
        for name in ('region_masks','visibility','person_owner'):
            value=getattr(self,name)
            if ((value<0)|(value>1)).any(): raise ValueError(name+' must be 0..1')
        if (self.person_owner.sum(1)>1.00001).any() or (self.body_height[self.person_valid]<=0).any():
            raise ValueError('Invalid ownership or physical scale')
        owners=self.person_valid[:,torch.tensor([0,1,0,0,1,1],device=device)]
        if (self.local_z_valid & ~owners[...,None,None]).any() or (self.person_owner*~self.person_valid[...,None,None]).any():
            raise ValueError('Inactive target Person B cannot own pixels or local depth')
        if (self.local_z[self.local_z_valid]<=0).any() or (self.local_z[~self.local_z_valid]!=0).any():
            raise ValueError('Local Camera-Z needs independent validity, zero only outside support')
        hand_z=self.local_z_valid[:,2:].flatten(2).any(-1).reshape(b,2,2)
        if (hand_z & (self.wrist_z<=0)).any(): raise ValueError('Local Hand Z requires a positive same-camera FK wrist Z')
        if (self.scene_z[self.scene_z_valid]<=0).any() or (self.scene_z[~self.scene_z_valid]!=0).any():
            raise ValueError('Scene Camera-Z needs independent validity')
        if len(self.metadata)!=b: raise ValueError('Local provenance batch differs')
        for meta in self.metadata:
            if meta.get('schema')!=LOCAL_SCHEMA or meta.get('expression_mode') not in ('source','driver','off'):
                raise ValueError('Explicit V4 local schema/expression mode required')
            for key in ('target_canvas_id','source_canvas_id','driver_record_sha256','camera_id','scale_group_id'):
                if not isinstance(meta.get(key),str) or not meta[key]: raise ValueError('Missing local '+key)
            if meta.get('normal_space')!='camera_cv' or meta.get('hand_pose_convention')!='full_axis_angle_after_pose_mean':
                raise ValueError('Local coordinate/hand rotation convention differs')
            if len(meta.get('source_person_ids',()))!=2 or len(meta.get('target_person_ids',()))!=2:
                raise ValueError('Declare stable Source and Driver person IDs, including inactive slots')
            if meta.get('expression_mode')=='source' and any(meta.get('expression_retargeted',(False,False))) and not meta.get('source_asset_sha256'):
                raise ValueError('Retargeted Source expression needs its asset fingerprint')
        return self


@dataclass(frozen=True)
class NativeLocalReferencesV4(_TensorRecord):
    arcface: torch.Tensor
    arcface_valid: torch.Tensor
    face_dino: torch.Tensor
    face_dino_valid: torch.Tensor
    hand_dino: torch.Tensor
    hand_dino_valid: torch.Tensor
    source_face_boxes: torch.Tensor
    source_hand_boxes: torch.Tensor
    metadata: tuple[dict, ...]

    def validate(self):
        b, p, r, d = self.arcface.shape
        device=self.arcface.device
        if p!=2 or d!=512 or not 1<=r<=3: raise ValueError('ArcFace requires [B,2,R,512], R<=3')
        for name, shape in (('arcface',(b,2,r,512)),('face_dino',(b,2,r,256,1536)),
            ('hand_dino',(b,2,2,r,256,1536)),('source_face_boxes',(b,2,4)),('source_hand_boxes',(b,2,2,4))):
            _tensor(getattr(self,name),shape,device,name)
        for name, shape in (('arcface_valid',(b,2,r)),('face_dino_valid',(b,2,r)),('hand_dino_valid',(b,2,2,r))):
            _bool(getattr(self,name),shape,device,name)
        if len(self.metadata)!=b or any(m.get('schema')!=REFERENCE_SCHEMA or m.get('role')!='source_only' for m in self.metadata):
            raise ValueError('Local appearance must have explicit source-only provenance')
        return self


@dataclass(frozen=True)
class PreparedNativeLocalV4(_TensorRecord):
    spatial: torch.Tensor                 # [B,6,512,16,16]
    tokens: torch.Tensor                  # [B,6,16,512], hand uses first 12
    token_valid: torch.Tensor
    geometry_token: torch.Tensor
    spatial_valid: torch.Tensor
    boxes: torch.Tensor
    pixel_weights: torch.Tensor
    region_valid: torch.Tensor
    metadata: tuple[dict, ...]

    def validate(self):
        b, _, h, w=self.pixel_weights.shape; d=self.spatial.device
        for name, shape in (('spatial',(b,6,512,16,16)),('tokens',(b,6,16,512)),('boxes',(b,6,4)),('pixel_weights',(b,6,h,w))):
            _tensor(getattr(self,name),shape,d,name)
        for name, shape in (('token_valid',(b,6,16)),('geometry_token',(b,6,16)),('spatial_valid',(b,6)),('region_valid',(b,6))):
            _bool(getattr(self,name),shape,d,name)
        if (self.pixel_weights<0).any() or (self.pixel_weights.sum(1)>1.00001).any():
            raise ValueError('Local region weights must not double-count overlapping pixels')
        return self


@dataclass(frozen=True)
class PreparedNativeFullControlV4:
    body: NativeGeometryFeatures
    local: PreparedNativeLocalV4 | None
    metadata: tuple[dict, ...]
    static_face: torch.Tensor | None = None
    static_hand: torch.Tensor | None = None

    def validate(self):
        b,n,d=self.body.geometry.shape
        grid=self.body.target_token_hw
        if (not isinstance(grid,tuple) or len(grid)!=2 or any(type(v) is not int or v<=0 for v in grid)
                or n%4 or d!=512 or n!=grid[0]*grid[1]):
            raise ValueError('Full V4 preserves exact-grid [B,N,512] control')
        _tensor(self.body.geometry,(b,n,512),self.body.geometry.device,'body geometry')
        _tensor(self.body.interaction,(b,n,512),self.body.geometry.device,'body interaction')
        _bool(self.body.person_valid,(b,2),self.body.geometry.device,'body person_valid')
        if len(self.metadata)!=b: raise ValueError('Prepared V4 provenance batch differs')
        if self.local is not None:
            self.local.validate()
            if self.local.spatial.shape[0]!=b or self.local.spatial.device!=self.body.geometry.device:
                raise ValueError('Body/local batch or device differs')
            if self.local.pixel_weights.shape[-2:]!=(grid[0]*16,grid[1]*16): raise ValueError('Local canvas differs from Body target grid')
        for value in (self.static_face,self.static_hand):
            if value is not None: _tensor(value,(b,n,512),self.body.geometry.device,'static local delta')
        return self

    def to(self, *args, **kwargs):
        from .depth_condition_v2 import _to_fp32
        body=replace(self.body,geometry=_to_fp32(self.body.geometry,*args,**kwargs),
            interaction=_to_fp32(self.body.interaction,*args,**kwargs),
            person_valid=self.body.person_valid.to(device=_to_fp32(self.body.geometry,*args,**kwargs).device))
        return replace(self,body=body,local=None if self.local is None else self.local.to(*args,**kwargs),metadata=deepcopy(self.metadata),
            static_face=None if self.static_face is None else _to_fp32(self.static_face,*args,**kwargs),
            static_hand=None if self.static_hand is None else _to_fp32(self.static_hand,*args,**kwargs))

    def index_select(self, indices):
        local_indices=indices.to(self.body.geometry.device)
        body=replace(self.body,geometry=self.body.geometry.index_select(0,local_indices),interaction=self.body.interaction.index_select(0,local_indices),
                     person_valid=self.body.person_valid.index_select(0,local_indices))
        return replace(self,body=body,local=None if self.local is None else self.local.index_select(indices),
                       static_face=None if self.static_face is None else self.static_face.index_select(0,local_indices),
                       static_hand=None if self.static_hand is None else self.static_hand.index_select(0,local_indices),
                       metadata=tuple(deepcopy(self.metadata[i]) for i in indices.cpu().tolist()))

    @classmethod
    def stack(cls, records):
        if not records or any(r.body.target_token_hw!=records[0].body.target_token_hw for r in records):
            raise ValueError('Bucket prepared controls by exact target canvas')
        if any((r.local is None)!=(records[0].local is None) for r in records):
            raise ValueError('Cannot mix local-off and local-on prepared controls')
        body=replace(records[0].body,geometry=torch.cat([r.body.geometry for r in records]),
            interaction=torch.cat([r.body.interaction for r in records]),person_valid=torch.cat([r.body.person_valid for r in records]))
        extra={}
        for name in ('static_face','static_hand'):
            if any((getattr(r,name) is None)!=(getattr(records[0],name) is None) for r in records): raise ValueError('Static modes differ')
            extra[name]=None if getattr(records[0],name) is None else torch.cat([getattr(r,name) for r in records])
        return cls(body,None if records[0].local is None else PreparedNativeLocalV4.stack([r.local for r in records]),
                   tuple(deepcopy(m) for r in records for m in r.metadata),**extra).validate()

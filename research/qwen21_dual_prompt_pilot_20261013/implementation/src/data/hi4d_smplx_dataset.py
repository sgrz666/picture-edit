"""SMPL-X controls and source-only generation data, separate from supervision."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from src.pose_control.qwen21_full.dual.schema import (
    ANATOMICAL_PART_JOINT_INDICES, SMPLX_JOINT_TO_PART, SMPLX14_PARTITION_SPEC,
    SMPLX14_PARTITION_SHA256, SOURCE_SCHEMA_VERSION, SOURCE_CROP_VERSION,
    SOURCE_ENCODER_VERSION, generation_versions, geometry_schema_metadata)


SIZE = 512
SMPLX_TO_PART = np.asarray(SMPLX_JOINT_TO_PART, dtype=np.int64)
GENERATION_KEYS = {'source_rgb','source_masks','source_features','target_geometry',
                   'source_features_sha256','person_uids','prompt','seed','resolution',
                   'versions','target_geometry_known'}
SUPERVISION_SCHEMA_VERSION = 'dual_supervision_v1'
SUPERVISION_KEYS = {'schema_version','target_rgb','target_masks','target_rgb_sha256',
    'target_mask_sha256','target_geometry','target_geometry_sha256','target_geometry_sidecar_sha256',
    'pair','action','source_frame','target_frame','camera','person_uids'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_cache_metadata(source_rgb, source_masks, person_uids):
    if len(source_masks) != 2 or len(person_uids) != 2:
        raise ValueError('source feature cache requires two masks and two person UIDs')
    return dict(schema_version=SOURCE_SCHEMA_VERSION, crop_version=SOURCE_CROP_VERSION,
        encoder_version=SOURCE_ENCODER_VERSION, source_rgb_sha256=file_sha256(source_rgb),
        source_mask_sha256=[file_sha256(path) for path in source_masks],
        person_uids=list(person_uids))


def validate_source_feature_cache(features, row):
    """Reject stale slot bindings before constructing an appearance condition."""
    metadata = features.get('metadata') if isinstance(features, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError('source feature cache requires embedded metadata')
    for name, expected_version in generation_versions().items():
        if name in ('source','crop','encoder') and row['versions'].get(name) != expected_version:
            raise ValueError(f'source feature cache has incompatible versions.{name}')
    expected = source_cache_metadata(row['source_rgb'], row['source_masks'], row['person_uids'])
    expected.update(schema_version=row['versions']['source'],
        crop_version=row['versions']['crop'], encoder_version=row['versions']['encoder'])
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f'source feature cache {name} does not match current inputs/manifest')
    return metadata


def validate_geometry_schema(payload, versions):
    """NPZ metadata is required; a manifest cannot relabel an old geometry cache."""
    expected = geometry_schema_metadata()
    for name, expected_version in generation_versions().items():
        if name in ('geometry','coordinate_convention','rotation6d_convention','partition_sha256') and versions.get(name) != expected_version:
            raise ValueError(f'geometry NPZ has incompatible versions.{name}')
    expected.update(schema_version=versions['geometry'],
        coordinate_convention=versions['coordinate_convention'],
        rotation6d_convention=versions['rotation6d_convention'],
        partition_sha256=versions['partition_sha256'])
    for name, value in expected.items():
        if name not in payload:
            raise ValueError(f'geometry NPZ missing {name}')
        actual = np.asarray(payload[name])
        if actual.shape != () or actual.item() != value:
            raise ValueError(f'geometry NPZ has incompatible {name}')
    if 'source_shape_shared' not in payload or np.asarray(payload['source_shape_shared']).shape != () or np.asarray(payload['source_shape_shared']).item() is not True:
        raise ValueError('geometry NPZ requires verified source_shape_shared=True')
    return expected


def anatomical_part_geometry(joints_world):
    """Unit world bone directions and the canonical part joint in SMPL-X order.

    The first 55 joints must be the standard SMPL-X skeleton, without a mapper.
    Invalid/degenerate bones have zero vectors, zero joints, and false validity.
    """
    joints = np.asarray(joints_world, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[0] != 2 or joints.shape[1] < 55 or joints.shape[2] != 3:
        raise ValueError('anatomical geometry requires [2,J>=55,3] standard SMPL-X world joints')
    starts = joints[:, [12,0,16,18,20,17,19,21,1,4,7,2,5,8]]
    ends = joints[:, [15,9,18,20,20,19,21,21,4,7,10,5,8,11]].copy()
    ends[:, 4] = joints[:, 25:40].mean(1)
    ends[:, 7] = joints[:, 40:55].mean(1)
    part_joints = joints[:, ANATOMICAL_PART_JOINT_INDICES].copy()
    direction = ends - starts
    norm = np.linalg.norm(direction, axis=-1)
    valid = np.isfinite(direction).all(-1) & np.isfinite(part_joints).all(-1) & (norm > 1e-8)
    orientation = np.zeros_like(direction)
    np.divide(direction, norm[..., None], out=orientation, where=valid[..., None])
    part_joints[~valid] = 0
    return orientation, part_joints, valid


def letterbox_affine(width, height, size=SIZE):
    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError('letterbox dimensions must be positive')
    scale = min(size/width, size/height)
    return np.array([[scale,0,(size-width*scale)/2], [0,scale,(size-height*scale)/2], [0,0,1]], np.float32)


def letterbox_array(value, *, nearest=False, size=SIZE):
    affine = letterbox_affine(value.shape[1], value.shape[0], size)
    return cv2.warpAffine(value, affine[:2], (size,size),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def read_rgb(path):
    array = letterbox_array(np.asarray(Image.open(path).convert('RGB')))
    return torch.from_numpy(array.copy()).permute(2,0,1).float()/255


def read_mask(path):
    array = np.asarray(Image.open(path).convert('L'))
    array = letterbox_array((array > 127).astype(np.uint8), nearest=True)
    return torch.from_numpy(array.copy()).float()[None]


def source_person_crops(rgb,masks):
    """Square source-only crops; suppress pixels belonging to the other person."""
    if rgb.shape!=(512,512,3) or masks.shape!=(2,512,512):
        raise ValueError('source crops require letterboxed RGB and two masks at 512')
    crops,coverage,boxes = [],[],[]
    for i in range(2):
        exclusive = (masks[i]>.5)&(masks[1-i]<=.5)
        y,x = np.nonzero(exclusive)
        if len(x)<16:
            raise ValueError(f'source person {i} has insufficient exclusive visible pixels')
        center = np.array([(x.min()+x.max()+1)/2,(y.min()+y.max()+1)/2])
        side = float(max(x.max()-x.min()+1,y.max()-y.min()+1)*1.2)
        lo = center-side/2
        scale = 512/side
        affine = np.array([[scale,0,-lo[0]*scale],[0,scale,-lo[1]*scale]],np.float32)
        masked = np.where(exclusive[...,None],rgb,128).astype(np.uint8)
        crops.append(cv2.warpAffine(masked,affine,(512,512),borderValue=(128,128,128)))
        coverage.append(cv2.warpAffine(exclusive.astype(np.float32),affine,(512,512),flags=cv2.INTER_LINEAR))
        boxes.append(np.r_[lo,lo+side])
    return np.stack(crops),np.stack(coverage),np.asarray(boxes,np.float32)


def hi4d_contact_pairs(contact):
    """Hi4D uses zero for absent and one-based opposite vertex indices."""
    contact = np.asarray(contact)
    if contact.ndim!=2 or contact.shape[0]!=2 or not np.isfinite(contact).all():
        raise ValueError('contact correspondences must have shape [2,V]')
    if np.any(contact<0) or np.any(contact>contact.shape[1]) or np.any(contact!=np.floor(contact)):
        raise ValueError('invalid one-based Hi4D contact index')
    ai = np.flatnonzero(contact[0]>0)
    bi = np.flatnonzero(contact[1]>0)
    pairs = np.concatenate((np.stack((ai,contact[0,ai].astype(np.int64)-1),-1),
                            np.stack((contact[1,bi].astype(np.int64)-1,bi),-1)))
    return np.unique(pairs,axis=0)


def smplx_part_labels(lbs_weights):
    if lbs_weights.shape != (10475,55):
        raise ValueError('SMPL-X requires 10475 vertices and 55 skinning joints; SMPL labels cannot be reused')
    return SMPLX_TO_PART[lbs_weights.argmax(-1)]


def visible_owner(amodal, depth):
    masks = np.asarray(amodal, dtype=bool)
    depth = np.asarray(depth)
    if masks.shape != depth.shape or masks.shape[0] != 2:
        raise ValueError('amodal and depth must have shape [2,H,W]')
    if not np.isfinite(depth[masks]).all() or np.any(depth[masks] <= 0):
        raise ValueError('valid camera depths must be positive and finite')
    da,db = np.where(masks,depth,np.inf)
    near = np.minimum(da,db)
    owners = masks & (np.abs(np.where(masks,depth,0)-np.where(np.isfinite(near),near,0)[None]) < 1e-5)
    return owners.astype(np.float32)/np.maximum(owners.sum(0,keepdims=True),1)


def _bbox(mask):
    y,x = np.nonzero(mask)
    if not len(x):
        raise ValueError('valid geometry has an empty amodal mask')
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],np.float32)/SIZE


def _rotation6d(matrix):
    return np.concatenate([matrix[:,0],matrix[:,1]])


def geometry_vectors(betas, root_axis_angle, root_world, R_cw, t_cw, K_512, amodal, heights):
    rotations = Rotation.from_rotvec(root_axis_angle).as_matrix()
    intrinsics = np.array([K_512[0,0],K_512[1,1],K_512[0,2],K_512[1,2]])/SIZE
    globals_ = np.stack([np.concatenate([betas[i],_rotation6d(rotations[i]),root_world[i],
                                         intrinsics,_rotation6d(R_cw),t_cw]) for i in range(2)])
    boxes = [_bbox(m) for m in amodal]
    centers = [(b[:2]+b[2:])/2 for b in boxes]
    dimensions = [np.maximum(b[2:]-b[:2],1/SIZE) for b in boxes]
    camera_roots = root_world @ R_cw.T + t_cw
    height = max(float(np.mean(heights)),.5)
    relative = []
    for i,j in ((0,1),(1,0)):
        delta = (camera_roots[j]-camera_roots[i])/height
        relative.append(np.concatenate([delta,_rotation6d(rotations[i].T@rotations[j]),
                         centers[j]-centers[i],np.log(dimensions[j]/dimensions[i]),[np.linalg.norm(delta)]]))
    return globals_.astype(np.float32),np.asarray(relative,np.float32)


def body25_heatmaps(xy, score, sigma=5., size=SIZE):
    if xy.shape != (24,2) or score.shape != (24,):
        raise ValueError('DWPose body/feet convention requires 24 points and scores')
    points = np.concatenate([xy,((xy[8]+xy[11])/2)[None]])
    scores = np.concatenate([score,[min(score[8],score[11])]])
    yy,xx = np.mgrid[:size,:size].astype(np.float32)
    heat = np.zeros((25,size,size),np.float32)
    for k,((x,y),confidence) in enumerate(zip(points,scores)):
        if confidence >= .25 and np.isfinite([x,y]).all() and 0<=x<size and 0<=y<size:
            heat[k] = np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2))*float(confidence)
    return heat


def validate_generation_manifest(row):
    unexpected = set(row)-GENERATION_KEYS
    if unexpected:
        raise ValueError(f'forbidden generation fields: {sorted(unexpected)}')
    required = GENERATION_KEYS
    missing = required-set(row)
    if missing:
        raise ValueError(f'missing generation fields: {sorted(missing)}')
    if row['resolution'] != 512 or row['target_geometry_known'] is not True:
        raise ValueError('pilot requires 512 and explicitly known target geometry')
    if len(row['source_masks']) != 2 or len(row['person_uids']) != 2:
        raise ValueError('one source frame with two person slots is required')
    if any(not isinstance(uid, str) or not uid.strip() for uid in row['person_uids']) or len(set(row['person_uids'])) != 2:
        raise ValueError('person_uids must identify two distinct source actors')
    if not isinstance(row['versions'], dict):
        raise ValueError('versions must be an explicit schema/coordinate/cache mapping')
    for name, expected in generation_versions().items():
        if row['versions'].get(name) != expected:
            raise ValueError(f'incompatible or missing versions.{name}')
    sha = row['source_features_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('source_features_sha256 must be the cached PT file SHA256')


def hi4d_asset_identity(path, *, target):
    """Read the documented Hi4D source/target identity from the asset path."""
    parts=Path(path).parent.parts
    if len(parts)<5:
        raise ValueError('supervision requires a Hi4D source/target asset path')
    kind,pair,second,third,frame=parts[-5:]
    camera,action=(third,second) if target else (second,third)
    if kind!=('targets' if target else 'sources') or not pair.startswith('pair') or not re.fullmatch(r'cam\d+',camera) or not frame.isdecimal():
        raise ValueError('supervision requires a Hi4D source/target asset identity')
    return dict(pair=pair,action=action,camera=int(camera[3:]),frame=int(frame))


def validate_supervision_manifest(row, generation):
    """Bind truth to content and to the matching generation geometry/actor slots."""
    if not isinstance(row,dict) or set(row)!=SUPERVISION_KEYS:
        missing=SUPERVISION_KEYS-set(row) if isinstance(row,dict) else SUPERVISION_KEYS
        raise ValueError(f'supervision schema fields are incomplete or unsupported: {sorted(missing)}')
    if row['schema_version']!=SUPERVISION_SCHEMA_VERSION:
        raise ValueError('supervision requires dual_supervision_v1')
    if any(type(row[key]) is not int or row[key]<0 for key in ('source_frame','target_frame','camera')):
        raise ValueError('supervision frame/camera values must be nonnegative integers')
    if row['person_uids']!=generation['person_uids']:
        raise ValueError('supervision person_uids do not match generation slots')
    if Path(row['target_geometry']).resolve()!=Path(generation['target_geometry']).resolve():
        raise ValueError('supervision target_geometry path does not match generation geometry')
    geometry_path=Path(generation['target_geometry'])
    actual_geometry_sha=file_sha256(geometry_path)
    if actual_geometry_sha!=row['target_geometry_sha256']:
        raise ValueError('supervision target_geometry_sha256 does not match actual geometry content')
    sidecar_path=geometry_path.with_suffix('.json')
    actual_sidecar_sha=file_sha256(sidecar_path)
    if actual_sidecar_sha!=row['target_geometry_sidecar_sha256']:
        raise ValueError('supervision target_geometry_sidecar_sha256 does not match actual sidecar content')
    sidecar=json.loads(sidecar_path.read_text())
    if sidecar.get('geometry_sha256')!=actual_geometry_sha:
        raise ValueError('supervision geometry sidecar SHA256 does not match geometry content')
    for key in ('pair','action','target_frame','source_frame','camera','person_uids'):
        if sidecar.get(key)!=row[key]:
            raise ValueError(f'supervision {key} does not match explicitly bound geometry sidecar')
    match=re.fullmatch(r'(.+)_(\d{6})_cam(\d+)_geometry\.npz',geometry_path.name)
    if match is None or (match[1],int(match[2]),int(match[3]))!=(row['action'],row['target_frame'],row['camera']):
        raise ValueError('supervision action/frame/camera does not match geometry asset identity')
    with np.load(geometry_path,allow_pickle=False) as payload:
        validate_geometry_schema(payload,generation['versions'])
        if list(map(str,payload['person_uids']))!=row['person_uids']:
            raise ValueError('supervision person_uids do not match geometry payload slots')
    source=hi4d_asset_identity(generation['source_rgb'],target=False)
    target=hi4d_asset_identity(row['target_rgb'],target=True)
    if (source['pair'],source['camera'],source['frame'])!=(row['pair'],row['camera'],row['source_frame']):
        raise ValueError('supervision pair/source_frame/camera does not match source asset identity')
    if (target['pair'],target['action'],target['camera'],target['frame'])!=(row['pair'],row['action'],row['camera'],row['target_frame']):
        raise ValueError('supervision pair/action/target_frame/camera does not match target asset identity')
    if not isinstance(row['target_masks'],list) or len(row['target_masks'])!=2 or not isinstance(row['target_mask_sha256'],list) or len(row['target_mask_sha256'])!=2:
        raise ValueError('supervision requires two ordered target masks and their SHA256 values')
    target_dir=Path(row['target_rgb']).parent
    for i,path in enumerate(row['target_masks']):
        if Path(path).parent!=target_dir or Path(path).name!=f'gt_mask_{"AB"[i]}.png':
            raise ValueError('supervision target mask paths must preserve target frame and actor slot order')
    actual_rgb_sha=file_sha256(row['target_rgb'])
    actual_mask_sha=[file_sha256(path)for path in row['target_masks']]
    if actual_rgb_sha!=row['target_rgb_sha256'] or actual_mask_sha!=row['target_mask_sha256']:
        raise ValueError('supervision target RGB/mask SHA256 does not match actual image content')
    return dict(row,target_rgb_sha256=actual_rgb_sha,target_mask_sha256=actual_mask_sha,
        target_geometry_sha256=actual_geometry_sha,target_geometry_sidecar_sha256=actual_sidecar_sha)


def bind_supervision_manifest(row, generation):
    """Create a reviewable truth manifest from actual files and bound geometry."""
    bound=dict(row,schema_version=SUPERVISION_SCHEMA_VERSION,
        person_uids=list(generation['person_uids']),target_geometry=generation['target_geometry'],
        target_rgb_sha256=file_sha256(row['target_rgb']),
        target_mask_sha256=[file_sha256(path)for path in row['target_masks']],
        target_geometry_sha256=file_sha256(generation['target_geometry']),
        target_geometry_sidecar_sha256=file_sha256(Path(generation['target_geometry']).with_suffix('.json')))
    validate_supervision_manifest(bound,generation)
    return bound


@dataclass(frozen=True)
class TrainingTarget:
    rgb: torch.Tensor
    person_masks: torch.Tensor | None
    metadata: dict


class Hi4DSMPLXDataset:
    def __init__(self, generation_manifest, supervision_manifest=None):
        self.path = Path(generation_manifest)
        self.rows = json.loads(self.path.read_text())
        if isinstance(self.rows,dict): self.rows = [self.rows]
        for row in self.rows: validate_generation_manifest(row)
        self.supervision_manifest = None if supervision_manifest is None else Path(supervision_manifest)

    def __len__(self): return len(self.rows)

    def generation(self,index=0,*,prompt=None,seed=None):
        from src.pose_control.qwen21_full.dual.conditions import SourceCondition,GeometryCondition,DualGenerationInput
        row = self.rows[index]
        if file_sha256(row['source_features']) != row['source_features_sha256']:
            raise ValueError('source feature cache SHA256 does not match generation manifest')
        features = torch.load(row['source_features'],map_location='cpu',weights_only=True)
        validate_source_feature_cache(features, row)
        uids = (tuple(row['person_uids']),)
        source = SourceCondition(rgb=read_rgb(row['source_rgb'])[None],
            masks=torch.stack([read_mask(p) for p in row['source_masks']])[None],person_uids=uids,
            person_latents=features['person_latents'][None],latent_masks=features['latent_masks'][None],
            dino_patches=features['dino_patches'][None],dino_valid=features['dino_valid'][None],
            version=row['versions']['source'])
        with np.load(row['target_geometry'],allow_pickle=False) as npz:
            validate_geometry_schema(npz, row['versions'])
            values = {k:torch.from_numpy(npz[k].copy())[None] for k in
                ('normal','depth','part','pose','amodal_mask','visible_owner','global32','relative14',
                 'part_positions','part_orientations','part_joints','part_valid','part_confidence','contact_endpoints','contact_features',
                 'contact_valid','contact_map','person_valid')}
            geometry_uids = (tuple(str(x) for x in npz['person_uids']),)
        geometry = GeometryCondition(**values,person_uids=geometry_uids,
                                     geometry_hash=file_sha256(row['target_geometry']),
                                     version=row['versions']['geometry'])
        generated = DualGenerationInput(source=source,geometry=geometry,
            prompts=(row['prompt'] if prompt is None else prompt,),seeds=(row['seed'] if seed is None else seed,))
        generated.validate()
        return generated

    def supervision(self,index=0):
        if self.supervision_manifest is None:
            raise ValueError('supervision requires its separate manifest')
        rows = json.loads(self.supervision_manifest.read_text())
        rows=rows if isinstance(rows,list) else [rows]
        if len(rows)!=len(self.rows):
            raise ValueError('supervision row count must match generation rows')
        metadata=validate_supervision_manifest(rows[index],self.rows[index])
        metadata['supervision_manifest_sha256']=file_sha256(self.supervision_manifest)
        masks=torch.stack([read_mask(p)for p in metadata['target_masks']])[None]
        return TrainingTarget(read_rgb(metadata['target_rgb'])[None],masks,metadata)


class SMPLXRenderer:
    """Independent amodal rasters in camera coordinates; no RGB/GT-mask reads."""
    def __init__(self,faces,labels,device='cuda'):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.device = torch.device(device)
        self.context = dr.RasterizeCudaContext(device=self.device)
        self.faces = torch.as_tensor(faces,dtype=torch.int32,device=self.device)
        self.labels = torch.as_tensor(labels,dtype=torch.long,device=self.device)

    def render(self,vertices,R_cw,t_cw,K_512,size=512):
        from torch.nn import functional as F
        v = torch.as_tensor(vertices,dtype=torch.float32,device=self.device)
        R = torch.as_tensor(R_cw,dtype=v.dtype,device=v.device)
        t = torch.as_tensor(t_cw,dtype=v.dtype,device=v.device)
        K = torch.as_tensor(K_512,dtype=v.dtype,device=v.device)
        camera = v@R.T+t
        z = camera[:,2]
        if bool((z<=.1).any()): raise ValueError('mesh intersects near plane')
        projection = camera@K.T
        xy = projection[:,:2]/z[:,None]
        a,b = (20+.1)/(20-.1),-2*20*.1/(20-.1)
        clip = torch.stack(((2*xy[:,0]/size-1)*z,(1-2*xy[:,1]/size)*z,a*z+b,z),-1)[None]
        raster,_ = self.dr.rasterize(self.context,clip,self.faces,resolution=(size,size))
        tri = self.faces.long()
        face_normal = torch.cross(camera[tri[:,1]]-camera[tri[:,0]],camera[tri[:,2]]-camera[tri[:,0]],dim=-1)
        normals = torch.zeros_like(camera)
        for k in range(3): normals.index_add_(0,tri[:,k],face_normal)
        normals = F.normalize(normals,dim=-1)
        attributes = torch.cat((normals,z[:,None],F.one_hot(self.labels,14).float()),-1)[None]
        interpolated,_ = self.dr.interpolate(attributes,raster,self.faces)
        # nvdiffrast raster rows use bottom-left origin; image/camera pixels use top-left.
        values = interpolated[0].flip(0)
        mask = (raster[0,:,:,3]>0).flip(0)
        normal = F.normalize(values[:,:,:3],dim=-1)*mask[:,:,None]
        part = F.one_hot(values[:,:,4:].argmax(-1),14).float()*mask[:,:,None]
        return {'normal':normal.permute(2,0,1).cpu().numpy(),
                'depth':(values[:,:,3]*mask).cpu().numpy(),
                'part':part.permute(2,0,1).cpu().numpy(),
                'mask':mask.cpu().numpy(), 'vertex_normals_camera':normals.cpu().numpy()}

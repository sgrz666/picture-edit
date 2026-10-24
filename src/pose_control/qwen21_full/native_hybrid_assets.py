"""Hi4D original-image observations; no estimator is imported on the reader path.

The legacy A/B filename is provenance, never identity. Assignment is injective and
uses official masks AND named body anchors from an already verified V3 record.
"""
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .native_full_assets import sha256, read_record, write_record

ASSET_SCHEMA = 'native_hi4d_hybrid_asset_v1'
OBS_SCHEMA = 'native_hi4d_observation_index_v1'
CONFIG_SCHEMA = 'native_hi4d_hybrid_config_v1'
LAYOUT_134 = 'dwpose134'
LAYOUT_133 = 'coco_wholebody133'
# DWPose index -> pelvis-inclusive Body21 index, by anatomical name.
ANCHORS = {2:17, 3:19, 4:21, 5:16, 6:18, 7:20, 8:2, 9:5,
           10:8, 11:1, 12:4, 13:7, 1:12}


@dataclass(frozen=True)
class HybridConfig:
    schema: str = CONFIG_SCHEMA
    defaults_status: str = 'engineering_defaults_not_accepted'
    point_score: float = .3
    min_body_anchors: int = 4
    assignment_score: float = .5
    assignment_margin: float = .15
    min_hand_points: int = 6
    min_face_points: int = 6
    wrist_min_px: float = 16.
    wrist_bbox_ratio: float = .1
    roi_expand: float = .1
    expression_mode: str = 'source'

    def __post_init__(self):
        if (self.schema != CONFIG_SCHEMA or self.expression_mode not in ('source','driver','off')
                or self.defaults_status!='engineering_defaults_not_accepted'):
            raise ValueError('Explicit Hybrid config / expression mode required')
        for name in ('point_score','assignment_score','assignment_margin','wrist_bbox_ratio','roi_expand'):
            if not np.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) <= 1:
                raise ValueError('Invalid Hybrid threshold: '+name)
        if self.point_score<=0 or self.assignment_score<=0 or any(type(getattr(self,n))!=int for n in ('min_body_anchors','min_hand_points','min_face_points')):
            raise ValueError('Positive point/matching thresholds and integer point counts required')
        if not 4 <= self.min_body_anchors <= len(ANCHORS) or not 6 <= self.min_hand_points <= 21 or not 1 <= self.min_face_points <= 68:
            raise ValueError('Invalid reliable-point count')
        if not np.isfinite(self.wrist_min_px) or self.wrist_min_px <= 0:
            raise ValueError('Invalid wrist pixel threshold')

    def record(self): return asdict(self)


def file_ref(path):
    path = Path(path).resolve()
    return {'path':str(path), 'sha256':sha256(path)}


def verified(record):
    path = Path(record['path'])
    if sha256(path) != record['sha256']: raise ValueError('Hybrid provenance SHA differs: '+str(path))
    return path


def frame_key(row):
    """Camera is part of the join even when all frame names happen to be unique."""
    return (str(row['pair']), str(row['action']), str(row['camera_id']), str(int(row['frame_id'])))


def geometry_key(meta):
    # V3 stores explicit pair/action/frame and a namespaced camera identifier.
    camera = str(meta['camera_id']).split(':')[-1]
    return frame_key({**meta, 'camera_id':camera})


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def canonical_points(xy, scores, layout):
    xy = np.asarray(xy, np.float32); scores = np.asarray(scores, np.float32)
    n = {LAYOUT_134:134, LAYOUT_133:133}.get(layout)
    if n is None or xy.shape != (n,2) or scores.shape != (n,):
        raise ValueError('Explicit 133/134 layout must match native coordinates and scores')
    if layout == LAYOUT_133:
        from src.pose_edit.geometry import cocowholebody133_to_dwpose134
        # Existing converter supplies the exact anatomy ordering and neck XY.
        xy = cocowholebody133_to_dwpose134(xy)
        order = [0,1,6,8,10,5,7,9,12,14,16,11,13,15,2,4,1,3]
        body = scores[order].copy(); body[1] = min(scores[5], scores[6])
        scores = np.concatenate((body, scores[17:]))
    finite = np.isfinite(xy).all(-1) & np.isfinite(scores)
    if ((scores[finite] < 0) | (scores[finite] > 1)).any(): raise ValueError('Detection scores must be in [0,1]')
    return np.where(np.isfinite(xy),xy,0).astype(np.float32), np.where(finite,scores,0).astype(np.float32), finite


def mask_array(ref, hw):
    with Image.open(verified(ref)) as im:
        if (im.height,im.width) != tuple(hw): raise ValueError('Official mask is not on the original RGB grid')
        return np.asarray(im.convert('L')).copy() > 0


def body_anchors(data, meta):
    """V3 stores ORIGINAL-image K; asset_to_original describes raster sampling only."""
    joints = np.asarray(data['joints_cam'],np.float64)
    k = np.asarray(data['K'],np.float64)
    image = joints @ k.T
    good = np.isfinite(image).all(-1) & (joints[...,2] > 1e-8)
    uv = image[...,:2]/np.where(good,image[...,2],1)[...,None]
    return uv, good


def assign_people(candidates, masks, anchors, anchor_valid, config):
    """Global one-to-one matching. A losing/ambiguous slot stays unavailable."""
    count = len(candidates); score = np.full((2,count),-1.,np.float64)
    for p in range(2):
        ys,xs = np.nonzero(masks[p]); height = float(ys.max()-ys.min()+1) if len(ys) else 0.
        if height == 0: continue
        tolerance = max(config.wrist_min_px,config.wrist_bbox_ratio*height)
        for d, (xy,confidence,finite) in enumerate(candidates):
            pairs = [(i,j) for i,j in ANCHORS.items() if finite[i] and confidence[i]>=config.point_score and anchor_valid[p,j]]
            if len(pairs)<config.min_body_anchors: continue
            points = np.array([xy[i] for i,j in pairs]); expected = np.array([anchors[p,j] for i,j in pairs])
            xi = np.floor(points[:,0]).astype(int); yi = np.floor(points[:,1]).astype(int)
            inside = (xi>=0)&(xi<masks.shape[2])&(yi>=0)&(yi<masks.shape[1])
            membership = np.zeros(len(pairs),bool); membership[inside] = masks[p,yi[inside],xi[inside]]
            anchor_score = np.exp(-.5*(np.linalg.norm(points-expected,axis=-1)/tolerance)**2).mean()
            score[p,d] = .5*membership.mean()+.5*anchor_score
    choices = []
    for a,b in product(range(-1,count),repeat=2):
        if a>=0 and a==b: continue
        if any(d>=0 and score[p,d]<config.assignment_score for p,d in enumerate((a,b))): continue
        choices.append((sum(score[p,d] if d>=0 else 0. for p,d in enumerate((a,b))),(a,b)))
    choices.sort(reverse=True)
    best = choices[0] if choices else (0.,(-1,-1)); result=[]
    for p,d in enumerate(best[1]):
        alternatives = [v for v,ds in choices if ds[p]!=d]
        gap = best[0]-max(alternatives) if alternatives else 1.
        # Identical duplicate caches of one physical detection must not allow reuse.
        duplicate = d>=0 and any(q!=d and np.allclose(candidates[d][0],candidates[q][0],atol=1e-4) for q in range(count))
        ok = d>=0 and gap>=config.assignment_margin and not duplicate
        ambiguous=duplicate or (gap<config.assignment_margin and (score[p]>=config.assignment_score).any())
        result.append({'candidate':d if ok else -1,'status':'matched' if ok else ('ambiguous' if ambiguous else 'missing'),
                       'score':float(score[p,d]) if d>=0 else 0.,'margin':float(gap),
                       'reason':None if ok else ('duplicate_detection' if duplicate else 'assignment_missing_or_ambiguous')})
    return result, score.tolist()


def observation_asset(data, meta, masks, candidates, provenance, config, *, rejected_reason=None):
    xy = np.zeros((2,134,2),np.float32); scores = np.zeros((2,134),np.float32); valid = np.zeros((2,134),bool)
    anchors,av = body_anchors(data,meta)
    assignment,matrix = assign_people(candidates,masks,anchors,av,config) if rejected_reason is None else (
        [{'candidate':-1,'status':'rejected','score':0.,'margin':0.,'reason':rejected_reason} for _ in range(2)],[])
    for p,item in enumerate(assignment):
        if item['candidate']>=0:
            xy[p],scores[p],finite = candidates[item['candidate']]
            valid[p] = finite & (scores[p]>=config.point_score)
    return {'xy_native':xy,'scores':scores,'point_valid':valid,
        'person_assigned':np.array([x['candidate']>=0 for x in assignment],bool),
        'metadata':{'schema':ASSET_SCHEMA,'key':list(geometry_key(meta)),'original_hw':meta['original_hw'],
            'person_ids':meta['person_ids'],'camera_id':meta['camera_id'],'scale_group_id':meta['scale_group_id'],
            'geometry_sha256':meta['record_sha256'],'layout':LAYOUT_134,'coordinate_space':'original_pixels',
            'assignment':assignment,'assignment_matrix':matrix,'config':config.record(),'provenance':provenance}}


def _import_frame(dataset, record, generation_rows, generation_ref, layout, config):
    """No RGB pixels are opened; the RGB/mask/camera bytes are SHA-bound."""
    data,meta = dataset.geometry_record(record)
    if meta['model_family_indices'] != [1,1]: raise ValueError('Hybrid Hi4D requires two original SMPL models')
    key = geometry_key(meta); rows = generation_rows.get(key,[])
    reason = None; candidates=[]; provenance={'generation_manifest':generation_ref,'geometry':record}
    masks = np.zeros((2,*meta['original_hw']),bool)
    if len(rows)!=1: reason='missing_or_ambiguous_generation_row'
    else:
        row=rows[0]; provenance['generation_row']=row
        # These declarations must be in the original generation manifest, never inferred from filenames.
        required=('rgb','camera_file','person_mask_A','person_mask_B','dwpose_native_height','dwpose_native_width','dwpose_pre_crop','dwpose_pre_resize')
        if any(k not in row for k in required): reason='incomplete_legacy_source_chain'
        elif row['dwpose_pre_crop'] or row['dwpose_pre_resize']: reason='unresolved_legacy_coordinate_transform'
        elif [row['dwpose_native_height'],row['dwpose_native_width']]!=list(meta['original_hw']): reason='native_grid_conflict'
        elif file_ref(row['rgb'])['sha256']!=meta['rgb']['sha256']: reason='rgb_source_conflict'
        elif file_ref(row['camera_file'])['sha256']!=meta['sources']['camera']['sha256']: reason='camera_source_conflict'
        elif (row.get('person_A'),row.get('person_B'))!=(0,1): reason='unresolved_official_subject_mask_order'
        else:
            provenance.update(rgb=file_ref(row['rgb']),camera=file_ref(row['camera_file']),
                masks=[file_ref(row['person_mask_A']),file_ref(row['person_mask_B'])],
                source_status='legacy_manifest_bound_reassociated_not_original_detector_hash')
            masks=np.stack([mask_array(r,meta['original_hw']) for r in provenance['masks']])
            provenance['detections']=[];provenance['candidate_failures']=[]
            for field in ('dwpose_A','dwpose_B'):
                if not row.get(field):
                    provenance['candidate_failures'].append({'field':field,'reason':'not_declared'});continue
                try:
                    ref=file_ref(row[field]);provenance['detections'].append(ref)
                    with np.load(verified(ref),allow_pickle=False) as f:
                        if 'xy_native' not in f or 'scores' not in f: raise ValueError('missing_native_xy_or_scores')
                        candidates.append(canonical_points(f['xy_native'],f['scores'],layout))
                except (FileNotFoundError,ValueError) as error:
                    # PermissionError deliberately propagates; never bypass access denials.
                    provenance['candidate_failures'].append({'field':field,'reason':str(error)})
    return observation_asset(data,meta,masks,candidates,provenance,config,rejected_reason=reason)


def import_frame(dataset, record, generation_rows, generation_ref, layout, config):
    # Validate the Body outside the recoverable legacy-observation operation.
    data,meta=dataset.geometry_record(record)
    try:
        return _import_frame(dataset,record,generation_rows,generation_ref,layout,config)
    except FileNotFoundError as error:
        return observation_asset(data,meta,np.zeros((2,*meta['original_hw']),bool),[],
            {'generation_manifest':generation_ref,'geometry':record,
             'generation_row':(generation_rows.get(geometry_key(meta)) or [None])[0]},
            config,rejected_reason='missing_legacy_source_file: '+str(error))


def read_observation(row, *, geometry_sha=None):
    if row.get('schema')!=OBS_SCHEMA: raise ValueError('Explicit Hybrid observation index required')
    asset=read_record(row['asset'],schema=ASSET_SCHEMA);m=asset['metadata']
    if list(row['key'])!=m['key'] or (geometry_sha is not None and m['geometry_sha256']!=geometry_sha):
        raise ValueError('Observation frame/geometry source conflicts')
    for name,shape,dtype in (('xy_native',(2,134,2),np.float32),('scores',(2,134),np.float32),
                            ('point_valid',(2,134),np.bool_),('person_assigned',(2,),np.bool_)):
        if asset[name].shape!=shape or asset[name].dtype!=dtype or not np.isfinite(asset[name]).all():
            raise ValueError('Invalid native observation '+name)
    if m.get('coordinate_space')!='original_pixels' or m.get('layout')!=LAYOUT_134: raise ValueError('Canonical original points required')
    if ((asset['scores']<0)|(asset['scores']>1)).any() or (asset['point_valid'] & ~asset['person_assigned'][:,None]).any():
        raise ValueError('Observation confidence/assignment validity conflicts')
    if len(m.get('person_ids',[]))!=2 or len(set(m['person_ids']))!=2: raise ValueError('Stable distinct Hi4D person IDs required')
    provenance=m['provenance']
    for key in ('generation_manifest','rgb','camera'):
        if provenance.get(key): verified(provenance[key])
    for key in ('masks','detections'):
        for ref in provenance.get(key,[]): verified(ref)
    for ref in provenance.get('detector_models',{}).values(): verified(ref)
    return asset


def detect_frame(dataset, record, prior, detector_config, config):
    """Explicit opt-in only; detector uses original RGB, never target appearance features."""
    from src.pose_edit.dwpose import detect_people, load_sessions
    import cv2
    data,meta=dataset.geometry_record(record);provenance=prior['metadata']['provenance']
    row=provenance.get('generation_row')
    if row is None or geometry_key(meta)!=frame_key(row): raise ValueError('Supplemental detection needs explicit original frame/mask/camera provenance')
    if file_ref(row['rgb'])['sha256']!=meta['rgb']['sha256'] or file_ref(row['camera_file'])['sha256']!=meta['sources']['camera']['sha256']:
        raise ValueError('Detection RGB/camera conflicts with V3')
    if (row.get('person_A'),row.get('person_B'))!=(0,1): raise ValueError('Official mask subject ordering required')
    mask_refs=[file_ref(row['person_mask_A']),file_ref(row['person_mask_B'])]
    masks=np.stack([mask_array(r,meta['original_hw']) for r in mask_refs])
    model_refs={name:file_ref(detector_config[name]) for name in ('dwpose_det_model','dwpose_pose_model')}
    image=cv2.imread(str(verified(meta['rgb'])))
    if image is None or tuple(image.shape[:2])!=tuple(meta['original_hw']): raise ValueError('Native detection RGB dimensions differ')
    detected=detect_people(detector_config,image,sessions=load_sessions(detector_config))
    candidates=[canonical_points(d['points'],d['scores'],LAYOUT_134) for d in detected]
    return observation_asset(data,meta,masks,candidates,{**provenance,'masks':mask_refs,'rgb':meta['rgb'],
        'camera':file_ref(row['camera_file']),'detector_models':model_refs,'source_status':'direct_native_detection',
        'prior_status':prior['metadata']['assignment']},config)

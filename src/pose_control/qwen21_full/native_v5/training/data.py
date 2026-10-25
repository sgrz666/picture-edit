"""Eligibility is route-aware; missing modalities remain missing, rows stay recorded."""
from dataclasses import asdict,replace
import torch
from torch.nn import functional as F
from ..contracts import NativeControlOptionsV5
from ..detail import inner_feather
from ..data import NativeFullDatasetV5
from ..local import gather_source
from ...native_dual_depth import DualControlOptions
from ...native_local_condition import RIGID_FACE_SLOTS
from ...native_local_control import resolve_local_ownership
from .io import verify,read_json_ref
from .config import digest

def route_options(route,kind):
    raw={}
    if route.branch=='body':raw.update(face=False,hand=False,detail=False,interaction=False)
    elif route.branch=='face':raw.update(hand=False,detail=False,interaction=False)
    elif route.branch=='hand':raw.update(face=False,detail=False,interaction=False)
    elif route.branch=='detail':raw.update(interaction=False,face_identity_gain=0.,face_texture_gain=0.,face_geometry_gain=0.,hand_geometry_gain=0.,hand_texture_gain=0.)
    elif route.branch=='interaction':raw.update(face=False,hand=False,detail=False)
    raw.update(route.controls)
    if kind=='single':raw['interaction']=False
    options=NativeControlOptionsV5(**raw)
    mode=route.base_controls.get('depth_mode','full_geometry');physical=mode=='full_geometry'
    base={'depth_mode':mode,'detail':physical,'interaction':options.interaction and kind=='dual',
        'raster':options.interaction and kind=='dual' and physical,'part_contact':options.interaction and kind=='dual' and physical}
    base.update(route.base_controls)
    return options,DualControlOptions(**base)

def source_detail_support(refs,sample):
    c=sample['local_condition_v4'];sh,sw=sample['source_token_hw'];out=torch.zeros(6,dtype=torch.bool)
    yy,xx=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij');unit=torch.stack((xx,yy),-1)
    for r,p in enumerate((0,1,0,0,1,1)):
        if not c.person_valid[0,p]:continue
        src=int(c.source_indices[0,p]);sr=src if r<2 else 2+2*src+(r-2)%2
        if not refs.valid[0,sr]:continue
        box=refs.boxes[0,sr];grid=(box[:2]+unit*(box[2:]-box[:2]))[None]*2-1
        support=F.interpolate(refs.support[0:1,sr:sr+1].float(),size=(sh,sw),mode='area')
        out[r]=(F.grid_sample(support,grid,align_corners=False)>1e-6).any()
    return out

def sample_status(sample,route,config):
    c=sample['local_condition_v4'];s=sample['training_supervision_v5'];r=sample['local_references_v4']
    options,base=route_options(route,sample['native_scene_kind']);reasons=[]
    if not s.paired.all():reasons.append('unpaired_target_rgb')
    if sample['native_scene_kind']=='dual' and base.depth_mode=='full_geometry' and any(m['mode']!='full_geometry' for m in sample['dual_depth'].metadata):reasons.append('explicit_depth_mode_mismatch')
    if base.identity:
        source_valid=sample['source_person_mask'].flatten(1).any(1)
        if not source_valid[c.source_indices[0,c.person_valid[0]]].all():reasons.append('body_identity_source_missing')
    weights=inner_feather(resolve_local_ownership(c,face=options.face,hand=options.hand),c.boxes,config.eligibility_feather_px)
    cells=(F.avg_pool2d(weights,16).flatten(2)>0).sum(-1)[0]
    pv=c.face_point_valid.clone();parameters=c.jaw_valid|c.expression_valid
    for p in range(2):
        meta=c.metadata[0]
        if meta['expression_mode']=='off' or (meta['expression_mode']=='source' and not meta.get('expression_retargeted',[False,False])[p]):
            rigid=torch.zeros(72,dtype=torch.bool);rigid[list(RIGID_FACE_SLOTS)]=True;pv[:,p]&=rigid;parameters[:,p]=False
    face_geo=(pv.any(-1)|parameters)&c.face_roi_valid
    av=gather_source(r.arcface_valid,c).any(-1)&c.face_roi_valid
    tv=gather_source(r.face_dino_valid,c).any(-1)&c.face_roi_valid
    hv=gather_source(r.hand_dino_valid,c).any(-1)&c.hand_roi_valid
    hand_geo=(c.hand_point_valid.any(-1)|c.hand_pose_valid|c.mesh_valid)&c.hand_roi_valid
    large=cells>=config.eligibility_min_spatial_cells;visible=cells>0
    face_usable=((av[0]&visible[:2]&(options.strength('face_identity')>0))|
        (tv[0]&visible[:2]&(options.strength('face_texture')>0))|(face_geo[0]&large[:2]&(options.strength('face_geometry')>0)))
    hand_usable=((hv[0].flatten()&visible[2:]&(options.strength('hand_texture')>0))|
        (hand_geo[0].flatten()&large[2:]&(options.strength('hand_geometry')>0)))
    detail_usable=source_detail_support(sample['detail_references_v5'],sample)&large&c.region_valid[0]
    detail_usable &= torch.tensor([options.face and options.face_strength>0]*2+[options.hand and options.hand_strength>0]*4)
    detail_usable &= options.strength('detail')>0
    interaction=sample['native_scene_kind']=='dual' and options.interaction and base.interaction and options.interaction_strength>0
    if interaction and not sample['dual_depth'].projected.any():interaction=False
    branch_ok={'body':options.body_strength>0,'face':bool(face_usable.any()),'hand':bool(hand_usable.any()),
        'detail':bool(detail_usable.any()),'interaction':bool(interaction)}
    branch_ok['joint']=any(branch_ok.values())
    if not branch_ok[route.branch]:reasons.append('route_has_no_usable_inputs')
    return {'eligible':not reasons,'reasons':reasons,'branch_usable':branch_ok,'cells':cells.tolist(),
        'effective_regions':{'face':face_usable.tolist(),'hand':hand_usable.tolist(),'detail':detail_usable.tolist()},
        'face_area':float(s.weights[:,:2].sum()),'hand_area':float(s.weights[:,2:].sum()),
        'auxiliary_faces':int(s.auxiliary_valid.sum()),'null_hand_context_only':bool(hand_geo.any() and not hv.any()),
        'present':{'face_geometry':face_geo.tolist(),'hand_geometry':hand_geo.tolist(),'arcface':av.tolist(),'face_dino':tv.tolist(),'hand_dino':hv.tolist()},
        'capabilities':{'jaw':c.jaw_valid.tolist(),'expression':c.expression_valid.tolist(),'hand_pose':c.hand_pose_valid.tolist(),
            'bps':c.mesh_valid.tolist(),'hand_2d':c.hand_point_valid.any(-1).tolist(),'face_2d':c.face_point_valid.any(-1).tolist()},
        'bucket':[list(sample['target_token_hw']),list(sample['source_token_hw']),sample['native_scene_kind'],sample['data_profile'],r.arcface.shape[2]]}

class DataPlan:
    def __init__(self,config):
        self.config=config;self.datasets={};self.group_indices={};self.bindings={};self.status={};self.records=[]
        split_groups={'train':set(),'validation':set()};identities={}
        for spec in config.sources:
            manifest=verify(spec.manifest);groups,bound=read_json_ref(spec.groups)
            dataset=NativeFullDatasetV5(manifest['path'],load_target=True,data_profile=spec.data_profile)
            ids={r['pair_id'] for r in dataset.rows}
            if set(groups)!=ids or any(not isinstance(v,str) or not v for v in groups.values()):raise ValueError('Explicit original-video/sequence group for every pair required')
            split_groups[spec.role]|=set(groups.values())
            self.datasets[spec.source_id]=dataset;self.group_indices[spec.source_id]=groups
            closure=[]
            for row in dataset.rows:
                closure.append({k:row.get(k) for k in ('pair_id','v4_manifest','detail','supervision','references')})
            identities[spec.source_id]={'manifest':manifest,'groups':bound,'closure':closure,'role':spec.role}
            for index in range(len(dataset)):
                sample=dataset[index]
                if spec.scene_kind!='any' and spec.scene_kind!=sample['native_scene_kind']:raise ValueError('Source scene kind assertion differs')
                for route in config.stage.routes:
                    status=sample_status(sample,route,config)
                    if spec.role=='train' and route.source_ids and spec.source_id not in route.source_ids:status={**status,'eligible':False,'reasons':status['reasons']+['source_not_selected_for_route']}
                    self.status[(route.name,spec.source_id,index)]=status
                    self.records.append({'source_id':spec.source_id,'role':spec.role,'index':index,'pair_id':dataset.rows[index]['pair_id'],'route':route.name,**status})
        if split_groups['train']&split_groups['validation']:raise ValueError('Original-video/sequence leakage between train and validation')
        self.identity={'sources':identities,'config_sha256':config.signature()};self.signature=digest(self.identity)
        self.pools={}
        for route in config.stage.routes:
            sources=[s for s in config.sources if s.role=='train' and (not route.source_ids or s.source_id in route.source_ids)]
            if not sources:raise ValueError('Route has no selected training sources: '+route.name)
            for s in sources:
                pool=[i for i in range(len(self.datasets[s.source_id])) if self.status[(route.name,s.source_id,i)]['eligible']]
                if not pool:raise ValueError('Positive-weight route/source pool is empty: '+route.name+'/'+s.source_id)
                self.pools[(route.name,s.source_id)]=pool
            selected=[self.status[(route.name,s.source_id,i)] for s in sources for i in self.pools[(route.name,s.source_id)]]
            for term,field in (('face','face_area'),('hand','hand_area'),('identity','auxiliary_faces'),('perceptual','auxiliary_faces')):
                if getattr(route.loss,term)>0 and not any(row[field]>0 for row in selected):raise ValueError('Requested loss has no eligible data: '+route.name+'/'+term)
            if not any(row['role']=='validation' and row['route']==route.name and row['eligible'] for row in self.records):raise ValueError('Route has no eligible held-out validation samples: '+route.name)
    def sample(self,source,index):return self.datasets[source][index]
    def report(self):return {'schema':'native_v5_data_preflight_v1','level':'data','passed':True,'signature':self.signature,'identity':self.identity,
        'rows':self.records,'all_rows_retained':True,'model_constructed':False,'gradient_passed':False}

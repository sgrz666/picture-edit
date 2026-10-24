"""V5 manifest sidecars reuse verified V4 records; inference never opens target RGB."""
from dataclasses import fields,replace
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from src.data.canvas_contract import CanvasSpec
from ..native_full_data import NativeFullDatasetV4,collate_native_full_v4
from ..native_full_assets import sha256,read_record,write_record
from ..native_local_condition import REFERENCE_SCHEMA
from ..native_local_control import resolve_local_ownership
from .contracts import NativeDetailReferencesV5,NativeTrainingSupervisionV5,DETAIL_SCHEMA,SUPERVISION_SCHEMA,valid_five_points
from .detail import inner_feather

PAIR_SCHEMA='native_full_pair_v5'


def verified(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or sha256(path)!=ref['sha256']: raise ValueError('V5 asset requires absolute path and exact SHA')
    return path


def read_sidecar(ref,cls,schema): return cls(**read_record(ref,schema=schema)).validate()
def write_sidecar(path,value,schema):
    value.validate()
    return write_record(path,{f.name:getattr(value,f.name) for f in fields(value)},schema=schema)


def validate_references(refs,local):
    refs.validate()
    if len(refs.metadata)!=len(local.metadata): raise ValueError('Reference/local batch differs')
    for i,(r,c) in enumerate(zip(refs.metadata,local.metadata)):
        count=refs.arcface.shape[2]
        if r.get('reference_count')!=count or len(r.get('reference_sources',[]))!=count: raise ValueError('Declare every offline Source reference')
        if count>1 and not r.get('explicit_multi_reference'): raise ValueError('R>1 requires explicit multi-reference declaration')
        for key in ('source_rgb_sha256','source_canvas_id','source_person_ids'):
            if r.get(key)!=c.get(key): raise ValueError('Primary Source reference differs: '+key)
        for source in r['reference_sources']:
            if source['person_ids']!=c['source_person_ids']: raise ValueError('Reference identities/order differ')
            if not source.get('geometry_sha256') or not source.get('canvas'): raise ValueError('Each reference needs geometry and its own Canvas')
            spec=CanvasSpec.from_record(source['canvas'])
            if spec.fingerprint()!=source['canvas_id']: raise ValueError('Offline reference Canvas fingerprint differs')
            verified(source['rgb'])
        for source in r['reference_sources'][1:]:
            if source['rgb']['sha256']==c.get('driver_rgb_sha256'):
                raise ValueError('Auxiliary references cannot use the supervision target RGB')
        if r['reference_sources'][0]['rgb']['sha256']!=c['source_rgb_sha256'] or r['reference_sources'][0]['canvas_id']!=c['source_canvas_id']:
            raise ValueError('First offline reference must be the actual Qwen primary Source')
        if c.get('source_asset_sha256') and r['reference_sources'][0]['geometry_sha256']!=c['source_asset_sha256']:
            raise ValueError('Primary local references and body Source geometry differ')
    return refs


class NativeFullDatasetV5(torch.utils.data.Dataset):
    def __init__(self,manifest,*,load_target=False,data_profile='auto'):
        self.path=Path(manifest);self.load_target=load_target
        self.rows=[json.loads(s) for s in self.path.read_text(encoding='utf8').splitlines() if s.strip()]
        self.readers={};self.indices={}
        ids=[r.get('pair_id') for r in self.rows]
        if not ids or len(set(ids))!=len(ids): raise ValueError('Unique V5 pair IDs required')
        for r in self.rows:
            if r.get('schema')!=PAIR_SCHEMA or 'detail' not in r or 'supervision' not in r: raise ValueError('Explicit V5 sidecars required')
            path=str(verified(r['v4_manifest']))
            if path not in self.readers:
                reader=NativeFullDatasetV4(path,load_target=load_target,data_profile=data_profile)
                self.readers[path]=reader;self.indices[path]={x['pair_id']:i for i,x in enumerate(reader.rows)}
            if r['pair_id'] not in self.indices[path]: raise ValueError('V5 pair absent from bound V4 manifest')

    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];path=str(verified(row['v4_manifest']))
        sample=self.readers[path][self.indices[path][row['pair_id']]]
        if row.get('references') is not None: sample['local_references_v4']=read_record(row['references'],schema=REFERENCE_SCHEMA).validate()
        validate_references(sample['local_references_v4'],sample['local_condition_v4'])
        detail=read_sidecar(row['detail'],NativeDetailReferencesV5,DETAIL_SCHEMA)
        detail=replace(detail,metadata=tuple({**m,'container_sha256':row['detail']['sha256']} for m in detail.metadata))
        c=sample['local_condition_v4'];m=detail.metadata[0]
        for k in ('source_canvas_id','source_rgb_sha256','source_person_ids'):
            if m.get(k)!=c.metadata[0].get(k): raise ValueError('Detail source differs: '+k)
        if not torch.equal(detail.source_indices,c.source_indices): raise ValueError('Detail identity mapping differs')
        sample['detail_references_v5']=detail
        if self.load_target:
            supervision=read_sidecar(row['supervision'],NativeTrainingSupervisionV5,SUPERVISION_SCHEMA)
            if supervision.weights.shape!=c.region_masks.shape: raise ValueError('Training masks must match this exact local batch/canvas')
            for key,value in (('target_canvas_id',c.metadata[0]['target_canvas_id']),('driver_rgb_sha256',c.metadata[0]['driver_rgb_sha256']),
                ('driver_record_sha256',c.metadata[0]['driver_record_sha256']),('target_person_ids',c.metadata[0]['target_person_ids'])):
                if supervision.metadata[0].get(key)!=value: raise ValueError('Training supervision source differs')
            if not supervision.paired.all(): raise ValueError('V5 training requires paired RGB; no cross-person supervision')
            sample['training_supervision_v5']=supervision
        return sample


collate_native_full_v5=collate_native_full_v4


def source_detail_sidecar(sample,declaration):
    c=sample['local_condition_v4'];refs=sample['local_references_v4'];spec=CanvasSpec.from_record(sample['source_canvas'])
    boxes=torch.cat((refs.source_face_boxes,refs.source_hand_boxes.flatten(1,2)),1).clone()
    support=torch.zeros(1,6,*spec.canvas_hw,dtype=torch.bool)
    if declaration['availability']=='available':
        if declaration.get('role')!='primary_source_only' or declaration.get('support_kind')!='visible_owned': raise ValueError('Trusted visible-owned Source masks required')
        for k in ('source_rgb_sha256','source_person_ids'):
            if declaration.get(k)!=c.metadata[0][k]: raise ValueError('Source mask provenance differs: '+k)
        if 'boxes_original_normalized' in declaration:
            if declaration.get('source_geometry_sha256')!=sample['identity_metadata']['source_geometry_sha256']:
                raise ValueError('Explicit Detail ROI must bind the original Source geometry')
            raw_boxes=torch.tensor(declaration['boxes_original_normalized'],dtype=torch.float32)
            if raw_boxes.shape!=(6,4) or not torch.isfinite(raw_boxes).all(): raise ValueError('Six explicit unclipped Source ROIs required')
            boxes=spec.points(raw_boxes.reshape(6,2,2)).reshape(1,6,4)
        if len(declaration['masks'])!=6: raise ValueError('Declare masks in FaceA,FaceB,HandAL,AR,BL,BR order')
        for r,ref in enumerate(declaration['masks']):
            with Image.open(verified(ref)) as im:
                if (im.height,im.width)!=spec.original_hw: raise ValueError('Source support must declare original-image pixels')
                value=torch.from_numpy(np.asarray(im.convert('L')).copy()>0)
            support[0,r]=spec.stack(value.float(),nearest=True)>.5
        # Extra masks cannot expand the trusted Source person visibility.
        physical=torch.tensor([0,1,0,0,1,1])
        support &= sample['source_person_mask'][physical,0][None]
        h,w=spec.canvas_hw;yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
        support &= (xx>=boxes[...,0,None,None])&(xx<boxes[...,2,None,None])&(yy>=boxes[...,1,None,None])&(yy<boxes[...,3,None,None])
        contested=support.sum(1,keepdim=True)>1;support &= ~contested
    elif declaration['availability']!='unavailable' or not declaration.get('reason'):
        raise ValueError('Missing Detail must have explicit unavailable reason')
    valid=support.flatten(2).any(-1)&(boxes[...,2:]>boxes[...,:2]).all(-1)
    m={'schema':DETAIL_SCHEMA,'role':'primary_source_only','source_canvas_id':spec.fingerprint(),
       'source_rgb_sha256':c.metadata[0]['source_rgb_sha256'],'source_person_ids':c.metadata[0]['source_person_ids'],
       'declaration':declaration,'regions':'face_a,face_b,hand_a_left,hand_a_right,hand_b_left,hand_b_right'}
    return NativeDetailReferencesV5(boxes,support,valid,c.source_indices.clone(),(m,)).validate()


def training_sidecar(sample,declaration,*,feather_px=8):
    c=sample['local_condition_v4'];meta=c.metadata[0];spec=CanvasSpec.from_record(sample['target_canvas'])
    paired=not sample['pair_metadata'].get('cross_person',False) and (not meta.get('counterfactual_geometry') or meta.get('paired_rgb_compatible',False))
    weights=inner_feather(resolve_local_ownership(c),c.boxes,feather_px)
    aux=torch.zeros(1,2,dtype=torch.bool);points=torch.zeros(1,2,5,2);rejections=[]
    if declaration['availability']=='available':
        raw=json.loads(verified(declaration['record']).read_text())
        for k,v in (('driver_rgb_sha256',meta['driver_rgb_sha256']),('target_person_ids',meta['target_person_ids'])):
            if raw.get(k)!=v: raise ValueError('Training eligibility belongs to another target')
        if raw.get('role')!='training_only' or raw.get('coordinate_space')!='original_pixels': raise ValueError('Training-only original alignment required')
        points[0]=spec.pixel_points(torch.tensor(raw['landmarks5'],dtype=torch.float32))
        aux[0]=torch.tensor(raw['valid'],dtype=torch.bool)
        aux &= sample['local_references_v4'].arcface_valid.any(-1).gather(1,c.source_indices.clamp_min(0))&c.person_valid&c.face_roi_valid
        h,w=spec.canvas_hw
        aux &= ((points[...,0]>=0)&(points[...,0]<w)&(points[...,1]>=0)&(points[...,1]<h)).all(-1)
        for person in range(2):
            if not valid_five_points(points[0,person]):
                aux[0,person]=False;rejections.append({'person':person,'reason':'invalid_or_degenerate_five_points'})
        points=torch.where(torch.isfinite(points),points,0.)
    elif declaration['availability']!='unavailable' or not declaration.get('reason'): raise ValueError('Explicit training eligibility declaration required')
    aux &= paired
    aux &= weights[:,:2].flatten(2).sum(-1)>0
    m={'schema':SUPERVISION_SCHEMA,'role':'training_only','driver_rgb_sha256':meta['driver_rgb_sha256'],
       'driver_record_sha256':meta['driver_record_sha256'],'target_canvas_id':spec.fingerprint(),
       'target_person_ids':meta['target_person_ids'],'declaration':declaration,'feather_px':feather_px,'eligibility_rejections':rejections}
    return NativeTrainingSupervisionV5(weights.float(),aux,points,torch.tensor([paired]),(m,)).validate()


def promote_manifest(job,output):
    """Future offline metadata/sidecar builder; does not render or extract features."""
    root=Path(output).resolve()
    if not str(root).startswith('/data/shangguanrz/'): raise ValueError('New V5 assets belong under /data/shangguanrz/')
    root.mkdir(parents=True,exist_ok=False)
    v4=job['v4_manifest'];reader=NativeFullDatasetV4(verified(v4),load_target=False,data_profile=job.get('data_profile','auto'))
    declarations=job['records']
    if set(declarations)!={r['pair_id'] for r in reader.rows}: raise ValueError('Declare every pair, including unavailable local data')
    rows=[];reports=[]
    for i,old in enumerate(reader.rows):
        sample=reader[i];decl=declarations[old['pair_id']]
        refs=decl.get('references')
        if refs is not None: sample['local_references_v4']=read_record(refs,schema=REFERENCE_SCHEMA).validate()
        validate_references(sample['local_references_v4'],sample['local_condition_v4'])
        detail=source_detail_sidecar(sample,decl['detail']);supervision=training_sidecar(sample,decl['supervision'],feather_px=job.get('feather_px',8))
        rows.append({'schema':PAIR_SCHEMA,'pair_id':old['pair_id'],'v4_manifest':v4,'references':refs,
            'detail':write_sidecar(root/f'detail_{i:06d}.npz',detail,DETAIL_SCHEMA),
            'supervision':write_sidecar(root/f'supervision_{i:06d}.npz',supervision,SUPERVISION_SCHEMA)})
        reports.append({'pair_id':old['pair_id'],'detail_valid':detail.valid.tolist(),'detail_support_pixels':detail.support.sum((-1,-2)).tolist(),
            'auxiliary_eligible':supervision.auxiliary_valid.tolist(),'paired':supervision.paired.tolist(),
            'declarations':decl,'supervision_status':supervision.metadata[0]})
    with (root/'pairs.jsonl').open('x') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    with (root/'report.json').open('x') as f: json.dump({'records':reports,'all_rows_retained':True,'runtime_acceptance':'not_run'},f,indent=2)
    return rows

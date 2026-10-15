import io,json,zipfile,hashlib,collections,re,time
from pathlib import Path
import numpy as np
root=Path('/data/shangguanrz/pic-edit-datasets/SynBody');out=root/'inspection_20261003'
report={}
with zipfile.ZipFile(root/'HumanNeRF-subset/smplx.zip') as z:
    data={n.split('/')[0]:dict(np.load(io.BytesIO(z.read(n)),allow_pickle=True)) for n in z.namelist() if n.endswith('.npz')}
    sx={k:d['smplx'].item() for k,d in data.items()}
    signatures=collections.defaultdict(list)
    for sid,x in sx.items():
        signatures[hashlib.sha256(x['body_pose'].tobytes()).hexdigest()].append(sid)
    report['human_nerf']={'subjects':len(sx),'body_frames':dict(collections.Counter(len(x['body_pose']) for x in sx.values())),'exact_body_sequence_groups':[v for v in signatures.values() if len(v)>1],'stale_data_len':[{'subject':k,'declared':int(d.get('__data_len__',-1)),'actual':len(sx[k]['body_pose'])} for k,d in data.items() if int(d.get('__data_len__',len(sx[k]['body_pose'])))!=len(sx[k]['body_pose'])]}
    a,b=sx['028'],sx['042']
    report['example']={'A':{'subject':'028','view':'00','frame':200},'B':{'subject':'042','view':'00','frame':150},'C':{'subject':'028','view':'00','frame':150},'beta_l2':float(np.linalg.norm(a['betas']-b['betas'])),'fields':{k:{'shape':list(a[k].shape),'exact_equal':bool(np.array_equal(a[k],b[k])),'max_abs_diff':float(np.max(np.abs(a[k]-b[k])))} for k in a},'source_to_target_body_rms':float(np.sqrt(np.mean((a['body_pose'][200]-a['body_pose'][150])**2)))}
    report['human_nerf']['schema']={k:list(v.shape) for k,v in sx['000'].items()}
    report['human_nerf']['unique_beta_count']=len(set(x['betas'].tobytes() for x in sx.values()))
with zipfile.ZipFile(root/'SMPL-X_subset/Synbody_v1_0_sampled-smplx.zip') as z:
    names=[n for n in z.namelist() if n.endswith('.npz')];first=dict(np.load(io.BytesIO(z.read(names[0])),allow_pickle=True))
    def desc(x):
        if x.dtype==object and x.size==1:return {k:list(np.asarray(v).shape) for k,v in x.item().items()}
        return list(x.shape)
    report['smplx_subset']={'person_annotation_files':len(names),'backup_files_excluded':len([n for n in z.namelist() if '.npz~' in n]),'camera_sequences':len(set(n.rsplit('/smplx/',1)[0] for n in names)),'schema':{k:desc(v) for k,v in first.items()}}
    seen=set();seqsign=collections.defaultdict(list);motions=collections.Counter();framecount=0;validcount=0
    for idx,n in enumerate(names):
        sequence=n.rsplit('/smplx/',1)[0];base=n.rsplit('/',1)[-1];key=re.sub('_CAM[0-9]+$','',sequence)+'/'+base
        if key in seen:continue
        seen.add(key);x=np.load(io.BytesIO(z.read(n)),allow_pickle=True)['smplx'].item();p=x['body_pose'];framecount+=len(p);validcount+=1
        sig=hashlib.sha256(p.tobytes()).hexdigest();seqsign[sig].append({'path':n,'frames':len(p),'beta_hash':hashlib.sha256(x['betas'][0].tobytes()).hexdigest(),'person':base.split('-',1)[0]})
        motions[base.split('-',1)[-1][:-4]]+=1
        if validcount%1000==0:print('ANNOTATIONS',validcount,flush=True)
    cross=[v for v in seqsign.values() if len(set(i['beta_hash'] for i in v))>1 and v[0]['frames']>2]
    report['smplx_subset'].update({'deduplicated_person_sequences':validcount,'deduplicated_pose_frames':framecount,'motion_names':len(motions),'most_common_motions':motions.most_common(8),'exact_body_sequence_cross_shape_groups':len(cross),'cross_shape_group_examples':cross[:3]})
(out/'annotation_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='human_nerf'},indent=2),flush=True)

import os,zipfile,json,time,stat,shutil
from pathlib import Path
root=Path('/data/shangguanrz/pic-edit-datasets/SynBody')
out=root/'inspection_20261003'
status={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'state':'running','archives':[]}
def save():
    t=out/'extraction_status.tmp'
    t.write_text(json.dumps(status,indent=2)); t.replace(out/'extraction_status.json')
save()
try:
    plans=[('HumanNeRF-subset',x,None) for x in ['camera_params.zip','smplx.zip','smpl.zip','img.zip','mask.zip']]
    plans += [('Assets',x,None) for x in ['SMPL-XL-100-meta.zip','SMPL-XL-1000-meta.zip']]
    plans += [('Assets','SMPL-XL-100-fbx.zip',{'0000001','0000002','0000006'}),('Assets','SMPL-XL-100-obj.zip',{'0000001','0000002','0000006'})]
    plans += [('SMPL-X_subset','Synbody_v1_0_sampled-smplx.zip',None)]
    for sub,arc,ids in plans:
        dest=out/'extracted'/sub
        if sub=='Assets': dest=dest/('1000' if '1000-' in arc else '100')
        dest.mkdir(parents=True,exist_ok=True); dest=dest.resolve()
        with zipfile.ZipFile(root/sub/arc) as z:
            infos=[i for i in z.infolist() if not i.is_dir() and (ids is None or i.filename.split('/')[0] in ids)]
            if sub=='SMPL-X_subset': infos=[i for i in infos if i.filename.endswith('.npz')]
            entry={'archive':sub+'/'+arc,'files':len(infos),'bytes':sum(i.file_size for i in infos),'written':0,'skipped_existing':0}
            status['archives'].append(entry);save()
            for i,info in enumerate(infos):
                target=(dest/info.filename).resolve()
                if not target.is_relative_to(dest): raise ValueError('Unsafe archive path: '+info.filename)
                if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink: '+info.filename)
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if target.stat().st_size!=info.file_size:raise ValueError('Existing size mismatch '+str(target))
                    entry['skipped_existing']+=1;continue
                temp=target.with_name(target.name+'.extracting')
                with z.open(info) as src,open(temp,'xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                temp.replace(target);entry['written']+=1
                if (i+1)%2000==0:save()
            entry['complete']=True;save();print('DONE',sub,arc,flush=True)
    status['state']='complete';status['finished']=time.strftime('%Y-%m-%d %H:%M:%S');save()
except Exception as e:
    status['state']='failed';status['error']=repr(e);save();raise

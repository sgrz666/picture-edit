#!/usr/bin/env python3
"""Run A/B evidence, full C estimation, and bounded shared-camera retargeting."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):digest.update(part)
    return digest.hexdigest()


def checkpointed_substage(marker,fingerprint,work,outputs):
    marker=Path(marker)
    if marker.is_file():
        evidence=json.loads(marker.read_text(encoding='utf8'))
        if evidence['fingerprint']!=fingerprint or any(not Path(p).is_file() or file_sha(p)!=sha for p,sha in evidence['outputs'].items()):
            raise RuntimeError(f'Substage evidence changed at {marker}; use a new --name')
        return True
    work();paths=list(outputs())
    if not paths or any(not Path(p).is_file() for p in paths):raise RuntimeError('Substage did not produce its declared evidence')
    marker.parent.mkdir(parents=True,exist_ok=True)
    with marker.open('x',encoding='utf8') as stream:
        json.dump({'fingerprint':fingerprint,'outputs':{str(p):file_sha(p) for p in paths}},stream,indent=2)
    return False


def image_path(value):
    if isinstance(value,dict):value=value.get('path',value.get('image',value.get('file')))
    if not isinstance(value,str):raise ValueError('Each ABC input must be an image path')
    path=Path(value).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    return str(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--retarget-only',action='store_true');args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf-8'))
    if args.retarget_only:
        from retarget.scene import build_scene
        build_scene(request);return
    root=Path(request['project_root']).resolve();run=Path(request['run_dir']).resolve();out=run/'geometry'
    if (out/'scene.json').exists() and not (out/'.stages/abc_retarget.json').is_file():
        raise FileExistsError('Unsealed geometry scene already exists; use a fresh --name')
    out.mkdir(parents=True,exist_ok=True);logs=run/'logs';logs.mkdir(parents=True,exist_ok=True)
    settings=request['settings']
    settings.setdefault('smplest_root',str(root/'add/tools/SMPLest-X'))
    settings.setdefault('smplest_checkpoint','smplest_x_h');settings.setdefault('person_detection_confidence',.5)
    settings.setdefault('dwpose_det_model',str(root/'add/tools/dwpose_mimicmotion_official/models/yolox_l.onnx'))
    settings.setdefault('dwpose_pose_model',str(root/'add/tools/dwpose_mimicmotion_official/models/dw-ll_ucoco_384.onnx'))
    geometry_python=settings.get('geometry_python',str(root/'add/envs/smplestx_bw/bin/python'))
    sam_python=settings.get('sam_python','/home/shangguanrz/miniconda3/envs/deepgen/bin/python')
    multi_python=settings.get('multihmr_python',str(root/'add/envs/multihmr/bin/python'))
    inputs={k:image_path(request['inputs'][k]) for k in ('source_a','source_b','interaction')}
    def execute(name,python,worker,payload,extra=()):
        path=out/(name+'_request.json')
        text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if path.is_file() and path.read_text(encoding='utf8')!=text:path=out/(name+'_request_'+uuid.uuid4().hex[:8]+'.json')
        environment=os.environ.copy();environment['PYTHONPATH']=str(root);environment['PYOPENGL_PLATFORM']='egl'
        started=time.perf_counter()
        def work():
            if not path.is_file():path.write_text(text,encoding='utf8')
            with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
                process=subprocess.run([python,'-u',str(HERE/worker),'--request',str(path),*extra],
                    cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                log_path=logs/(name+'.log')
                tail='\n'.join(log_path.read_text(encoding='utf8',errors='replace').splitlines()[-25:])
                raise RuntimeError(f'{name} failed; log={log_path}\n{tail}')
        def outputs():
            if name=='abc_sources':
                return [out/'sources.json']+[out/('source_'+role)/filename for role in ('A','B')
                    for filename in ('source.json','original.png','canvas.png','mask.png','mask_original.png','keypoints.npz')]
            if name.startswith('abc_fit_'):
                folder=out/('source_'+name[-1]);return [folder/'fit.json',folder/'params.npz']
            if name=='abc_interaction':return [out/'interaction'/filename for filename in ('fit.json','original.png','person_0.npz','person_1.npz')]+(
                [out/'interaction/official_contact.npz',out/'interaction/official_stick_contact.png'] if request.get('official_record') else [])
            if name=='abc_native_conditions':return list((out/'native').glob('*.npz'))+[out/'native/index.json']
            return [out/'scene.json']+list((out/'render').glob('*'))+[out/('target_'+role)/filename
                for role in ('A','B') for filename in ('params.npz','mesh.npz')]
        key=request.get('fingerprint',hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest())+':'+name
        cached=checkpointed_substage(out/'.stages'/(name+'.json'),key,work,outputs)
        print(json.dumps({'stage':name,'cache_hit':cached,'seconds':round(time.perf_counter()-started,2)}),flush=True)
    evidence={'project_root':str(root),'settings':settings,'manifest':str(out/'sources.json'),
        'sources':[{'role':role,'image':inputs['source_'+role.lower()],'out_dir':str(out/('source_'+role))} for role in ('A','B')]}
    if request.get('source_cache'):
        records=request['source_cache']
        for source in records:
            if any(file_sha(path)!=sha for path,sha in source['cache_artifacts'].items()):raise ValueError('Frozen source cache SHA changed')
        def reuse_sources():
            with (out/'sources.json').open('x',encoding='utf8') as stream:
                json.dump({'schema':'abc_source_evidence_v1','source_only_cache':True,'sources':records},stream,indent=2)
        checkpointed_substage(out/'.stages/abc_sources_cached.json',request['fingerprint'],reuse_sources,lambda:[out/'sources.json'])
        print('SOURCE_ONLY_CACHE_REUSED',flush=True)
    else:
        execute('abc_sources',sam_python,'abc_source_worker.py',evidence)
        for role in ('A','B'):
            execute('abc_fit_'+role,geometry_python,'abc_estimate_worker.py',{'kind':'source','project_root':str(root),
                'settings':settings,'image':str(out/('source_'+role)/'original.png'),'out_dir':str(out/('source_'+role))})
    official=request.get('official_record')
    execute('abc_interaction',geometry_python if official else multi_python,
        'abc_hi4d_worker.py' if official else 'abc_estimate_worker.py',{'kind':'interaction','project_root':str(root),
        'settings':settings,'image':inputs['interaction'],'out_dir':str(out/'interaction'),'official_record':official})
    execute('abc_retarget',geometry_python,'abc_geometry_worker.py',request,('--retarget-only',))
    execute('abc_native_conditions',geometry_python,'abc_native_assets.py',
        {**request,'scene':str(out/'scene.json'),'native_geometry_dir':str(out/'native')})


if __name__=='__main__':main()

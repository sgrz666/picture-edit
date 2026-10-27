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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent


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
    if (out/'scene.json').exists():raise FileExistsError('Geometry already complete; use a new run or orchestrator cache verification')
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
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        environment=os.environ.copy();environment['PYTHONPATH']=str(root);environment['PYOPENGL_PLATFORM']='egl'
        started=time.perf_counter()
        with (logs/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([python,'-u',str(HERE/worker),'--request',str(path),*extra],
                cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'stage':name,'seconds':round(time.perf_counter()-started,2)}),flush=True)
    evidence={'project_root':str(root),'settings':settings,'manifest':str(out/'sources.json'),
        'sources':[{'role':role,'image':inputs['source_'+role.lower()],'out_dir':str(out/('source_'+role))} for role in ('A','B')]}
    execute('abc_sources',sam_python,'abc_source_worker.py',evidence)
    for role in ('A','B'):
        # Fit the EXIF-normalised original that DWPose/SAM2 actually saw.
        execute('abc_fit_'+role,geometry_python,'abc_estimate_worker.py',{'kind':'source','project_root':str(root),
            'settings':settings,'image':str(out/('source_'+role)/'original.png'),'out_dir':str(out/('source_'+role))})
    execute('abc_interaction',multi_python,'abc_estimate_worker.py',{'kind':'interaction','project_root':str(root),
        'settings':settings,'image':inputs['interaction'],'out_dir':str(out/'interaction')})
    execute('abc_retarget',geometry_python,'abc_geometry_worker.py',request,('--retarget-only',))


if __name__=='__main__':main()

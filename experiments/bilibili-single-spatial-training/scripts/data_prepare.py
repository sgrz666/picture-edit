"""Resumable Bilibili preparation; no Qwen loading, optimizer or training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

DATA = Path('/data/shangguanrz/datasets/bilibili')
OLD = Path('/data/shangguanrz/pic-edit-experiments/bilibili-native-v5-training-20261106-v1')
BODY = Path('/data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1/full/index.json')
SPATIAL = BODY.parent.parent / 'full_visible_region_v2_20261107'
BASELINE = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
EXPECTED_COUNTS = {'train': {'videos':39,'frames':1560,'pairs':9520},
                   'val': {'videos':5,'frames':200,'pairs':1490},
                   'test': {'videos':5,'frames':200,'pairs':1146}}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class StatHashMemo:
    """Only reuse SHA within this process while all observed stat fields agree."""
    def __init__(self):
        self.cache = {}
        self.hash_reads = 0

    @staticmethod
    def signature(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def __call__(self, path):
        path = Path(path).resolve()
        before = self.signature(path)
        entry = self.cache.get(str(path))
        if entry is not None and entry[0] == before:
            return entry[1]
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024), b''):
                h.update(block)
        if self.signature(path) != before:
            raise ValueError('File changed during SHA verification: ' + str(path))
        self.hash_reads += 1
        self.cache[str(path)] = (before, h.hexdigest())
        return h.hexdigest()

    def verify(self, ref):
        path = Path(ref['path'])
        if not path.is_absolute() or self(path) != ref['sha256']:
            raise ValueError('Artifact SHA differs: ' + str(path))
        return ref


SHA = StatHashMemo()


def ref(path, schema=None):
    result = {'path':str(Path(path).resolve()), 'sha256':SHA(path)}
    if schema:
        result['schema'] = schema
    return result


def write_exclusive_bytes(path, raw):
    """Publish a complete inode without overwriting; retain pending for audit."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(pending,path)
    if os.name=='posix':
        descriptor=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish(path, value, jsonl=False):
    path = Path(path)
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if (rows(path) if jsonl else load(path)) != normalized:
            raise ValueError('Existing immutable publication differs: ' + str(path))
        return ref(path)
    raw=(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in normalized) if jsonl
         else json.dumps(normalized,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    try:
        write_exclusive_bytes(path,raw)
    except FileExistsError:
        if (rows(path) if jsonl else load(path))!=normalized:
            raise ValueError('Concurrent immutable publication differs: '+str(path))
    return ref(path)


def check_frame_binding(frame, record, canvas):
    for key in ('frame_id','clip_id','split','loss_mask'):
        if record.get(key) != frame[key]:
            raise ValueError('Source frame differs: ' + key)
    if record.get('group_id') != frame['clip_id']:
        raise ValueError('Source original-video group differs')
    if record.get('rgb') != {'path':frame['rgb_path'],'sha256':frame['rgb_sha256']}:
        raise ValueError('Source RGB binding differs')
    if record.get('canvas') != canvas or record.get('geometry_method') != 'sapiens_v1':
        raise ValueError('Source Canvas/geometry differs')
    if not record.get('source_features_complete') or record.get('feature_protocol') != 'FP32_same_models_and_preprocessing_batched_16':
        raise ValueError('Source frozen FP32 features incomplete or protocol differs')


def check_spatial_config(config, frames_ref, body_ref, frame_ids):
    for key, expected in {'frames':frames_ref,'body_cache_index':body_ref,'hand_policy':'visible_region_v2',
                          'canvas_long_side':512,'geometry_method':'sapiens_v1'}.items():
        if config.get(key) != expected:
            raise ValueError('Spatial configuration differs: ' + key)
    selected = config.get('selected_frame_ids', [])
    if len(selected) != len(set(selected)) or set(selected) != set(frame_ids):
        raise ValueError('Spatial full-frame coverage differs')


def make_prepared(manifests, spatial, initialization, model_manifest, run_root, counts, audits):
    def source(split, role):
        return {'source_id':'bilibili_'+split,'role':role,'manifest':manifests[split]['manifest'],
                'groups':manifests[split]['groups'],'data_profile':'full_smplx','scene_kind':'single',
                'weight':1.,'spatial_weights':spatial}
    return {'schema':'bilibili_single_prepared_v1',
            'sources':[source('train','train'), source('val','validation')],
            'test':source('test','validation'), 'test_holdout':True,
            'initialization':initialization, 'model_path':'/data/shangguanrz/pic-edit-models/Qwen-Image-2.1',
            'model_manifest':model_manifest,'cache_root':str(run_root/'frozen_qwen'),
            'counts':{**counts,'frames':sum(c['frames'] for c in counts.values()),
                      'pairs':sum(c['pairs'] for c in counts.values())},
            'bindings':{'canvas_long_side':512,'qwen_condition_profile':'single_2ref',
                        'geometry_version':'sapiens_v1','hand_policy':'visible_region_v2'},
            'audits':audits, 'all_original_rows_preserved':True,
            'model_constructed':False,'optimizer_created':False,'training_started':False}


@contextmanager
def native_hash_memo():
    """Scope the verified-stat SHA optimization to this private CPU/data process."""
    import importlib
    targets = {'src.pose_control.qwen21_full.native_full_assets':'sha256',
               'src.pose_control.qwen21_full.native_full_data':'sha256',
               'src.pose_control.qwen21_full.native_v5.data':'sha256',
               'src.pose_control.qwen21_full.native_v5.conditioning':'sha256',
               'src.pose_control.qwen21_full.native_v5.hi4d_guide_binding':'sha256',
               'src.pose_control.qwen21_full.native_v5.training.io':'file_sha'}
    originals = []
    for name, attribute in targets.items():
        module = importlib.import_module(name)
        originals.append((module,attribute,getattr(module,attribute)))
        setattr(module,attribute,SHA)
    try:
        yield
    finally:
        for module,attribute,value in reversed(originals):
            setattr(module,attribute,value)


def spatial_resume_command(args, config):
    if config['device']!='cpu' or config['torch_threads']!=4:
        raise ValueError('Only the approved independent four-thread CPU spatial job may resume')
    return [sys.executable,'-u',str(args.repo_root/'scripts/data/build_native_spatial_weights.py'),
            '--frames',str(args.data_root/'frames/frames.jsonl'),'--output-root',str(args.spatial_root),
            '--body-cache-index',str(args.body_index),'--hand-policy',config['hand_policy'],
            '--device','cpu','--torch-threads','4','--canvas-long-side',str(config['canvas_long_side']),
            '--geometry-method',config['geometry_method'],'--seed',str(config['seed']),
            '--sam-config',config['sam_config'],'--limit','0','--resume']


def spatial_ready(args):
    """Observe the CPU writer, or resume that exact immutable configuration."""
    import fcntl
    args.run_root.mkdir(parents=True,exist_ok=True)
    with (args.run_root/'spatial_prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _spatial_ready(args)


def _spatial_ready(args):
    frames_ref, body_ref = ref(args.data_root/'frames/frames.jsonl'), ref(args.body_index)
    frame_ids = [f['frame_id'] for f in rows(args.data_root/'frames/frames.jsonl')]
    config=load(args.spatial_root/'config.json')
    check_spatial_config(config,frames_ref,body_ref,frame_ids)
    for name, expected in config.get('producer_files',{}).items():
        if SHA(args.repo_root/name)!=expected:
            raise ValueError('Spatial producer changed; cannot resume: '+name)
    resumed=False
    while not (args.spatial_root/'index.json').exists():
        failures = sorted(args.spatial_root.glob('failure_*.json'))
        if failures:
            raise RuntimeError('External spatial builder reported failure: ' + str(failures[-1]))
        processes = subprocess.check_output(['ps','-eo','pid,args'],text=True)
        matches = [line for line in processes.splitlines() if 'build_native_spatial_weights.py' in line
                   and str(args.spatial_root) in line]
        if not matches:
            if resumed:
                raise RuntimeError('Resumed spatial CPU builder exited before publishing its full index')
            command=spatial_resume_command(args,config)
            tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            log=args.run_root/'data_audits'/('spatial_resume_'+tag+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('xb') as output:
                process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                                         cwd=args.repo_root,start_new_session=True)
            publish(log.with_suffix('.process.json'),{'pid':process.pid,'command':command,'log':str(log),
                    'config':ref(args.spatial_root/'config.json'),'device':'cpu','training_started':False})
            resumed=True
            matches=[str(process.pid)+' '+' '.join(command)]
            print(json.dumps({'event':'resumed_external_spatial','pid':process.pid,'log':str(log),'command':command}),flush=True)
        progress = sorted(args.spatial_root.glob('progress_*.jsonl'))
        tail = rows(progress[-1])[-1] if progress else {}
        print(json.dumps({'event':'waiting_external_spatial','processes':matches,'progress':tail}),flush=True)
        time.sleep(30)
    index = load(args.spatial_root/'index.json')
    check_spatial_config(index['config'],frames_ref,body_ref,frame_ids)
    if len(index['entries']) != len(frame_ids):
        raise ValueError('Completed spatial index misses full coverage')
    report = args.run_root/'data_audits/spatial_v2.json'
    if report.exists():
        audit = load(report)
        if not audit.get('passed') or audit.get('index') != ref(args.spatial_root/'index.json'):
            raise ValueError('Existing spatial audit differs or failed')
    else:
        from scripts.audit.audit_native_spatial_weights import audit
        publish(report,audit(args.spatial_root/'index.json',args.data_root/'frames/frames.jsonl'))
    return ref(args.spatial_root/'index.json'),ref(report)


def verify_prepared(path):
    prepared = load(path)
    if prepared.get('schema') != 'bilibili_single_prepared_v1':
        raise ValueError('Prepared schema differs')
    visited=set()
    def walk(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                SHA.verify(value)
                target=Path(value['path'])
                if str(target) not in visited:
                    visited.add(str(target))
                    if target.suffix in ('.json','.jsonl'):
                        walk(rows(target) if target.suffix=='.jsonl' else load(target))
            for item in value.values():
                walk(item)
        elif isinstance(value,list):
            for item in value:
                walk(item)
    walk(prepared)
    return prepared


def prepare(args):
    if not str(args.run_root.resolve()).startswith('/data/shangguanrz/'):
        raise ValueError('New artifacts must remain on /data/shangguanrz/')
    args.run_root.mkdir(parents=True,exist_ok=True)
    lock = args.run_root/'data_prepare.lock'
    import fcntl
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another data preparation process owns this run') from error
        if (args.run_root/'prepared.json').exists():
            result = verify_prepared(args.run_root/'prepared.json')
            print(json.dumps({'event':'prepared_verified','prepared':ref(args.run_root/'prepared.json')}),flush=True)
            return result
        sys.path.insert(0,str(args.repo_root))
        import torch
        torch.set_num_threads(4)
        from prepare_dataset import prepare_pairs, full_native_audit
        from npz_codec import private_numpy_codec
        with native_hash_memo(),private_numpy_codec():
            manifests, counts, pair_report = prepare_pairs(args)
            spatial, spatial_audit = spatial_ready(args)
            native_report = full_native_audit(args,manifests,spatial)
        initialization = ref(BASELINE/'initialization/native_v5_init_v2.pt')
        model_manifest = ref(BASELINE/'source/qwen_model_files.json')
        audits = {'pairs':pair_report,'native_data':native_report,'spatial':spatial_audit}
        for name in ('split.json','split_verification_groups_v2.json','pixel_isolation_decoded_v1.json','sapiens_v1_asset_binding.json'):
            path=args.old_root/'report'/name
            if path.exists():
                audits[name] = ref(path)
        result = make_prepared(manifests,spatial,initialization,model_manifest,args.run_root,counts,audits)
        publish(args.run_root/'prepared.json',result)
        print(json.dumps({'event':'prepared','prepared':ref(args.run_root/'prepared.json'),'counts':result['counts']}),flush=True)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','spatial','verify-reuse'))
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--old-root',type=Path,default=OLD)
    parser.add_argument('--body-index',type=Path,default=BODY)
    parser.add_argument('--spatial-root',type=Path,default=SPATIAL)
    parser.add_argument('--legacy-root',type=Path,default=Path(__file__).resolve().parents[1]/'source/legacy')
    parser.add_argument('--legacy-repo-root',type=Path,default=Path('/home/shangguanrz/project/pic-edit'),
                        help='Original producer path for verifying historical Source identities; never used for new output')
    parser.add_argument('--device',choices=('cuda',),default='cuda',
                        help='Missing pair assembly uses the unchanged CUDA rasterizer; spatial/verify-reuse are CPU only')
    parser.add_argument('--verify-limit',type=int,default=0,help='Only limits CPU verify-reuse; never limits preparation')
    args=parser.parse_args()
    try:
        if args.action=='spatial':
            sys.path.insert(0,str(args.repo_root))
            spatial_ready(args)
        elif args.action=='verify-reuse':
            sys.path.insert(0,str(args.repo_root))
            from prepare_dataset import verify_reuse
            with native_hash_memo():
                verify_reuse(args)
        else:
            prepare(args)
    except Exception:
        import traceback
        publish(args.run_root/'data_audits'/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                {'error':traceback.format_exc(),'model_constructed':False,'optimizer_created':False,'training_started':False})
        raise


if __name__=='__main__':
    main()

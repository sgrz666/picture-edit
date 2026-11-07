"""Qualified pixel-only inputs for frozen single-person Qwen encoding.

This plan cannot train an Adapter or substitute for full data/gradient acceptance.
It retains the exact full-plan source identity, but never decodes Native arrays.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from src.data.canvas_contract import CanvasSpec
from .config import digest
from .io import file_sha,verify,read_json_ref
from .. import conditioning
from ..spatial_weights import read_spatial_index


def progress(event,**values):
    print(json.dumps({'event':event,**values},ensure_ascii=False),flush=True)


class VerifiedSHA:
    """Reuse a computed SHA only while device/inode/size/mtime/ctime all agree."""
    def __init__(self):self.cache={}
    @staticmethod
    def signature(path):
        s=Path(path).stat()
        return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def __call__(self,path):
        path=Path(path);before=self.signature(path);key=str(path.resolve())
        cached=self.cache.get(key)
        if cached is not None and cached[0]==before:return cached[1]
        sha=file_sha(path)
        if self.signature(path)!=before:raise ValueError('Input changed during SHA verification')
        self.cache[key]=(before,sha);return sha
    def verify(self,ref):
        if not Path(ref['path']).is_absolute() or self(ref['path'])!=ref['sha256']:
            raise ValueError('Frozen input SHA differs: '+str(ref['path']))
        return ref


@contextmanager
def condition_sha_memo(memo):
    original=conditioning.sha256;conditioning.sha256=memo
    try:yield
    finally:conditioning.sha256=original


class PixelDataset:
    def __init__(self,plan,source,rows):self.plan=plan;self.source=source;self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        from ...native_full_data import _rgb
        info=self.plan.ensure_metadata(self.source,index)
        sample={'native_scene_kind':'single','data_profile':'full_smplx',
                'pair_metadata':{'pair_id':self.rows[index]['pair_id']},
                'local_condition_v4':SimpleNamespace(metadata=(dict(info['meta']),)),
                'source_token_hw':info['source_canvas'].token_hw,'target_token_hw':info['target_canvas'].token_hw}
        ref,expected=self.plan.conditions[(self.source,index)]
        with condition_sha_memo(self.plan.sha):actual=conditioning.read_condition(ref,sample)
        if actual!=expected:raise ValueError('Frozen condition changed after input planning')
        sample['tgt_image']=_rgb(info['target_rgb'],info['target_canvas'])
        return sample


class ConditionTable(dict):
    def __init__(self,plan):super().__init__();self.plan=plan
    def __missing__(self,key):
        self.plan.ensure_metadata(*key)
        return self[key]


class FrozenInputPlan:
    def __init__(self,config,qualification,*,lazy=False):
        if config.qwen_condition_profile!='single_2ref' or any(
            s.scene_kind!='single' or s.data_profile!='full_smplx' or s.role not in ('train','validation') for s in config.sources):
            raise ValueError('Qualified frozen pixel plan supports explicit single_2ref/full_smplx sources only')
        report,bound=read_json_ref(qualification)
        if (report.get('schema')!='native_v5_data_preflight_v1' or report.get('level')!='data' or
            report.get('passed') is not True or report.get('all_rows_retained') is not True or
            report.get('signature')!=digest(report.get('identity'))):
            raise ValueError('Complete matching data qualification required for frozen pixel plan')
        self.config=config;self.qualification=bound;self.qualified_report=report
        self.datasets={};self.conditions=ConditionTable(self);self.metadata={};self.sha=VerifiedSHA();self.v4_records={}
        identities={};total=0
        for source in config.sources:
            manifest=verify(source.manifest);groups,group_ref=read_json_ref(source.groups)
            rows=[json.loads(line) for line in Path(manifest['path']).read_text(encoding='utf8').splitlines() if line.strip()]
            ids=[row.get('pair_id') for row in rows]
            if not ids or len(set(ids))!=len(ids) or set(ids)!=set(groups):raise ValueError('Frozen manifest/groups coverage differs')
            if any(row.get('schema')!='native_full_pair_v5' for row in rows):raise ValueError('Frozen pixel plan needs exact Native V5 manifests')
            identity={'manifest':manifest,'groups':group_ref,'role':source.role,
                      'closure':[{k:row.get(k) for k in ('pair_id','v4_manifest','detail','supervision','references','qwen_condition')} for row in rows]}
            if source.spatial_weights is not None:
                spatial,index_ref=read_spatial_index(source.spatial_weights)
                identity['spatial_weights']={'index':index_ref,'entries':spatial['entries']}
            if identity!=report['identity']['sources'].get(source.source_id):
                raise ValueError('Frozen inputs differ from full data qualification: '+source.source_id)
            identities[source.source_id]=identity
            self.datasets[source.source_id]=PixelDataset(self,source.source_id,rows);total+=len(rows)
        if set(identities)!=set(report['identity']['sources']):raise ValueError('Frozen source set differs from full qualification')
        self.identity={'sources':identities,'config_sha256':config.signature()};self.signature=digest(self.identity)
        self.condition_bindings={source+'\0'+row['pair_id']:row['qwen_condition']['sha256']
                                 for source,dataset in self.datasets.items() for row in dataset.rows}
        progress('frozen_input_qualification_passed',pairs=total,qualification=bound,lazy=lazy)
        if not lazy:
            count=0
            for source,dataset in self.datasets.items():
                for index in range(len(dataset)):
                    self.ensure_metadata(source,index);count+=1
                    if count==1 or count%250==0 or count==total:progress('frozen_input_binding',completed=count,total=total)

    def ensure_metadata(self,source,index):
        if (source,index) in self.metadata:return self.metadata[(source,index)]
        row=self.datasets[source].rows[index];ref=row['v4_manifest'];self.sha.verify(ref)
        if ref['path'] not in self.v4_records:
            values=[json.loads(line) for line in Path(ref['path']).read_text(encoding='utf8').splitlines() if line.strip()]
            if len({v.get('pair_id') for v in values})!=len(values):raise ValueError('Duplicate V4 frozen pair IDs')
            self.v4_records[ref['path']]={v.get('pair_id'):v for v in values}
        raw=self.v4_records[ref['path']][row['pair_id']]
        if raw.get('schema')!='native_full_pair_v4' or raw.get('scene_kind')!='single' or raw.get('data_profile','full_smplx')!='full_smplx':
            raise ValueError('Frozen V4 pair profile differs')
        with condition_sha_memo(self.sha):
            spec=conditioning.read_condition(row['qwen_condition'])
            if spec['profile']!='single_2ref':raise ValueError('Frozen source condition profile differs')
            sc=CanvasSpec.from_record(spec['images'][0]['canvas']);tc=CanvasSpec.from_record(spec['images'][1]['canvas'])
            meta={'source_rgb_sha256':raw['source_rgb']['sha256'],'driver_rgb_sha256':raw['target_rgb']['sha256'],
                  'source_canvas_id':sc.fingerprint(),'target_canvas_id':tc.fingerprint()}
            minimal={'native_scene_kind':'single','pair_metadata':{'pair_id':row['pair_id']},
                     'local_condition_v4':SimpleNamespace(metadata=(meta,))}
            conditioning.read_condition(row['qwen_condition'],minimal)
            for rgb,canvas in ((raw['source_rgb'],sc),(raw['target_rgb'],tc)):
                self.sha.verify(rgb)
                with Image.open(rgb['path']) as im:
                    if (im.height,im.width)!=canvas.original_hw:raise ValueError('Frozen original RGB/Canvas dimensions differ')
            self.metadata[(source,index)]={'meta':meta,'source_canvas':sc,'target_canvas':tc,'target_rgb':raw['target_rgb']}
            self.conditions[(source,index)]=(row['qwen_condition'],spec)
        return self.metadata[(source,index)]

    def binding(self,source,index,model_sha,prompt_sha):
        info=self.ensure_metadata(source,index);ref,spec=self.conditions[(source,index)]
        return {'key':source+'\0'+self.datasets[source].rows[index]['pair_id'],**info['meta'],
                'source_token_hw':list(info['source_canvas'].token_hw),'target_token_hw':list(info['target_canvas'].token_hw),
                'model_manifest_sha256':model_sha,'condition_sha256':ref['sha256'],'prompt_sha256':spec['prompt']['sha256']}

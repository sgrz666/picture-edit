"""Immutable per-pair frozen-cache commits; verified recovery never overwrites."""
from pathlib import Path
import json,os,uuid
import torch
from .config import digest
from .io import verify,file_sha,write_json,sync_directory


def validate_cached_pair(pair,binding):
    for side in ('source','target'):
        grid=tuple(binding[side+'_token_hw'])
        if tuple(pair[side+'_token_hw'])!=grid or tuple(pair[side+'_latent'].shape)!=(1,grid[0]*grid[1],64):
            raise ValueError('Frozen cache latent grid differs: '+side)
    if binding.get('condition_sha256') is not None:
        from ..conditioning import validate_pair
        validate_pair(pair)
        if pair.get('prompt_sha256')!=binding['prompt_sha256']:raise ValueError('Frozen cache prompt binding differs')
    for value in pair.values():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Frozen cache has nonfinite tensors')


class FrozenCacheStore:
    def __init__(self,root,identity,*,resume=False):
        self.root=Path(root).resolve()
        expected={'schema':'native_v5_cache_preparation_v1','identity':identity,'identity_sha256':digest(identity)}
        if resume:
            if not self.root.is_dir():raise ValueError('Cache resume requires an existing directory')
            path=self.root/'preparation.json'
            if not path.is_file() or digest(json.loads(path.read_text(encoding='utf8')))!=digest(expected):
                raise ValueError('Frozen cache preparation identity differs; use a new directory')
        else:
            self.root.mkdir(parents=True,exist_ok=False)
            write_json(self.root/'preparation.json',expected)
        self.records=self.root/'entries';self.records.mkdir(exist_ok=True)

    def _record(self,binding):return self.records/(digest(binding['key'])+'.json')

    def get(self,binding):
        record=self._record(binding)
        if not record.exists():return None
        entry=json.loads(record.read_text(encoding='utf8'))
        if entry.get('schema')!='native_v5_frozen_pair_commit_v1' or entry.get('binding')!=binding:
            raise ValueError('Frozen cache pair binding differs: '+binding['key'])
        ref=verify(entry['file'])
        if self.root not in Path(ref['path']).resolve().parents:raise ValueError('Frozen cache entry escapes cache root')
        validate_cached_pair(torch.load(ref['path'],map_location='cpu',weights_only=True),binding)
        expected={'file':ref,**{k:binding[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')}}
        if binding.get('condition_sha256') is not None:expected['condition_sha256']=binding['condition_sha256']
        if entry.get('entry')!=expected:self._invalid_entry()
        return expected

    @staticmethod
    def _invalid_entry():raise ValueError('Frozen cache commit entry/file binding differs')

    def put(self,binding,pair):
        if self._record(binding).exists():raise FileExistsError('Frozen pair already committed')
        validate_cached_pair(pair,binding)
        path=self.root/('pair_'+digest(binding['key'])[:24]+'_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as f:
            torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in pair.items()},f)
            f.flush();os.fsync(f.fileno())
        sync_directory(self.root)
        ref={'path':str(path),'sha256':file_sha(path)}
        entry={'file':ref,**{k:binding[k] for k in ('source_rgb_sha256','driver_rgb_sha256','source_canvas_id','target_canvas_id')}}
        if binding.get('condition_sha256') is not None:entry['condition_sha256']=binding['condition_sha256']
        write_json(self._record(binding),{'schema':'native_v5_frozen_pair_commit_v1','binding':binding,'file':ref,'entry':entry})
        return entry

    def publish(self,value):
        path=self.root/'index.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=value:raise ValueError('Frozen final index differs; never overwrite')
            return {'path':str(path),'sha256':file_sha(path)}
        # Link a fully flushed immutable publication into place. The ready record
        # remains for auditing; no previous artifact is deleted or renamed.
        ready=write_json(self.root/('index_publish_'+uuid.uuid4().hex+'.json'),value)
        os.link(ready['path'],path);sync_directory(self.root)
        return {'path':str(path),'sha256':file_sha(path)}

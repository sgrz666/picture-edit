"""Snapshot missing pure-Python SAM dependencies without modifying any environment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys

PACKAGES=(('hydra','hydra-core'),('omegaconf','omegaconf'),('antlr4','antlr4-python3-runtime'),
          ('iopath','iopath'),('portalocker','portalocker'))


def require_pure_python(root):
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.so','.pyd','.dll','.dylib'):
            raise ValueError('Refusing a binary dependency overlay: '+str(p))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not str(a.output.resolve()).startswith('/data/shangguanrz/'):raise ValueError('Persistent run directory required')
    if a.output.exists():raise FileExistsError('Dependency snapshots are immutable')
    sources=[]
    for module,dist in PACKAGES:
        spec=importlib.util.find_spec(module);distribution=importlib.metadata.distribution(dist)
        source=Path(spec.origin).parent;require_pure_python(source)
        sources.append((module,dist,distribution.version,source,Path(distribution._path)))
    a.output.mkdir(parents=True)
    records=[]
    for module,dist,version,source,metadata in sources:
        shutil.copytree(source,a.output/module,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(metadata,a.output/metadata.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        records.append({'module':module,'distribution':dist,'version':version,'original_path':str(source)})
    files=[{'path':str(p.relative_to(a.output)),'sha256':sha(p)} for p in sorted(a.output.rglob('*')) if p.is_file()]
    value={'schema':'sam_pure_python_dependency_snapshot_v1','source_python':sys.executable,'packages':records,
           'files':files,'training_frameworks_copied':False,'environment_modified':False,
           'producer_sha256':sha(__file__)}
    with (a.output/'manifest.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps({'output':str(a.output),'packages':records,'verified_files':len(files),'manifest_sha256':sha(a.output/'manifest.json')}),flush=True)


if __name__=='__main__':main()

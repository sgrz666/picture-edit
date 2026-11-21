"""Pack a byte-pinned code snapshot without model/data artifacts or deletions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[4]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment',type=Path,action='append',default=[]);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    selected=[]
    for folder in ('src','third_party','scripts','tests'):
        selected.extend(x for x in (ROOT/folder).rglob('*') if x.is_file() and x.suffix in ('.py','.json','.md','.sh','.yaml') and '__pycache__' not in x.parts)
    selected.extend(x for x in Path(__file__).parents[1].rglob('*') if x.is_file() and x.suffix in ('.py','.json','.md','.sh') and '__pycache__' not in x.parts)
    for folder in a.experiment:
        folder=folder.resolve()
        if ROOT/'experiments' not in folder.parents:raise ValueError('Extra experiment must belong to this repository')
        selected.extend(x for x in folder.rglob('*') if x.is_file() and x.suffix in ('.py','.json','.md','.sh') and '__pycache__' not in x.parts)
    if (ROOT/'conftest.py').exists():selected.append(ROOT/'conftest.py')
    selected=sorted(set(selected));records=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in selected:
            data=file.read_bytes();name=file.relative_to(ROOT).as_posix()
            archive.writestr(name,data);records.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':records}
        archive.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps({'path':str(a.output.resolve()),'files':len(records),'bytes':a.output.stat().st_size,
                      'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()

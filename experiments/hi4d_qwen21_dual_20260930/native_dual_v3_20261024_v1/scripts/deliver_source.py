"""Guarded scp delivery; only source/docs/static reports and syntax compilation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parents[4]
    run = Path(__file__).resolve().parents[1]
    repo = '/home/shangguanrz/project/pic-edit'
    remote_run = '/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/native_dual_v3_20261024_v1'
    baseline = json.loads((run/'source/baseline.json').read_text())
    prior = {}
    for p in sorted((run/'report').glob('remote_delivery*.json'), key=lambda p: p.stat().st_mtime_ns):
        prior.update({r['target']: r['sha256'] for r in json.loads(p.read_text())['files']})
    names = set(subprocess.check_output(['git', 'diff', '7c5f0a1', '--name-only'], cwd=root, text=True).splitlines())
    names.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines())
    entries = [(root/n, repo+'/'+n, baseline['files'].get(n)) for n in sorted(names)
               if not n.startswith('experiments/') and Path(n).suffix in ('.py', '.md', '.json')]
    entries.append((run/'README.md', remote_run+'/README.md', None))
    for folder, extension in (('scripts', '*.py'), ('report', '*.json')):
        entries += [(p, remote_run+'/'+folder+'/'+p.name, None) for p in (run/folder).glob(extension)]
    tag = 'delivery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage = run/'source'/tag
    stage.mkdir(exist_ok=False)
    rows = []
    with tarfile.open(stage/'source.tar.gz', 'x:gz') as archive:
        for i, (p, target, before) in enumerate(entries):
            archive.add(p, arcname=str(i))
            rows.append({'name': str(i), 'target': target, 'before_sha256': prior.get(target, before),
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (stage/'manifest.json').write_text(json.dumps({'baseline_head': baseline['head'], 'files': rows}, indent=2))
    apply = stage/'apply_delivery.py'
    apply.write_text('''from pathlib import Path
import hashlib,json,sys,tarfile,subprocess,py_compile
stage=Path(sys.argv[1]); manifest=json.loads((stage/'manifest.json').read_text()); rows=manifest['files']
repo='/home/shangguanrz/project/pic-edit'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
if head!=manifest['baseline_head']: raise RuntimeError('Remote HEAD changed; inspect and snapshot again')
for r in rows:
 p=Path(r['target'])
 assert p.suffix in ('.py','.md','.json')
 assert str(p).startswith((repo+'/','/data/shangguanrz/pic-edit-experiments/qwen21_hi4d_dual/'))
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() not in (r['before_sha256'],r['sha256']):
  raise RuntimeError('Concurrent source drift: '+str(p))
with tarfile.open(stage/'source.tar.gz') as archive:
 for r in rows:
  p=Path(r['target']); p.parent.mkdir(parents=True,exist_ok=True)
  p.write_bytes(archive.extractfile(r['name']).read())
compiled=[]
for r in rows:
 p=Path(r['target']); assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 if p.suffix=='.py':
  cache=stage/'bytecode'/(r['name']+'.pyc');cache.parent.mkdir(exist_ok=True)
  py_compile.compile(str(p),cfile=str(cache),doraise=True);compiled.append(str(p))
result={'status':'delivered_syntax_only_not_runtime_accepted','baseline_head':head,'files':rows,'compiled':compiled,
 'not_executed':['tests','project_imports','models','forward','backward','training','cache_export','weight_migration','package_installation']}
with (stage/'delivery_result.json').open('x') as f:json.dump(result,f,indent=2)
print('Delivered',len(rows),'source/docs/evidence files; syntax compiled',len(compiled),'Python files; no project imports')
''', encoding='utf8')
    remote = remote_run+'/source/'+tag
    subprocess.run(['ssh', 'sg', 'mkdir', '-p', remote], check=True)
    for p in stage.iterdir():
        subprocess.run(['scp', str(p), 'sg:'+remote+'/'], check=True)
    subprocess.run(['ssh', 'sg', '/usr/bin/python3', remote+'/apply_delivery.py', remote], check=True)
    subprocess.run(['scp', 'sg:'+remote+'/delivery_result.json', str(run/'report'/('remote_'+tag+'.json'))], check=True)


if __name__ == '__main__':
    main()

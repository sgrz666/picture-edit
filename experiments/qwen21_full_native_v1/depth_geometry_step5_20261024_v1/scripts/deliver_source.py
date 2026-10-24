"""Source/docs delivery with per-file baseline SHA guard; syntax compile only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path(__file__).resolve().parents[4]
    run = Path(__file__).resolve().parents[1]
    family = '/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1'
    remote_run = family + '/depth_geometry_step5_20261024_v1'
    repo = '/home/shangguanrz/project/pic-edit'
    baseline = json.loads((run / 'source/baseline.json').read_text())['files']
    prior = {}
    previous = sorted((run / 'report').glob('remote_syntax_delivery*.json'), key=lambda p: p.stat().st_mtime_ns)
    for path in previous:
        prior.update({r['target']: r['sha256'] for r in json.loads(path.read_text())['files']})
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
    new = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()
    entries = []
    for name in sorted(set(changed + new)):
        if name.startswith('experiments/'):
            continue
        p = root / name
        if p.suffix in ('.py', '.md', '.json'):
            entries.append((p, repo + '/' + name, prior.get(repo + '/' + name, baseline.get(name))))
    entries += [(run / 'README.md', remote_run + '/README.md', None),
                (root / 'experiments/qwen21_full_native_v1/README.md', family + '/README.md',
                 hashlib.sha256((run / 'source/baseline_experiment_README.md').read_bytes()).hexdigest())]
    for folder, suffix in (('scripts', '.py'), ('report', '.json')):
        entries += [(p, remote_run + '/' + folder + '/' + p.name, None) for p in (run / folder).glob('*' + suffix)]
    tag = 'delivery_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    stage = run / 'source' / tag
    stage.mkdir(exist_ok=False)
    summary = root / 'experiments/qwen21_full_native_v1/report/STEP5_ARCHITECTURE.md'
    if summary.exists():
        previous_summary = run / 'source/previous_STEP5_ARCHITECTURE.md'
        known_summary_sha = (hashlib.sha256(previous_summary.read_bytes()).hexdigest()
                             if previous_summary.exists() else None)
        entries.append((summary, family + '/report/STEP5_ARCHITECTURE.md',
                        prior.get(family + '/report/STEP5_ARCHITECTURE.md', known_summary_sha)))
    rows = []
    with tarfile.open(stage / 'source.tar.gz', 'x:gz') as tar:
        for i, (path, target, before) in enumerate(entries):
            before = prior.get(target, before)
            tar.add(path, arcname=str(i))
            rows.append({'name': str(i), 'target': target, 'before_sha256': before,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (stage / 'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    apply = stage / 'apply_delivery.py'
    apply.write_text('''from pathlib import Path
import hashlib,json,sys,tarfile,py_compile,shutil
stage=Path(sys.argv[1]); run=stage.parent.parent
rows=json.loads((stage/'manifest.json').read_text())
for row in rows:
 p=Path(row['target'])
 assert p.suffix in ('.py','.md','.json')
 assert str(p).startswith(('/home/shangguanrz/project/pic-edit/','/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/'))
 if p.exists():
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if digest not in (row['before_sha256'],row['sha256']): raise RuntimeError('Concurrent source drift: '+str(p))
with tarfile.open(stage/'source.tar.gz') as archive:
 for row in sorted(rows,key=lambda r:r['before_sha256'] is not None):
  p=Path(row['target']); p.parent.mkdir(parents=True,exist_ok=True)
  p.write_bytes(archive.extractfile(row['name']).read())
compiled=[]
for row in rows:
 p=Path(row['target'])
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 if p.suffix=='.py':
  target=stage/'bytecode'/(row['name']+'.pyc');target.parent.mkdir(exist_ok=True)
  py_compile.compile(str(p),cfile=str(target),doraise=True); compiled.append(str(p))
formal=run/'source';formal.mkdir(exist_ok=True)
for filename in ('baseline.json','source_baseline.tar.gz'):
 src=run/filename;dest=formal/filename
 if not dest.exists(): shutil.copyfile(src,dest)
result={'status':'source_delivered_syntax_only_not_runtime_accepted','files':rows,'compiled':compiled,
        'not_executed':['tests','project_import','model_forward','model_backward','training','rendering','cache_export','weight_migration']}
with (stage/'delivery_result.json').open('x') as stream:json.dump(result,stream,indent=2)
print('Delivered',len(rows),'source/docs/evidence files; syntax compiled',len(compiled),'Python files; no project imports')
''', encoding='utf8')
    remote = remote_run + '/source/' + tag
    subprocess.run(['ssh', 'sg', 'mkdir', '-p', remote], check=True)
    for p in stage.iterdir():
        subprocess.run(['scp', str(p), 'sg:' + remote + '/'], check=True)
    subprocess.run(['ssh', 'sg', '/usr/bin/python3', remote + '/apply_delivery.py', remote], check=True)
    result_path = run / 'report' / ('remote_syntax_' + tag + '.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    subprocess.run(['scp', 'sg:' + remote + '/delivery_result.json', str(result_path)], check=True)


if __name__ == '__main__':
    main()

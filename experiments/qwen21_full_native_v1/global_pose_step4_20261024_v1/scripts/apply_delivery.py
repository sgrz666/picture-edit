import json, pathlib, hashlib, tarfile, sys, py_compile
repo=pathlib.Path('/home/shangguanrz/project/pic-edit')
stage=pathlib.Path(sys.argv[1])
baseline=json.loads((stage/'baseline.json').read_text())['files']
files=json.loads((stage/'delivery.json').read_text())
prior=stage.parent/'last_delivery.json'
old=json.loads(prior.read_text()) if prior.exists() else {}
with tarfile.open(stage/'delivery.tar.gz') as tar:
 for row in files:
  dest=repo/row['path']
  if dest.exists():
   actual=hashlib.sha256(dest.read_bytes()).hexdigest()
   allowed={baseline.get(row['path']),old.get(row['path']),row['sha256']}
   if actual not in allowed: raise RuntimeError('Concurrent source drift: '+row['path'])
 for row in files:
  dest=repo/row['path']; dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_bytes(tar.extractfile(row['path']).read())
  py_compile.compile(str(dest),doraise=True)
  old[row['path']]=row['sha256']
prior.write_text(json.dumps(old,indent=2))
print('Delivered and compiled',len(files),'Python files')

import json,pathlib,sys,tarfile,re
from urllib.parse import unquote
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
sys.path.insert(0,str(run/'code'))
from src.pose_control.qwen21_full.prompt_study.protocol import sha256_file
from src.pose_control.qwen21_full.prompt_study.results import verify_report_relative_links
errors=verify_report_relative_links(run/'report/index.html',run)
if errors:raise ValueError(errors[:10])
paths={p for p in (run/'report').rglob('*') if p.is_file()}
for page in (run/'report').rglob('*.html'):
 for url in re.findall(r'(?:href|src)=["\']([^"\']+)["\']',page.read_text(),re.I):
  if url.startswith(('#','http:','https:','data:','javascript:')):continue
  target=(page.parent/unquote(url.split('?')[0].split('#')[0])).resolve()
  if not target.is_relative_to(run.resolve()) or not target.is_file():raise ValueError(f'Invalid referenced artifact: {url}')
  paths.add(target)
manifest=[]
for path in sorted(paths):
 if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):raise ValueError('Unexpected report symlink')
 if path.suffix.lower() in ('.pt','.safetensors','.bin','.onnx','.py'):raise ValueError('Unexpected binary/code reference')
 manifest.append({'path':path.relative_to(run).as_posix(),'bytes':path.stat().st_size,'sha256':sha256_file(path)})
export=run/'export';export.mkdir(exist_ok=True)
archive=export/'progress_report_20261011.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for row in manifest:tar.add(run/row['path'],arcname=row['path'],recursive=False)
(export/'progress_report_20261011.files.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'report_files':len(manifest),'archive_bytes':archive.stat().st_size,'relative_links_verified':True,'formal_experiment_completed':False}))

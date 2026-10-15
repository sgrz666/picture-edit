import hashlib,json,pathlib,tarfile,shutil
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
audit=run/'repair_audit'
records=json.loads((audit/'code_pe_instruction_v3_patch.json').read_text())
for record in records:
 if hashlib.sha256((run/'code'/record['path']).read_bytes()).hexdigest()!=record['original_sha256']:
  raise ValueError('Remote PE baseline changed')
 if any(p.is_file() for p in (run/'training').rglob('*')):raise RuntimeError('Training code must not change during formal training')
backup=audit/'pe_instruction_v3_before'
if backup.exists():raise RuntimeError('PE patch backup already exists; inspect first')
backup.mkdir()
expected={r['path']:r['repaired_sha256'] for r in records}
with tarfile.open(audit/'code_pe_instruction_v3_patch.tar.gz') as archive:
 for member in archive.getmembers():
  if not member.isfile() or member.name not in expected:raise ValueError('Unexpected patch member')
  payload=archive.extractfile(member).read()
  if hashlib.sha256(payload).hexdigest()!=expected[member.name]:raise ValueError('Patch bytes changed')
  target=run/'code'/member.name
  if not target.resolve().is_relative_to((run/'code').resolve()):raise ValueError('Patch escapes code')
  before=backup/member.name;before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,before)
  target.write_bytes(payload)
print(json.dumps({'installed':len(expected),'scoring_code_unchanged':True}))

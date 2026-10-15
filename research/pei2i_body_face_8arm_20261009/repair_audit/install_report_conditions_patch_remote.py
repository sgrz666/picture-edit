import hashlib,json,pathlib,tarfile
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
audit=run/'repair_audit'
records=json.loads((audit/'code_report_conditions_patch.json').read_text())
for record in records:
 if hashlib.sha256((run/'code'/record['path']).read_bytes()).hexdigest()!=record['original_sha256']:
  raise ValueError('Remote report patch baseline changed')
expected={r['path']:r['repaired_sha256'] for r in records}
with tarfile.open(audit/'code_report_conditions_patch.tar.gz') as archive:
 for member in archive.getmembers():
  if not member.isfile() or member.name not in expected:raise ValueError('Unexpected patch member')
  payload=archive.extractfile(member).read()
  if hashlib.sha256(payload).hexdigest()!=expected[member.name]:raise ValueError('Patch bytes changed')
  target=run/'code'/member.name
  if not target.resolve().is_relative_to((run/'code').resolve()):raise ValueError('Patch escapes code')
  target.write_bytes(payload)
print(json.dumps({'installed':len(expected),'scoring_and_training_code_unchanged':True}))

import hashlib,json,pathlib,tarfile
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
audit=run/'repair_audit'
records=json.loads((audit/'code_snapshot_repaired.json').read_text())
allowed={r['path']:r['repaired_sha256'] for r in records}
with tarfile.open(audit/'code_snapshot_repaired.tar.gz') as archive:
 for member in archive.getmembers():
  if not member.isfile() or member.name not in allowed:
   raise ValueError('Unexpected archive member')
  target=run/'code'/member.name
  if not target.resolve().is_relative_to((run/'code').resolve()):
   raise ValueError('Archive path escapes code directory')
  payload=archive.extractfile(member).read()
  if hashlib.sha256(payload).hexdigest()!=allowed[member.name]:
   raise ValueError('Archive file hash differs')
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(payload)
print(json.dumps({'installed':len(allowed),'code_root':str(run/'code')}))

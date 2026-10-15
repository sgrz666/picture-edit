import json,pathlib,tarfile
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
destination=run/'report'
destination.mkdir(exist_ok=True)
with tarfile.open(run/'repair_audit/qa8_v1_review.tar.gz') as archive:
 for member in archive.getmembers():
  target=destination/member.name
  if not target.resolve().is_relative_to(destination.resolve()) or member.issym() or member.islnk():
   raise ValueError('Invalid review archive path')
  if member.isdir():target.mkdir(parents=True,exist_ok=True)
  elif member.isfile():
   target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(archive.extractfile(member).read())
print(json.dumps({'review_page':str(destination/'development_qa8_v1/index.html')}))

import json, subprocess, pathlib
run = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python = '/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
code = "from pathlib import Path; from scripts.data.prepare_qwen21_prompt_study_faces import validate_face_assets; import json; r=Path(" + repr(str(run)) + "); result=validate_face_assets(r); print(json.dumps(result,ensure_ascii=False) if result is not None else 'Face asset verification passed')"
result = subprocess.run([python, '-c', code], cwd=run/'code', capture_output=True, text=True)
print(result.stdout[-3000:]); print(result.stderr[-1000:]); print('exit_code',result.returncode)
for split in ('train512', 'val24', 'diagnostic6'):
 row = json.loads((run/'data'/f'{split}_pairs.jsonl').read_text().splitlines()[0])
 print(split, {k:v for k,v in row.items() if 'mask' in k})
payload = dict(command=code, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
(run/'repair_audit/face_validation.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
raise SystemExit(result.returncode)

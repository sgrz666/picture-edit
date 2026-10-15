import hashlib,json,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
audit=run/'repair_audit'
expected={}
for name in ('code_snapshot_repaired.json','code_report_conditions_patch.json','code_pe_instruction_v3_patch.json'):
 expected.update({r['path']:r['repaired_sha256'] for r in json.loads((audit/name).read_text())})
for name,sha in expected.items():
 if hashlib.sha256((run/'code'/name).read_bytes()).hexdigest()!=sha:raise ValueError(f'Current code differs: {name}')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
tests=sorted(p.relative_to(run/'code').as_posix() for p in (run/'code/tests').glob('test_qwen21_prompt_study*.py'))
tests+=['tests/test_qwen21_joint_pilot_visuals.py','tests/test_qwen21_native_resolution.py','tests/test_qwen21_frozen_bfs_lora.py']
with (audit/'verification_current_remote.log').open('w') as output:
 result=subprocess.run([python,'-m','pytest',*tests,'-q'],cwd=run/'code',stdout=output,stderr=subprocess.STDOUT)
summary={'current_scoped_files':len(expected),'code_hashes_verified':True,'test_exit_code':result.returncode,
 'test_log':'repair_audit/verification_current_remote.log','tail':(audit/'verification_current_remote.log').read_text()[-2200:]}
(audit/'verification_current_remote.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));raise SystemExit(result.returncode)

import json,pathlib,subprocess
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
python='/home/shangguanrz/nihaochen/envs/qwen21/bin/python'
subprocess.run([python,str(run/'code/scripts/reports/build_qwen21_pei2i_study_report.py'),
 '--run',str(run)],cwd=run/'code',check=True)
summary=json.loads((run/'report/summary.json').read_text())
print(json.dumps({key:summary[key] for key in ('status','is_completed','train_updates','pe_success','completed_eval_jobs','scoring_errors_count','provenance_errors_count')},ensure_ascii=False))

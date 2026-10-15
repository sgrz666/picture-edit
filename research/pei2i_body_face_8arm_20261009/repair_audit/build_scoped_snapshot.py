"""Package only reviewed repair files, recording original and repaired SHA256."""
import hashlib,json,tarfile
from pathlib import Path
root=Path('C:/Users/sg/.codex/worktrees/pei2i-experiment-repair/pictureedit')
audit=Path(__file__).resolve().parent
original={r['path']:r['sha256'] for r in json.loads((root/'.pei2i_repair_source_snapshot.json').read_text(encoding='utf-8'))}
names=['scripts/data/prepare_qwen21_prompt_study.py','scripts/data/prepare_qwen21_prompt_study_faces.py',
 'scripts/precompute_qwen21_pei2i_prompts.py','scripts/precompute_qwen21_prompt_study_cache.py',
 'scripts/train/train_qwen21_joint_pilot.py','scripts/eval/eval_qwen21_joint_pilot.py',
 'scripts/run/run_qwen21_pei2i_study.py','scripts/run/check_qwen21_prompt_study_resume.py',
 'scripts/reports/build_qwen21_pei2i_study_report.py','src/pose_control/qwen21_full/data_runtime.py',
 'tests/test_qwen21_joint_pilot_visuals.py','docs/experiments/2026-10-10-pei2i-eight-arm-runbook.md']
names+= [p.relative_to(root).as_posix() for p in (root/'src/pose_control/qwen21_full/prompt_study').glob('*.py')]
names+= [p.relative_to(root).as_posix() for p in (root/'tests').glob('test_qwen21_prompt_study*.py')]
records=[]
with tarfile.open(audit/'code_snapshot_repaired.tar.gz','w:gz') as archive:
 for relative in sorted(set(names)):
  p=root/relative
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  records.append({'path':relative,'original_sha256':original.get(relative),'repaired_sha256':digest})
  archive.add(p,arcname=relative,recursive=False)
(audit/'code_snapshot_repaired.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(records),'archive_bytes':(audit/'code_snapshot_repaired.tar.gz').stat().st_size}))

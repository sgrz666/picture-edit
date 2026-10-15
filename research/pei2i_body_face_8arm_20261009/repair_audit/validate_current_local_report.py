import json,hashlib,sys
from pathlib import Path
root=Path('D:/codeplus/pictureedit')
run=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from src.pose_control.qwen21_full.prompt_study.results import verify_report_relative_links
manifest=json.loads((run/'repair_audit/progress_report_20261011.files.json').read_text())
for row in manifest:
 if hashlib.sha256((run/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Local artifact changed')
errors=verify_report_relative_links(run/'report/index.html',run)
errors+=verify_report_relative_links(run/'report/development_qa8_v1/index.html',run)
if errors:raise ValueError(errors[:10])
summary=json.loads((run/'report/summary.json').read_text(encoding='utf-8'))
print(json.dumps({'verified_files':len(manifest),'missing_links':0,'formal_complete':summary['is_completed'],
 'local_page':str(run/'report/index.html'),'images_in_history':len(list((run/'report/history').glob('*.png')))}))

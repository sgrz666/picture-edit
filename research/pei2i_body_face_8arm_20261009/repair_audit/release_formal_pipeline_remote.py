"""Bind the completed visual review to remote evidence and release the live controller."""
import json
import os
import pathlib
import shutil
import sys
from datetime import datetime, timezone

run = pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
sys.path.insert(0, str(run / 'code'))
from scripts.precompute_qwen21_pei2i_prompts import qa8_review_provenance, validate_qa8_review
from src.pose_control.qwen21_full.prompt_study.prompts import request_sha256
from src.pose_control.qwen21_full.prompt_study.protocol import atomic_json, sha256_file

state = json.loads((run / 'repair_audit/full_pipeline_state.json').read_text())
assert state['status'] == 'waiting_for_signed_qa_review', state
os.kill(state['pid'], 0)
launch = json.loads((run / 'repair_audit/qa8_v3_launch.json').read_text())
assert launch['status'] == 'complete' and launch['exit_code'] == 0
visual_path = run / 'repair_audit/qa8_v3_visual_review.json'
visual = json.loads(visual_path.read_text())
assert visual['reviewed_records'] == 24
assert visual['raw_answer_parse_and_hash_checks_passed'] is True
assert visual['decision'] == 'release_for_controlled_experiment_with_recorded_prompt_errors'
reviewed = {(row['protocol'], row['pair_id']): row for row in visual['records']}
assert len(reviewed) == 24
for proto in ('pg', 'pb', 'pbg'):
    rows = [json.loads(line) for line in (run / f'prompts/qa8/{proto}/train.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == 8
    for row in rows:
        observed = reviewed[proto, row['pair_id']]
        for key in ('request_sha256', 'raw_sha256', 'compiled_prompt_sha256'):
            assert observed[key] == row[key], (proto, row['pair_id'], key)

payload = qa8_review_provenance(run)
payload.update(
    reviewed=True,
    reviewed_at=datetime.now(timezone.utc).isoformat(),
    notes=visual['notes'],
    decision=visual['decision'],
    visual_review_path='repair_audit/qa8_v3_visual_review.json',
    visual_review_sha256=sha256_file(visual_path),
    controller_pid=state['pid'],
)
payload['review_signature_sha256'] = request_sha256(payload)
review_path = run / 'prompts/qa8.json'
if review_path.exists():
    original_digest = sha256_file(review_path)
    backup = run / f'repair_audit/qa8_before_v3_signed_review_{original_digest}.json'
    if not backup.exists():
        shutil.copyfile(review_path, backup)
atomic_json(review_path, payload)
validate_qa8_review(run)
print(json.dumps({
    'reviewed': True, 'reviewed_records': 24,
    'review_sha256': sha256_file(review_path),
    'decision': visual['decision'], 'controller_pid': state['pid'],
    'controller_will_continue': True,
    'formal_optimizer_updates_at_release': 0,
}, indent=2))

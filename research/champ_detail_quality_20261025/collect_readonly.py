"""Snapshot existing CHAMP inputs/keypoints and hashes without writing the dataset."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path('/data/shangguanrz/pic-edit-datasets/CHAMP_pilot_20261024')
directory = root / 'cleaned_training_images/train_rgb_pristine_v2'
project = Path('/home/shangguanrz/project/pic-edit')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for path in sorted(directory.glob('*.png')):
    seq, stem = path.stem.split('_', 1)
    original = root / f'frames/{seq}/rgb/{stem}.png'
    keypoints = root / f'assets/{seq}/dwpose/{stem}.json'
    rows.append({
        'frame_id': path.stem, 'sequence_id': seq, 'stem': stem,
        'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path),
        'source_path': str(original), 'source_sha256': sha(original),
        'dwpose_path': str(keypoints), 'dwpose_sha256': sha(keypoints),
        'dwpose': json.loads(keypoints.read_text(encoding='utf-8')),
    })

files = ['qc/face_hand_quality_audit.json', 'qc/detail_roi/detail_roi.jsonl',
         'qc/hand_gate/per_frame.jsonl', 'manifests/frame_quality.jsonl',
         'CHAMP_v1/manifests/pairs/train_pairs_pristine_v2.jsonl']
snapshot = {
    'schema': 'champ_detail_quality_readonly_snapshot_v1',
    'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'source_directory': str(directory), 'frame_count': len(rows),
    'all_exported_pngs_byte_identical_to_source': all(r['sha256'] == r['source_sha256'] for r in rows),
    'frames': rows,
    'metadata_hashes': {rel: sha(root / rel) for rel in files},
    'git_log': subprocess.run(['git', '-C', str(project), 'log', '-4', '--oneline'], capture_output=True, text=True, check=True).stdout,
    'script_sha256': sha(Path(__file__)),
    'actions': ['read existing PNG bytes', 'read existing JSON/keypoints', 'read git log'],
    'dataset_files_modified': False, 'gpu_models_loaded': False,
}
print(json.dumps(snapshot, ensure_ascii=False, indent=2))

from pathlib import Path
import json
r=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008')
for filename in ['evaluation/J_common_tiktok/config.json','evaluation/J_common_tiktok/checkpoint_metadata.json','training/J/config.json']:
 p=r/filename
 print(filename,p.read_text()[:6000] if p.exists() else 'missing')
for p in (r/'evaluation/J_common_tiktok').glob('*prompt*'): print(p)

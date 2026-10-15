import json,pathlib,sys,time
run=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009')
sys.path.insert(0,str(run/'code'))
from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock,atomic_json,sha256_file
from src.pose_control.qwen21_full.prompt_study.measurement import validate_scoring_runtime,LOSS_FILES
from src.evaluation.joint_pilot_metrics import IdentityMetric,PoseMetric
from PIL import Image
import cv2,numpy as np,torch
started=time.time()
lock=load_study_lock(run/'data/lock.json',verify_assets=False)
signature=validate_scoring_runtime(lock,run/'code/models',run/'code')
identity,pose=IdentityMetric(run/'code/models'),PoseMetric(run/'code/models')
pairs=[json.loads(line) for line in (run/'data/train512_pairs.jsonl').read_text().splitlines() if line.strip()][:8]
records=[]
for row in pairs:
 with Image.open(row['source_rgb']) as saved:source=saved.convert('RGB')
 with Image.open(row['target_rgb']) as saved:target=saved.convert('RGB')
 boxes,_=identity.det.detect(cv2.cvtColor(np.asarray(source),cv2.COLOR_RGB2BGR))
 detected=pose.detect(target)
 records.append({'pair_id':row['pair_id'],'source_size':list(source.size),'target_size':list(target.size),
  'source_faces':len(boxes),'source_identity_self_cosine':identity.cosine(identity.embedding(source),identity.embedding(source)) if len(boxes)==1 else None,
  'target_pose_detected':detected is not None,'target_pose_visible_joints':int((detected[1]>.3).sum()) if detected is not None else 0})
losses={}
for relative in LOSS_FILES:
 model=torch.jit.load(str(run/'code/models'/relative),map_location='cpu').eval()
 losses[relative]={'loaded':True,'sha256':sha256_file(run/'code/models'/relative)}
 del model
result={'schema':'measurement_cpu_input_smoke_v1','passed':True,'formal_generation_count':0,
 'scoring_signature_sha256':signature,'loss_models':losses,'input_checks':records,'seconds':time.time()-started}
atomic_json(run/'preflight/measurement_cpu_smoke.json',result)
print(json.dumps(result,indent=2))

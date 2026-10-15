import json,os
R="/home/shangguanrz/project/pic-edit/experiments"
D=os.path.join(R,"qwen21_hi4d_dual")
def tail(f,n=2):
    try: return "".join(open(f).readlines()[-n:])
    except Exception as e: return "ERR "+str(e)
print("### action_prompt_v2/completion_status.json ###");print(open(os.path.join(D,"action_prompt_v2","completion_status.json")).read())
print("### action_prompt_v2 train_metrics tail ###");print(tail(os.path.join(D,"action_prompt_v2","train_metrics.jsonl")))
print("### action_prompt_v2 train.log tail ###");print(tail(os.path.join(D,"action_prompt_v2","train.log"),8))
print("### action_prompt_v2 run_manifest args ###")
m=json.load(open(os.path.join(D,"action_prompt_v2","run_manifest.json")));print(json.dumps(m.get("args",m),indent=1)[:900])
# full_detailed numbers from pair14 eval
print("### pair14 eval_seed42 all modes ###")
s=json.load(open(os.path.join(D,"single_pair14_action_v2_overfit","eval_seed42","summary.json")))
for mode,v in s["mean"].items():
    print(mode,"::",{k:round(v[k],4) for k in ["pose_pck_a","pose_pck_b","contact_distance_error","body_identity_a","body_identity_b","background_mae","identity_swap","detected_person_count"] if k in v and v[k] is not None})
print("### single_hug_eval full mode ###")
sh=json.load(open(os.path.join(D,"single_hug_eval","summary.json")))
for mode,v in sh["mean"].items():
    print(mode,"::",{k:round(v[k],4) for k in ["pose_pck_a","pose_pck_b","contact_distance_error","body_identity_a","body_identity_b","background_mae","identity_swap","detected_person_count"] if k in v and v[k] is not None})
print("### single_hug_overfit/report.md head ###");print(open(os.path.join(D,"single_hug_overfit","report.md")).read()[:1500])
print("### single_hug_overfit train_metrics tail ###");print(tail(os.path.join(D,"single_hug_overfit","train_metrics.jsonl")))
print("### single_pair14_overfit train_metrics tail ###");print(tail(os.path.join(D,"single_pair14_action_v2_overfit","train_metrics.jsonl")))

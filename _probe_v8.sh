import json,os
R="/home/shangguanrz/project/pic-edit/experiments"
V8=os.path.join(R,"qwen21_dual_v8")
for sub in ["pilot_pair14_v8_300_v1","eval_pair14_v8_300_v1","eval_pair14_v8_300_rescore_v2","gates_pair14_v1","gates_pair14_v2","gates_pair14_v3","gates_pair14_v4","mesh_worker_smoke_v1"]:
    p=os.path.join(V8,sub); print(f"\n=== {sub} ===")
    if not os.path.isdir(p): print("MISSING",os.path.islink(p)); continue
    print(" files:",os.listdir(p))
print("### rescore_v2.log tail ###")
print("".join(open(os.path.join(V8,"eval_pair14_v8_300_rescore_v2.log")).readlines()[-25:]))
print("### cache_precision_probe_v1.json ###")
print(json.dumps(json.load(open(os.path.join(V8,"cache_precision_probe_v1.json"))),indent=1)[:1200])
print("### eval rescore_v2 dir files ###",os.listdir(os.path.join(V8,"eval_pair14_v8_300_rescore_v2")))
for f in os.listdir(os.path.join(V8,"eval_pair14_v8_300_rescore_v2")):
    if f.endswith(".json"):
        print(" --",f,"--");print(json.dumps(json.load(open(os.path.join(V8,"eval_pair14_v8_300_rescore_v2",f))),indent=1)[:1500])

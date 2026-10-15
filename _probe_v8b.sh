import json,os
V8="/home/shangguanrz/project/pic-edit/experiments/qwen21_dual_v8"
def show(p,lim=1400):
    fp=os.path.join(V8,p)
    if os.path.exists(fp): print(f"\n##### {p} #####");print(json.dumps(json.load(open(fp)),indent=1)[:lim])
    else: print("\n##### %s MISSING #####"%p)
show("pilot_pair14_v8_300_v1/real_model_gates.json")
show("pilot_pair14_v8_300_v1/training_complete.json")
show("pilot_pair14_v8_300_v1/process_exit.json")
show("pilot_pair14_v8_300_v1/run_manifest.json",900)
show("gates_pair14_v3/real_model_gates.json",2500)
# tail pilot train metrics
f=os.path.join(V8,"pilot_pair14_v8_300_v1","train_metrics.jsonl")
if os.path.exists(f):
    L=open(f).readlines(); print("\npilot train lines:",len(L));print("".join(L[-2:]))
# rescore summary
for sub in ["eval_pair14_v8_300_v1","eval_pair14_v8_300_rescore_v2"]:
    p=os.path.join(V8,sub)
    for f in os.listdir(p):
        if "summary" in f or f=="results.json" or f=="metrics.json":
            print(f"\n== {sub}/{f} ==");print(json.dumps(json.load(open(os.path.join(p,f))),indent=1)[:1800])

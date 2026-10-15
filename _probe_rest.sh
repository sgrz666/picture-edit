import json,os
R="/home/shangguanrz/project/pic-edit/experiments"
print("########## 16-test results.json per stage ##########")
for sub in ["stage1_baselines","stage2_adapters","stage3_detailed_prompt","stage4_bfs_body","stage5_bfs_bracket_norm","stage6_pei2i"]:
    p=os.path.join(R,"16-test",sub)
    files=[f for f in os.listdir(p)] if os.path.isdir(p) else []
    print(f"\n=== {sub} === files:",files)
    for f in files:
        if f.endswith(".json") and ("result" in f or "metric" in f or "summary" in f):
            print(" --",f,"--");print(json.dumps(json.load(open(os.path.join(p,f))),indent=1)[:900])
print("\n########## 16-test prompts.json head ##########");print(open(os.path.join(R,"16-test","prompts.json")).read()[:600])
print("\n########## hi4d_overfit_archive ##########")
for sub in ["hug_1000steps","hug_120steps"]:
    p=os.path.join(R,"hi4d_overfit_archive",sub); print(sub,"files:",os.listdir(p))

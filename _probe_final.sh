import json,os
R="/home/shangguanrz/project/pic-edit/experiments"
for sub in ["hug_1000steps","hug_120steps"]:
    p=os.path.join(R,"hi4d_overfit_archive",sub)
    print(f"\n##### hi4d_overfit_archive/{sub} summary.json #####")
    print(json.dumps(json.load(open(os.path.join(p,"summary.json"))),indent=1)[:1200])
    c=os.path.join(p,"experiment_config.json")
    if os.path.exists(c): print("cfg:",json.dumps(json.load(open(c)),indent=1)[:600])
# 16-test counts
print("\n##### 16-test case counts #####")
for sub in ["stage1_baselines","stage2_adapters","stage3_detailed_prompt","stage4_bfs_body","stage5_bfs_bracket_norm","stage6_pei2i"]:
    p=os.path.join(R,"16-test",sub)
    png=len([f for f in os.listdir(p) if f.endswith(".png")])
    imgs=len(os.listdir(os.path.join(p,"images"))) if os.path.isdir(os.path.join(p,"images")) else 0
    rc=0
    for f in os.listdir(p):
        if f.endswith(".json") and ("result" in f):
            try: rc+=len(json.load(open(os.path.join(p,f))))
            except: pass
    print(sub,"png_top",png,"images/",imgs,"result_cases",rc)
# qwen-2.1-test
Q=os.path.join(R,"qwen-2.1-test")
print("\n##### qwen-2.1-test tree #####")
for root,ds,fs in os.walk(Q):
    if root.count(os.sep)-Q.count(os.sep)>=2: ds[:]=[]
    rel=root.replace(Q,"")
    print(rel or "/","dirs",[d for d in ds],"files",[f for f in fs if not f.endswith('.png')][:12])

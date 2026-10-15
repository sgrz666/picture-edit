import json,os
R="/home/shangguanrz/project/pic-edit/experiments"
D=os.path.join(R,"qwen21_hi4d_dual")
print("##### hi4d_dual/run_manifest.json #####");print(open(os.path.join(D,"run_manifest.json")).read())
print("##### train_metrics.jsonl tail #####")
print("\n".join(open(os.path.join(D,"train_metrics.jsonl")).readlines()[-3:]))
for sub in ["single_hug_eval","single_hug_eval_seed43","single_hug_overfit","single_pair14_action_v2_overfit","action_prompt_v2","action_prompt_v2_smoke","prompt_audit_pair14","prompt_audit_pair14_v2"]:
    p=os.path.join(D,sub)
    print(f"\n===== {sub} =====")
    if not os.path.isdir(p): print("MISSING"); continue
    print(" files:",os.listdir(p))
    for f in ["summary.json","eval_summary.json","results.json"]:
        fp=os.path.join(p,f)
        if os.path.exists(fp):
            print(f" --{f}--"); print(json.dumps(json.load(open(fp)),indent=1)[:1600])
    # nested eval_seed42
    es=os.path.join(p,"eval_seed42")
    if os.path.isdir(es):
        print(" eval_seed42 files:",os.listdir(es))
        s42=os.path.join(es,"summary.json")
        if os.path.exists(s42): print(" eval_seed42 summary:",json.dumps(json.load(open(s42)),indent=1)[:1600])

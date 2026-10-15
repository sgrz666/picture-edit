import json,glob,os
B="/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body"
keys=["psnr","ssim","mae","pose_pck_002","pose_pck_005","identity_cosine","identity_detected","human_mae","background_mae_source","seconds","peak_cuda_gib"]
for d in ["four_arm_eval_50_s40","four_arm_eval_50_s30","four_arm_eval_50"]:
    p=os.path.join(B,d,"summary.json")
    if not os.path.exists(p): print("NO",p); continue
    s=json.load(open(p))
    print("\n=====",d,"num_samples",s.get("num_samples"),"=====")
    for arm,v in s["arms"].items():
        row=" | ".join(f"{k}={v[k]['mean']:.4f}(n{v[k]['count']})" for k in keys if k in v)
        print(arm,"::",row)
    pd=s.get("paired_deltas_second_minus_first")
    if pd: print("paired_deltas:",json.dumps(pd))
# eval_50 & deepgen_eval_50 & pilot_dev_eval_8 summaries
for d in ["eval_50","eval_50_s30","deepgen_eval_50","deepgen_eval_50_s40","pilot_dev_eval_8"]:
    p=os.path.join(B,d,"summary.json")
    if not os.path.exists(p):
        print("\n",d,"NO summary.json; files:",os.listdir(os.path.join(B,d))[:12]); continue
    s=json.load(open(p))
    print("\n=====",d,"=====","topkeys",list(s.keys())[:15])
    for k in ("psnr","ssim","mae","pose_pck_002","pose_pck_005","identity_cosine","seconds","peak_cuda_gib"):
        if k in s: print("  ",k,s[k])
    if "arms" in s:
        for arm,v in s["arms"].items():
            print("  arm",arm,{kk:round(v[kk]['mean'],4) for kk in ["psnr","ssim","mae","pose_pck_002","pose_pck_005","identity_cosine"] if kk in v})
    if "overall" in s: print("  overall",s["overall"])

import json,os
B="/home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body"
for d in ["eval_50","eval_50_s30","pilot_dev_eval_8"]:
    s=json.load(open(os.path.join(B,d,"summary.json")))
    print("\n====",d,"steps",s.get("steps"),"seed",s.get("seed"),"num_samples",s.get("num_samples"),"ckpt",os.path.basename(str(s.get("checkpoint"))))
    m=s.get("mean",{})
    for k in ["psnr","ssim","mae","pose_pck_002","pose_pck_005","identity_cosine","human_mae","background_mae_source","seconds","peak_cuda_gib"]:
        if k in m: print("   ",k,round(m[k],4) if isinstance(m[k],(int,float)) else m[k])
for d in ["deepgen_eval_50","deepgen_eval_50_s40"]:
    p=os.path.join(B,d,"eval_metrics.json")
    s=json.load(open(p))
    print("\n====",d,"eval_metrics.json topkeys",list(s.keys())[:20])
    for k in ("mean","summary","aggregate","psnr"):
        if k in s: print("  ",k,json.dumps(s[k])[:600])
# images count
for d in ["four_arm_eval_50_s40","four_arm_eval_50","deepgen_eval_50","eval_50","pilot_dev_eval_8","controlled_smoke","native_smoke"]:
    n=len([f for f in os.listdir(os.path.join(B,d)) if f.lower().endswith(".png")]) if os.path.isdir(os.path.join(B,d)) else 0
    print("PNG at top of",d,"=",n)

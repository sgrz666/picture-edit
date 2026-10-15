import json,os,glob
R="/home/shangguanrz/project/pic-edit/experiments"
Q=os.path.join(R,"qwen-2.1-test","experiment","results")
print("### qwen-2.1-test results ###")
for sub in sorted(os.listdir(Q)):
    p=os.path.join(Q,sub)
    if not os.path.isdir(p): continue
    js=[os.path.basename(x) for x in glob.glob(p+"/*.json")]
    print(f"-- {sub} -- json:",js,"pngs:",len(glob.glob(p+"/**/*.png",recursive=True)))
    for j in js:
        try: print("   ",j,json.dumps(json.load(open(os.path.join(p,j))),indent=1)[:700])
        except Exception as e: print("   ",j,"ERR",e)
# overfit summary search anywhere
for f in glob.glob(R+"/qwen-2.1-test/**/*summary*.json",recursive=True)+glob.glob(R+"/qwen-2.1-test/**/metrics*.json",recursive=True):
    print("FOUND",f);print(json.dumps(json.load(open(f)),indent=1)[:800])
print("\n### tiktok_v3_smoke ###")
T=os.path.join(R,"qwen21_tiktok_v3_smoke")
for sub in ["body","face","face_features","qwen21_cache"]:
    p=os.path.join(T,sub)
    print(f"-- {sub} --",os.listdir(p)[:15])
    for f in os.listdir(p):
        if f.endswith(".json"):
            print("   ",f,json.dumps(json.load(open(os.path.join(p,f))),indent=1)[:500])

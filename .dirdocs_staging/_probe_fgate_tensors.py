import torch, os, json
os.chdir('/home/shangguanrz/project/pic-edit/experiments')

def desc(path):
    try:
        obj=torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return "load-fail: %s"%e
    def fmt(v):
        import torch as T
        if isinstance(v,T.Tensor): return "Tensor%s %s"%(tuple(v.shape),str(v.dtype))
        if isinstance(v,dict): return "dict(%d){%s}"%(len(v),",".join(list(v.keys())[:8]))
        if isinstance(v,(list,tuple)): return "%s len=%d"%(type(v).__name__,len(v))
        s=str(v); return s[:60]
    if isinstance(obj,dict):
        return {k:fmt(v) for k,v in obj.items()}
    return fmt(obj)

for f in ['tiktok_overfit_sanity_full/condition_cache/00001_0014.pt',
          'iper_v65_face_overfit_024_6/condition_cache/024_6_source_motion_f000375.pt',
          'tiktok_00001_condition_repair/conditions/pose_only.pt',
          'tiktok_00001_condition_repair/conditions/pose_part.pt']:
    print("==",f,"==")
    print(json.dumps(desc(f),ensure_ascii=False,default=str))

print("== train_pose_part_1500 tree ==")
for root,dirs,files in os.walk('tiktok_00001_condition_repair/train_pose_part_1500'):
    for fn in files:
        if fn in ('README.md','目录说明.md'): continue
        p=os.path.join(root,fn)
        print("  %10d  %s"%(os.path.getsize(p),p))

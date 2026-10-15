import json, os
os.chdir('/home/shangguanrz/project/pic-edit/experiments')

def keys(d): 
    return list(d.keys()) if isinstance(d,dict) else type(d).__name__

print("###### iper_v65_face_overfit_024_6/metrics.json ######")
d=json.load(open('iper_v65_face_overfit_024_6/metrics.json'))
print("topkeys:", keys(d)[:60])
for k in ('architecture_version','dataset','appearance','source_stem','target_stem','seed','total_steps','max_steps','steps','final','verdict','checks','loss_terms'):
    if isinstance(d,dict) and k in d:
        v=d[k]
        print(" ",k,"=",json.dumps(v,ensure_ascii=False)[:300])
# if there is a per-step list, print first and last
if isinstance(d,dict):
    for k,v in d.items():
        if isinstance(v,list) and v and isinstance(v[0],dict):
            print(" LIST",k,"len",len(v),"first",json.dumps(v[0],ensure_ascii=False)[:200],"last",json.dumps(v[-1],ensure_ascii=False)[:200])

print("###### iper_v65_face_overfit_smoke/metrics.json ######")
d=json.load(open('iper_v65_face_overfit_smoke/metrics.json'))
print("topkeys:", keys(d)[:60])
for k,v in (d.items() if isinstance(d,dict) else []):
    if isinstance(v,(int,float,str,bool)): print("  ",k,"=",v)
    elif isinstance(v,list) and v and isinstance(v[0],dict):
        print("  LIST",k,"len",len(v),"first",json.dumps(v[0],ensure_ascii=False)[:200],"last",json.dumps(v[-1],ensure_ascii=False)[:200])

print("###### tiktok_v65_face_overfit_00001/metrics.json ######")
d=json.load(open('tiktok_v65_face_overfit_00001/metrics.json'))
for k,v in (d.items() if isinstance(d,dict) else []):
    if isinstance(v,(int,float,str,bool)): print("  ",k,"=",v)
    elif isinstance(v,list) and v and isinstance(v[0],dict):
        print("  LIST",k,"len",len(v),"first",json.dumps(v[0],ensure_ascii=False)[:220],"last",json.dumps(v[-1],ensure_ascii=False)[:220])
    else:
        print("  ",k,"=",json.dumps(v,ensure_ascii=False)[:160])

print("###### tiktok_v65_test_2steps/metrics.json ######")
d=json.load(open('tiktok_v65_test_2steps/metrics.json'))
for k,v in (d.items() if isinstance(d,dict) else []):
    if isinstance(v,(int,float,str,bool)): print("  ",k,"=",v)
    elif isinstance(v,list) and v and isinstance(v[0],dict):
        print("  LIST",k,"len",len(v),"first",json.dumps(v[0],ensure_ascii=False)[:220],"last",json.dumps(v[-1],ensure_ascii=False)[:220])

print("###### train_pose_part_1500/metrics.json (condition_repair) ######")
d=json.load(open('tiktok_00001_condition_repair/train_pose_part_1500/metrics.json'))
for k,v in (d.items() if isinstance(d,dict) else []):
    if isinstance(v,(int,float,str,bool)): print("  ",k,"=",v)
    elif isinstance(v,list) and v and isinstance(v[0],dict):
        print("  LIST",k,"len",len(v),"first",json.dumps(v[0],ensure_ascii=False)[:220],"last",json.dumps(v[-1],ensure_ascii=False)[:220])

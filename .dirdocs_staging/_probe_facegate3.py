import json, os
os.chdir('/home/shangguanrz/project/pic-edit/experiments')

print("===== iper_v65_face_overfit_024_6 fixed_off_on_comparisons (full) =====")
d=json.load(open('iper_v65_face_overfit_024_6/metrics.json'))
for e in d['fixed_off_on_comparisons']:
    print(json.dumps(e,ensure_ascii=False))
print("non_face changed?", d.get('non_face_sha256_before')==d.get('non_face_sha256_after'), "elapsed", d.get('elapsed_seconds'))
print("optimizer_groups keys:", list(d.get('optimizer_groups',{}).keys()))

print("===== iper_v65_face_overfit_024_6 loss terms at step 1,250,500,750,1000,1250,1500 =====")
ms=d['metrics']
for want in (1,250,500,750,1000,1250,1500):
    for m in ms:
        if m['step']==want:
            print(json.dumps(m,ensure_ascii=False))

print("===== train_pose_part_1500 fixed_off_on_comparisons (full) =====")
d=json.load(open('tiktok_00001_condition_repair/train_pose_part_1500/metrics.json'))
for e in d['fixed_off_on_comparisons']:
    print(json.dumps(e,ensure_ascii=False))

print("===== iper_v64_detail_overfit_024_6/overfit_summary.json =====")
d=json.load(open('iper_v64_detail_overfit_024_6/overfit_summary.json'))
def show(d,depth=0):
    if isinstance(d,dict):
        for k,v in d.items():
            if isinstance(v,(dict,list)):
                print("  "*depth+str(k)+":", ("list len "+str(len(v))) if isinstance(v,list) else "")
                if isinstance(v,list) and v and isinstance(v[0],(int,float,str)):
                    print("  "*depth+"  sample:",json.dumps(v[:5],ensure_ascii=False))
                elif isinstance(v,dict) and depth<1:
                    show(v,depth+1)
            else:
                print("  "*depth+str(k)+" =",v)
show(d)

print("===== iper_v64 eval_metrics.csv head =====")
print(open('iper_v64_detail_overfit_024_6/eval_metrics.csv').read()[:1200])

print("===== tiktok_objective_eval/objective_eval.json =====")
d=json.load(open('tiktok_objective_eval/objective_eval.json'))
show(d)

print("===== tiktok_objective_eval/timestep_sweep.json =====")
d=json.load(open('tiktok_objective_eval/timestep_sweep.json'))
print(json.dumps(d,ensure_ascii=False)[:1500])

print("===== tiktok_smoke_00001/smoke_report.json =====")
print(open('tiktok_smoke_00001/smoke_report.json').read())

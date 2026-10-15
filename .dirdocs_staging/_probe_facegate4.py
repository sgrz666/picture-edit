import json, os, glob, subprocess
os.chdir('/home/shangguanrz/project/pic-edit/experiments')

print("===== iper_v64_detail_overfit_024_6 criteria / eval / val =====")
d=json.load(open('iper_v64_detail_overfit_024_6/overfit_summary.json'))
print("criteria:")
for c in d['criteria']: print("  ",json.dumps(c,ensure_ascii=False))
print("all_criteria_passed=",d.get('all_criteria_passed'))
print("eval_history:")
for e in d['eval_history']: print("  ",json.dumps(e,ensure_ascii=False)[:240])
print("val_history first/last:")
print("  ",json.dumps(d['val_history'][0],ensure_ascii=False)[:240])
print("  ",json.dumps(d['val_history'][-1],ensure_ascii=False)[:240])
print("train_history first/last:")
print("  ",json.dumps(d['train_history'][0],ensure_ascii=False)[:220])
print("  ",json.dumps(d['train_history'][-1],ensure_ascii=False)[:220])

print("===== objective_eval snapshots =====")
d=json.load(open('tiktok_objective_eval/objective_eval.json'))
for s in d['snapshots']: print("  ",json.dumps(s,ensure_ascii=False)[:220])
print("===== timestep_sweep all rows (facesim_delta / lpips) =====")
t=json.load(open('tiktok_objective_eval/timestep_sweep.json'))
print("checkpoint:",t.get('checkpoint'),"rows:",len(t['rows']))
for r in t['rows']:
    print("  t=%s mult=%s fsim_off=%.4f fsim_on=%.4f delta=%.4f lpips_off=%.4f lpips_on=%.4f"%(
        r['timestep'],r.get('face_schedule_multiplier'),r['facesim_off'],r['facesim_on'],r.get('facesim_delta',0),r['lpips_off'],r['lpips_on']))

set -u
cd /home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body || exit 1
echo '@@@ run_manifest.command_args'
python3 -c "
import json
m=json.load(open('full_20k/run_manifest.json'))
ca=m.get('command_args',{})
print('command_args keys:',sorted(ca)[:60])
for k in ('cache','v6_checkpoint','resume','checkpoint','max_steps','accumulation','micro_batch_size','save_every','seed','pairs'):
    if k in ca: print(' ',k,'=',str(ca[k])[:160])
"
echo '@@@ deepgen_off 臂完整结构（找 PCK/FaceID/耗时/显存 的真实键名）'
python3 -c "
import json
d=json.load(open('four_arm_eval_50_s40/summary.json'))
a=d['arms']['qwen21_on']
print('arm keys:',sorted(a))
for k in sorted(a):
    if k not in ('psnr','ssim','mae'): print(' ',k,'=',str(a[k])[:220])
"
echo '@@@ paired_deltas'
python3 -c "
import json
d=json.load(open('four_arm_eval_50_s40/summary.json'))
pd=d['paired_deltas_second_minus_first']
print(type(pd), list(pd)[:8] if isinstance(pd,dict) else pd[:2])
print(json.dumps(pd, ensure_ascii=False)[:900])
"
echo '@@@ generation_settings + appearance overlap'
python3 -c "
import json
d=json.load(open('four_arm_eval_50_s40/summary.json'))
print('gen:',json.dumps(d['generation_settings'],ensure_ascii=False)[:300])
print('train/test/appear:',d['train_appearance_count'],d['test_appearance_count'],d['appearance_overlap_count'])
"

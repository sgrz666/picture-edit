set -u
cd /home/shangguanrz/project/pic-edit/experiments/qwen21_iper_body || exit 1
echo '@@@ s40 summary.json 顶层键'
python3 -c "
import json
d=json.load(open('four_arm_eval_50_s40/summary.json'))
print(sorted(d.keys()))
for k in ('num_samples','steps','seed','pairs','checkpoint'):
    if k in d: print(k,'=',str(d[k])[:120])
arms=d.get('arms') or d.get('metrics') or {}
print('arms keys:', list(arms)[:6])
for a,v in list(arms.items())[:4]:
    if isinstance(v,dict):
        print(a, {kk:v.get(kk) for kk in ('avg_psnr','psnr','avg_ssim','ssim','avg_mae','mae','pck@0.02','pck_0_02','face_id','peak_vram_gib','seconds_per_image') if kk in v})
"
echo '@@@ full_20k run_manifest 关键字段'
python3 -c "
import json
m=json.load(open('full_20k/run_manifest.json'))
for k in ('model','v6_checkpoint','resume','cache','max_steps','accumulation','micro_batch_size','save_every','seed','git_head'):
    if k in m: print(k,'=',str(m[k])[:150])
print('other keys:',[k for k in m if k not in ('model','v6_checkpoint','resume','cache','max_steps','accumulation','micro_batch_size','save_every','seed','git_head')][:12])
"
echo '@@@ metrics 行数与末行'
wc -l full_20k/train_metrics.jsonl 2>/dev/null; tail -c 300 full_20k/train_metrics.jsonl 2>/dev/null
echo '@@@ 目录计数'
find . -maxdepth 1 -type d | wc -l; du -sh --si . 2>/dev/null
echo '@@@ smoke metrics 行数'
wc -l smoke/train_metrics.jsonl 2>/dev/null; ls full_20k/ | head

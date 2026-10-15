import zipfile, json, os, glob
E='/home/shangguanrz/project/pic-edit/experiments/iper_adapter_finetune_v1'
f=E+'/condition_cache/001_10_source_motion_f000050.pt'
z=zipfile.ZipFile(f)
names=[n for n in z.namelist()]
print('zip entries', len(names))
for n in names[:40]: print('  ', n, z.getinfo(n).file_size)
print('...')
# read data/ tensor records
import re
dds=[n for n in names if n.startswith('data/')]
print('tensor count', len(dds))
for n in dds[:6]:
    print(n, z.getinfo(n).file_size)
print()
print('--- stage metrics summary ---')
d=json.load(open(E+'/stage_evolution/stage_evolution_metrics.json'))
for k in sorted(d, key=lambda x:int(x)):
    v=d[k]
    print(k, 'avg_psnr %.2f avg_ssim %.4f avg_mae %.2f n=%d'%(v['avg_psnr'],v['avg_ssim'],v['avg_mae'],len(v['details'])))
print()
print('--- eval_results/eval_metrics.json summary ---')
m=json.load(open(E+'/eval_results/eval_metrics.json'))
print(m['summary'])
ps=m['per_sample']
print('per_sample n', len(ps))
print('min psnr', min(x['psnr'] for x in ps), 'max psnr', max(x['psnr'] for x in ps))
print()
print('--- results png count ---')
print(len(glob.glob(E+'/eval_results/results/*.png')))
print()
print('--- train.log head lines about args (grep) ---')
t=open(E+'/train.log',errors='replace').read()
import re
for kw in ['resolution','batch','lr','learning','Adam','adapter','Epoch','dataset']:
    lines=[l for l in t.splitlines() if kw.lower() in l.lower()][:3]
    for l in lines: print('  *',l[:160])
print()
print('--- eval.log tail 30 ---')
print(''.join(open(E+'/eval.log',errors='replace').readlines()[-30:])[:2000])

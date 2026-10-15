import json, glob, os
R='/home/shangguanrz/project/pic-edit/experiments/review_20260908'
for d in sorted(glob.glob(R+'/runs/*')):
    s=json.load(open(os.path.join(d,'summary.json')))
    print(os.path.basename(d), '| status',s.get('status'),
          '| wall %.2fs'%s['wall_seconds'],'| proc %.2fs'%s.get('process_wall_seconds',0),
          '| peak_nvml %.4f'%s['peak_nvml_gib'],'| peak_alloc %.4f'%s['peak_allocated_gib'],
          '| peak_res %.4f'%s['peak_reserved_gib'],
          '| dit %.0fms'%s['component_cuda_ms']['dit'],
          '| vae_enc %.1f'%s['component_cuda_ms']['vae_encoder'],
          '| load %.2f'%s['stages'][0]['seconds'],
          '| edit %.2f'%s['stages'][1]['seconds'],
          '| nvmlerr',s.get('nvml_error'), '| interp', s.get('interpretation'))
print()
print('--- manifest entries:', len(json.load(open(R+'/quality_manifest.json'))))
print('--- logs dir stderr sizes & S1 letterbox stderr:')
p=R+'/logs/Q0_S1_42_letterbox_35_4.5.stderr.log'
print(open(p).read()[-700:])
print('--- stdout example:')
print(open(R+'/logs/Q0_S0_42_letterbox_35_4.5.stdout.log').read())
print('--- S1 stdout:')
print(open(R+'/logs/Q0_S1_42_letterbox_35_4.5.stdout.log').read())

import json, glob, os
R='/home/shangguanrz/project/pic-edit/experiments/review_20260908'
for d in sorted(glob.glob(R+'/runs/*')):
    s=json.load(open(os.path.join(d,'summary.json')))
    print('=====',os.path.basename(d))
    for k in ['status','wall_seconds','process_wall_seconds','peak_nvml_gib','peak_allocated_gib','peak_reserved_gib','interpretation','nvml_error','error','stage','reason']:
        if k in s: print('   ',k,'=',s[k])
    if 'component_cuda_ms' in s: print('    comp=',{k:round(v,1) for k,v in s['component_cuda_ms'].items()})
    if 'stages' in s: print('    stages=',[(x.get('stage'),round(x.get('seconds',0),2),x.get('peak_allocated_gib')) for x in s['stages']])
    print('    allkeys=',sorted(s.keys()))
    for f in ['stderr.log','stdout.log']:
        p=os.path.join(d,f)
        if os.path.exists(p):
            t=open(p,errors='replace').read()
            print('    %s len=%d tail:'%(f,len(t)), repr(t[-300:]))
print('manifest entries:', len(json.load(open(R+'/quality_manifest.json'))))

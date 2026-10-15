import json, os, glob
E='/home/shangguanrz/project/pic-edit/experiments'
def ls(d,depth=1):
    print('### ls -la',d)
    os.system('ls -la %s'%d)
for d in ['iper_v6_native_20k_v1/heldout_test_step_20000','iper_v6_native_20k_v1/multistage_evolution_showcase','iper_v6_native_20k_v1/three_row_showcase']:
    ls(os.path.join(E,d))
p=os.path.join(E,'iper_v6_native_20k_v1/three_row_showcase')
for f in sorted(os.listdir(p)):
    if f.endswith('.json'):
        print('=== ',f)
        print(open(os.path.join(p,f),errors='replace').read()[:4000])
print('=== png count per showcase dir ===')
for d in ['iper_v6_native_20k_v1/heldout_test_step_20000','iper_v6_native_20k_v1/multistage_evolution_showcase','iper_v6_native_20k_v1/three_row_showcase']:
    pp=os.path.join(E,d)
    fs=sorted(os.listdir(pp))
    print(d, 'files',len(fs), 'png',sum(1 for x in fs if x.endswith('.png')))
    print('   sample names',[x for x in fs if x.endswith('.png')][:8])
print('=== top_candidate_pairs.jsonl head ===')
print(open(os.path.join(E,'iper_v6_native_20k_v1/top_candidate_pairs.jsonl')).read()[:1500])

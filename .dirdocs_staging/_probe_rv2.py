import json, os, glob
V='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2'
def show(f,n=2500):
    p=os.path.join(V,f)
    print('======',f, os.path.getsize(p) if os.path.exists(p) else 'MISSING')
    if os.path.exists(p):
        print(open(p,errors='replace').read()[:n])
for f in ['ACCEPTANCE.json','budget_carryover.json','budget_report.json','capacity.json','evidence_index.json','reproducibility.json','model_inventory.json']:
    show(f, 2000)
print('====== manifest counts')
for f in ['main_manifest.json','infer_manifest.json','q4_manifest.json','smoke_manifest.json','stable_manifest.json','sensitivity_manifest.json','quality_manifest.json','quality_selection.json','adapter_visual_review.json']:
    o=json.load(open(os.path.join(V,f)))
    if isinstance(o,list):
        print(f,'LIST len',len(o), 'entry keys', sorted(o[0].keys()) if o and isinstance(o[0],dict) else '')
    else:
        print(f,'DICT keys',sorted(o.keys()))
        for k,v in o.items():
            if isinstance(v,list): print('   ',k,'list len',len(v))
            elif isinstance(v,dict): print('   ',k,'dict keys',sorted(v.keys())[:15])
            else: print('   ',k,'=',v)
print('====== dirs')
for d in ['runs','logs','configs','review_sheets','cache']:
    p=os.path.join(V,d)
    if os.path.isdir(p):
        ch=sorted(os.listdir(p))
        print(d,'count',len(ch))
        print('   first5',ch[:5]); print('   last3',ch[-3:])

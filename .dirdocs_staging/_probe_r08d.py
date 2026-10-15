import json, os, glob
R='/home/shangguanrz/project/pic-edit/experiments/review_20260908'
m=json.load(open(R+'/quality_manifest.json'))
print('manifest len', len(m))
print('ids', [x['id'] for x in m])
print('keys of entry0', sorted(m[0].keys()))
print('kinds', sorted(set(x['kind'] for x in m)))
print('resolutions', sorted(set(x['resolution'] for x in m)))
print('tasks', sorted(set(x['task'] for x in m)))
print('groups', sorted(set(x['group'] for x in m)))
print('seeds', sorted(set(x['seed'] for x in m)))
print('preprocess', sorted(set(x['preprocess'] for x in m)))
print('control_injected', sorted(set(str(x.get('control_injected')) for x in m)))
print('steps', sorted(set(x['steps'] for x in m)), 'cfg', sorted(set(x['cfg'] for x in m)))
print('prompts', sorted(set(x['prompt'][:60] for x in m)))
print()
print('--- run dir files ---')
for d in sorted(glob.glob(R+'/runs/*')):
    if os.path.isdir(d):
        print(os.path.basename(d), sorted(os.listdir(d)))
print()
print('=== review_20260908_v2 (archived entity) top ===')
V='/data/shangguanrz/pic-edit-archive/cleanup_20261003/experiments/review_20260908_v2'
for p in sorted(glob.glob(V+'/*')):
    print(('DIR ' if os.path.isdir(p) else 'FILE'), os.path.getsize(p) if os.path.isfile(p) else '-', os.path.basename(p))

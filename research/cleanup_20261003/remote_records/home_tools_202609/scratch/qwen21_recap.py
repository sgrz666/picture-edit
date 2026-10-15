import json
import os

ROOT = '/home/shangguanrz/project/pic-edit'

g = json.load(open(os.path.join(ROOT, 'experiments/qwen21_iper_body/pilot_dev_eval_8/gate_summary.json')))
print('=== gate_summary top keys ===', list(g.keys()))
for arm in ('off', 'on'):
    if arm in g:
        print('--- arm:', arm, '---')
        for k, v in g[arm].items():
            m = v.get('mean') if isinstance(v, dict) else v
            print('   %-24s %s' % (k, m))
    else:
        print('--- arm:', arm, 'MISSING ---')

print()
print('=== 目录存在性 ===')
for p in ['experiments/qwen21_iper_body/four_arm_eval_50',
          'experiments/qwen21_iper_body/eval_50',
          'experiments/qwen21_iper_body/deepgen_eval_50',
          'experiments/qwen21_iper_body/pilot_3k_acc4_resumed',
          'experiments/qwen21_iper_body/pilot_dev_eval_8',
          'experiments/qwen21_iper_body/full_20k']:
    fp = os.path.join(ROOT, p)
    n = len(os.listdir(fp)) if os.path.isdir(fp) else 0
    print('  %-58s exists=%s files=%d' % (p, os.path.isdir(fp), n))

print()
print('=== experiments/qwen21_iper_body 清单 ===')
base = os.path.join(ROOT, 'experiments/qwen21_iper_body')
for f in sorted(os.listdir(base)):
    fp = os.path.join(base, f)
    tag = 'dir' if os.path.isdir(fp) else 'file'
    print('  [%s] %s' % (tag, f))

print()
print('=== 其它新增文件大小 ===')
for p in ['train_iper_qwen21_body.py', 'eval_iper_qwen21_body.py', 'eval_iper_four_arm.py',
          'build_tiktok_v3_candidate_pool.py', 'requirements_qwen21.txt',
          'tests/test_qwen21_control_contract.py', 'add/tools/build_3dpw_frame_quality_v1.py']:
    fp = os.path.join(ROOT, p)
    if os.path.isfile(fp):
        print('  %-52s %8d bytes' % (p, os.path.getsize(fp)))
    else:
        print('  %-52s MISSING' % p)

import inspect
import sys
sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
from src.pose_control.v6 import condition_injector as ci
from src.data import iper_dataset as ds

print('=' * 74)
print('PART_NAMES  (%d 类)' % len(ci.PART_NAMES))
for i, n in enumerate(ci.PART_NAMES):
    print('   %2d  %s' % (i, n))
print('=' * 74)
print('iper_dataset 中与 v6 条件相关的函数')
for name in dir(ds):
    if 'v6' in name.lower() or 'heat' in name.lower() or 'kps' in name.lower() or 'render' in name.lower():
        print('   ', name)
print('=' * 74)
print(inspect.getsource(ds.IPERPoseDataset._get_v6_conditions))

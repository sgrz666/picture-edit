import inspect
import sys

sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
from src.pose_control.v6 import condition_injector as ci

src = inspect.getsource(ci.SMPLXConditionInjector.forward)
print('=' * 74)
print('SMPLXConditionInjector.forward  —— 校验段（从 mask_a 起）')
print('=' * 74)
i = src.find('mask_a =')
print(src[i:i + 2200])

print()
print('=' * 74)
print('SpatialConditionEncoder.forward 全文')
print('=' * 74)
print(inspect.getsource(ci.SpatialConditionEncoder.forward))

print()
print('=' * 74)
print('SpatialConditionEncoder 是否自带校验方法')
print('=' * 74)
for name in dir(ci.SpatialConditionEncoder):
    if 'valid' in name.lower() or 'check' in name.lower() or 'require' in name.lower():
        print('  ', name)

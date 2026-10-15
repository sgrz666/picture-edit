import inspect
import re
import sys

sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
from src.pose_control.v6 import condition_injector as ci

TARGETS = {
    'SpatialConditionEncoder': 'forward',
    'GlobalConditionTokenEncoder': 'forward',
    'RelativeGeometryTokenEncoder': 'forward',
    'ContactRasterEncoder': 'forward',
}

for cls_name, meth in TARGETS.items():
    cls = getattr(ci, cls_name)
    src = inspect.getsource(getattr(cls, meth))
    print('=' * 74)
    print(cls_name + '.' + meth)
    print('-' * 74)
    lines = src.split('\n')
    shown = 0
    for ln in lines:
        s = ln.strip()
        keep = (
            'ndim' in s or 'shape' in s or 'ValueError' in s or 'raise' in s
            or 'def forward' in s or 'must' in s or s.startswith('if ')
        )
        if keep and shown < 26:
            print('   ', s[:150])
            shown += 1
    print()

# 注入器 forward 里对 A/B 侧形状的校验
src = inspect.getsource(ci.SMPLXConditionInjector.forward)
print('=' * 74)
print('SMPLXConditionInjector.forward 形状校验')
print('-' * 74)
for ln in src.split('\n'):
    s = ln.strip()
    if ('shape' in s or 'ndim' in s or 'must' in s) and len(s) < 170:
        print('   ', s)

import inspect
import sys

sys.path.insert(0, '/home/shangguanrz/project/pic-edit')

from src.pose_control.v6 import condition_injector as ci

for name in ['TaskType', 'ConditionBundle', 'SpatialConditionEncoder',
             'GlobalConditionTokenEncoder', 'RelativeGeometryTokenEncoder',
             'ContactRasterEncoder', 'StructuredContactRelationEncoder',
             'ContactRelationBatch']:
    obj = getattr(ci, name, None)
    print('=' * 72)
    if obj is None:
        print(name, '-> NOT FOUND')
        continue
    if hasattr(obj, '__members__'):
        try:
            print(name, '=', {k: int(v) for k, v in obj.__members__.items()})
        except Exception:
            print(name, '=', list(obj.__members__.keys()))
        continue
    print(name)
    for meth in ('__init__', 'forward'):
        f = getattr(obj, meth, None)
        if f is None:
            continue
        try:
            print('   ', meth, inspect.signature(f))
        except Exception as exc:
            print('   ', meth, 'sig-err', exc)

print('=' * 72)
print('ConditionBundle 字段')
cb = getattr(ci, 'ConditionBundle', None)
if cb is not None:
    if hasattr(cb, '__dataclass_fields__'):
        for k, v in cb.__dataclass_fields__.items():
            print('   %-24s %s' % (k, v.type))
    else:
        print('   ', [a for a in dir(cb) if not a.startswith('_')])

import inspect
import sys
sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
from src.pose_control.qwen21.body_adapter import Qwen21BodyAdapter, Qwen21ControlContext
print('encode_geometry:', inspect.signature(Qwen21BodyAdapter.encode_geometry))
print('__init__:', inspect.signature(Qwen21BodyAdapter.__init__))
print('ControlContext.__init__:')
for n, p in inspect.signature(Qwen21ControlContext.__init__).parameters.items():
    if n != 'self':
        print(f'   {n} = {p.default!r}')

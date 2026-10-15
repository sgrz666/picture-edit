import inspect
import sys
sys.path.insert(0, '/home/shangguanrz/project/pic-edit')
from src.pose_control.v6 import condition_injector as ci
print(inspect.getsource(ci.SMPLXConditionInjector._validate_person))

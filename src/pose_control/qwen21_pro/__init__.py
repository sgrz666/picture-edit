"""完整官方 Union 上的可训练人物任务分支。"""
from .adapter import ProTaskAdapter
from .contracts import ContactRelations, ProAdapterConfig, TaskCondition

__all__ = ["ContactRelations", "ProAdapterConfig", "ProTaskAdapter", "TaskCondition"]

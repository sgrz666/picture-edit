"""完整官方 Union 上的可训练人物任务分支。"""
from .adapter import ProTaskAdapter
from .contracts import ContactRelations, ProAdapterConfig, TaskCondition

__all__ = ["ContactRelations", "ProAdapterConfig", "ProTaskAdapter", "TaskCondition",
           "install_pro_adapter", "bind_task_condition", "set_pro_enabled",
           "build_pro_transformer", "build_pro_pipeline", "ProPipeline"]


def __getattr__(name):
    if name in {"install_pro_adapter", "bind_task_condition", "set_pro_enabled"}:
        from . import integration
        value = getattr(integration, name)
    elif name in {"build_pro_transformer", "build_pro_pipeline", "ProPipeline"}:
        from . import pipeline
        value = getattr(pipeline, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value

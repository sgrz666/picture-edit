#!/usr/bin/env python
"""探针：反复导入 videox_fun 的 pipeline 包，记录它为了与本实验无关的模型而要求的缺失依赖。

本实验只需要 QwenImage21 的 transformer / vae / control pipeline。VideoX-Fun 的包 __init__
会把 CogVideoX、fantasytalking 等全部导入一遍，其中音频/视频模型依赖 librosa、decord 这类
与本任务无关的包。这里只把缺什么列出来，不安装、不改动 qwen21 环境。
"""
import importlib
import sys
import types

STUB = {}


def try_import():
    for name in list(sys.modules):
        if name.startswith("videox_fun"):
            del sys.modules[name]
    return importlib.import_module("videox_fun.pipeline")


for _ in range(30):
    try:
        mod = try_import()
        print("IMPORT_OK after stubbing:", sorted(STUB))
        break
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e).split("'")[1]
        if missing.startswith("videox_fun"):
            print("VIDEOX-INTERNAL missing module:", missing)
            break
        stub = types.ModuleType(missing)
        stub.__path__ = []
        stub.__getattr__ = lambda attr, _m=missing: (_ for _ in ()).throw(
            ImportError(f"{_m} 是为跳过与本实验无关的模型而刻意打桩的空模块"))
        sys.modules[missing] = stub
        STUB[missing] = str(e)[:120]
        print("stub:", missing, "|", str(e)[:100], flush=True)
    except Exception as e:
        print("OTHER_ERROR", type(e).__name__, str(e)[:400])
        print("stubbed so far:", sorted(STUB))
        break

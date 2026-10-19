# -*- coding: utf-8 -*-
"""固定版本源码与权重的导入引导、加载与 180 键审计。

本实验只在 /data 下建独立 venv（--system-site-packages 复用 qwen21 的 torch/diffusers/transformers），
不改动现有 qwen21 环境，也不改动 VideoX-Fun 的固定 commit。
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

VIDEOX_PIN = "13bb3c625e8b1e457d7124d92207b49e00e4a6fe"
UNION_REPO = "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"
UNION_REVISION_PIN = "8a4702014d4dabb5f896fcba917e2ee0a961465f"

WORK = pathlib.Path(os.environ.get("UNION_WORK", "/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1"))
VIDEOX = WORK / "source/VideoX-Fun"
CONFIG = VIDEOX / "config/qwenimage21/qwenimage21_control.yaml"
BASE_MODEL = pathlib.Path(os.environ.get("UNION_BASE", "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"))
UNION_DIR = WORK / "models/Qwen-Image-2.1-Fun-Controlnet-Union"
RUNS = WORK / "runs"


def sha256_file(p: pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def union_checkpoint() -> pathlib.Path:
    cands = sorted(UNION_DIR.rglob("*.safetensors"))
    if not cands:
        raise FileNotFoundError(f"Union safetensors 尚未下载到 {UNION_DIR}")
    return max(cands, key=lambda p: p.stat().st_size)


def verify_videox_checkout() -> str:
    head = subprocess.run(["git", "-C", str(VIDEOX), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != VIDEOX_PIN:
        raise RuntimeError(f"VideoX-Fun HEAD {head} 与固定 commit {VIDEOX_PIN} 不一致")
    return head


def _stub_unrelated_optional() -> list[str]:
    """给与本实验无关的可选依赖打空桩。

    VideoX-Fun 的包 __init__ 会把 CogVideoX / fantasytalking 等全部模型导入一遍，其中音频模型
    需要 librosa。装它等于往环境里塞与本任务无关的东西，所以只在缺失时挂一个空模块：任何代码
    真去用它会立刻抛 ImportError，不会静默出错。
    """
    import importlib.util
    import types
    from importlib.machinery import ModuleSpec

    # 先让 diffusers / transformers 完成"librosa 是否可用"的探测，再挂空桩；
    # 顺序反了会让它们把桩当成真包，从而改变后端选择。
    import diffusers  # noqa: F401
    import transformers  # noqa: F401

    stubbed = []
    for name in ("librosa",):
        if importlib.util.find_spec(name) is not None:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__file__ = f"<stub:{name}>"
        mod.__spec__ = ModuleSpec(name, loader=None, is_package=True)

        def _blocked(attr, _n=name):
            # 双下划线探测（inspect / hasattr）必须正常失败，否则会把无关模块的导入机制带崩
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            raise ImportError(f"{_n} 是为跳过与本实验无关的音频模型而刻意挂的空模块，不应被真正调用")

        mod.__getattr__ = _blocked
        sys.modules[name] = mod
        stubbed.append(name)
    return stubbed


def bootstrap():
    """把固定版本源码挂到 sys.path 最前，并校验 commit 与基座路径。"""
    verify_videox_checkout()
    _stub_unrelated_optional()
    if str(VIDEOX) not in sys.path:
        sys.path.insert(0, str(VIDEOX))
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"缺少 Qwen-Image-2.1 基座：{BASE_MODEL}")
    return VIDEOX


def control_param_names(model) -> list:
    return sorted(n for n, _ in model.named_parameters()
                  if n.startswith("control_blocks.") or n.startswith("control_img_in."))


def load_union_transformer(device="cpu", dtype=None, expect_control_keys=180):
    """按官方顺序加载：from_pretrained（会把 control 投影重新置零）→ 再加载 Union 权重。

    返回 (transformer, audit)。audit 里必须看到 zeroed_before_load 为真、
    restored_after_load 等于全部控制键数、且加载后没有任何控制张量仍为零，
    否则等于在用随机/零控制权重跑实验。
    """
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    bootstrap()
    from videox_fun.models import QwenImage21ControlTransformer2DModel

    dtype = dtype or torch.bfloat16
    kwargs = OmegaConf.to_container(OmegaConf.load(CONFIG)["transformer_additional_kwargs"])

    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
        str(BASE_MODEL), subfolder="transformer", low_cpu_mem_usage=True,
        torch_dtype=dtype, transformer_additional_kwargs=kwargs,
    ).to(dtype)

    ckpt_path = union_checkpoint()
    state = load_file(str(ckpt_path))
    state = state["state_dict"] if "state_dict" in state else state

    named = dict(transformer.named_parameters())
    ctrl_names = [n for n in named if n.startswith("control_blocks.") or n.startswith("control_img_in.")]

    # from_pretrained 之后、加载 Union 之前：官方 loader 会把 before/after_proj 归零
    zeroed = [n for n in ctrl_names if float(named[n].detach().float().abs().sum()) == 0.0]

    missing, unexpected = transformer.load_state_dict(state, strict=False)

    named_after = dict(transformer.named_parameters())
    ctrl_keys_in_ckpt = sorted(k for k in state if k.startswith("control_blocks.") or k.startswith("control_img_in."))
    mismatches, still_zero, differing = [], [], []
    for k in ctrl_keys_in_ckpt:
        if k not in named_after:
            mismatches.append((k, "absent-in-model"))
            continue
        a, b = state[k], named_after[k]
        if tuple(a.shape) != tuple(b.shape):
            mismatches.append((k, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        if float(a.to(b.dtype).detach().float().abs().sum()) == 0.0:
            pass  # checkpoint 自己就是零（不应出现在 proj 上）
        if not torch.equal(a.to(b.dtype), b.detach()):
            differing.append(k)
        if float(b.detach().float().abs().sum()) == 0.0 and float(a.detach().float().abs().sum()) != 0.0:
            still_zero.append(k)

    restored = len(ctrl_keys_in_ckpt) - len(differing) - len(mismatches)
    audit = {
        "videox_commit": verify_videox_checkout(),
        "union_checkpoint": str(ckpt_path),
        "union_sha256": sha256_file(ckpt_path),
        "union_bytes": ckpt_path.stat().st_size,
        "control_keys_in_checkpoint": len(ctrl_keys_in_ckpt),
        "expected_control_keys": expect_control_keys,
        "control_params_in_model": len(ctrl_names),
        "zeroed_by_loader_before_union": len(zeroed),
        "restored_after_union_load": restored,
        "still_zero_after_load": still_zero[:20],
        "shape_or_presence_mismatches": mismatches[:20],
        "bitwise_differing": differing[:20],
        "loader_missing_keys": len(missing),
        "loader_unexpected_keys": len(unexpected),
        "loader_missing_sample": sorted(str(m) for m in missing)[:10],
        "loader_unexpected_sample": sorted(str(u) for u in unexpected)[:10],
        "control_layers": list(transformer.control_layers),
        "control_in_dim": transformer.control_in_dim,
        "num_control_blocks": len(transformer.control_blocks),
        "union_revision_pin": UNION_REVISION_PIN,
    }
    audit["gate_pass"] = (
        len(ctrl_keys_in_ckpt) == expect_control_keys
        and restored == expect_control_keys
        and not still_zero
        and not mismatches
        and len(zeroed) > 0
    )
    if device != "cpu":
        transformer = transformer.to(device)
    return transformer, audit


def build_pipeline(kind="reference", device=None, dtype=None, memory_mode="model_group_offload",
                   load_union=True):
    """按官方 predict_t2i_control.py 的构造顺序建 pipeline。

    kind="official"   : 原版 QwenImage21ControlPipeline（官方复现关用）
    kind="reference"  : 本实验的 ReferenceUnionPipeline（接入关与矩阵用）
    kind="native"     : 原生 Qwen-Image-2.1 pipeline（工作流基线，不加载 Union）
    """
    bootstrap()
    import torch
    from omegaconf import OmegaConf
    from diffusers import FlowMatchEulerDiscreteScheduler
    from videox_fun.models import (AutoencoderKLQwenImage21, Qwen3VLForConditionalGeneration,
                                   Qwen3VLProcessor)
    from videox_fun.pipeline import QwenImage21ControlPipeline
    from videox_fun.utils import apply_gpu_memory_mode

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.bfloat16
    audit = None

    if kind == "native":
        from videox_fun.models import QwenImage21Transformer2DModel
        from videox_fun.pipeline import QwenImage21Pipeline
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            str(BASE_MODEL), subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=dtype,
        ).to(dtype)
        pipe_cls = QwenImage21Pipeline
    else:
        if load_union:
            transformer, audit = load_union_transformer(device="cpu", dtype=dtype)
        else:
            kwargs = OmegaConf.to_container(OmegaConf.load(CONFIG)["transformer_additional_kwargs"])
            from videox_fun.models import QwenImage21ControlTransformer2DModel
            transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
                str(BASE_MODEL), subfolder="transformer", low_cpu_mem_usage=True,
                torch_dtype=dtype, transformer_additional_kwargs=kwargs,
            ).to(dtype)
        if kind == "reference":
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            from reference_union_pipeline import ReferenceUnionPipeline
            pipe_cls = ReferenceUnionPipeline
        else:
            pipe_cls = QwenImage21ControlPipeline

    vae = AutoencoderKLQwenImage21.from_pretrained(str(BASE_MODEL), subfolder="vae").to(dtype)
    processor = Qwen3VLProcessor.from_pretrained(str(BASE_MODEL), subfolder="processor")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL), subfolder="text_encoder", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(BASE_MODEL), subfolder="scheduler")

    pipeline = pipe_cls(vae=vae, text_encoder=text_encoder, processor=processor,
                        transformer=transformer, scheduler=scheduler)

    if memory_mode and memory_mode != "none":
        apply_gpu_memory_mode(pipeline, memory_mode, device, dtype,
                              exclude_module_name=["img_in", "txt_in", "time_text_embed", "modulation"])
    else:
        pipeline = pipeline.to(device)
    return pipeline, audit

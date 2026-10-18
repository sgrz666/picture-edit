"""复用完整 Union 加载器与 ReferenceUnionPipeline 的 Pro 构建入口。"""
from __future__ import annotations

import inspect
from collections.abc import Mapping

import torch

from .contracts import ProAdapterConfig
from .integration import bind_task_condition, install_pro_adapter


def _setup(config, control_lora, lora_options, *, checkpoint_path=None, expected_metadata=None, transformer_setup=None):
    if type(control_lora) is not bool:
        raise ValueError("control_lora must be bool")
    if lora_options is not None and not isinstance(lora_options, Mapping):
        raise ValueError("lora_options must be a mapping")
    if lora_options and not control_lora:
        raise ValueError("lora_options requires control_lora=True")
    if config is not None and not isinstance(config, ProAdapterConfig):
        raise ValueError("config must be ProAdapterConfig")
    if (checkpoint_path is None) != (expected_metadata is None):
        raise ValueError("checkpoint_path and caller-computed expected_metadata are required together")
    if transformer_setup is not None and not callable(transformer_setup):
        raise ValueError("transformer_setup must be callable")

    def install(transformer):
        lora_audit = None
        if control_lora:
            from ..qwen21_union.weights import install_control_lora
            lora_audit = install_control_lora(transformer, **dict(lora_options or {}))
        pro_audit = install_pro_adapter(transformer, config)
        composition = transformer_setup(transformer) if transformer_setup is not None else None
        checkpoint_audit = None
        if checkpoint_path is not None:
            from .checkpoint import load_adapter_weights
            checkpoint_audit = load_adapter_weights(checkpoint_path, transformer, expected_metadata=expected_metadata)
        return {"control_lora": lora_audit, "pro_adapter": pro_audit,
                "composition": composition, "checkpoint": checkpoint_audit}

    return install


def build_pro_transformer(scope="joint", device="cpu", dtype=None, *, config: ProAdapterConfig | None = None,
                          control_lora=False, lora_options=None, source_root=None, base_model=None,
                          union_checkpoint=None, checkpoint_path=None, expected_metadata=None, transformer_setup=None):
    """返回 (model,audit)：验证/加载原180参数 → 可选控制LoRA → FP32任务模块。

    source_root/base_model/union_checkpoint 与旧 build_transformer 同义。默认不装
    LoRA；control_lora=True 复用旧130目标白名单。原网络冻结但不使用 no_grad。
    """
    setup = _setup(config, control_lora, lora_options, checkpoint_path=checkpoint_path,
                   expected_metadata=expected_metadata, transformer_setup=transformer_setup)
    from ..qwen21_union import pipeline as union
    transformer, audit = union.build_transformer(scope, "cpu", dtype, source_root=source_root,
                                                  base_model=base_model, union_checkpoint=union_checkpoint)
    extra = setup(transformer)
    return transformer.to(device), dict(audit, **extra)


class ProPipeline:
    """薄 composition；委派原 pipeline，并把原始任务 pack 绑定到每次采样。

    支持原 image/mask_image/reference_images。true CFG 的正负 prompt 分别执行
    transformer.forward，因此 mask 分别按本次 text 长度构建。
    """
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def to(self, *args, **kwargs):
        """保留原 pipeline 的转换语义，并让链式调用继续绑定任务条件。"""
        self._pipeline.to(*args, **kwargs)
        return self

    def __call__(self, *args, task_condition=None, task_enabled=True, **native_kwargs):
        native_kwargs.setdefault("cfg_norm_rescale", True)
        bound = inspect.signature(self._pipeline.__call__).bind_partial(*args, **native_kwargs)
        values = dict(native_kwargs, **{k: v for k, v in bound.arguments.items() if k != "kwargs"})
        if values.get("num_images_per_prompt", 1) != 1:
            raise ValueError("Pro requires num_images_per_prompt=1; implicit B>1 CFG expansion is unsupported")
        for name in ("prompt", "negative_prompt"):
            value = values.get(name)
            if isinstance(value, (tuple, list)) and len(value) != 1:
                raise ValueError("Pro pipeline supports prompt batch B=1")
        for name in ("prompt_embeds", "negative_prompt_embeds"):
            value = values.get(name)
            if value is not None and (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1):
                raise ValueError("Pro pipeline supports embedding batch B=1")
        if values.get("use_kv_cache", False):
            raise ValueError("Pro pipeline requires KV cache disabled")
        if getattr(self._pipeline, "text_encoder", object()) is None:
            if values.get("reference_images"):
                raise ValueError("cached pipeline has no text_encoder for raw reference_images; build with cached=False")
            if values.get("prompt_embeds") is None:
                raise ValueError("cached pipeline requires encoded prompt_embeds; build with cached=False for raw prompts")
            from ..qwen21_union.pipeline import normalize_inference_options
            cfg_scale = normalize_inference_options(values.get("true_cfg_scale", 1.),
                                                     values.get("guidance_scale"), None, self.transformer)
            if (cfg_scale > 1
                    and values.get("negative_prompt") is not None and values.get("negative_prompt_embeds") is None):
                raise ValueError("cached CFG requires encoded negative_prompt_embeds")
        with bind_task_condition(self.transformer, task_condition, enabled=task_enabled):
            return self._pipeline(*args, **native_kwargs)


def build_pro_pipeline(scope="joint", device="cpu", offload="model_group_offload", dtype=None, *,
                       config: ProAdapterConfig | None = None, control_lora=False, lora_options=None,
                       source_root=None, base_model=None, union_checkpoint=None, cached=False,
                       checkpoint_path=None, expected_metadata=None, transformer_setup=None):
    """返回 (ProPipeline,audit)，所有新参数在原加载器 offload 扫描之前注册。

    cached=True 省略 VLM，只适合已编码 coarse 输入；原始 ROI reference_images
    必须 cached=False。沿用原 group_offload 的 device hooks，不转换任务 FP32 存储。
    """
    setup = _setup(config, control_lora, lora_options, checkpoint_path=checkpoint_path,
                   expected_metadata=expected_metadata, transformer_setup=transformer_setup)
    from ..qwen21_union import pipeline as union
    requested_offload = offload
    synchronous = (offload == "sync_group_offload" or
                   (str(device).startswith("cuda") and offload == "model_group_offload"))
    if synchronous and not str(device).startswith("cuda"):
        raise ValueError("sync_group_offload requires CUDA onload device")
    pipeline, audit = union.build_pipeline("union", scope, "cpu" if synchronous else device,
                                           "none" if synchronous else offload, dtype, source_root=source_root,
                                           base_model=base_model, union_checkpoint=union_checkpoint,
                                           transformer_setup=setup, cached=cached)
    audit = dict(audit, **audit["transformer_setup"])
    audit["requested_offload"] = requested_offload
    if synchronous:
        _, _, _, utils = union._upstream(source_root)
        # Shared Pro adapter executes at stages0/4/8/12. Synchronous groups avoid
        # stream prefetch replacing a shared parameter before its current use.
        utils.register_auto_device_hook(pipeline.transformer)
        utils.safe_enable_group_offload(pipeline, onload_device=str(device), offload_device="cpu",
                                        offload_type="leaf_level", use_stream=False)
        audit["offload"] = "sync_group_offload"
        pipeline.union_audit = audit
    result = ProPipeline(pipeline)
    result.pro_audit = audit
    result.trained_model_signature = None if audit["checkpoint"] is None else {
        "state_sha256": audit["checkpoint"]["state_sha256"],
        "metadata": audit["checkpoint"]["metadata"]}
    return result, audit



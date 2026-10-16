# Qwen 2.1 Union 数据集适配

本模块把方案 A 已验证的参考图接入层提升为共享代码，为方案 B 的冻结基线、控制 LoRA 训练和 joint/target 消融提供同一套输入。方案 A 的官方源码、独立环境和权重继续从 `/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/` 读取；本模块不下载模型、不安装依赖，也不修改官方 checkout。

`pipeline.py` 惰性导入固定 VideoX-Fun commit `13bb3c625e8b1e457d7124d92207b49e00e4a6fe`。`build_transformer(scope, device, dtype)` 只加载基座 DiT 和 Union，不分配 VLM/VAE；Union 的 SHA256 必须为 `65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd`，180 个控制 tensor 在官方 loader 归零后重新加载并逐个核对。`ScopedUnionTransformer` 只覆盖 `forward_control`，target 模式在官方 `target_token_mask` 上屏蔽输出残差，保留控制链计算图。没有 hook 或对上游类的全局替换。

`build_pipeline(kind='union'|'native'|'encoder', scope='joint'|'target', device='cpu', offload='model_group_offload')` 返回 `(pipeline, audit)`。union 使用 `_reference_pipeline.py` 中的方案 A 参考图适配器，默认输出分辨率改为 512；native 使用现有环境的 `diffusers.QwenImage21Pipeline`，audit 明确记录后端与版本。encoder 只加载被冻结的 VAE、Qwen3-VL 和 processor。缓存推理传 `cached=True` 可跳过 VLM；processor 仍用于初始化 image token ID。LoRA 安装和恢复放进 `transformer_setup` 回调，它在冻结基座之后、offload 收集参数之前执行。编码完成后调用方应释放 encoder pipeline，再加载训练 DiT。

`conditions.py` 校验每个 VLM image slot 对应 4 个 latent token，构建 `[control64, zero65]` 的 129 通道目标控制输入，并按真实补边内容面积计算 fractional token 权重。所有参考图控制输入均为零，但保留官方 affine bias。`contracts.py` 定义条件、监督、画布变换和运行配置；provider 负责准备 `[source_rgb, pose...]` 的 512 图像与人物顺序。`weights.py`、`training.py`、`checkpoint.py` 和 `evaluation.py` 分别负责精确 LoRA 模块、flow loss、恢复契约和指标，不由 pipeline 隐式开启训练。

`cache.py` 提供以下 API：

```python
from src.pose_control.qwen21_union.pipeline import build_pipeline, model_asset_signature
from src.pose_control.qwen21_union.cache import (
    precompute_conditions, precompute_supervision, load_condition, load_supervision,
)

base = '/data/shangguanrz/pic-edit-models/Qwen-Image-2.1'
signature = model_asset_signature(base)
pipe, audit = build_pipeline(kind='encoder', device='cuda')
condition_index = precompute_conditions(pipe, condition_records, cache_root, signature)
supervision_index = precompute_supervision(pipe, train_supervision_records, cache_root, signature)
condition = load_condition(cache_root, sample_id)
supervision = load_supervision(cache_root, sample_id)
```

输入为 dict。条件必需 `sample_id`、`source_rgb`、有序 `reference_pose_images`、`control_image`、`prompt`，支持 `person_order`、`source_transform`/`target_transform` 或 `transforms`。source、pose、control 必须已经为 512×512；条件编码不会打开 `target_rgb` 或 `target_masks`。监督输入额外带原始 `target_rgb`、精确 `target_transform` 和 `split='train'`；调用方应从条件清单补入 split，`SupervisionRecord.to_dict()` 本身不含 split。监督图按 `original_wh` 核对尺寸，再按 `resized_wh`/`offset_xy` 做 LANCZOS 缩放与白色补边，与共享 RGB 预处理一致，全程在内存中处理。预处理版本固定为 `qwen21-union-prepared512-target-lanczos-white-v2`。

缓存 tensor 使用同一个 `latent_dist.mode()` 和 `(latent - latents_mean) / latents_std`，不消耗 denoising RNG。condition 含 `[1,Nref,64]` reference、`[1,1024,64]` control、`[1,L,4096]` prompt、prompt mask、已追加 256 个目标 slot 的 img mask、完整 img shapes，以及 latent 尺寸 `target_hw=(32,32)`、像素尺寸 `canvas_hw=(512,512)`。supervision 只含 `[1,1024,64]` target 与 `[1,1024]` valid target tokens。两类文件分放 `conditions/` 和 `supervision/`，旧内容寻址文件保留；index 指向当前指纹。加载核对 payload SHA256 并 clone 成普通 CPU tensor，损坏缓存立即报错。

训练或推理在加载前用 `expected_condition_fingerprint(record, model_signature, vae_config, dtype)` 重新计算条件指纹，并传给 `load_condition(..., expected_hash=...)`；训练监督对应 `expected_supervision_fingerprint` 和 `load_supervision`。这两个 helper 只读输入字节和 VAE 配置 dict，不加载编码器；不能用 cache index 自己声明的 hash 代替预期指纹校验。

`model_asset_signature(base, components=...)` 对配置和实际权重字节做 SHA256；`encoder_asset_signature(base)` 只选 VAE/VLM/processor。相同进程中 stat 未变的文件不重复读取。应把该结果及官方源码/预处理版本写入实验 lock，不能只给缓存传模型路径。缓存指纹还包括有序图像内容、prompt、变换、dtype 和 VAE 编码策略。

可在 CPU 执行：

```powershell
python -m pytest tests/test_qwen21_union_layout.py tests/test_qwen21_union_cache.py tests/test_qwen21_union_pipeline.py -q
```

测试覆盖三参考图与文本/padding 交错、129 通道、残差与梯度范围、内容指纹、缓存损坏、无目标读取、VAE mode 归一化、矩形目标精确变换和 LoRA/offload 顺序。没有挂载固定源码时跳过唯一的上游导入测试。940×1280 的目标画布变换为 376×512、left=68；有效内容权重之和为 752，边缘 token 保留 fractional coverage。

本模块上游是 `experiments/qwen21_union_migration_v1/scripts/reference_union_pipeline.py` 的已验证方案 A 接入层；下游为方案 B 的数据准备、缓存、训练与评估 CLI。实验选择、预算和归因限制见 `docs/superpowers/plans/2026-10-15-qwen21-union-dataset-adaptation-plan-b.md`。GPU 出图和训练效果需由实验门禁验证，CPU 契约通过本身不代表图像质量提升。

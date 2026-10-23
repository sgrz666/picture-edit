# 原生512维几何前端：方案C第一步

本文保留 Native v1 的复现说明。当前默认 Body21 v2，见 [GLOBAL_GEOMETRY_V2.md](GLOBAL_GEOMETRY_V2.md)；以下旧路径均显式选择 legacy。

本模块把现有Full的几何编码和推理独立取出。输出是高低分辨率融合后的geometry/interaction token，默认 `[B,N,512]`；主干反馈接入另见 [NATIVE_BODY.md](NATIVE_BODY.md)，本模块本身不生成残差。

## 保留的内容

`native_geometry.py` 的 `Qwen21NativeGeometry` 保留native Normal/Pose25/Part14 stems、原384→256 fusion、global/task/relative/contact编码、完整reasoner，以及geometry/interaction两张stride2的高分辨率融合卷积。它不构造旧SD3控制核、不执行128维投影、不把latent变为16通道，也不加载Union。

输出等于旧bridge的128维投影之前的特征；经过旧projection后应与旧bridge逐值相同。前端包含双人推理能力，单人行interaction保持零；这不是双人生成效果证明。

## 继承已有Full权重

```python
import torch
from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry

payload = torch.load(full_checkpoint, map_location="cpu", weights_only=False, mmap=True)
frontend = Qwen21NativeGeometry.from_full_checkpoint(payload, legacy=True)
print(frontend.inheritance_report)
```

输入必须是已有的可信 `qwen21_full_v1` checkpoint。只导入四个前缀：`legacy.condition_injector.`、`legacy.reasoner.`、`legacy.condition_bridge.geometry_high_downsample.`、`legacy.condition_bridge.interaction_high_downsample.`。其他Body控制核、Face、Hand、optimizer等不会恢复；报告明确加载数量与忽略键数。缺键、选中前缀内多余键、错误shape、非有限值都拒绝，不用 `strict=False` 蒙混加载。

如果已有完整Full对象，可用 `Qwen21NativeGeometry.from_full_adapter(full, legacy=True)`。返回独立FP32 CPU参数，不与旧模型共享存储；之后显式 `.to(device)`。非默认reasoner须传原 `reasoning_config`。本阶段拒绝Depth-enabled和非native stem的模型。

原生前端自己的 `state_dict()` 带 `qwen21_full_native_geometry_v1` schema及reasoner配置，构造同配置实例后strict恢复。Full→native是warm-start，不是恢复旧optimizer/RNG。

## 输入与输出

```python
from src.pose_control.qwen21_full.data_runtime import build_condition_bundle

bundle = build_condition_bundle(frontend, batch, device="cpu", dtype=torch.float32)
features = frontend(bundle, target_token_hw=(target_height // 16, target_width // 16))
geometry = features.geometry       # [B,N,512]
interaction = features.interaction # 单人行全零
```

`batch`沿用Full已准备好的Normal/Pose25/Part14/global/mask/task输入。它们必须已经变换到目标画布。当前要求reasoner输出网格**恰好**等于目标网格且token数可构成Qwen四token图像slot；不会偷偷resize。相同尺寸也不能代替上游坐标/相机对齐证明。

该出口保留计算图，可训练原编码器与高低尺度分支；如需冻结前端，用 `.requires_grad_(False)`。不把人体mask作为整个生成图mask，背景附近的几何特征仍保留；padding和target token写入由后续Qwen接入阶段处理。

## 测试与CPU审计

从本工作副本根目录运行：

```bash
python -m pytest tests/test_qwen21_native_geometry.py -q
python scripts/audit/audit_qwen21_native_geometry.py --checkpoint /path/to/existing/full.pt
```

服务器现有默认Body checkpoint为 `/home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt`。审计不加载Qwen/VAE、不用GPU、不修改checkpoint；使用确定性的合成条件核验已训练前端的数值继承，不评测真实图像质量。`--output`可将JSON写到一个**尚不存在**的文件，其父目录必须已存在；已有输出拒绝覆盖。

这些token的显式主干接入见 [NATIVE_BODY.md](NATIVE_BODY.md)。Depth及控制层数消融保持独立，现有Full训练/推理默认入口保持原样。

# Full Native Body：Qwen主干实时反馈接口

Body21 前端见 [GLOBAL_GEOMETRY_V2.md](GLOBAL_GEOMETRY_V2.md)。本页低层接口示例显式使用旧 Geometry，Body 接口保持不变。

`native_body.py` 接收第一步的原生geometry特征；`native_integration.py` 在每个选定Qwen block完成后，读取当层目标hidden并计算残差。默认32个位置、6组输出head；新head初始全零。没有Union控制链或SD3控制核。

## 计算与权重

前端沿用现有Full；新接口使用 `4096→512 query`，与512维geometry、局部卷积、geometry调制、layer embedding和真实sigma融合，再由所属组的 `512→4096 head`写回target。

geometry局部编码在每次transformer.forward只算一次；每层query读取不同的主干当前hidden。只屏蔽padding等无效画布token，不把所有背景更新都裁掉。`NativeGeometryFeatures.interaction`尚未接独立残差分支，不能据此宣称双人交互训练已经完成。

参数存储和计算要求FP32，主干hidden可以BF16。先加载/转换基座dtype，再安装新模块；之后可`.to(device)`，整体`.bfloat16()`会改变新模块存储，入口将拒绝继续启用。新Body使用独立的 `qwen21_full_native_body_v1`配置和state，旧Full权重只供几何前端warm-start；新head需要训练。

## 显式接入

```python
import torch
from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body

# pipe须为现有QwenImage21Pipeline，并已按原生流程加载BF16基座。
frontend = Qwen21NativeGeometry.from_full_checkpoint(full_payload, legacy=True).eval().requires_grad_(False).to("cuda")
body = Qwen21NativeBody().to("cuda")
audit = install_native_body(pipe.transformer, body)
features = frontend(condition_bundle, target_token_hw=(height // 16, width // 16))

with bind_native_body(pipe.transformer, features):
    result = pipe(prompt=prompt, image=[source_image], height=height, width=width,
                  output_resolution=512, num_inference_steps=30, use_kv_cache=False)
```

`condition_bundle`应由这个frontend的condition injector构造，条件已经变换到目标画布；不是把目标RGB或骨架参考图偷偷混进condition。图像、提示词、原生VAE与scheduler仍由现有pipeline负责。本例展示接线方式，尚不能把未训练新head的出图作为姿态效果。

示例按现有512基准固定参考图的`output_resolution=512`；正式对照须同时锁住参考尺寸和目标尺寸，不能只固定height/width而让参考图悄悄改成原生默认1024。

直接调用transformer也支持：

```python
prediction = pipe.transformer(**native_inputs, native_geometry=features)[0]
baseline = pipe.transformer(**native_inputs, native_enabled=False)[0]
```

绑定可接受 `valid_mask`（bool `[1,Ntarget]`）和非负`strength`；每次forward仍从实际 `img_mask/img_shapes` 重建target metadata，支持不同参考数量、参考/目标网格和正负CFG文本长度。第一版限定B=1、KV cache关闭；禁止同时开启旧Full控制context。设备offload/compile与完整7B采样质量尚需各自验收，不由CPU小模型结果代替。

## 反向重算

使用 `transformer.enable_gradient_checkpointing()` 的标准non-reentrant实现，或显式`partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)`。拒绝reentrant和未经验证的custom helper，避免冻结主干时丢失新分支梯度。

每次forward的target mask、sigma和geometry执行对象被显式捕获进checkpoint调用。退出绑定或第二次forward之后，第一次的backward仍用第一次条件。原生forward本身不重写，原block参数对象和原有state key保留，新模块只增加 `native_body.*`。

保存训练成果时分别保存前端、body、配置及训练状态；不以 `pipe.save_pretrained()`导出整个临时安装实例。动态绑定/执行上下文不是checkpoint内容。

## 验证

```bash
python -m pytest tests/test_qwen21_native_body.py tests/test_qwen21_native_geometry.py -q
python scripts/audit/audit_qwen21_native_body.py
```

CPU工具使用真实32层Qwen block结构、缩小hidden；可加 `--full-checkpoint /path/full.pt` 连现有训练前端一起核验。检查零输出/disabled等价、prefix不变、target响应、所有head梯度及两次forward后的checkpoint梯度一致。没有优化器更新。

真实7B CUDA入口：

```bash
python scripts/audit/verify_qwen21_native_body_runtime.py \
  --base-model models/Qwen-Image-2.1 \
  --full-checkpoint /home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt \
  --resolution 512
```

该工具要求至少40GiB可用显存，仅做固定合成输入的前向/反向，不采样图片、不更新optimizer、不保存权重。所有工具的`--output`均只写不存在的新JSON，父目录必须已存在。完整出图、迁移训练、Depth和稀疏层数效果属于后续验证。

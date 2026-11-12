# Native Full V6.1：空间支持与多尺度 PoseLink

本版只实现 P1＋P2：Geometry-aware Key Mask、PoseLink 编辑区域 Gate、多尺度 Pose Tokens。保留 V6 的 Body／Interaction／Face／Hand／Detail、原生 Qwen Attention、Role／Contact Bias 和 Loss。默认仍是 512 维、8 heads、六层 5／10／15／21／26／31、Query 分块 128，不增加 Qwen Tokens。

## 里面有什么

- `contracts.py`／`spatial.py`：读取已有 Body UV、投影支持和目标 Face／Hand ROI；构建逐人物软支持与条件 Gate。Body UV 按像素转为 Canvas 规范化坐标；局部坐标、Mask 保持 FP32。没有有效关节或 ROI 时保持缺失，不运行检测器或填伪标签。
- `tokens.py`／`poselink.py`：每张 Pose 的上限为 256 Tokens，默认 global 128、joint 64、local 64。局部顺序是左手、右手、左足、右足、面部；按有效人物均分，再按解剖顺序轮询分配。每个采样保留全 Canvas `xy`、人物、参考图及类型，不把局部 patch 坐标当作全局网格。
- `runtime.py`／`prepared.py`／`integration.py`：版本化初始化、严格回读、空间记录批处理；各分支仍读同一份未控制 Hidden，只向 Target 写残差。新增 Gate 只作用于 PoseLink。
- `data.py`／`training/`：已有 V5／V6 manifest 加可选原始 repair 侧车；独立 V6.1 training schema，复用原有 Loss、参数目录和稳定 optimizer 超集。

支持权重在 Pose Hidden 与几何的池化／插值前应用并归一化。Key 支持默认向外扩展 16 像素、内部羽化 8 像素；远处背景、padding 与无效人物不能成为 Key。编辑 Gate 使用目标人体／关节／局部支持和可选 repair 的并集，同样扩展 16 像素、羽化 8 像素。Gate 保存的是有效像素条件均值，最终只乘一次原有 coverage；Gate 外新增 PoseLink 残差严格为零。

## 修复原人物位置

可选 manifest 字段 `pose_repair_v61` 是绝对路径＋SHA＋schema 的 NPZ 引用，由既有 `write_record(..., schema='native_pose_repair_v61')` 保存原始字典：

```python
{
    'schema': 'native_pose_repair_v61',
    'role': 'source_repair',
    'mask': mask,  # FP32 [B,2,H_target,W_target]，已经对齐目标 Canvas
    'metadata': ({
        'target_canvas_id': target_canvas_id,
        'source_rgb_sha256s': source_rgb_sha256s,  # 按 Qwen Source 图像顺序，1 或 2 个
        'target_person_ids': target_person_ids,
        'driver_record_sha256': driver_record_sha256,
    },),
}
```

这不是 Training Supervision，不能把目标 RGB 或训练监督 Mask 当作该条件。reader、collator 与 runtime 都校验 Canvas、尺寸、人物顺序、Driver 和 Source SHA；5Ref 还绑定各自的 Source Bank。缺失时使用目标区域并记录 `source_repair=missing`，不自动缩放 Source Mask 或猜测跨 Canvas 映射。直接调用 runtime 时也可通过 batch 的同名字段提供这份经过校验的原始字典。

## 如何检查和使用

默认准备不构建模型、optimizer 或训练任务：

```bash
cd /home/shangguanrz/project/pic-edit
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
$PY scripts/train/train_qwen21_native_full_v61.py prepare
$PY scripts/train/train_qwen21_native_full_v61.py prepare --profile poselink-full
```

CPU 回归（不读取真实权重或真实数据）：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m pytest \
  tests/test_native_v61_spatial.py tests/test_native_v61_poselink.py \
  tests/test_native_v61_data.py tests/test_native_v61_runtime.py \
  tests/test_native_v61_integration.py -q
```

Windows 本地在 main 根目录使用 `python` 执行同一文件列表。集成测试使用随机初始化的小尺寸真实 Qwen Blocks，checkpoint 测试使用合成 Adapter 权重；不运行 optimizer step。

后续需要初始化时，显式使用以下工具；本次只实现工具，没有迁移真实权重：

```bash
$PY scripts/data/migrate_qwen21_native_full_v61.py \
  --v6 /data/shangguanrz/已有V6.pt \
  --output /data/shangguanrz/新目录/native_full_v61.pt \
  --report /data/shangguanrz/新目录/migration.json
```

`--pose-config` 可指定 V6.1 配置 JSON。`geometry_key_mask/edit_region_gate/multiscale_tokens` 三项独立开关默认开启；全部关闭精确走 V6，单独开启 Gate 时保留 V6 Attention 结果，只乘 Gate。新迁移继承旧 Encoder 和 PoseLink 参数、将新增类型 embedding 置零并记录 adapted/new；不继承 optimizer／RNG，不接受跨版本 resume。已训练的 V6 在新功能开启时会改变残差语义，不宣称迁移后数值等价。

推理入口 `scripts/run/run_qwen21_native_full_v61.py` 参数沿用 V6，包括 `--poselink-strength`。必须有明确 2／4／5Ref 的 `qwen_condition` 和新输出目录。训练的 Head／Full／Joint、数据划分、空间 FM、显式阶段切换与梯度门禁沿用 V6；训练状态 schema 为 `native_v61_training_v1`，尚未执行真实 preflight 或训练。

## 与 V6 的关系与验收界限

V6 文件、checkpoint header 和入口保持不变。V6.1 是独立 `native_full_v61`，不改 Qwen 原生 Self-Attention、文本／参考前缀、V5 训练快照、Loss 或已有冻结 Qwen 缓存。repair 引用进入数据身份签名；改变 repair 不需要重算冻结 Qwen latents，但严格恢复必须重新核对原始条件。

本轮只验证接口、支持归一化、预算、坐标、Gate、前缀、梯度、版本和冻结契约。没有做 P3 分层 Attention、P4 顶点 Contact／自适应 Gate，没有做真实迁移、7B、训练或生成实验。限制新增 PoseLink 残差不等于保证最终生成图背景不变，其他分支与 Qwen Attention 仍可传播目标变化。

2026-11-12 验证：原 V6 **70 passed**，新增 V6.1 完整套件 **65 passed**，累计 **135** 项 CPU 回归。Combined Pose 的人物绑定稀疏采样在插值前使用各自支持；其他人物的污染值与梯度回归均通过。规格与独立代码审查已通过。完全没有有效几何支持时，空间 metadata 明确记录 `key_support_empty=true`、`key_support_reason=no_valid_geometry`。

每参考图 256 是配置硬上限。若旧 V6 使用更大的自定义预算，初始化工具要求显式提供新的低预算配置，不静默削减，也不承诺更改预算后的数值等价。

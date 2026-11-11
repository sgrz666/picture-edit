# Native Full V6：Geometry-Guided PoseLink

V6 在 V5 的 Body／Interaction／Face／Hand／Source Detail 之外增加独立 PoseLink。目标是让当前生成 Hidden 动态读取有三维几何、空间和人物约束的 Pose Reference。单人与双人共用参数，Qwen 原生 Attention 不改。

本轮仅交付代码与 CPU 合成回归；没有运行真实权重迁移、数据提取、缓存构建、7B 前反向、训练或生成实验。V5 代码和正在运行的训练快照不修改。

## 架构与文件

|阶段|实现|行为|
|---|---|---|
|P0|`pose_layout.py`、`conditioning.py`|按原生 image IDs 提取 Pose；核对图像 SHA、角色、Canvas 和网格|
|P1|`poselink.py`|Target Hidden→512维 Query；Pose Hidden→K/V；独立零初始化输出|
|P2|`frontend.py`、`contracts.py`|一次 Reasoner 保存身份融合前的 Geometry／Depth 与逐人物 Tokens，按 Canvas 对齐|
|P3|同一 PoseLink|可见 owner 的软 Role Bias、空间距离 Bias、接触部位的局部跨人放宽|

`runtime.py` 保留 V5 参数名称，新增 `poselink.*`；`prepared.py` 在 V5 Prepared Control 上增加纯几何。`integration.py` 的各分支读取同一份未控制 block 输出，FP32 求和后只更新 Target Tokens。`generation.py` 复用原生 scheduler、64 通道 latent 与 VAE。`data.py`／`source_bank.py` 保留双来源各自的 Canvas 与显式身份映射。训练配置、参数策略、梯度门禁和恢复状态位于 [training](training/README.md)。

默认控制宽度 512、8 heads，注入 0-based **5、10、15、21、26、31**；PoseLink 参数跨层共享，输出 Head 与 log gain 分层。每张 Pose 最多池化为 **256** Tokens，Query 每块 **128** Tokens，双人最多 3 张 Pose，分块 Attention 的 key 数最多 768；不构造全图永久 Attention 矩阵。梯度模式用非重入 checkpoint 重算分块，不缓存跨时间步 K/V。小尺寸 CPU 测试配置不代表生产 4096 Hidden 的效果或显存验收。

三种参考角色依次为：

- 2Ref：`source, target_pose`。
- 4Ref：`source, combined_pose, person_0_pose, person_1_pose`。
- 5Ref：`source_a, source_b, person_0_pose, person_1_pose, combined_pose`。

PoseLink 索引不选 Source 或文本；原生 Qwen Pose Hidden 本身已经上下文化，仍可能间接含 Source 信息。`owner` 来自真实可见性和 Depth 排序，独立 `projected` 保留遮挡后的完整投影支持；不使用 Bounding Box 替代。未知 contact 填零，仅有正标注的接触部位可放宽跨人代价。

## 如何检查与使用

在服务器主仓目录运行准备命令，只打印配置与缺失绑定，不加载模型：

```bash
PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
cd /home/shangguanrz/project/pic-edit
$PY scripts/train/train_qwen21_native_full_v6.py prepare
```

CPU 回归命令：

```bash
CUDA_VISIBLE_DEVICES="" $PY -m pytest \
  tests/test_native_v6_poselink.py tests/test_native_v6_runtime.py \
  tests/test_native_v6_integration.py tests/test_native_v6_data.py \
  tests/test_native_v6_conditioning.py -q
```

`test_native_v6_integration.py` 使用随机初始化的真实小尺寸 Qwen21 Blocks，不读取预训练权重。覆盖 2／4／5Ref 零输出相对 V5 的逐值一致性、非零残差保持参考前缀、两次 forward 后 checkpoint 反向。其余测试检查交错文本、相邻图像、矩形网格、padding 污染、无效输入、Source 排除、接触局部性、梯度、版本回读与 typed Source Bank。Windows 本地可直接用 `python` 运行同一组测试。

后续需要迁移时显式调用以下入口；本轮没有执行：

```bash
$PY scripts/data/migrate_qwen21_native_full_v6.py \
  --v5 /data/shangguanrz/已有V5.pt \
  --output /data/shangguanrz/新目录/native_full_v6.pt \
  --report /data/shangguanrz/新目录/migration.json
```

工具严格继承 V5 模块、零初始化 PoseLink，并严格回读保存文件；V5 optimizer／RNG 不进入 V6 初始化。`--pose-config` 可读取 `PoseLinkConfig` JSON，独立设置 `use_geometry/use_interaction/use_spatial/use_role/use_contact`、层列表和内存预算。这些结构设置随 V6 checkpoint 保存。

推理入口为 `scripts/run/run_qwen21_native_full_v6.py`，参数沿用 V5，增加 `--poselink-strength`，设为 0 可关闭新增分支。必须使用有显式 `qwen_condition` 的 2／4／5Ref manifest；不接受 Source-only legacy 条件。输出目录必须是新目录。实际模型与数据验收后再运行：

```bash
$PY scripts/run/run_qwen21_native_full_v6.py --checkpoint /data/shangguanrz/新目录/native_full_v6.pt \
  --manifest /data/shangguanrz/已验证的pairs.jsonl --pair-index 0 \
  --poselink-strength 1 --output /data/shangguanrz/新推理目录
```

## 与 V5 的关系与当前结论

V6 复用 V5 的静态编码器与局部分支，独立维护编排和 checkpoint schema `native_full_v6`。2／4Ref 可读取既有 V5 manifest；5Ref 用 V6 row 与独立 Source Bank。已有冻结 V5 多参考缓存仅在 RGB SHA、Prompt、条件索引、角色和 Canvas 全部吻合时复用；缓存不含可学习 Adapter 输出。

5Ref 新 row 使用 `schema=native_full_pair_v6`、`scene_kind=dual`、`data_profile=full_smplx`，并引用 `sample/source_bank/supervision/qwen_condition/source_rgb/target_rgb`。`sample` 是 `native_full_sample_v6` 原始几何 NPZ，必须保留完整旧前端输入和 `local_condition_v4`；`source_bank` 通过 `source_bank.write_source_bank()` 保存，schema 为 `native_v6_source_bank_v1`。每源保留自己的 mask、Canvas、稳定 ID、冻结 ArcFace／DINO references 与可选 Detail；不保存 Qwen latent，运行时由真实 Reference Latent 绑定。`valid_image_mask` 必须属于目标 Canvas。推理不打开 `target_rgb` 或 `supervision`。新独立双来源 row 当前限定 `full_smplx`；既有 `hybrid_hi4d` 的 2／4Ref 继续走兼容 reader。准确字段与序列化方法见 `data.py`、`source_bank.py` 模块说明。

2026-11-11 本地完整 CPU 回归 **70 passed**；包含真实小尺寸 Qwen21 Blocks、V5 数值一致性、Source Bank／Canvas 拒绝、完整梯度和 optimizer 阶段参数恢复（不执行 optimizer step）。规格审查与独立代码审查通过。

目前能够证明的是代码接口、合成数值与梯度契约；尚不能证明姿态精度、身份保持、接触合理性、训练速度或显存收益。下一轮应另行做真实数据和 7B 有界验收，再决定训练配置；本轮不创建实验目录。

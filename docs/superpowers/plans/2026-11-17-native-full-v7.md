# Native Full V7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. User explicitly requires edits on main; do not create a worktree. The preceding design discussion is approved by the request to implement it.

**Goal:** 将五篇论文中适合 4Ref 双人编辑的机制接入现有 Native 架构，交付可迁移、训练、推理、消融的 V7。

**Architecture:** V7 继承 V6.2 的采样、前端、局部分支与原生安装器，只增加可学习融合、身份调制、侧路位置编码、纯几何接触桥和监督项。训练引擎通过默认保持旧语义的扩展点复用，不能复制整套 Trainer。规范见 `docs/superpowers/specs/2026-11-17-native-full-v7-design.md`。

**Tech Stack:** Python、PyTorch、现有 Qwen Image 2.1 Diffusers、pytest；不安装依赖。

---

### Task 1：模型组件与运行时

**Files:** 新建 `src/pose_control/qwen21_full/native_v7/{__init__,contracts,pose_fusion,poselink,local,runtime,integration}.py`；仅为必要无状态 hook 修改 `native_v62/poselink.py`、`native_v5/local.py`；新增 `tests/test_native_v7_{components,runtime,integration}.py`。

- [ ] 先写失败测试，证明未交付的 API 与所需行为：

```python
def test_v7_configuration_is_strict_and_roundtrips():
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    cfg = PoseLinkConfigV7()
    assert cfg.mask_pose_fusion and cfg.spatial_rope
    assert cfg.geometry_contact and cfg.identity_modulation
    assert PoseLinkConfigV7.from_dict(cfg.to_dict()) == cfg
```

- [ ] 执行 `python -m pytest tests/test_native_v7_components.py -q`，确认缺少 V7 而失败。
- [ ] 实现 `PoseLinkConfigV7(PoseLinkConfigV62)`，保留已有字段并验证四个 bool；每 head XY 维度必须能分成偶数对。新增融合和接触模块只能使用已存在的采样证据。
- [ ] 实现零初始化 Pose/Mask 残差与逐人物 Source identity scale/shift，继承现有局部身份 Attention；以下边界逐项先加测试再实现：

```python
def test_v7_switches_reject_non_boolean_values():
    import pytest
    from src.pose_control.qwen21_full.native_v7.contracts import PoseLinkConfigV7
    for name in ('mask_pose_fusion', 'spatial_rope', 'geometry_contact', 'identity_modulation'):
        with pytest.raises(ValueError):
            PoseLinkConfigV7(**{name: 'true'})
```

污染/梯度测试直接复用 `tests/test_native_v62_poselink.py` 的 case，暖输出 head 后污染 person_1_pose：断言 person 0 非接触输出逐值相等、被禁止 Hidden 梯度为零；Contact 测试将输出 bias 设非零，断言未知/禁止接触处仍为零。

- [ ] Contact K/V 使用纯几何/部位数据，Pose 路径关闭跨人 Hidden 放宽；known-positive/forbidden/unknown、无支持、单人场景都有数值和梯度测试。
- [ ] 复用 V6.2 安装器与 Prepared 记录；`NativeFullRuntimeV7.from_v62_checkpoint()` 显式保留旧参数，登记所有新参数；严格 `from_checkpoint()` 不接收 V6.2 header。
- [ ] 所有开关关闭后迁移 V6.2 输出一致；每个新参数进入正确 controller/head 分组，测试 S1/S2/S3 优化后确实改变。
- [ ] 在小型真实 Qwen Blocks 上验证 2/4/5Ref、Target-only、checkpoint 反向；提交任务代码。

### Task 2：监督与共享 Trainer 扩展

**Files:** 新建 `native_v7/{losses,training}.py`；按需修改 `native_v62/training/{config,backend,trainer,validation,probes,checkpointing,data}.py` 的公共扩展点；新增 `tests/test_native_v7_{losses,training}.py`。

- [ ] 先写不构建模型的配置与 Trainer 测试：

```python
def test_v7_prepare_does_not_build_model():
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, NativeV7Trainer
    cfg = TrainingConfigV7()
    report = NativeV7Trainer(cfg).prepare()
    assert report['schema'] == 'native_v7_training_v1'
    assert not report['model_constructed']
    assert not report['optimizer_created']
    assert not report['training_started']
    assert TrainingConfigV7.from_dict(cfg.record()) == cfg
```

- [ ] 运行失败测试，再实现纯配置与训练引擎注入点。默认 V6.2 路径继续使用原类和原 schema。
- [ ] 实现逐人物可见区域、接触区域的归一化 FM 项；空支持为可微零，部分 coverage 不重复相乘，不能使用 projected 当作可见外观区域。
- [ ] 复用 `native_v5.losses` 的五点对齐、decode 和冻结 evaluator；固定对应的 GT/Source 身份对比需检测有效负样本、低噪声和间隔，保留生成侧梯度。
- [ ] 数据资格预检发现零有效 ID 监督时显式失败并说明原因；提供各条件计数，不修改旧 manifests 或伪造标签。
- [ ] Trainer、grad preflight、loss probe 必须调用同一个 objective 扩展；snapshot/strict resume/advance-stage 必须使用当前配置类型与 schema。
- [ ] 测试 Loss 数值、梯度、无资格/无负样本、参数/optimizer 状态和旧 schema 精确回归；提交任务代码。

### Task 3：入口、文档与服务器验收

**Files:** 新增 `scripts/{train/train_qwen21_native_full_v7,run/run_qwen21_native_full_v7,data/migrate_qwen21_native_full_v7,audit/audit_qwen21_native_full_v7}.py`；新增 `native_v7/README.md`；更新本计划进度。

- [ ] 先写 CLI `--help`、`prepare`、迁移新文件/严格读回/拒绝覆盖、审计入口测试；再复用现有 CLI 控件、frozen cache、生成函数与 SHA 工具实现。
- [ ] README 中文说明每个论文机制具体接在哪里、与原论文不同之处、初始化/消融和训练状态；列出可运行测试命令：

```bash
python -m pytest tests/test_native_v7_components.py tests/test_native_v7_runtime.py tests/test_native_v7_integration.py tests/test_native_v7_losses.py tests/test_native_v7_training.py -q
python scripts/train/train_qwen21_native_full_v7.py prepare --diagnostics
python scripts/data/migrate_qwen21_native_full_v7.py --help
```

- [ ] 运行 V6/V6.1/V6.2 与 V7 的完整相关套件；复审规格覆盖后独立代码质量审查，有问题先修复再复审。
- [ ] 本地明确路径提交到 main；投递前核对远端 HEAD、建立时间戳源快照，将本任务文件通过 scp 精确覆盖至服务器，不运行 `git reset/clean` 或删移原产物。
- [ ] 服务器编译、CPU 套件、真实 Hi4D 两条样本的数据资格审计；记录本地/服务器 SHA 一致。任何训练准备输出不应启动训练。
- [ ] 最终逐项核对规范，报告完成内容、测试结果、使用入口、真实 7B/效果验证界限；所有必需项满足后才能标记 goal complete。

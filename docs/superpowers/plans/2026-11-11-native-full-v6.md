# Native Full V6 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans; execute continuously within the approved scope.

**Goal:** 实现独立 V6 的完整单／双人 PoseLink，本轮只做代码与 CPU 验证。

**Architecture:** 复用 V5 能力，新建结构化 Pose Layout、纯几何前端和独立 PoseLink 残差。版本化 Runtime、checkpoint 和入口隔离当前训练。

**Tech Stack:** Python、PyTorch SDPA、现有 diffusers Qwen 2.1 接口、pytest。

## 任务与验收

- [x] P0：新增 `native_v6/contracts.py`、`pose_layout.py`，先写 `tests/test_native_v6_poselink.py`。用交错 img_mask 与真实 metadata 规则验证 2／4／5Ref 的 Pose／target 索引，未知角色和错误网格拒绝。运行 `python -m pytest tests/test_native_v6_poselink.py -q` 先失败再通过。
- [x] P1–P3：新增 `poselink.py`，512 维共享多头 Attention、坐标对齐、Spatial／Role／Contact 软 Bias、零 Head、可配置层与分块内存预算。验证无效区域零残差、首步 Head 梯度、非零 Head 上游梯度、独立开关与 Source 排除。
- [x] Runtime：新增 `frontend.py`、`runtime.py`、`integration.py`；前端一次 Reasoner，几何在身份融合前保存；独立版本化 V5 编排并复用其模块，不改 V5。新增 `tests/test_native_v6_runtime.py` 先验证 migration/header、参数完整覆盖、零输出与 prefix／checkpoint 行为。
- [x] 入口：新增 V6 training 与 run／data 入口；独立 prepare 默认，不自动启动模型、optimizer 或实验。支持 Head／Full／Joint、V5 显式初始化、严格 V6 回读、旧空间 FM。配置缺绑定时明确报告。
- [x] 交付：中文 README 写清运行与测试命令、默认层和开关、尚未执行的真实验收。独立审查规格与实现，修复后重新测试。向服务器 scp 新文件并核对摘要、语法和 V5 源码未变。

## 关键实现约束

```python
# 只能增加目标残差，所有分支读取相同原始 Hidden。
result = original.clone()
result[:, target_mask] = identity_safe_add(original[:, target_mask], delta)
```

```python
# 初始化是新版本，绝不冒充恢复旧 optimizer。
assert v6_payload['architecture'] == 'native_full_v6'
assert 'training_state' not in migrated_payload
```

只运行专门的 CPU 合成测试和静态检查；不安装依赖、不构建数据缓存、不加载真实 checkpoint、不使用 GPU。

验收：完整 CPU 套件 70 passed；另有阶段 optimizer 参数恢复的 RED→GREEN 回归（无 optimizer step）。服务器投递审计保留在 `/data/shangguanrz/code-snapshots/native-full-v6-20261111T193254/`，其中 `before.json`／`after.json` 记录旧 V5 和新文件摘要，`v5-source-before.tar.gz` 是源代码快照。投递前服务器 HEAD 已更新到 `0d48cfa`，只新增 V6 文件，不覆盖该版本的 V5 修改。

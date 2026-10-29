# docs/ —— 设计规范、迁移记录与规划索引

`docs/` 是 pic-edit 的**设计规范与演进记录层**，收纳面向项目自身的架构说明、训练计划、运维迁移执行记录与工程规格方案。

---

## 一、核心文档导航总表

| 文档名称 | 主题与核心内容 | 归档日期 | 适用技术线 / 场景 |
|---|---|---|---|
| [`模型跨盘迁移执行记录_20261025.md`](模型跨盘迁移执行记录_20261025.md) | **根盘空间危机解决方案与模型跨盘迁移**<br>将 8 组大模型/数据（54GB）从 `/` 迁移至 `/data/shangguanrz/`，原位软链回填，双端逐文件 SHA256 全等 | 2026-10-25 | 基础设施与环境维护 |
| [`远端空间清理执行记录_20261025.md`](远端空间清理执行记录_20261025.md) | **远端系统盘 A 组空间安全清理记录**<br>冒烟残渣清理、字节码缓存清理，根盘可用空间恢复至 31GB | 2026-10-25 | 基础设施与环境维护 |
| [`Qwen21_V5固定样本结果报告工具_20261026.md`](Qwen21_V5固定样本结果报告工具_20261026.md) | **V5 固定样本实验结果报告工具与指南**<br>只读核对训练谱系、只读证据包导出、单文件 HTML 构建与离线 PDF 渲染规范 | 2026-10-26 | [`scripts/reports/`](../scripts/reports/README.md) |
| [`Qwen21_V5双人双源训练与使用说明_20261026.md`](Qwen21_V5双人双源训练与使用说明_20261026.md) | **Qwen 2.1 Native V5 双人双源训练与使用规范**<br>双源独立编码、五图协议输入、接触约束与损失配置 | 2026-10-26 | [`src/pose_control/qwen21_full/native_v5/`](../src/pose_control/qwen21_full/native_v5/) |
| [`Qwen2.1_项目框架总览_20261015.md`](Qwen2.1_项目框架总览_20261015.md) | 项目现行核心框架、各模块契约与端到端流水线总览 | 2026-10-15 | 全局框架参考 |
| [`Qwen21_双人Adapter适配性与实验调查报告_20261017.md`](Qwen21_双人Adapter适配性与实验调查报告_20261017.md) | 远端 `4.two man ` 实验深度调查，Adapter 架构适配性与 SOTA 双人方案对比调研 | 2026-10-17 | 双人控制技术线 |
| [`空间清理审计与清单_20261017.md`](空间清理审计与清单_20261017.md) | 2026-10-17 远端根盘空间释放（+126GB）与权重迁移对账审计 | 2026-10-17 | 历史清理记录 |
| [`workflow_multiple_设计_20261013.md`](workflow_multiple_设计_20261013.md) | `workflow-multiple/`（双人姿态编辑与交互迁移）设计记录与一期状态 | 2026-10-13 | [`workflow-multiple/`](../workflow-multiple/README.md) |
| [`workflow_PE蒸馏到Qwen3VL任务分支_20261015.md`](workflow_PE蒸馏到Qwen3VL任务分支_20261015.md) | workflow PE-I2I 蒸馏与小模型选型调研 | 2026-10-15 | 提示词流水线 |
| [`workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md`](workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md) | GGUF 社区量化评估报告 | 2026-10-14 | 推理加速与量化 |
| [`workflow_PE重写成本_量化与蒸馏调研_20261014.md`](workflow_PE重写成本_量化与蒸馏调研_20261014.md) | 提示词重写低延迟优化与成本测算 | 2026-10-14 | 推理优化 |
| [`workflow_无GT定量评测调研_20261013.md`](workflow_无GT定量评测调研_20261013.md) | 单人 workflow 无 GT 定量评测多轴指标体系 | 2026-10-13 | 评测指标设计 |
| [`workflow审查与整改_20261012.md`](workflow审查与整改_20261012.md) | workflow 流水线规范审查与整改记录 | 2026-10-12 | 流水线整改 |
| [`tiktok_qwen21_full_training.md`](tiktok_qwen21_full_training.md) | TikTok 数据集及 Qwen 2.1 全量训练规范 | 2026-10-03 | 训练规范 |
| [`qwen21_hi4d_dual.md`](qwen21_hi4d_dual.md) | Qwen-Image-2.1 双人主体控制说明 | 2026-09-29 | 双人控制规范 |

---

## 二、专项子目录导航

- [`superpowers/`](superpowers/README.md)：工程规划与设计规格层：
  - `plans/2026-10-26-hia-face-hand-dataset.md`：HIA 双人手脸高质量子集构建计划。
  - `plans/2026-10-26-hia-scaled-v2.md`：HIA 规模化数据集扩展方案。
- [`experiments/`](experiments/README.md)：实验专项说明与交接 runbook（BFS Body+Face 联合实验总结与历史操作记录）。

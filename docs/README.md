# docs/ —— 项目设计规范、训练计划与演进记录层

`docs/` 是 pic-edit 的**设计文档与演进记录层**，收纳面向项目自身的架构说明、训练规范、对照实验结论、清理执行记录，以及 `superpowers/`（工程方法论：计划与规格）和 `experiments/`（实验专项说明与交接 runbook）。它记录的是「项目自己的决策与现状」，与 `research/`（带来料证据包的专题调研/审计）分工明确。远端访问：`ssh sg` → `/home/shangguanrz/project/pic-edit/docs`。

## 里面有什么

内容 Markdown 文件（不含本文）：

| 文件 | 主题 | 日期 |
|---|---|---|
| `架构审查_20261012.md` | 对 node01/sg 工作区的**只读架构审查**（目录统计、git 状态、`diff`/`md5sum`、pytest 收集与运行），文内统计快照数字唯一来源于审查时刻实况 | 2026-10-12 |
| `目录清理执行记录_20261003.md` | 首轮目录大清理执行记录：远端辅助脚本归类入 `scripts/`，HOME 旧诊断/临时脚本逐字节归档，本地报告与 TikTok 历史成果迁移 | 2026-10-03 |
| `目录清理执行记录_20261009.md` | 二轮整理（R1–R8，缓存清理等），验收脚本输出 `passed: true`、`pending_assets: []`、`problems: []`，无 git 提交/推送 | 2026-10-09 |
| `adapter_v6_architecture.md` | DeepGen **Unified SMPL-X Adapter V6.3** 架构：`SMPLXConditionInjector → ConditionBundle → SMPLXAdapterReasoner → SharedRecurrentControlCore` 的上下文感知循环控制管线 | 9-17 |
| `BFS_扩大样本对照_20261004.md` | BFS（Body-Face-Skeleton）扩大样本对照（30 对 / 8 位人物 / 10 衣着组），身份有改善但存头部方向取舍 | 2026-10-04 |
| `BFS_面部保持移用评估_20261004.md` | BFS 面部保持移用评估：Qwen 2.1 LoRA 与当前底座结构兼容，建议「姿态生成 → 局部头部修复」两阶段方案 | 2026-10-04 |
| `champ_deepgen_single_sample_runbook.md` | CHAMP + DeepGen PoseAdapter **单样本运行手册**：数据契约（normal / depth / openpose / semantic 四张控制图） | 10-12 |
| `iper_v6_training_plan.md` | DeepGen V6.3 iPER 单人姿态控制**训练实施计划**（方案 A）：废弃旧 `train_iper_adapter.py`、数据隔离、分阶段推进 | 10-12 |
| `PROJECT_STRUCTURE_20261003.md` | 2026-10-03 的**工作线与目录快照**（DeepGen V6/V6.5/V6.6、Qwen 2.1 简化/完整、数据工程），历史结构快照 | 2026-10-03 |
| `qwen21_hi4d_dual.md` | Qwen-Image-2.1 **双人主体控制**说明：Hi4D 渲染几何作目标，A/B 源掩码绑外观，512 等比画布，`main_pairs` 的 train/val/test 分流 | 9-29 |
| `tiktok_qwen21_full_training.md` | TikTok → Qwen 2.1 完整版**训练使用说明**：512 小规模 Body 试训、受控 Face 验证，测试集只查分组交叉不进训练 | 10-03 |
| `workflow审查与整改_20261012.md` | 单人/双人 workflow 的审查发现与整改对应关系（stage 指纹拆成 identity/runtime、PE 产物只读、主体选择显式化） | 2026-10-12 |
| `workflow_无GT定量评测调研_20261013.md` | 无 GT 条件下的指标选择推导（第二读者 PCK、strict PCK、骨长比漂移、SMPL-X β、背景/人体保真、非主体位移） | 2026-10-13 |
| `workflow_multiple_设计_20261013.md` | 双人线设计决策与 85 格矩阵口径、swap/import 结论的来龙去脉 | 2026-10-13 |
| `workflow_大方向优化调研_20261014.md` | workflow 方向层缺口 P0–P6 与合同验收条目的对照 | 2026-10-14 |
| `workflow_PE重写成本_量化与蒸馏调研_20261014.md` | **PE-I2I 重写成本解剖**（占单人一次 run 前三段 GPU 时间 75.4%；4479 token 只换 310 词提示词）、量化与蒸馏两条路的难度评级、社区现成产物能否直接用、Qwen Research License 提示 | 2026-10-14 |
| `workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md` | **社区 GGUF 量化权重选型**：读文件头证明 plain Q4_K_M 把线性注意力敏感层压到 4-bit（作者消融里 KLD 0.146 那一档），真正收益来自换 llama.cpp + MTP（i2i 3.49–4.39×），附四级验收闸门与 sm_120/无 nvcc 的前置条件 | 2026-10-14 |
| `workflow_PE蒸馏到Qwen3VL任务分支_20261015.md` | **共享 Qwen3-VL 的 Prompt LoRA 可行性**：SSH 核验单人三图到两图契约、已有 lm_head、4479→320 token 答案比例、43.65M 参数分支预算，以及社区 T2I 蒸馏先例与分阶段验收 | 2026-10-15 |

子目录：

- `experiments/` —— 实验专项说明与交接 runbook（BFS Body+Face 联合实验的最终总结与历史 runbook）。
- `superpowers/` —— 工程方法论目录，下含 `plans/`（阶段计划）与 `specs/`（接口规格）。

## 与上下层的关系

- **技术线 1（DeepGen + Unified SMPL-X Adapter）**：`adapter_v6_architecture.md`、`iper_v6_training_plan.md` 对应 `src/pose_control/v6/`，训练入口在 `scripts/train/train_iper_v6_native.py`、`train_iper_v65_face.py`、`train_iper_v66_hand.py`。
- **技术线 2（Qwen-Image-2.1 迁移）**：`qwen21_hi4d_dual.md`、`tiktok_qwen21_full_training.md` 对应 `src/pose_control/qwen21/`（简化）与 `src/pose_control/qwen21_full/`（全量），入口见 `scripts/train/train_iper_qwen21_body.py`、`train_iper_qwen21_full.py`、`train_hi4d_qwen21_dual.py`。
- **BFS 落地路线**：`BFS_*` 两篇 + `experiments/` 是「Adapter 控姿 + BFS 768 ROI 精修面部」两阶段级联的落地记录，对接 `research/bfs_transfer_audit_20261004/`。
- 同级相关：`research/README.md`（调研/审计证据层）、`research/qwen21_dual_upgrade_20261011/`（双人升级设计稿与审计证据）。

## 注意与坑

- **`PROJECT_STRUCTURE_20261003.md` 是 2026-10-03 的旧结构快照**，已非现状（它未含后续 V6.5/V6.6 hand 分支、qwen21 双人升级等演进），新人勿当现状引用，以 `架构审查_20261012.md` 的实况统计为准。
- `架构审查_20261012.md` 内所有数字的唯一来源是文件内的统计快照（2026-10-12 00:2x 实况），带时间截面性，与后续改动不自动同步。
- `tiktok_qwen21_full_training.md` 明确：TikTok **测试集 1,156 对全程未用于训练 / Face 特征提取 / 采样**，只用于检查分组交叉，引用该数据线评测时勿把 test 当 train。

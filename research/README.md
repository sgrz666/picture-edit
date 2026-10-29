# research/ —— 专题技术调研与审计证据导航

`research/` 是 pic-edit 的**调研与审计证据层**：每个子目录对应一个专题技术调研，且均附带完整的来料佐证（JSON 结论、日志、验证脚本、第三方代码副本与渲染切片），可完全独立核验。

---

## 一、专题调研目录导航总表

| 调研子目录 | 专题方向与核心问题 | 核心证据形态 | 关联技术线 / 成果 |
|---|---|---|---|
| [`champ_detail_quality_20261025/`](champ_detail_quality_20261025/README.md) | **CHAMP 数据集细节质量与清洗规范**<br>肢体模糊度门禁、手部可用性判定、人脸清晰度多尺度过滤与训练对构建 | 30+ QC 脚本、`inspection/records.json` (7173 行质检记录)、`snapshot.json` | 数据清洗标准与高质量训练集构建 |
| [`union_adapter_architecture_20261017/`](union_adapter_architecture_20261017/README.md) | **Qwen21 Pro 与 Union 架构取舍与设计证据**<br>保留 16 个原生控制块、挂载 14.6M FP32 ProTaskAdapter、解耦外观 Bank 与显式接触 | `ARCHITECTURE.zh-CN.md`、`task_v3_requirements.json`、`official_matrix_records.jsonl` | [`src/pose_control/qwen21_pro/`](../src/pose_control/qwen21_pro/) |
| `canvas_native_portrait_acceptance_20261022/` | **竖屏原生画布与尺度适配验收**<br>Letterbox 变换、高宽比保持与生成留边裁切对齐 | 画布测绘脚本与尺度评估 JSON | `src/pose_edit/geometry.py` |
| `qwen21_depth_champ_20261023/` | **Depth 与 CHAMP 空间注意力编码融合**<br>核对深度图读取方向、10.7MB Depth 权重与 Native 融合分支 | 深度特征样本、注意力热图与复查脚本 | `src/pose_control/qwen21_full/` |
| `qwen21_full_native_geometry_20261022/` | **Full 方案 C 几何条件与实时反馈**<br>以现有 Full 为基准研究类型化几何条件编码 | 几何编码探针与张量对比 | `src/pose_control/qwen21_full/native_v5/` |
| `bfs_transfer_audit_20261004/` | **BFS 控制移用性与两阶段级联审计**<br>768 局部采样探针、LoRA 有无对照、CPU 兼容性核验 | 6 份 JSON 结果、对比面板 PNG | 两阶段面部精修方案 |
| `hi4d_smplx_pilot_20261011/` | **Hi4D 双人 SMPL-X 拟合与接触图修复**<br>Multi-HMR 拟合、标定相机外参对齐与穿透修正 | 网格参数、投影对比图与相机矩阵 | `workflow-multiple/` |
| `qwen21_dual_upgrade_20261011/` | **Qwen-Image-2.1 双人控制架构升级调研**<br>论文审查、第三方仓库副本、Token 绑定机制设计 | 8 个第三方仓库副本、设计草案、`readiness.json` | `src/pose_control/qwen21_full/` |
| `qwen21_dual_prompt_pilot_20261013/` | **双人交互提示词 Pilot 实验证据包**<br>双人动作文本编码、角色对齐与交付快照 | 训练/交付完整证据包、数据切片、快照 SHA256 | `experiments/4.two man /` |
| `smplx_condition_audit_20261008/` | **SMPL-X 条件输入质量审计**<br>视锥裁剪误差、身体关节坐标归一化核查 | `body_inputs.json` | `src/controls/generator.py` |
| `tiktok_training_20261003/` | **TikTok 数据集接入与训练方案调研**<br>单人舞蹈序列划分规范、基线代码快照 | `REPORT.zh-CN.md`、`verification.json` | `src/data/tiktok_dataset.py` |
| `cleanup_20261003/` | **2026-10-03 全仓整理现场只读快照**<br>迁移对照、备份清单与清理对账 | 14 份 JSON、4 份日志、`relocation_map.csv` | 历史归档依据 |
| `deepgen_assessment/` | **DeepGen-1.0 纯基座生成能力评估**<br>无控制条件下文本响应度与原始画质 | 渲染图、PDF 报告与自包含 pipeline 脚本 | 基座性能对照 |

---

## 二、与上下层的调用边界

- `research/` 属于**不可修改的只读证据库**，用于支撑 `docs/` 中的技术决策与标准制定。
- 严禁将 `research/` 内部的快照代码当做可执行的主干源码运行；生产运行代码一律位于 `src/` 与 `scripts/`。

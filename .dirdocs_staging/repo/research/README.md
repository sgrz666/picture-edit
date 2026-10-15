# research/ —— 调研与审计证据层：每条工作线"为什么这样选"的历史快照

> 工作区路径：`/home/shangguanrz/project/pic-edit/research`　|　本级：8 个子目录 / 仅 1 个说明文件　|　子树规模：约 141 MB（`du -sh --si` 实测合计：bfs 472k、cleanup 189k、deepgen 17M、hi4d 59M、legacy 115k、qwen21_dual 62M、smplx_condition 25k、tiktok 1.7M），统计时间 2026-10-12

## 一、这一层在干什么

`research/` 不属于训练或推理代码线，而是**调研与审计证据层**：每当项目要做一个有争议的决策（选基座、上 768 分辨率、清理目录、接入 TikTok、给 Hi4D 造 SMPL-X 双人标注、升级双人 Adapter），就先在这里留下"当时看了哪些方案、比较维度是什么、数字证据是什么、最终采纳/否决了谁"的现场记录。上游是各类探针脚本与人工审查（产物 JSON/日志/图板），下游消费者是 `experiments/` 的训练决策和 `docs/` 的选型文档。它是 `experiments/README.md` 的"为什么"侧账本。

## 二、层级地图

| 子目录 | 装什么 | 内容实证（真实文件名/数量） | 与其他子目录的关系 |
|---|---|---|---|
| `bfs_transfer_audit_20261004/` | BFS（body-face-skeleton）LoRA 控制移用到 768 分辨率的探针审计 | 7 个文件：`results.json`（512 探针指标）、`probe768.json`、`existing_probe_summary.json`、`panel_004.png` 等 | 服务于 qwen21 单人控制线；被后续 `qwen21_dual_upgrade` 的注入预算讨论引用 |
| `cleanup_20261003/` | 2026-10-03 目录大清理的完整验收证据 | 23 个文件：`final_verification.json`、`relocation_map.csv`（176 行）、`home_classification.json`（66 项）等 | 元证据：其他目录的位置变动以它为准（`navigation_operations.json` 记录了软链去向） |
| `deepgen_assessment/` | DeepGen-1.0 适配性评估报告的撰写与 PDF 渲染三版快照 | 顶层 `build_report.py`、审计用 pipeline 副本；三代 `render*/` 共 32 张页面 PNG + 3 份 PDF | 结论沉淀为 `legacy_202609/` 的 docx/md 报告 |
| `hi4d_smplx_pilot_20261011/` | Hi4D 两帧 SMPL-X 双人拟合试点（inputs→initialization→observations→fit_v1 流水线） | 59 MB；父级 `README.md` 为用户手写（本层唯一保留的手写件） | 产物 `fit_v1` 直接成为 `qwen21_dual_upgrade` pilot 控制资产的数据源 |
| `legacy_202609/` | 2026-09 基座选型调研归档（显存预算 md、checkpoint 审计、评估 docx） | 4 个文件：`第三阶段图像编辑调研与显存预算.md`、`checkpoint_audit.json` 等 | 是 `deepgen_assessment` 同期工作的文字主报告 |
| `qwen21_dual_upgrade_20261011/` | Qwen2.1 双人 Adapter（dual_v8）调研 + 8 个参考论文仓库快照 + 试点控制资产 v1→v3 | 62 MB / 880+ 文件；**活跃开发中**（`pilot_controls_v2/v3`、`implementation/` 写于 10-12） | 消费 `hi4d_smplx_pilot` 的拟合结果；引用 `smplx_condition_audit` 的数据语义 |
| `smplx_condition_audit_20261008/` | TikTok v6 条件张量里 SMPL-X 参数是否真正进入训练通路的核查 | 仅 `body_inputs.json` 1 个文件（passed=true，2 个样本） | 上承 `tiktok_training_20261003`，下证 qwen21_full 条件设计 |
| `tiktok_training_20261003/` | TikTok 接入 qwen21_full 训练链路的连通性冒烟与就绪审计 | 6 个文件：`REPORT.zh-CN.md`、`readiness.json`、`verification.json`（30 万字符级）、`comparison.png` | 其 `readiness.json` 的 sha 被 `smplx_condition_audit` 引用为同一清单 |

## 三、注意事项 / 调用路径

- **远端为准**：本地 Windows 镜像 `D:\codeplus\pictureedit\research` 还存有 16 个远端已不存在的目录（`abc_showcase_20261003`、`adapter_audit_20261002`、`smplx_scan_20261003`、`structure_audit_20261003`、`synbody_audit_20261003`、`tiktok_smoketest_202609`、`hi4d_dual_audit_20261011`、`pei2i_body_face_8arm_20261009`、`dfm_preview_20261008`、`dna_rendering_20261008`、`geometry_prompt_coordination_20261009`、`bfs_body_*` 4 个、`adapter_value_diagnosis_20261011`）。2026-10-12 在远端 `pic-edit/`、`pic-edit/archive/`、`/data/shangguanrz/pic-edit-archive/` 均实测不到它们，应视为本地历史镜像残留，勿据此推断远端结构。
- 本层各目录的 `目录说明.md` 与旧 `README.md`（10-11 版）均为 `scripts/gen_folder_docs.py` 自动生成，无参考价值；本层 README 为人工实证版本。唯一例外：`hi4d_smplx_pilot_20261011/README.md` 是用户手写，保留。
- 多数结论引用了 `/home/shangguanrz/nihaochen/**` 下的运行产物（baseline、envs、qwen21_full 训练目录）：该路径属于另一工作区，本层只记录指针，不复制内容。
- `qwen21_dual_upgrade_20261011/` 与 `/data` 部分实验目录正被另一会话实时写入，本层统计为 2026-10-12 快照。

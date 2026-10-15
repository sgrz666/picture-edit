# scripts/audit/ —— 只读审计与验收入口：不训练、不生成图像，只核验数据、权重与工程结构
> 工作区路径：/home/shangguanrz/project/pic-edit/scripts/audit　|　本级：0 个子目录 / 22 个 Python 实体文件（约 0.4 MB，`du -sh --si` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md 与 `__pycache__`（编译缓存，一行带过）

## 一、这一层在干什么
本目录收纳全部"验收类"入口：CPU-only 的数据质量审计、权重谱系核验、结构冒烟和工程整理验收。所有脚本的 docstring 都强调同一纪律——不构造优化器、不保存权重、不下载外部模型、不改训练入口，因此可以在 GPU 被训练占用时随时运行。上游是 `src/` 各控制线和 `datasets/` 清单，下游产物是 JSON/CSV 审计结果（多数写到调用者指定的 `--output`，历史运行落在 `experiments/` 与 `/home/shangguanrz/nihaochen/qwen21_full/` 下）。

## 二、层级地图（按用途分类）
| 类别 | 文件 | 内容实证 | 与其他类的关系 |
|---|---|---|---|
| Qwen21 权重/继承审计 | `audit_qwen21_weight_compatibility.py`、`inventory_qwen21_full_weights.py`、`check_qwen21_full_preservation.py`、`evaluate_qwen21_weight_inheritance.py`（在 eval/）、`verify_qwen21_full_runtime.py`、`verify_qwen21_full_resume.py` | docstring 逐条声明 CPU-only / 逐张量 provenance / 一步 CUDA 恢复复现性检查（reload checkpoint、恢复全部 RNG 后比对模型、AdamW 状态与梯度） | 服务于 `scripts/train/train_iper_qwen21_full.py` 一线；runtime/resume 要求 Body/Face checkpoint 已存在且 GPU 空闲 |
| Face 单样本诊断 | `probe_qwen21_face_single_sample.py`、`verify_qwen21_face_single_sample.py`、`audit_qwen21_face_alignment.py`、`verify_qwen21_face_repair.py`、`check_qwen21_native_value_prediction.py` | probe：固定噪声+区域误差+一步重建；alignment：对既有生成图做离线裁剪/对齐审计，"target 五点是离线监督元数据" | 与 `scripts/run/run_qwen21_face_single_sample.py`、`run/finalize_*` 构成"跑→验→收尾"闭环 |
| Hi4D 泄漏/条件审计 | `audit_hi4d_qwen21_prompt_leak.py`（target-RGB 隔离与 target-geometry 敏感性，需 `--checkpoint --control-cache --latent-cache`）、`verify_qwen21_condition_combinations.py`（原生条件因子补充实验的产物验收） | — | 消费 Hi4D dual 线的缓存与检查点 |
| TikTok V3 数据质量 | `audit_tiktok_v3_blur_sample.py`、`_final_pairs.py`、`_pose_redundancy.py`、`_strict_final.py`、`_visual_quality_v2.py`、`_visual_quality_v3.py` | 输入/输出固定在 `datasets/TikTokDataset/TikTok_v3/manifests/` 与 `TikTok_v3/qc/`；v2→v3 是同一视觉质量判据的两轮迭代 | 是 `scripts/data/` TikTok V3 建池流水线的质检端，v3 为最终版 |
| V6 结构冒烟 | `smoke_adapter_reasoner.py`、`smoke_adapter_v6.py`、`smoke_deepgen_control_interface.py` | 合成条件、零初始化头等价性（`1e-6`），明确"不训练、不存权重、不下载" | 对应 `docs/adapter_v6_architecture.md` 的验证命令 |
| 工程整理验收 | `verify_project_organization.py` | 只读校验 2026-10-03 整理日志（`archive/cleanup_20261003/organization.json` 的保护清单/别名），`--full-assets` 全量哈希复查约 9 分钟 | **注意**：10-12 入口层迁到 scripts/ 真实文件后，它断言的还是"根目录实体+scripts/ 链接"旧布局（`docs/架构审查_20261012.md` P1-2 后续），现会报大量 problem，属校验器过期而非工程被破坏 |

## 三、关键文件与实证内容
- `verify_project_organization.py`：整理验收的单一入口。`docs/目录清理执行记录_20261003.md` 记载当时输出 `passed: true, pending_assets: [], problems: []`，全量复查读取约 157.5 GiB 归档。
- `audit_qwen21_weight_compatibility.py`（21.7 KB）：权重语义映射审计，产出 JSON 供 `scripts/reports/write_qwen21_weight_audit_report.py` 生成 Word 报告。
- `audit_tiktok_v3_visual_quality_v3.py`（50.4 KB）：本目录最大脚本，含边缘锐度、掩码、面板拼接的完整质检逻辑，输出 CSV/JSONL 到 `TikTok_v3/qc/`。

## 四、注意事项 / 调用路径
- `scripts/` 根目录的 6 个软链接（`probe_qwen21_face_single_sample.py`、`verify_qwen21_face_repair.py`、`verify_qwen21_face_single_sample.py`、`verify_qwen21_full_resume.py`、`eval_qwen21_face_repair.py`→eval/、`write_qwen21_face_single_sample_report.py`→reports/）是 10-03 过渡兼容入口，10-09 批量撤除链接时因被 finalize runner 哈希清单引用而保留（见 `docs/目录清理执行记录_20261009.md` R7），实体都在本目录或同级目录，不是重复副本。
- 复现示例：`python3 scripts/audit/verify_project_organization.py [--full-assets]`；Face 对齐复查 `python scripts/audit/audit_qwen21_face_alignment.py --single-root <run> --repair-root <run> --output <json>`（用 `/home/shangguanrz/nihaochen/envs/qwen21/bin/python`）。
- 本目录无符号链接；`__pycache__` 为运行产生的字节码缓存。

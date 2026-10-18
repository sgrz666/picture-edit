# research/tiktok_training_20261003/ —— TikTok 数据集接入与 Qwen 2.1 完整版训练调研

本目录是 2026-10-03 对 **TikTok 数据集接入 Qwen-Image-2.1 完整版（qwen21_full）训练链路**的检查与交付证据包。结论：训练链路已打通，适合小规模试训，暂不适合直接大规模长训。对应 `docs/tiktok_qwen21_full_training.md` 的使用说明。验证主机 `ssh sg`，GPU RTX 6000D ~84 GiB，基线 HEAD `b1a00c2a3b2955779af40ab19b5bab71ac19539a`，远端输出 `/home/shangguanrz/nihaochen/tiktok_qwen21_full_20261003`。

## 里面有什么

| 文件 | 内容 |
|---|---|
| `REPORT.zh-CN.md` | 交付报告：链路检查与修改记录、结论（Body→Face 真实前向/反向/更新/保存/恢复/采样均验证；生成质量未达目标姿态准确+人脸改善，全量 Face 缓存未提取） |
| `verification.json` | 完整验证结果（296 KB）：`captured_utc`/`ssh_alias`/`run_root`/`architecture`/`resolution`/`readiness`/`dataset`/`splits` 等 |
| `readiness.json` | 就绪度摘要：`dataset`/`architecture`/`splits`/`train`/`input_pairs`/`eligible_pairs`/`excluded_pairs`/`exclusion_reasons`/`sequences`/`video_groups`/`val`/`test`/`pairs` |
| `comparison.png` | 对照图（1.3 MB） |
| `unit_tests_fresh.log` | 全新单元测试日志（811 B，pytest 进度） |

## 与上下层的关系

- 设计说明：`docs/tiktok_qwen21_full_training.md`、`docs/superpowers/plans/`（项目整理）。
- 实现层：`src/pose_control/qwen21_full/`、`src/data/`（TikTok v3 条件）、`scripts/train/train_iper_qwen21_full.py`、`scripts/eval/eval_iper_qwen21_full.py`。
- 条件质量上游：`research/smplx_condition_audit_20261008/`。

## 注意与坑

- **TikTok 测试集 1,156 对全程未用于训练 / Face 特征提取 / 采样**——`readiness.json` 与报告都只把 test 用于检查分组交叉（video 组无跨划分）。引用该数据线指标时勿把 test 当 train 评测集。
- 严格数据池原 train/val/test 为 8,574 / 1,120 / 1,156 对；Body 条件筛选后 train/val 为 8,548 / 1,112 对（排除 26/8 对身体有效关键点不足样本），比较规模时口径要对齐。
- 报告明确「暂不建议直接大规模 Body＋Face 长训」：本次仅验证 4 训练对 / 2 验证对，先用固定验证集短训观察再扩规模。
- 这是 2026-10-03 的**链路检查快照**，后续 qwen21_full 已接入 Hi4D 双人等，引用前核对基线 HEAD 与当前主树差异。

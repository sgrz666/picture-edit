# docs/experiments/ —— BFS Body+Face 联合试点的结论文档（runbook 与最终交接）
> 工作区路径：/home/shangguanrz/project/pic-edit/docs/experiments　|　本级：0 个子目录 / 2 份实体文档（7.2 KB + 6.0 KB，`ls -la` 2026-10-12 实测）　|　另含生成物 README.md、目录说明.md

## 一、这一层在干什么
记录 2026-10-08 那条"Body+Face 联合训练 + 冻结 BFS LoRA 三分支对照"实验线的两份文档：一份是历史 512 协议 runbook（已停用），一份是原生画幅改造后的最终交接。关系是取代：先看 handoff，runbook 只用于理解旧结果怎么看。

## 二、层级地图
| 文档 | 工作线 | 结论（正文实数） |
|---|---|---|
| `2026-10-08-bfs-body-face-joint-handoff.md` | 当前权威：原生分辨率（TikTok 604×1080 pad 到 608×1088）联合底座 J 256 步 + Qwen/BFSBody/BFSHead 各 256 步，共 1,024 次正式更新 | 36 例全量：初始 ID 0.4285 → J 底座 0.4602（全场最高）；PCK@0.05 BFSHead 0.9206（全场最高）；长程微调姿态↑身份↓（Qwen 0.4047/BFSBody 0.3583/BFSHead 0.3333）。落地建议：Face ROI 加独立身份梯度损失。远端 run `/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008`，本地图与报告 `research/bfs_body_face_joint_pilot_20261008/report/index.html` |
| `2026-10-08-bfs-body-face-joint-pilot-runbook.md` | 同一实验的旧 512 letterbox 协议操作记录（顶部明确"旧 512 协议已停止，先读 handoff"） | 查看方法、锁定资产（`data/lock.json`、runtime 源码 tar 与 SHA256 `45f09c62…`）、重评三命令（eval_qwen21_joint_pilot --score-only、summarize、build_report）与边界声明（短程单 seed、不能证明跨人物身份迁移） |

## 三、注意事项
两文档共同引用的产物目录都在 `research/`（本地）与 `nihaochen/qwen21_full/`（远端），不在本目录；本目录只存结论。

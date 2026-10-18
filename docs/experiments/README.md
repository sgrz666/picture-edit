# docs/experiments/ —— BFS Body+Face 联合实验的专项说明与交接 runbook

`docs/experiments/` 是 `docs/` 下的**实验专项说明子目录**，专门收纳 BFS（Body-Face-Skeleton）Body+Face 联合架构实验的总结与历史 runbook。它的作用不是存放生成图像或指标原始数据（那些在本地 `D:/codeplus/pictureedit/research/bfs_body_face_joint_pilot_20261008/`），而是把「做了什么、在哪看结果、怎么复现」写成可交接的文字记录。

## 里面有什么

| 文件 | 主题 | 日期 |
|---|---|---|
| `2026-10-08-bfs-body-face-joint-handoff.md` | BFS Body+Face 联合实验**最终总结与交接**：已废除旧 512 letterbox/resize，全面转入原生画幅（Native Resolution），完成 1,024 步模型更新与全量 36 例评测对照；给出核心工作目录、训练服务器、Python 环境（RTX 6000D 80GB）与远端/本地产物路径 | 2026-10-08 |
| `2026-10-08-bfs-body-face-joint-pilot-runbook.md` | 同实验的**历史 runbook**：TikTok Body+Face 联合训练与「冻结 BFS 底座」对照的查看方法（本地 `report/index.html`、`summary.md`、`evaluation/`、`quality_reviews/`、`data/lock.json` 与运行时快照哈希）。注明旧 512 协议已停止，保留原始像素宽高的改造尚未实现 | 2026-10-08 |


## 与上下层的关系

- 承接 `docs/BFS_扩大样本对照_20261004.md` 与 `docs/BFS_面部保持移用评估_20261004.md` 的选型结论，落地到「Adapter 控姿 + BFS 768 ROI 精修面部」两阶段级联（当前落地路线）。
- 对应训练线：`src/pose_control/qwen21_full/`（冻结 BFS 底座对照）与 `research/bfs_transfer_audit_20261004/`（BFS 移用性审计证据）。
- 原始指标与图像：本地 `D:/codeplus/pictureedit/research/bfs_body_face_joint_pilot_20261008/`，可离线双击 `report/index.html` 完整查看。

## 注意与坑

- **阅读顺序**：先看 `handoff.md`（结论与现状），`runbook.md` 描述的是历史配置与历史结果查看法；runbook 开头已声明「旧 512 协议已停止」，勿按其中历史配置当成当前协议。
- 两篇均指向**本地** `research/bfs_body_face_joint_pilot_20261008/` 看结果，远端 `ssh sg` 上是代码快照（`runtime_source_20261008.tar.gz`），不含渲染图。
- 评估资产用固定监看下标与 `lock.json` 哈希锁定，复现前先核对哈希再动样本池。

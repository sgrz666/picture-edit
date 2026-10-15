# tiktok_00001_condition_repair/ —— 对比不同 SMPL-X 条件通道对 TikTok 00001 姿态漂移的修复效果

实验目的
- 修复 TikTok 00001 上因 SMPL-X 条件通道差异导致的姿态漂移，对比「仅姿态 / 姿态+部位 / 训练 1500 步 / 训练 1500 步+face025」等条件配置下的重定向质量，找出最优条件修复方案。

里面有什么
- `conditions/`：不同条件通道的离线缓存（pose_only / pose_part / face025 等）。
- `inference_pose_only/`、`inference_pose_part/`：仅用姿态、姿态+部位条件下的推理结果图。
- `inference_trained_1500/`、`inference_trained_1500_face025/`：用训练 1500 步权重 + 是否带 face025 通道的推理结果图。
- `train_pose_part_1500/`：pose_part 条件 1500 步训练的权重与日志。

关键结论
- 四类条件（pose_only / pose_part / inference_trained_1500 / face025）共产出 44 张分阶段修复对比图。结论指向：`pose_part` 与训练后条件相比 `pose_only` 更能抑制姿态漂移；具体指标以 [../README.md](../README.md) 第 4.3 节为准。

与其他实验的关系
- 属于 DeepGen 技术线 TikTok 条件修复分支，服务于后续 `tiktok_overfit_sanity_full` 的门禁实验。回到 [../../README.md](../../README.md)。

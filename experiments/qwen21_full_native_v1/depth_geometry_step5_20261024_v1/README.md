# Native Depth V2：架构实现，未运行验收

本轮完成 P0–P3：保留 CHAMP 的 Relative＋Valid 分支，新增 Camera-Z 几何细节、Normal/Part 感知融合及 runtime v3。沿用目标帧同源配对，不加入 A/B 重渲染或几何损失。

远端同名目录在 `/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/`。`source/baseline.json` 和 `source/source_baseline.tar.gz` 记录开始时的源码、UTC 时间、HEAD、未提交状态和 SHA256。`report/` 只保存语法编译、AST 与尺寸接线的静态核对记录，不把它们称为运行测试。

本轮不运行测试、模型前反向、训练、渲染、缓存构建或 checkpoint 迁移。新增架构尚未运行验收；上一轮 Body21 的测试数字不适用于本轮。最终接口与命令见 `src/pose_control/qwen21_full/NATIVE_DEPTH_V2.md`。

上游为 `global_pose_step4_20261024_v1` 的 Body21，保留此前 Depth V1 作为独立对照。后续验收再检查混合模态、相机坐标、边界、零投影、checkpoint 和 Qwen 梯度。

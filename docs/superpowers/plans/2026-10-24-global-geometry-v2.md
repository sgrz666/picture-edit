# Global Geometry v2 执行记录

执行依据为用户批准的 21 关节改造计划。本人执行，不派 subagent。远端源码是基线；loss、optimizer 和训练算法保持原样；本轮不启动训练。

1. 快照 `sg` 当前 HEAD、未提交 canvas 工作及源码 SHA256，复用 `codex/full-native-geometry` 工作树。按 baseline/前次投递 SHA 检查后再更新远端源码。
2. 固定 Body63、SMPL-X 1–21、完整 22 节点树、前两列 rotation6D、Global26 两个同步平移槽。原 Full、Native v1 显式 legacy。
3. 增加原始 FK 缓存和严格 reader；TikTok/iPER 各 8 帧，模型重建顶点误差门槛 1e-4。只读既有 SMPL-X 参数，使用现有 SMPLest-X 环境。
4. 增加共享 BodyPoseTokenEncoder：11→256→256、Joint Embedding、2 层 Skeleton Attention。新增可独立冻结的参数前缀。
5. 原 Global Attention 后接 Joint Cross-Attention 增量；仅最终投影置零。关节投影按目标像素构造 Gaussian bias，区分 FK 与可投影有效性，全不可投影跳过。
6. 扩展 ConditionBundle、双人子批次、数据读取、Native runtime、checkpoint 迁移与重建、推理和既有训练入口的接线；不增加 Qwen token。
7. 测试覆盖旋转、拓扑、canonical canvas、缓存漂移、拒绝旧 schema/SMPL69、双人、零增量、strict 继承、梯度和冻结状态。用实际小型 Qwen 与 7B 矩形输入做有界前反向，无 optimizer。
8. 保存维度表、轻量报告、复查/使用命令；报告 iPER 相机输入变化，不将其混同为结构误差。不根据本轮结构验收宣称质量改善。

# V6 训练接口

训练目标默认仅 Velocity Loss。保留服务器 V5 当前离线空间 FM：`spatial_body/spatial_joint/spatial_hand` 默认 0，`spatial_max` 默认 2.5；按 GT SHA＋Canvas 索引加载，像素叠加并截断后再池化，不重复乘 padding coverage。Face／Hand 局部 Velocity 与非零空间 FM 不能同时启用。

## 配置与入口

`config.py` 是纯配置；`backend.py` 负责真实 Qwen／冻结缓存；`parameters.py` 完整登记参数和固定 optimizer 超集；`trainer.py` 负责累积、日志与更新边界；`validation.py` 检查冻结、前缀与梯度；`checkpointing.py` 使用独立 `native_v6_training_v1` 保存恢复状态。原 V5 入口和恢复状态均不修改。

默认 `prepare` 是 **S1 PoseLink Head**，不会构造模型或 optimizer。缺少 checkpoint、数据、缓存、模型 SHA、输出目录、预算或验证源时只列出缺失项。`--profile poselink-full` 对应 S2；`joint` 对应 S3。阶段转换继续使用显式 `advance-stage`，不自动训练。

|策略|活动参数|其余参数|
|---|---|---|
|S1 / poselink-heads|`poselink.output_heads.*`|全部冻结|
|S2 / poselink-full|`poselink_heads`＋`poselink_controller`|V5 与 Qwen 冻结|
|S3 / joint|全部 Native 活动分支；共享 Geometry 需显式解冻|Qwen／VAE／Qwen3-VL 冻结|

S2／S3 检查活动 Head 已完成 warmup，包括 PoseLink。全零 Head 初次反向只要求 Head 梯度，临时非零 Head 探针验证完整上游后恢复。新训练、阶段转换与严格恢复都要求匹配的真实梯度报告；CPU 合成测试不会伪造 production 报告。

Route `poselink_strength` 默认：`poselink/joint` 为 1，Body／Face／Hand／Detail／Interaction 为 0，可显式覆盖。原模态强度仍位于 `controls`，旧 Depth／Identity 设置位于 `base_controls`。阶段默认都不加入局部或辅助 Loss；需要时显式配置。

```bash
$PY scripts/train/train_qwen21_native_full_v6.py prepare
$PY scripts/train/train_qwen21_native_full_v6.py prepare --profile poselink-full
```

2Ref、4Ref、5Ref 分别用 `qwen_condition_profile=single_2ref/dual_4ref/dual_source_5ref`；同一配置的实际来源必须符合该 profile。B1、workers0、单进程。Source Bank 必须由 V6 reader 保持 typed，不能通过普通字典 collate 变成列表。

## 后续操作顺序

未来绑定真实数据时使用配置 JSON；`TrainingConfig().record()` 给出完整字段，`prepare` 输出同一结构和缺失绑定。文件引用使用绝对路径＋SHA。模型与冻结缓存、manifest、原视频／序列 group、验证划分与独立 `/data/shangguanrz/` 输出目录均需明确。

1. 显式 V5→V6 初始化；不恢复 V5 optimizer。
2. `prepare --config 配置.json` 检查配置；`preflight --level data` 核对来源、资格、划分和空间权重。
3. `preflight --level grad` 运行真实 7B 梯度／冻结／前缀验收，保存匹配报告。
4. `train` 启动 S1；后续通过 `advance-stage` 进入 S2／S3，`resume` 只恢复同版本同配置。

以上真实操作本轮均未执行。恢复要求相同代码／模型／缓存／数据／环境签名，保存 optimizer 边界、RNG、Sampler、日志已提交前缀与冻结摘要；旧 V5 training schema 不可作为 V6 resume。

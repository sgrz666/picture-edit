# 小红书 Native 装配与缓存并行

用户明确要求显存充足时不再串行等待。保留原 512／40／64、SFT 各 1000、后续各 50、Bilibili20k→V7 固定 SHA 和 SDPose。当前装配实际在第 130 帧 SAM 覆盖不足退出，129 帧已验收记录只读复用。

- [ ] 从 selection 和固定 Canvas 独立发布最终 Qwen 条件、SDPose guide、prompt bank；它们不依赖 Native mesh、SAM 或 DINO。Native manifest 必须引用同一份条件，禁止编码后悄悄改 SHA。
- [ ] 用共享原始 `encode_pair` 独立编码所有 616 对，保存独立 entry commit；索引与模型／输入／条件 SHA 完整绑定，允许恢复但不覆盖旧文件。
- [ ] Native 装配与 Qwen 编码使用两个实际 GPU 子进程；最终发布 cache_ready 前，以真实 Native DataPlan 核对完整条件集合与逐条 provenance。全部就绪后交回原 SFT／RL 编排。
- [ ] 第 130 帧为前景植物遮挡人像，13 个 SDPose body 提示点只有 8 个落在 SAM mask 中（.61538），三种提示诊断均不满足 .65。保留 .65 资格门槛；不合格分割显式记为 unavailable，所有依赖该分割的空间监督与背景评价支持置零，不丢样本、不重新分配缺失区域权重，global FM 保留。Source 条件的非空 SAM 代理另作弱条件记录，不冒充合格 GT。
- [ ] CPU 回归覆盖资格判断、缺失区域、条件稳定与索引对齐；真实缓存和资产并行进度、显存、PID 及恢复后状态记录到现有实验 README。

新代码用新的冻结源码目录，旧快照和失败输出保留。主线权重、数据集原图、预算、画布大小、Qwen／evaluator 模型均不变。

# 小红书 Native 装配与缓存并行

用户明确要求显存充足时不再串行等待。保留原 512／40／64、SFT 各 1000、后续各 50、Bilibili20k→V7 固定 SHA 和 SDPose。当前装配实际在第 130 帧 SAM 覆盖不足退出，129 帧已验收记录只读复用。

- [x] 从 selection 和固定 Canvas 独立发布全部 616 个最终 Qwen 条件、SDPose guide、prompt bank；Native manifest 引用同一份条件 SHA。
- [x] 共享原始 `encode_pair` 已完成 616／616 对；独立 entry commit 和缓存索引按模型／输入／条件 SHA 固定，峰值 allocated 17.49 GiB，索引 SHA `852db52f99065a53b39b870bad5648a47ea73e059be75b3a9313f15095b04620`。
- [x] 两个实际子进程已分别启动；编码进程不中断交接且已完成。Native 装配恢复至 153／747，主管 PID 3078223、资产 PID 3079034，旧已验收帧逐 SHA 复用。
- [x] 第 130 帧 .61538 覆盖不合格，空间标注按 unavailable 处理，不降低 .65 门槛、不丢样本、不分配缺失权重；global FM 保留。Source 代理另行记录。
- [x] 新增资格、缺失区域、typed schema 复用、缓存／Native 绑定回归通过；服务器冻结核心＋准备补丁共 104 passed，945 个文件 SHA／编译通过。README 已记录 PID、实际进度和显存。
- [ ] Native 全部完成后，以真实 DataPlan 核对完整条件集合与 provenance，再发布 cache_ready 并继续既定 SFT／RL／test 流程。

新代码用新的冻结源码目录，旧快照和失败输出保留。主线权重、数据集原图、预算、画布大小、Qwen／evaluator 模型均不变。

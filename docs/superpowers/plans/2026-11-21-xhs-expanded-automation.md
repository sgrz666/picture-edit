# 小红书扩大规模与一次性自动训练

用户授权：等待 PID 1916769 结束后自动训练，扩大已有小红书对照，生成独立测试集图片作为评价。仍固定最新 Native V7 与已核验的 Bilibili S3 20,000 步迁移权重，不更换 Qwen、SDPose、evaluator 或原始数据。

- [x] 在 `scripts/waiting.py` 实现 `/proc` PID＋boot id＋start ticks 等待和 NVML 显存探测；测试活进程、退出、zombie、PID 复用、探针瞬时失败。等待不创建 CUDA context。
- [x] `prepare.py` 新增扩大模式：512 train／40 val／64 test，保持原 note／重复 SHA 隔离，按 note 轮流取样、保留资格与未选中记录。两臂 SFT 各 1,000 updates（200／150／400／250），后续继续 SFT／GRPO 各 50。
- [x] 在 V6.2／V7 SourceSpec 增加独立 `test` 角色；禁止测试源进入训练路由和 sampler／validation loss probe。检查 test 与 train／val 的 note、Source 和 Target RGB SHA 均互斥。
- [x] `geometry.py`／`assets.py` 已接入逐 SHA 复用与三类数据发布；747 个原图中 172 帧可复用、575 帧待 GPU 补算，图片只增不覆。几何和资产 GPU 执行仍待依赖释放。
- [x] `stage.py`／`run.py` 已接入 val 阶段图和五份实际权重的 64 对 test 图；强制全部优化完成后才评价 test，测试不参与奖励、阶段预算、checkpoint 选择或调参。
- [x] `summarize.py` 按 prepared 预算验证真实采样轨迹，分别发布 validation／test 指标和面板；测试 split／pair／实际权重 SHA 强检与两套合成产物流冒烟通过。
- [x] 新目录串行进程已启动并复核 PID 2226038，确实等待 PID 1916769；本地／服务器各 98 项针对性测试，938 个源码 SHA 与编译验收通过，包 SHA `1eace7a3e9d793230bcb51c282b220f7dee969b0ed0dd7a9d1d9df00078bea70`。旧 78／10 目录保留并写替代记录。
- [ ] 1916769 结束后自动检查资源，实际完成新几何、SAM／Native 资产、cache、data／grad gate、SFT、后续对照与测试集评价；当前尚未开始这些 GPU 阶段。

参考 checkpoint：源 SHA `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`；共同 V7 初始化 SHA `57022f5cdcae2931f30ec93090c6e53626089eb69f8893761d8e9c901448060e`。此前 1,054 个继承张量逐值一致的 CPU 验收保留，正式启动前再次核对实际字节。

复测重点：`tests/test_region_pilot_automation.py`、`tests/test_native_test_split.py`、`tests/test_xiaohongshu_region_prepare.py`，加现有区域目标／Native V7 集成回归。此处是服务器一次性实验编排，不创建定时提醒或周期任务。

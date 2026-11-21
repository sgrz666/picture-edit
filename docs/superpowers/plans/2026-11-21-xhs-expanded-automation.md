# 小红书扩大规模与一次性自动训练

用户授权：等待 PID 1916769 结束后自动训练，扩大已有小红书对照，生成独立测试集图片作为评价。仍固定最新 Native V7 与已核验的 Bilibili S3 20,000 步迁移权重，不更换 Qwen、SDPose、evaluator 或原始数据。

- [ ] 在 `scripts/waiting.py` 实现 `/proc` PID＋boot id＋start ticks 等待和 NVML 显存探测；测试活进程、退出、zombie、PID 复用、探针瞬时失败。等待不创建 CUDA context。
- [ ] `prepare.py` 新增扩大模式：512 train／40 val／64 test，保持原 note／重复 SHA 隔离，按 note 轮流取样、保留资格与未选中记录。两臂 SFT 各 1,000 updates（200／150／400／250），后续继续 SFT／GRPO 各 50。
- [ ] 在 V6.2／V7 SourceSpec 增加独立 `test` 角色；禁止测试源进入训练路由和 sampler／validation loss probe。检查 test 与 train／val 的 note、Source 和 Target RGB SHA 均互斥。
- [ ] `geometry.py` 只为新增物理帧估计，逐 SHA 复用上一轮合法 SMPLest-X 结果；`assets.py` 为 train／val／test 独立发布绑定、测试缓存与固定预算，保持所有图片只增不覆。
- [ ] `stage.py`／`run.py` 生成 val 阶段图，并为 initial、uniform 最终、regional 最终、继续 SFT、GRPO 五份实际权重生成完整 64 对 test 图。测试不参与奖励、阶段预算、checkpoint 选择或调参。
- [ ] `summarize.py` 改为按 prepared 预算验证实际采样轨迹，并单独发布 validation 和 test 指标／图像面板；不把 validation 图片叫 test。
- [ ] 使用新的 `one/xiaohongshu-region-sdpose-sft-rl-20261121-expanded-v1`，原 78／10 run 只读保留并记录被替代。CPU 测试、源码 SHA／编译验收后，启动一个带文件锁和 PID 凭据的串行进程。1916769 结束后，再检查实际资源，执行新增几何、SAM／Native 资产、cache、真实 data／grad gate、两臂 SFT、后续对照和评价。

参考 checkpoint：源 SHA `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`；共同 V7 初始化 SHA `57022f5cdcae2931f30ec93090c6e53626089eb69f8893761d8e9c901448060e`。此前 1,054 个继承张量逐值一致的 CPU 验收保留，正式启动前再次核对实际字节。

复测重点：`tests/test_region_pilot_automation.py`、`tests/test_native_test_split.py`、`tests/test_xiaohongshu_region_prepare.py`，加现有区域目标／Native V7 集成回归。此处是服务器一次性实验编排，不创建定时提醒或周期任务。

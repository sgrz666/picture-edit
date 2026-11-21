# 小红书空间加权实验切换计划

目标：按用户要求把当前空间目标／SDPose／GRPO 对照切到 `/data/shangguanrz/datasets/xiaohongshu(single)`，旧 Bilibili 运行只读保留。本人在 main 执行，不使用 subagent。

设计沿用已确认的 uniform 300 updates／regional 300 updates，再从同一 regional 父权重继续 SFT 10 updates／GRPO 10 updates。Qwen 和 evaluator 固定；SDPose 同时提供骨架输入和评价。新数据没有 Native 几何，必须显式补算 SMPLest-X 与可追溯的区域侧车，不能伪造三维输入。

- [x] `tests/test_native_v62_conditions.py`：复现合法单人 Depth 的 legacy canvas hash 与 canonical pose hash 不同导致拒绝；修正消费者逐一验证两种绑定，保留错画布拒绝。
- [x] `experiments/xiaohongshu-region-sdpose-sft-rl/scripts/prepare.py` 与相应测试：笔记级 seed42 划分，重复 RGB SHA 笔记合并为同一个 split；锁定 78 train／10 val pilot 与完整 split 清单；所选图片只读。
- [x] `scripts/geometry.py`：在现有 SMPLest-X 环境批量估计，输入复制保留原字节，导出逐帧参数、索引与 SHA；第二轮固定 SDPose 框后 176／176 帧成功，首轮漏检失败原样保留。
- [ ] `scripts/assets.py`：复用经过验证的 Native importer／raster／Source appearance，构建真实 3D、SDPose guide、逐手 ROI、独立条件和配置；Canvas 长边 512 保持比例。保留原脚本快照及哈希，Source 身份只从 Source 提取。
- [x] 复用既有区域实验的串行 runner，移除验证数据源硬编码，按实际 manifest 展示小红书名称与样本数；改用当前 SDPose 环境路径，原图、候选图、量化与面板保存逻辑已接入，GPU 实测仍待队列释放。
- [x] 新 run 根：`/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-v1`。用户最新指示覆盖早先 SourceBG192 选择，固定 Bilibili S3 20,000 步（SHA `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`）作为共同起点，显式迁移到最新 Native V7；保留所有已训练参数、新参数按 seed42 初始化，逐版本严格读回，不继承旧优化器。不将旧架构的 20,000 步声称为新增 V7 模块已训练 20,000 步。迁移后 SHA `57022f5cdcae2931f30ec93090c6e53626089eb69f8893761d8e9c901448060e`，1,054 个继承张量逐值一致。
- [ ] 运行 CPU／真实 data／真实 grad 门禁，再启动实际训练。逐步检查子进程与新产物；任何失败保留证据，修复后重新固定源码与 gate。只将实际通过的阶段写作完成。

复测：`python -m pytest tests/test_native_v62_conditions.py tests/test_xiaohongshu_region_prepare.py tests/test_region_objective.py tests/test_region_supervision.py tests/test_native_region_backend.py -q`。原始数据不写入，图片永不删移，旧 run 不覆盖。

已投递完整串行任务 PID 1967731，冻结源码 `runtime-v2`（包 SHA `1fbbaf39346954747524b850d8253ea7fb78aa401b96308103e9af895e93ff15`），本地／服务器各 87 项针对性回归通过。当前因为其他 GPU 作业占用而等待，SAM／Native 侧车、真实 data／grad 门禁与 SFT／GRPO 尚未执行；不把任务投递视为训练完成。

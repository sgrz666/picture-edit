# Bilibili 区域 FM、SDPose 与 GRPO 接入计划

用户授权接入新空间目标，使用 `/data/shangguanrz/datasets/bilibili` 训练；骨架条件和生成评价均使用 SDPose，保留生成图，并增加此前 Pose/ID GRPO 思路的对照。全部工作在 main，本人执行，不使用 subagent。

**目标**：完成可重复的 uniform/RN SFT pilot，以及同一个 RN SFT 父检查点上的继续 SFT/GRPO pilot；保留训练、采样、评价和失败证据。Pilot 不替代全量多 seed 收益实验。

**实现**：共享 stdlib 区域配置、纯 Torch 目标、SHA/Canvas/人物绑定的区域 sidecar 接入已有 Trainer；Bilibili 使用已有 Native reader 和两图条件，Qwen 权重、旧样本及缓存不覆盖。新 SDPose 图片进入独立条件版本，并建立新冻结缓存。RL 采用可重放的单步 Flow-SDE/GRPO，奖励由真实生成图的 SDPose 和已校准身份评价给出，不用监督 FM 伪装成 RL。

- [x] 添加有意义的失败回归：uniform 回退、分区分母、置信度、空手/部分 padding、来源错绑定、旧配置签名、新旧目标互斥。
- [x] 实现区域目标、sidecar、配置与窗口计数，验证旧训练默认行为及 V7 重复区域项拒绝。
- [x] 准备 Bilibili pilot 的按视频分离样本，覆盖身体/手部；新 SDPose 条件、独立逐手区域、固定验证清单，记录准确样本数和资格。
- [x] 固定 source snapshot，CPU 回归和真实数据审计；原 qwen21/SDPose/RL 环境均不安装包或改版本。
- [ ] 在 GPU 可用且不干扰其他作业时编码新条件缓存、生成未训练图、执行真实 grad preflight 与有界 SFT；两个模式同初始化和随机序列。
- [ ] 复用原 Pose/ID GRPO 的 replay、组内优势、PPO clip 与固定 SFT reference 约束；完成 SDPose reward 对照审计和无更新 replay gate，再运行 bounded RL。
- [ ] 从同一 RN SFT checkpoint 运行等 optimizer-update 的继续 SFT；另外报告 RL 的额外采样/墙钟成本，不声称计算量相等。
- [ ] 保存训练前、阶段/周期、最终、RL 候选图，固定验证 seed；SDPose body/arm/wrist/hand 指标、检测失败、身份与背景代价并列报告。
- [ ] 验证 SHA、生成清单、训练计数和原产物不可变；更新 README 和实际结果，提交本次文件。

GPU 上已检测到另一项 Hi4D Native V7 作业。先开展独立 CPU 工作；GPU 投递以实际进程/显存状态为准，不终止他人作业。数据范围、更新预算与资源选择在实际 pilot manifest/config 中写明，不把子集试跑宣称为完整 9520 对训练。

## 已核对的执行状态

- 运行根：`/data/shangguanrz/pic-edit-experiments/bilibili-region-sdpose-sft-rl-20261120-v1`。
- 78 train pairs / 39 train videos；10 val pairs / 5 val videos；88 个独立 GT/Canvas 区域侧车。
- 真实 S1 数据预检通过：352 条样本×路由，344 条具备手部 ROI。失败的首轮报告保留；JSON/JSONL prompt bank 兼容缺陷经失败回归修复，53 项相关测试通过。
- 本轮相关 CPU/Native V7 集成回归 94 项通过；后续补充的 ROI 有效系数日志和 RL 参考状态异常恢复也已单独回归。
- 冻结源码 `source/runtime-v3`，代码包 SHA `285bf72c5491e986eac5c8f767f66443a000d88e27d010019706f49480ab9e47`。
- 用户明确选择“保持排队，Hi4D 完成后跑 Bilibili”。本实验启动器 PID **2526964** 已验证存活，等待已确认的 Hi4D PID **2370979** 退出，然后串行执行缓存、生成、两臂 SFT、继续 SFT、GRPO 与 SDPose 评价。
- 观察入口：`logs/pilot-all-v1.log`、`logs/events.jsonl`；终态为 `complete.json`。必须查看进程和产物，不因启动器已经投递就宣称训练完成。

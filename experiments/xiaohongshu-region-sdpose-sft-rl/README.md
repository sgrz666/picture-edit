# 小红书：空间区域目标、SDPose 与 GRPO

## 实验目的

将已确认的无加权／空间加权及继续 SFT／GRPO 对照切到 `/data/shangguanrz/datasets/xiaohongshu(single)`。SDPose 同时用于训练骨架和生成评价；保留所有生成图。第一轮选择 78 train／10 val 配对作集成 pilot，不称为全量收益实验。

## 内容与使用

`scripts/prepare.py` 锁定笔记级划分、重复 RGB SHA 分组和 pilot；`scripts/geometry.py` 补算真实 Native 几何；`scripts/assets.py` 构建 Native 训练数据、新 SDPose 条件与区域侧车。训练复用现有区域实验的共享 Trainer 与串行编排，两个 SFT 臂各 300 updates，后续两个臂各 10 updates；最终保存原图、逐图评价与对比面板。

运行根为 `/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-v1`。后续实际进度、样本数量、门禁与运行命令以本目录更新和服务器凭据为准。

## 当前状态

磁盘已核对：原始数据 1,307 pairs／198 notes／942 frames，SDPose 942 帧已完成。seed42 的完整划分为 train 1,091 对／158 notes、val 88 对／20 notes、test 128 对／20 notes；同一 note 和重复 RGB SHA 不跨 split。Pilot 锁定 78 train／10 val、176 张不同原图，已完整导出真实 SMPLest-X 参数。

首轮旧 YOLO 检测器漏帧被完整性门禁拦下，失败日志与结果保留。第二轮只在 run-local estimator 副本中，把 crop 框输入替换为已固定的 SDPose／YOLO11x 人体框；不改 SMPLest-X 模型或裁剪／投影算法，不改原图字节。176／176 帧尺寸与参数顺序均通过核对。

按用户最新指示，共同初始化改为 Bilibili 2 万步：`/data/shangguanrz/pic-edit-experiments/one/bilibili-single-spatial-train-20261108-ungated-v1/stages/S3/checkpoints/update_00020000_f51df6cd2ace4a97b06ab037bd03b939.pt`，SHA256 `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`。实际 S3 已完成 4,000／4,000，累计 20,000，commit／父链／journal 已核验；不再使用 SourceBG192。通过 V5→V6→V6.1→V6.2→V7 显式迁移保留已训练参数，新增架构参数按 seed42 初始化，逐次序列化严格读回；不继承旧优化器。目录迁移前的旧路径只作来源记录，原 checkpoint／commit 不改写。

这轮测的是从共同 2 万步起点出发的小红书增量微调收益；旧起点本身已有历史训练与旧空间策略，不解释成从零训练的因果对照。两个新臂只有区域目标不同，必须用各臂新产物评价。

迁移后共同初始化为 `<run>/initialization/v7.pt`，SHA256 `57022f5cdcae2931f30ec93090c6e53626089eb69f8893761d8e9c901448060e`。CPU 实权重验收确认 1,054 个继承张量／225,955,980 个元素与源权重逐值相等；176 个 mesh 均为正相机深度，真实一帧 importer／完整 mesh FK 检查通过，表达基绑定一致。小红书新架构训练更新数当前为 0。

真实 SDPose 原始 heatmap score 最大 1.14435，ROI 的 q 显式裁到 `[0,1]`；不改原 SDPose JSON。本地与服务器最新针对性回归各 87 项通过，包含此真实分数范围的失败回归。GPU 当前被其他作业使用，本实验按此前队列选择等待足够空闲，随后自动补齐 SAM／Native 侧车、运行 data／grad 门禁和全部训练对照。尚无小红书训练收益结果。

当前冻结源码 `source/runtime-v2`，包 SHA256 `1fbbaf39346954747524b850d8253ea7fb78aa401b96308103e9af895e93ff15`。等待进程 PID **1967731**，凭据在 `logs/pilot-all-v2.pid.json`，日志 `logs/pilot-all-v2.log`；每 30 秒探测空闲显存，70 GiB 以上才启动 GPU 阶段，不中断其他作业。初版等待器已在 SAM／训练开始前替换，旧源码、迁移权重和日志均保留。代码和权重实际状态应以进程与 SHA 再核对。

查看进度：

```powershell
ssh sg "tail -n 12 /data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-v1/logs/events.jsonl"
```

本地复测数据隔离、置信范围与单人 Canvas 门禁：

```powershell
python -m pytest tests/test_xiaohongshu_region_prepare.py tests/test_native_v62_conditions.py tests/test_region_objective.py tests/test_region_supervision.py tests/test_native_region_backend.py -q
```

`scripts/audit_initialization.py --root <run> --repo <冻结源码目录> --output <新的报告路径>` 可在 Qwen21 环境用 CPU 复核继承张量、176 帧原图／几何和真实 importer／FK；第一次审计结果在 `report/initialization-acceptance-v1.json`，本地镜像在 `report/live-run/`。不把该 CPU 验收称为真实 GPU 梯度验收。

最终原图在 `<run>/output/generation/`，RL 候选图在 `output/grpo/rollouts/`，逐图指标在 `report/evaluation/`；`report/comparison-ready.json` 指向指标表和 7 列对比面板。重新出图和复测报告必须使用新名字，旧产物保留。

## 与已有实验的关系

沿用 `bilibili-single-spatial-train-20261107/region-sdpose-sft-rl` 的目标设计与评估，旧 Bilibili run 已在初始生成的单人 Depth Canvas 门禁报错退出，没有新增 SFT 权重；其缓存、源码和日志保留。小红书是新 run，不复用旧样本或旧条件缓存。Native 几何由 SMPLest-X 另行估计，SDPose 只承担 2D 骨架／区域／评价职责。

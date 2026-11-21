# 小红书：空间区域目标、SDPose 与 GRPO

## 实验目的

将已确认的无加权／空间加权及继续 SFT／GRPO 对照切到 `/data/shangguanrz/datasets/xiaohongshu(single)`。SDPose 同时用于训练骨架和生成评价；保留所有生成图。第一轮选择 78 train／10 val 配对作集成 pilot，不称为全量收益实验。

## 内容与使用

`scripts/prepare.py` 锁定笔记级划分、重复 RGB SHA 分组和 pilot；`scripts/geometry.py` 补算真实 Native 几何；`scripts/assets.py` 构建 Native 训练数据、新 SDPose 条件与区域侧车。训练复用现有区域实验的共享 Trainer 与串行编排，两个 SFT 臂各 300 updates，后续两个臂各 10 updates；最终保存原图、逐图评价与对比面板。

运行根为 `/data/shangguanrz/pic-edit-experiments/one/xiaohongshu-region-sdpose-sft-rl-20261121-v1`。后续实际进度、样本数量、门禁与运行命令以本目录更新和服务器凭据为准。

## 当前状态

磁盘已核对：原始数据 1,307 pairs／198 notes／942 frames，SDPose 942 帧已完成。数据未带 Native 3D 资产，不能只替换 Bilibili 路径后直接训练。当前正在补齐数据接口，尚无小红书训练收益结果。

按用户最新指示，共同初始化改为 Bilibili 2 万步：`/data/shangguanrz/pic-edit-experiments/one/bilibili-single-spatial-train-20261108-ungated-v1/stages/S3/checkpoints/update_00020000_f51df6cd2ace4a97b06ab037bd03b939.pt`，SHA256 `d1a655a86079194def922b13fdd39888f5c7f6db4ea6734e9c617d364f9db7f3`。实际 S3 已完成 4,000／4,000，累计 20,000，commit／父链／journal 已核验；不再使用 SourceBG192。通过 V5→V6→V6.1→V6.2→V7 显式迁移保留已训练参数，新增架构参数按 seed42 初始化，逐次序列化严格读回；不继承旧优化器。目录迁移前的旧路径只作来源记录，原 checkpoint／commit 不改写。

这轮测的是从共同 2 万步起点出发的小红书增量微调收益；旧起点本身已有历史训练与旧空间策略，不解释成从零训练的因果对照。两个新臂只有区域目标不同，必须用各臂新产物评价。

## 与已有实验的关系

沿用 `bilibili-single-spatial-train-20261107/region-sdpose-sft-rl` 的目标设计与评估，旧 Bilibili run 已在初始生成的单人 Depth Canvas 门禁报错退出，没有新增 SFT 权重；其缓存、源码和日志保留。小红书是新 run，不复用旧样本或旧条件缓存。Native 几何由 SMPLest-X 另行估计，SDPose 只承担 2D 骨架／区域／评价职责。

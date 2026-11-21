# Native 离线构建器原样快照

`hia_training.py` 原样来自服务器 `experiments/6.qwen-controlnet union/hia_geometry_training_20261029/scripts/hia_training.py`；本实验只复用其中 SMPL-X importer、固定 Canvas、局部可见性与 Source ROI 构建函数，不执行它的 HIA 训练入口。

`appearance.py` 原样来自 `/data/shangguanrz/pic-edit-experiments/one/bilibili-native-v5-training-20261106-v1/scripts/appearance.py`，复用 Source-only ArcFace／DINO 特征提取。旧标签名 hand_a/right、hand_b/left 只作接口映射，小红书输入明确来自 SDPose 的 geometry 顺序。

快照未改写。运行时 `asset-producer.json` 固定两份文件 SHA、模型 SHA、SMPL-X 和 SAM2 checkpoint；数据由本实验的 `scripts/assets.py` 显式绑定，小红书原图与原 SDPose 不改动。

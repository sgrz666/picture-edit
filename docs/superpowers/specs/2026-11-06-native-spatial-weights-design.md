# Native V5 离线空间监督设计

2026-11-06 用户批准实施：先用现有模型生成 20 张 GT 的 Mask／权重预览，通过后批量缓存，不直接启动 Adapter 训练。

GT RGB 的可见穿衣人体由已有 SAM2.1 提取；关节来自已有有效 DWPose 原图像素坐标；手部使用掌部与手指软支持区域，并分别服从原 loss_mask 与 hand_raw_pose_usable。不上可微姿态估计器，不重新拟合 SMPL-X，不安装模型包。几何方法保留 sapiens_v1 标签，GT 分割和 2D 监督不冒充 Mesh 渲染或精确手部分割。

缓存三个独立 Canvas 分量与有效内容区、GT SHA256、Canvas 指纹及生成配置；索引按 GT SHA＋Canvas 指纹去重。权重在像素层以 1＋.5Body＋.5Joint＋1Hand 组合并截断到 2.5，乘一次有效内容区，再 area pooling 到实际 [B,N,64] 网格。每图按 C×sum(weight) 归一化后求平均。背景基础权重 1，Padding 0，系数为零保持旧路径。原 face/hand 区域 velocity loss 与新加权互斥。

所有产物独立落 /data/shangguanrz/pic-edit-experiments/native-spatial-supervision-20261106-v1。中断尝试和截断 JSON 保留原样，恢复只发布新文件，不删除或移动图像、视频与旧产物。训练接口显式绑定空间索引，SHA／Canvas／实际 coverage 不一致即失败，不过滤样本。

验收包括：20 张 GT/区域/权重/latent 对照图人工查看；全量来源、数组与身份审计；Qwen64 网格的零系数数值基线；既有训练回归与新增缓存测试。它们不等同于真实模型前反向、训练或 PCK/ID/背景改善，后者须另做受控消融。

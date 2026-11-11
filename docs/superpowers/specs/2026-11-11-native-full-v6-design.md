# Native Full V6：Geometry-Guided PoseLink

2026-11-11，用户已确认：完整实现 P0–P3，共用单／双人参数，保留 Qwen 原生 Attention；本轮只改代码与 CPU 验证，不训练、不生成、不迁移真实权重。

## 范围与基线

在 `src/pose_control/qwen21_full/native_v6/` 建立独立版本，复用 V5 的 Body、Interaction、Face、Hand、Detail 和 Reasoner。V5 源码、入口、checkpoint 和正在运行的 `/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1/source/runtime_repo_v7` 均不修改。

本地 HEAD 为 d4265fb，服务器主仓 HEAD 为 bbc729d。2026-11-11 只读核对：两侧 V5 顶层 Python 文件逐字规范化换行后相同；服务器 training 有未提交的空间加权 FM 改动。V6 的训练能力必须保留这些空间权重语义，不覆盖现有代码。

## 数据流

`img_mask` 展开每个四-token图像槽，用实际 Qwen `build_token_metadata` 得到 image IDs。结合显式 `reference_roles` 选出 Pose，支持 2Ref／4Ref／5Ref，不用连续前缀猜位置。Source 和文本不直接作为 PoseLink K/V。

V6 前端复用同一次 Reasoner，保留尚未加入 Source Identity 的几何／Depth、逐人物空间 Tokens、可见 owner、独立投影支持、canvas coverage 和接触部位支持；Body 仍使用原 V5 身份融合路径。Pose Hidden 在每个指定 block 动态读取，不缓存跨 sigma 的 K/V。Qwen 的 Pose Hidden 已经过原生 Attention，可能间接包含先前 Source 的上下文；本设计只保证 K/V 选取的 token 位置不包含 Source，不能宣称语义完全去外观。

PoseLink 内部使用 512 维、8 heads。Pose Reference Hidden 先投影，再按实际 Canvas 归一化坐标对齐几何与逐人物 Tokens。空间距离 Bias 与可见 owner 的软 Role Bias 共用一个 Attention，接触区域放宽跨人物惩罚，未知接触保持未知。只对 padding／无效 key 做硬屏蔽，避免把 A/B 完全隔离。

默认注入 0-based 5、10、15、21、26、31，允许独立选择层。Body／Interaction／局部分支与 PoseLink 都读同一份未控制 block 输出，FP32 累加后只写目标 Tokens。Qwen 原生 Q/K/V、RoPE、causal mask 和参考 Tokens 不改；新增 Qwen token 数为 0。输出 Head 零初始化，非重入 gradient checkpointing 显式绑定不可变 forward 上下文。

## 版本、训练和验证

V6 有独立 architecture/schema、严格 checkpoint 重建和显式 V5→V6 初始化；不继承 optimizer／RNG 为续训状态。提供 PoseLink Head、PoseLink Full、Joint 参数策略，Qwen 永久冻结，默认 velocity Loss，支持已有离线空间 FM。推理与训练显式透传 reference roles／网格信息；缺失或矛盾时拒绝执行。

本轮 CPU 合成验证覆盖：交错文本、相邻参考图、2／4／5Ref、矩形网格与 padding；Source 不入选；关闭／零输出与 V5 一致；reference prefix 不变；全无效无 NaN；零 Head 首梯度与非零 Head 完整梯度；跨 forward checkpoint 重算；严格回读与初始化边界。真实 7B、真实数据、显存与生成效果留待另行实验，不把单元测试当效果证据。

## 交付

新增 V6 核心模块、训练／推理／迁移入口、CPU 回归与中文 README。在服务器主仓仅新增 V6 对应文件；投递前记录时间戳与旧文件摘要，使用 scp，投递后 py_compile 和 SHA 校验。所有新运行产物使用独立 `/data/shangguanrz/` 路径。

# Native Full V7：复用现有模块的双人架构改造

## 目标与边界

用户已选择 4Ref（同一双人 Source、Combined Pose、Person 0 Pose、Person 1 Pose），并授权将前一轮论文讨论落地为架构修改。以 `00fb054d` 的 V6.2 为基础，新增 `native_v7`，保留旧入口与旧 checkpoint 的精确语义。2Ref、5Ref 继续走现有条件、Source Bank 与 Canvas 校验。所有新模型参数、配置、训练状态必须可保存、严格重建、明确初始化；不宣称未经真实训练验证的图像质量改善。本任务不启动大规模训练。

不拷贝 V6.2 的整套运行时和 Trainer。不新增人脸检测器、PoseNet、VAE、SMPL 编码器或 Qwen 原生 Tokens。复用现有逐人物采样、GT 五点人脸对齐、冻结 ArcFace/DINO 参考、区域 owner、Stage 参数目录、optimizer 超集与恢复机制。旧代码只增加默认保持旧行为的扩展点。

## 论文到实现的对应

|来源|采用的机制|本仓落点与区别|
|---|---|---|
|[DanceTogether §3.3](https://arxiv.org/html/2505.18078v1)|逐人物 Pose/Mask 特征融合、区域重建加权|沿用 V6.2 逐人物 Key 采样，以可见 owner、独立投影支持、部位和有效性产生轻量可学习 Pose/Mask 融合；不复制其 SVD UNet/PoseNet|
|[AnyPhoto §3.3–3.4](https://arxiv.org/html/2603.14770v1)|身份自适应调制、减少身份分支干扰|使用现有逐人物 Source 身份 Tokens 调制本人物局部控制器的归一化 Query；不重新提取身份，不修改 Qwen 原生 Attention|
|[PeopleComposer §3.2](https://arxiv.org/html/2605.23178v1)|跨模态空间对应|仅在 PoseLink 侧 Attention 使用目标 Canvas XY RoPE；保留硬人物路由，不挪用 Qwen Frame 轴，也不声称等同论文的文本/图像/姿态联合角色编码|
|[WithEveryone §3.3](https://arxiv.org/html/2608.20336v3)|已知位置上的逐人物身份监督|继续复用已有 GT 五点对齐与低噪声 one-step 解码，不在训练中匹配噪声生成脸；检查监督资格，避免配置打开但零有效样本|
|[WithAnyone §5.1–5.2](https://arxiv.org/html/2510.14975v1)|身份对比监督、同身份不同帧配对|增加固定人物对应的 ID 对比项，正样本为已配对 GT/可用 Source，其他人物为负样本；不以背景/姿态相近作为身份相同依据，不重写现有 paired reader|

以上为思想改编，论文原始模型和权重没有被复制。数学和接口与当前 Qwen 任务适配；文档不得称作论文完整复现。

## 架构

1. `PoseLinkAttentionV7` 复用 V6.2 参数投影、逐人物采样、query 分块、梯度 checkpoint、coverage 和 edit Gate。新配置独立控制 `mask_pose_fusion`、`spatial_rope`、`geometry_contact`、`identity_modulation`。全部关闭时，迁移后的 V7 与同配置 V6.2 数值一致。
2. Pose/Mask 融合在已采样的 512 维 Key 上使用对应人物的 owner/projected/part 特征，不能改变固定 Key 预算或将背景/无效人物变为合法 Key。新融合输出采用零初始化残差，已有权重完整继承。
3. 同人物 Pose 路径不因 contact 而读取另一人物的 Qwen Pose Hidden。独立 Contact Bridge 的 K/V 仅来自身份融合前的几何、人物和部位 Tokens。仅已知正接触、未被 forbidden 覆盖、双方有效、对应支持存在时开放跨人物几何。未知归属不自动扩大接触权限。无合法 Key 时包括输出 bias 在内必须严格为零。
4. XY RoPE 在侧 Attention 每个 Head 内分配 X/Y 偶数维，使用采样保留的全 Canvas 坐标；不使用 ROI 内局部坐标。计算保持稳定，支持矩形网格、padding、单人/双人和不等 Q/K 长度。
5. 身份调制复用 `PreparedLocalV5` 已绑定到人物的 identity Tokens。在本人物 Face/外观控制器中做零初始化 scale/shift；无身份 Token 或开关关闭时精确回退；不将 Face ID 扩张到背景修复区。局部 owner/ROI 和旧控制器本身继续复用。
6. 保留 Target-only FP32 残差合成和冻结 Qwen/VAE/文本编码器。新模块进入真实参数目录、Head Warmup、Full、Joint 阶段，不能作为从不优化的孤立模块交付。

## 训练与数据

共享 Trainer 增加明确的 runtime/backend/objective/config 扩展点，V6.2 默认行为保持不变。V7 新训练 schema 为 `native_v7_training_v1`，模型 header 为 `native_full_v7`。不以跨版本 resume 代替权重初始化；保存/恢复继续绑定代码、数据、缓存、配置、参数名和 optimizer/RNG。

训练目标复用现有 FM 和 GT 对齐 ID，增加可单独关闭的逐人物区域 FM、已知接触区域 FM 与身份对比项。新增加权支持按实际 Canvas/coverage 归一化一次，不能重复乘 padding 或用遮挡投影监督可见纹理。ID 对比遵守现有 `auxiliary_valid`、低噪声和更新间隔，必须保留生成图→VAE→预测的梯度，参考特征冻结。缺乏合法双人负样本时不伪造负样本。

数据预检报告人脸资格、真实双人/接触支持与 repair 覆盖；明确请求 ID/对比监督但整个源/路由没有有效样本时拒绝训练。推理不读取 `target_rgb` 或 training supervision。`hybrid_hi4d` 的 4Ref 继续沿兼容 reader；不伪造 SMPL-X 参数或将 SMPL24 当作 SMPL-X 手部标签。

## 验收

- V7 组件：开关回退、初始化、矩形 XY 对齐、同人物污染/梯度隔离、正/未知/禁止接触、无 Key 输出零、身份调制逐人隔离及梯度。
- 集成：真实随机小型 Qwen Blocks 的 2/4/5Ref、Target-only、前缀保持、梯度 checkpoint 前反向、参数分组及新模块优化、严格权重读回和旧版本拒绝。
- 训练：配置 JSON 往返、只准备不创建模型、Loss 有效资格、区域归一化、对比损失的数值/梯度、版本化 snapshot/resume 与阶段门禁。
- 回归：原 V6/V6.1/V6.2 套件；不因新增扩展点改变旧输出或原状态结构。
- 服务器：带时间戳源码快照、`scp` 投递、本地/服务器文件 SHA256 一致、编译与 CPU 回归、当前两条 Hi4D 4Ref 的真实 reader/条件资格审计。保留所有原产物、其他人的未提交修改和环境依赖。
- README 提供论文映射、迁移、prepare、preflight、训练/阶段切换、推理与消融命令，明确真实 7B/图像效果的验证界限。

# Full、Pro、Union：如何在几何控姿时保住人物身份

研究日期：2026-10-20。本文中的 Pro 是本仓 `qwen21_pro`，不是 Qwen 官方的 Pro API；Union 特指 Alibaba PAI / VideoX-Fun 的 **Qwen-Image-2.1-Fun-Controlnet-Union**，不是其他 Qwen 版本的同名模型。

## 1. 选型结论

**如果以“姿态与 ID 同时做好”为第一目标，我建议把现有 Pro/Union 接入作为下一轮实验平台，优先改身份路径、残差作用范围和训练监督；不建议继续原样扩训 Full，也不建议直接认定现有 Pro 已优于 Full。**

你的“保留我们的人物任务能力、借鉴 Union 对 Qwen DiT 的适配”方向有根据，但需要拆成两件事：

- **值得保留**：我们的源体型重定向、SMPL(-X) 几何、人物归属、Face/Hand ROI、按人物读取源外观的能力；Union 对 Qwen 原生 token 排布、位置编码、时间调制和控制层的适配。
- **需要改变**：让几何和身份共同竞争同一个残差输出、仅凭 flow MSE 判断身份保持、把冻结基座或保持参考 token 等同于保持生成脸。这些问题不会因为把旧核心换成原生 Qwen block 就自动消失。

当前三个选项的定位是：**Full 是已有任务能力与对照基线；Union 是结构控制先验；Pro 是更合适的改进起点，仍是未证明效果的候选。** 参数量属于成本，不作为否决完整 Union 的首要理由。若后续公平实验中原生轻量 Adapter 的姿态–ID 组合更好，也应选它，而不是为了“官方结构”保留 Union。

现有数据不能给三者排一个可信的最终效果榜：它们没有在同一数据划分、输入、采样设置、训练预算下完成统一图像评测。

## 2. Qwen 2.1 的结构到底要求什么

本次核对了公开的 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1)、[官方代码入口](https://github.com/QwenLM/Qwen-Image-2.1)、仓内固定版本源码及当前部署代码。以下底层细节以 [固定版本原生 Transformer](source/official_base_transformer.py) 为准。

| 项目 | 代码事实 | 对任务的含义 |
|---|---|---|
| 视觉生成骨干 | 32 个 single-stream DiT block，hidden 4096，32 heads × 128，SwiGLU 中间维 12288 | 不能沿用 SD3 双流 block 的语义假设，仅做维度投影不代表对齐了表征 |
| VAE | RGBA 输入，64 通道，空间下采样 16，DiT patch size 1 | 512 图像对应 32×32 latent token；小脸只有很少空间位置。多层控制不能凭空增加局部可寻址分辨率 |
| 图文序列 | Qwen3-VL 条件中的一个 image slot 展开为 4 个 VAE token；目标图像附在末端 | 必须依据实际 mask/img_shapes 找目标段，参考图可以不同尺寸，不能假定固定 offset |
| attention | `(q_idx >= kv_idx) or same_image_block`；同一图像内部双向 | 目标能读取源参考；源参考不反向读取目标噪声。双人目标内部仍能交互，不能为了隔离 ID 把接触区域全切断 |
| 时间调制 | 共享 modulation 投影给各层 scale/gate；目标用当前 t，文本/参考用 t=0；gate 经 tanh | 身份条件应作为稳定源信息；与噪声相关的任务增量应只作用目标，避免破坏 prefix 的条件语义 |
| 位置编码 | 三轴 RoPE，axes=(16,56,56)，每个图像有自己的空间网格 | 几何、ROI 与 reference bank 的坐标必须显式对齐，不能把任意重采样的图当成已对应 |
| 缓存 | 原生 prefix 在上述条件下可缓存；当前 Union 控制路径禁用 base KV cache | 保留 block-causal 规则与确认缓存有效性是两件事，不能因“是原生 block”就打开缓存 |

原生 block 可概括为：

```text
u = h + tanh(g_attn(t, token_type)) · Attention(LN(h) · (1+s_attn), RoPE, causal_mask)
h_next = u + tanh(g_mlp(t, token_type)) · SwiGLU(LN(u) · (1+s_mlp))
```

这解释了一个关键区别：**冻结权重只是阻止训练改权重；残差仍然会改变激活，继而改变后续 attention 和输出。** 同样，源参考 token 保持不变，也不保证目标 token 会继续采用源身份。

原生图像编辑同时利用参考图的语义上下文与 VAE 空间信息。我们的目标应是让几何分支补足“怎么动”，而不是把原生参考图能力一并替换掉。

## 3. 三种实现的真实差别

### Full：Qwen 外围接入正确，但内部仍是 DeepGen/SD3 控制核心

[qwen_adapter.py](../../src/pose_control/qwen21_full/qwen_adapter.py) 与 [control_core.py](../../src/pose_control/qwen21_full/control_core.py) 表明：

1. 内部保留1536维的V6控制结构；Qwen的4096维文本经外围text_bridge（默认4096→4096）与旧context_embedder进入旧核心，输出残差再由1536映射到4096。Qwen 64通道latent先经1×1卷积与pixel shuffle变成旧核心使用的16通道布局。pixel shuffle本身可逆，不能把问题说成“这一步必然丢信息”；问题是Qwen与DeepGen latent的语义没有因此自动对齐。
2. 控制核心由 `SD3ControlNetModel(num_layers=1)` 取得一个共享 joint block，几何/交互等分支递归运行 6 stage，并有各自的 stage adapter/zero head。不是独立复制 32 层 Qwen。
3. 每个去噪 timestep 先计算 6 份残差，再按 `6*layer//32` 重复注入 32 个 Qwen block，组内覆盖次数为 6、5、5、6、5、5。组内残差不重新读取该层当前的 Qwen hidden。
4. 主干注入只改目标段，原生 prefix 保留。这是优点。控制核心自己的位置、时间和双流语义仍来自旧架构；主干的原生机制没有被替换，但控制网络并不原生。
5. Full 已有独立 Face、Hand、appearance、contact 模块和某些可微身份辅助 loss，不能误说它“完全没有身份监督”。实际监督覆盖及有效增量要看训练日志。

**判断**：它有可复用的人物任务知识，但组内重复注入、旧 latent 语义和旧双流核心让“强控姿但身份变化”的诊断更复杂。增加 Full 训练步数不是优先级最高的动作。保留其前端与对照价值，不把旧控制核心整套搬进新结构。

### Union：原生结构控制适配更完整，但默认不是人物身份编辑器

[官方模型卡](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) 发布的是通用多条件控制与 inpainting 分支。固定实现见 [Union Transformer](source/official_union_transformer.py) 和 [官方 control pipeline](source/official_control_pipeline.py)。

- 16 个独立控制 block，继承 Qwen 原生 block；在 base 层 0,2,…,30 注入。沿用原生 RoPE、shared modulation、block-causal segments 和 key-valid 规则。
- 控制输入 129 通道是 `control latent 64 + keep mask 1 + masked-image latent 64`。它不是可任意填入 normal/owner/FaceID 的 129 维特征槽；官方 mask 图白色表示重画，送入网络前转换为 keep 语义。
- control chain 从初始 joint hidden 与控制 embedding 出发，逐层形成自己的 `c_k`，输出 `after_proj(c_k)`。它不是每层都回读当前 base hidden。
- base 在对应层把 hint **加到整个 joint sequence**。参考段控制输入补零不等于该段 hint 为零：控制块还读取 joint hidden，且线性层可有 bias。
- 官方 pipeline 把 `image` 当 inpaint 图，不走原生 reference-image 编辑路径。我们自建的 reference pipeline 恢复了该路径，并为参考 latent 补零控制输入；这已通过接入等价检查，但不是官方发布的人物身份保持训练证明。

**判断**：Union 的优势是原生接口与预训练结构先验，不是“保证不伤 ID”。不能把 reference 接入成功、zero-scale 等价与人物任务效果混为一谈。全 joint hint 对参考表征的影响值得消融，但还没有证据证明它是身份下降的唯一或主要原因。

### Pro：已有正确的人物绑定骨架，但当前身份路径仍依附于 Union

当前 [adapter.py](../../src/pose_control/qwen21_pro/adapter.py)、[integration.py](../../src/pose_control/qwen21_pro/integration.py)、[banks.py](../../src/pose_control/qwen21_pro/banks.py) 的实际逻辑是：

```text
u_k = 原生 Union control block(c_k)
query = projection(LN(u_k[target]))
geometry = typed normal/depth/mask/visibility/contact/part + ROI + contact relations
appearance = 目标 query 按人物读取 source bank
identity = FaceID projection，仅在该人物 face mask 内加入
delta = zero_head_k(fusion(query + geometry + appearance + timestep))
c_k' = u_k + target_person_support · delta
hint_k = 原 Union after_proj(c_k')
```

stage 0/4/8/12 是 **control stage 编号**，对应 base 层 0/8/16/24。完整 16 层 Union 都在。正式 pilot 的 source bank 是 VAE64＋DINO1536，source_dim=1600，不是配置类默认的 64。

值得保留的部分：每个人独立 K/V；显式 owner、visibility、null/invalid 处理；ID 只在脸部直接写入；typed contact 有人物/部位端点；零输出和完整恢复经过工程验证；生成条件不读目标 RGB。

尚未解决的部分：

1. **ID 与几何最后共用 fusion/输出头，再穿过冻结的 Union after_proj 与后续控制层。** 它增加了纠正 Union 的能力，但没有给身份一个独立、可控的输出通路。冻结 after_proj 不是严格容量上限；是否阻碍 ID 必须用同预算实验验证。
2. **任务 delta 的 target-only 保证不等于整个 Union hint 的 target-only。** `prefix_unchanged` 的工程结果应按实际比较对象理解，不能升级为“相对无控制原生模型，所有参考表征都不变”。
3. bank 隔离约束的是读哪个人的外观，不会自动实现同一人的脸、衣服、头发之间的部位对应；后续目标 self-attention 仍可传播变化。
4. [training.flow_loss](../../src/pose_control/qwen21_pro/training.py) 实际调用 [regional_flow_loss](../../src/pose_control/qwen21_pro/losses.py)。它优化 flow 误差，没有调用该文件里另一个可选 feature loss；本轮没有直接生成脸 ArcFace/身份 loss。模块存在与训练生效不能混用。

### 参数核算纠正

本次用当前源码在 CPU 的 `torch.device('meta')` 上构造模块，不加载大权重、不占 GPU，并从官方 safetensors header 重算 Union：

| 对象 | 精确计数 | 含义 |
|---|---:|---|
| Full，Face/Hand 都启用 | 373,099,568 | 去重后的模型参数，不等于某个训练阶段全部可训练 |
| Full buffer | 226,492,428 | 包括固定位置等 buffer，不应算成可训练参数 |
| Full state tensor 元素总数 | 599,591,996 | 历史约“600M”口径包含上述 buffer |
| Union 增量 | 3,775,479,808 | 180 tensor；BF16 权重约 7.032 GiB，不含 Qwen 基座/VLM/VAE |
| Pro pilot 额外任务参数 | 14,646,784 | source_dim1600，四个 task head；另保留完整 Union，pilot 没开 control LoRA |

Full 的历史联合训练实际开放 250,602,128 个参数，见 [旧 checkpoint 审计](../adapter_value_diagnosis_20261011/evidence/checkpoint_parameter_counts.json)。这些口径分开后，才能公平谈训练成本；较少可训练参数不等于较小推理网络。

## 4. 实验到底支持了什么

### 4.1 我们自己的控制也会损伤身份，不能把问题归给 Union 大

50 对 iPER 的 [Body20k 原始汇总](../union_adapter_architecture_20261017/evidence/body20k_50_summary.json)：

| 相同 Qwen 基座 | PCK@0.02，50 对 | 源脸 ArcFace，47 个有效样本 |
|---|---:|---:|
| 关闭控制 | 0.2031 | 0.8187 |
| 开启 Body20k | 0.9800 | 0.3510 |

这里是 **`qwen21/body_adapter.py` 轻量 Body，不是 Full**。它只在 6 个 base 层注入、读取当前 hidden、保留 prefix，仍发生明显 ID 下降。因此“只写 target”“减少控制层”“冻结基座”都不是充分条件。原生不动或少动姿态也会得高源脸相似度，不能拿 0.8187 作为所有目标姿态都必须达到的分数。

### 4.2 源脸信息的供给，比旧 Face 分支本身更有说服力

从远端重取了 [512 四臂原始 summary](evidence/face_four_arm_512.json)，严格共同脸集合 n=30：

| 同一轻量 Body/Face 实验 | 源脸 ArcFace | 同集合 PCK@0.02 |
|---|---:|---:|
| Body | 0.3953 | 0.9765 |
| Body＋A 脸裁图/角色提示 | 0.4719 | 0.9784 |
| Body＋Face 分支 | 0.4142 | 0.9745 |
| Body＋Face＋A 脸裁图 | 0.4802 | 0.9804 |

加入脸裁图/对应角色提示的组合增量约 **+0.0766**；已有裁图时，额外 Face 分支约 **+0.0083**。这不是证明 Face 分支永远没用，而是说明下一轮应先确保源脸可见、足够清晰且被模型读取，不能只堆网络。这里新增的是“裁图＋角色提示”组合，不能把数值全部归因于分辨率单一变量。

新取到的 [1024 val 原始结果](evidence/face_val_1024.json) 也应修正旧文档的简写：严格集合 n=21 中，Body / Body+Face / Body+Face+crop 的 ID 为 **0.6600 / 0.6661 / 0.7049**，对应 PCK 为 **0.9972 / 0.9972 / 0.9944**；全部可评分脸 n=42 的 ID 则为 0.5170 / 0.5200 / 0.5394。512 test 与1024 val 不是同集合，不能把二者均值相减当作分辨率因果收益，也不能宣称全量任务已达到 ID≥0.65。

### 4.3 Full 的现有 Face 训练增量有限

同一批 24 对的 [step0](../union_adapter_architecture_20261017/evidence/full_face_step0_summary.json) 与 [step3000](../union_adapter_architecture_20261017/evidence/full_face_step3000_summary.json)：可靠身份字段均 n=22，0.4328→0.4391；PCK 0.4434→0.4438。汇总顶层还写了 reliable_identity_n=21，和字段分母不一致，因此不把它做成正式显著性结论。

[联合 pilot](../../experiments/bfs_body_face_joint_pilot_20261008/report/summary.md) 在36个固定样本上，init→J 的 ID 0.4125→0.4369、PCK 0.6552→0.6513；继续到 Qwen final 后 ID 回落到0.3974、PCK 0.6254。256步 J 只实际使用20次有效辅助脸监督，Qwen分支26次。Full 有身份 loss 的代码，但有效覆盖稀疏；不能把“加了 identity loss”视为已解决。

### 4.4 Union 的失败有实物，但指标需要分开解释

最新 [48张矩阵报告](../../experiments/qwen21_union_migration_v1/report/METRICS.md) 已是48/48评估完成，旧10月17研究里的31/48状态已过期。6案例×2 seeds×4档，其中一档为 native、一档为 Union scale0。

hard_02 中：native→scale0.5→scale1 的严格躯干归一 PCK@0.2 为 **0.750→0.857→0.964**；外观 DINO 为 **0.916→0.908→0.585**。这是 DINO，**不是脸部 ArcFace**。单人和双人评分实现的有效关节分母也有差别，不能混合成一个 PCK 排名。

本次直接查看原图：scale1 有明显骨架线进入结果、脸部与体型变化；scale0.5 的源外观更接近，但还存在手臂/手部异常。对照 [10月19参考图消融](../../experiments/qwen21_union_migration_v1/output/probe_ref_ablation/summary.json)，只留源照片后，scale1 仍有明显骨架线，见 [现有探针图](evidence/single__hard_02__norefs_ctrl10__seed42.png)。所以“骨架也作为 reference”不是泄漏的充分解释；该探针只有一例一种 seed，且原提示词仍可能保留旧角色描述，不能定位唯一原因。

**骨架线能否诱导姿态检测器高分、是否把检测覆盖当成了真实人体成功，必须人工审查。** 当前图像已经足以说明这些高 PCK 输出并不满足任务；尚不能量化有多少分数来自画出的骨架。

该报告用“最佳 Union 档”且候选包含 scale0，因此“退化0例”受选择规则影响，不能解读为启用 Union 没有退化。双人低检出也须区分模型只画一人和检测器未分开遮挡人物。

### 4.5 Pro 的新证据：局部损失下降，不等于保住 ID

本次从原 run 只读重取 [step0](evidence/fixed_eval_000000.json)、[step200](evidence/fixed_eval_000200.json)、[gate](evidence/gate.json) 并逐条重算18个固定噪声位置。6对样本、8个人物、512画布，200次更新，完整Union冻结，无control LoRA。

| Pro 指标 | step0 | step200 | 相对下降 |
|---|---:|---:|---:|
| global flow MSE | 0.15983 | 0.15077 | 5.67% |
| face区域 flow MSE | 0.24681 | 0.16818 | 31.86% |
| contact区域 flow MSE | 0.04646 | 0.01655 | 64.37% |
| hand区域 flow MSE | 0.26094 | 0.18413 | 29.44% |
| body区域 flow MSE | 0.23406 | 0.19718 | 15.75% |
| global＋四区域总loss | 0.94810 | 0.71681 | 24.39% |

工程 gate 通过，global下降20%的旧效果门槛没过。旧 Union-LoRA B200下降17.79%，也没过；两者训练参数与目标不同，不能据此得出“Pro架构更差”的因果结论。

更直接的问题是 [样本记录](evidence/pro_results_20261018.json)：FaceID **仅2/8人物有效**，都来自TikTok；两组Hi4D的四个人全部无有效FaceID，iPER两人也无有效FaceID。不能用这轮训练证明双人ID保持能力。Hi4D normal/part在本轮legacy条件中无效，实际接触边也只集中在一例；不要把已实现的current几何重渲染说成这轮训练已经使用。

10月18报告没有完整40步成图质量结果。**10月20本次磁盘核对发现新的 `valpair_main200_20261020T001400` 正在运行**，已保存 [启动记录](evidence/pro_valpair_started_20261020.json)。00:33（北京时间）复核后，首个iPER样本的 [40步成图](evidence/pro_val_iper_coarse.png) 已落盘；[sampling记录](evidence/pro_val_iper_sampling.json) 确认512画布、seed43、CFG1、200步任务checkpoint、current/driver_exact几何。已查看 [源图](evidence/pro_val_iper_source.png) 和成图，但尚无该样本匹配的Union/step0对照与定量身份评分，双人结果也未完成，故不把单张出图当成Pro更优的证据。这里的验证几何是current，200步训练则是legacy，也需在比较中注明。本研究没有启动、终止或修改该进程。

### 4.6 两阶段精修值得保留，但“上 BFS 就好”没有依据

[BFS30例原始结果](../../experiments/bfs_compare_30_20261004/results.json) 的共同有效24例：Body的ID/PCK为0.3591/0.9899；无LoRA二次采样为0.5210/0.9498；BFS为0.4422/0.9629；alternative为0.4527/0.9678。说明身份可以改善，但姿态会回退，现有BFS也未赢过无LoRA精修。

因此保留“有目标几何约束的局部精修”，不把泛用换脸或全图二次生成当成答案。已有Pro的768 ROI链路仅通过2步工程探针、mask外像素保持检查，效果仍需完整采样验证。

## 5. 我认为更可能的冲突来源

按目前证据强弱排序：

1. **身份条件和监督不足——有直接证据。** Pro多数训练人物无FaceID，有效训练路径无直接ID loss；历史脸裁图收益明显大于旧Face分支边际增量。优先补足身份信息供给和生效路径。
2. **几何条件也可能携带身份形状——机制上合理，需逐样本确认。** target normal/depth/轮廓、密集脸关键点既表达动作也表达B的体型和脸型。保留A身份却强制拟合B形状，会形成矛盾。应尽可能用A体型/脸部身份形状承载B的关节旋转、头向、表情；SMPL-X粗形状不等于完整真实FaceID，不应过度依赖它。
3. **目标激活被强残差主导——多条控制线的现象一致，具体位置未定。** Full重复写入，Union全joint写入，Pro身份与几何共用输出，各有不同风险。但目前没有消融证明具体某个layer或某个after_proj就是根因。
4. **脸部空间与监督质量不足——结构与局部实验支持。** 16倍VAE压缩让小脸只占很少token。源脸高清裁图能改善条件读取，目标高分辨率/ROI能改善生成空间；二者不能互相替代。
5. **control图分布与参考输入混杂——已有泄漏反例。** 自定义骨架配色/拓扑/体型映射与官方控制分布的差异值得测；去掉骨架reference后仍泄漏，必须保留这一反证。没有证据把某一种配色直接定罪。

这几项可以同时存在。参数量大只解释资源成本，不能直接解释身份破坏。

## 6. 建议的下一版：身份与几何分开负责

这是架构建议，**本次没有实现，也没有声称它已经胜出**。

**第一层：保持源参考路径清楚。** A原图/按人物裁图走Qwen原生reference；合格的A脸裁图明确标注身份角色；B只提供动作、头向、表情、接触信息。Union/Pro主实验把骨架放控制路径；另保留原生Qwen“骨架当reference”的独立baseline，不混算输入收益。

**第二层：保留Union原生控制骨架，给几何输出可观测的作用范围。** 先消融“原版全joint hint”对比“只写target hint”；后者应保护base prefix相对native的表征，但修改了预训练使用方式，必须测姿态和身份两端，不能默认无损。保留整个人体动作支撑，脸部几何分成头向/位置/表达与身份形状。只把脸部残差全清零可能使头向和表情失控。

**第三层：把源身份/外观恢复从几何fusion中拆出。** 复用Pro人物bank和owner，做两个可独立关闭的分支：身体外观/衣服读取人物bank；脸部ID与高分辨率脸特征读取该人物face bank。优先比较当前“写进control state、再过Union after_proj”和“在选定base层直接写target ROI”的两种路径；其余条件、容量和训练预算尽量一致。

推荐验证的表达式：

```text
h_next = QwenBlock(h)
       + M_target · g_geo(layer,t,region) · R_geo
       + Σ_person M_face/person · g_id(layer,t) · R_id(source bank, current target hidden)
```

这里的gate按layer、去噪时间与区域分别考察。不能把“早期层”和“早期去噪步”当成同一个概念，也不直接规定“前16层一定最好”。先用少量可解释的固定设置测响应，再决定是否学习gate。层间残差RMS、脸ROI局部RMS和源bank响应要记录，但RMS小也不保证身份安全。

**第四层：训练必须直接关心身份与可编辑性。** 保留全局/局部flow监督，加入冻结、可微的FaceID网络和固定几何对齐的生成脸损失；生成端不能走ONNX/no_grad截断梯度。源ID是主要保持目标，同身份目标帧可辅助匹配目标视角，但跨身份driver B只能提供姿态/表情，不能作为ID真值。对每一类辅助损失记录有效人物数、有效更新次数和对任务参数的梯度。

`x0 = x_t - sigma * v_pred` 在高噪声处只是单步估计，身份监督应在可信噪声区间或有限步实际生成轨迹上验证，不能因为单步低MSE就认定完整采样好。几何与ID梯度若长期冲突，应调区域/权重/训练阶段；没有证据前不急着上复杂梯度投影算法。

身份提升也须受头向/表情/姿态约束，防止把A原脸正面复制回去取巧；背景约束只在源/目标人体占用并集以外的可保持区域施加，不能用整图像素一致性强迫人物不动。双人只隔离外观bank访问，保留身体接触所需的目标交互。

**第五层：必要时做局部高分辨率精修。** 精修从真实stage1成图启动，保留同一人物ID、目标头向/表达和几何；合成回整图后检查姿态、边界、头部尺寸与背景。只有在同一比较中确实增益才保留BFS LoRA。

这些建议与 [InstantID](https://arxiv.org/abs/2401.07519) 的身份语义/空间条件分工、[PuLID](https://arxiv.org/abs/2404.16022) 对身份与原模型行为保持的共同约束、[XVerse](https://arxiv.org/abs/2506.21416) 的主体特定控制思路一致；这些论文没有替我们证明Qwen2.1本任务效果。尤其XVerse的text-stream调制不能原样搬到Qwen2.1 single-stream、shared modulation和t=0 prefix上。借鉴机制，不承诺其已有权重可直接加载。

## 7. 应怎样测试，才能回答选型问题

按顺序推进，避免同时改结构、提示词、分辨率和监督后无法归因：

| 顺序 | 实验 | 回答的问题 |
|---|---|---|
| 0 | 先完成并读取已有Pro验证；统一native、Full、Union、Pro输入/分辨率/seed/步数，记录真实模型与控制配置 | 当前差异究竟来自哪里；不把尚未完成的run算成果 |
| 1 | 在现有Union上单独比较骨架control-only、原双路径、官方预处理一致的控制图；固定其他输入 | 输入分布是否导致骨架泄漏；参考通道是否有收益/干扰 |
| 2 | 原Union全joint hint vs target-only hint；单独扫控制强度 | 是否能在相同姿态水平下保留更多源身份 |
| 3 | 当前Pro vs 独立ID输出路径；都补齐同样的source脸与有效ID监督；增加“只改监督、不改架构”臂 | 改进来自信息/监督还是注入架构 |
| 4 | 同一方案512 vs1024/局部ROI，统一样本和有效脸集合 | 目标分辨率的净增益；不是跨集合均值比较 |
| 5 | 在选定设置上评双人遮挡/接触、身份交换与新身份泛化 | 单人收益能否迁到最终任务 |

每次先用一小组固定诊断样本和多个seed排除明显失败，再用隔离人物/视频的验证集选设置；test不参与调参。对同身份动作迁移和跨身份driver分别报告，新增训练应先通过同输入的过拟合完整采样检查。200步global下降20%可保留为调试指标，但不能替代最终的姿态与ID联合验收。

应报告的最小结果包括：

- body PCK/NME、head pose、表达、手部/接触、人体数量及人工伪影检查；所有PCK写清归一化与漏检口径。
- 脸ID使用ArcFace或独立身份评价器，身体/服饰外观用DINO等另列；脸检出率、脸尺寸与有效分母同时报告。小脸或大侧脸不能静默丢掉。
- source-target同身份跨视角相似度作为数据难度参照，不能称为数学上限；大姿态变换中ArcFace下降的一部分可能来自视角/检测，需要人工同看。
- 同seed、同输入的配对差值；独立人物/视频分组统计，不能把帧或seed当独立身份扩充置信度。
- **姿态–ID的Pareto取舍**：优先看相同姿态水平的ID是否提高，或相同ID水平的姿态是否提高；均值之外看双人和大姿态失败率。阈值由验证集预先确定，不用一个loss排所有模型。
- 固定几何后切换source身份/交换两个人的bank，确认生成跟随正确source；固定source后改姿态，确认模型确实改动作。训练数据中不要随意换source而仍拿原人物target监督，这会制造矛盾标签。

## 8. 复核范围与使用

本次本地HEAD为`292490bc`，服务器HEAD为`2ebd026b8568737d2f0e13c4ebea6a0634c54fff`。两侧Pro adapter/integration/contracts和Full qwen_adapter/control_core共5个关键文件SHA256一致；不能据此宣称两整个仓库一致。官方固定源码commit为`13bb3c625e8b1e457d7124d92207b49e00e4a6fe`。公开模型卡用于确认当前发布定位，历史实验只解释其固定版本。

交付文件：本报告；`evidence/`中的原始JSON、两张既有参考消融图及核算结果；`source/`中的固定官方源码与DeepGen配置；`manifest.json`中的来源、时间与SHA256；[只读核算脚本](scripts/audit_evidence.py)。源码快照用于阅读，不是新训练入口。

在仓库根目录复核（Python标准库即可，不调用GPU）：

```powershell
python -X utf8 research/qwen21_adapter_selection_20261020/scripts/audit_evidence.py --verify
```

需要重算当前Full/Pro模块参数，可额外用本机已有torch/diffusers：

```powershell
python -X utf8 research/qwen21_adapter_selection_20261020/scripts/audit_evidence.py --parameters --verify
```

后续使用时先按第7节的诊断实验推进，不直接以本报告替换训练架构。现有Pro失败quality gate的checkpoint只作工程/研究探针。本文没有修改任何模型或训练代码，没有发起新训练或GPU推理，没有移动、删除或覆盖既有图片与视频。

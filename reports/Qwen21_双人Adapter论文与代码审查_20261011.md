# Qwen-Image-2.1 双人 Adapter：论文、代码与接口审查

日期：2026-10-11。远端：`ssh sg`，项目 `/home/shangguanrz/project/pic-edit`。

**本轮已完成研究和设计，没有实施新版网络，也没有启动双人全量训练。** 本报告区分论文提出的机制、公开代码实现和我们自己的设计推论。详细实施设计见 [设计规格](D:/codeplus/pictureedit/docs/superpowers/specs/2026-10-11-qwen21-dual-v8-design.md)。

## 1. 结论

建议在 `qwen21_full` 中增加独立的 `dual_v8` 模式：复用现有单人几何编码的权重和数据语义，保留逐人物状态，用部位关系与接触图协调几何，在 Qwen 实际的目标 hidden state 上查询各自外观，最后只对目标图像 token 注入零初始化残差。

这项升级有三条依据：

1. 原有 full 已经包含共享人物编码、双向注意力、接触条件和槽位绑定。单纯再加同名模块不能说明升级价值。
2. full 的双人状态在进入继承自 DeepGen 的控制核心之前已经融合；外观 token 与其他条件在该核心内共同参与注意力，缺少逐人物的空间读取限制。
3. Qwen 2.1 的因果参考前缀、原生 RoPE、共享调制和前缀 KV 缓存构成明确的保护边界。外部目标残差可以利用这些边界，照搬 FLUX 的文本调制或角色 RoPE 则会改变它们。

推荐升级是**在现有 full 中增加 Qwen 原生双人控制路径**。旧单人和旧双人轻量路径保留用于兼容和比较；新双人模式不叠加原 full 的双人几何、交互残差。

## 2. 对原方案的修正

| 原方案中的表述 | 核查结果 | 对设计的影响 |
|---|---|---|
| 需要新增共享人物编码与 Mutual Attention | full 的 `PersonGeometryReasoner`、`BidirectionalCrossPersonReasoner` 已具备这些能力 | 升级关系表示、状态保留与注入方式，不能把已有能力当作新增贡献 |
| 接触图已经是完善的 Graph Transformer | 现有接触模块是接触栅格投影及关系 token cross-attention；没有显式部位节点邻接与边几何 | 增加可解释的部位节点、接触端点、法向、距离和置信度 |
| DanceTogether 的 adapter 可以直接迁移代码 | 检查的公开仓库有 0 个 Python 文件 | 借鉴论文的 mask/pose 绑定与融合思想，需要自行实现 |
| XVerse 是冻结主干的轻量调制即可 | 论文基于 FLUX.1-dev，并配合 rank 128 LoRA；代码改写文本归一化/调制 | 主体条件的分层控制可借鉴；不能据此宣称直接修改 Qwen 调制安全 |
| PeopleComposer 的角色 RoPE 可以原样搬入 Qwen | 它基于 FLUX；Qwen 的第一 RoPE 轴已参与文本位置和图像块定位 | 角色绑定放在 adapter 的人物 bank、部位节点和路由内 |
| CoShMDM 的 SDF/接触网络可直接复制 | 公开核心 `nets.py`、`blocks.py`、`losses.py` 与 InterGen 对应文件 SHA256 相同；未找到论文完整 VQ-VAE、SDF 策略和 CMC 实现 | 接触/体型设计作为研究启发，不能称为已经复现该论文模块 |
| full 仅有六个注入位置 | 轻量 `qwen21` 是六个；`qwen21_full` 是六组覆盖全部 32 个 block | 新旧注入预算必须分别计量 |
| 训练输入必须是两张独立人物源图 | 与本项目先前约定不一致 | 首版用一张同框源图派生 A/B 外观 bank，原生 Qwen 仍只接收这张参考图 |
| 有零初始化就不会影响原模型能力 | 只保证初始残差为零；训练后目标自注意力会传播改变 | 必须测指令遵循、背景、身份与关闭控制时的数值一致性 |

## 3. 实际 Qwen 2.1 计算路径

### 3.1 参数与 token 尺寸

本节依据远端已安装源码和本地模型配置，优先于通用 Qwen/FLUX 图示。

| 项目 | 当前值 | adapter 约束 |
|---|---|---|
| Transformer | `QwenImage21Transformer2DModel` | 不能按 Qwen-Image 1.0 的双流 block 接口设计 |
| Block 数 | 32 | 编号使用 0–31 |
| Hidden width | 4096 = 32 heads × 128 | 外部控制的输出维度为 4096 |
| 图像 token 输入/输出 | 64 / 64 | 不采用旧 16 通道 latent 或 256 通道 packing 假设 |
| `patch_size` | 1 | latent 网格的每个位置就是一个图像 token |
| VAE | 64 通道，RGBA 输入/输出，空间缩小 16 倍 | 512×512 图像对应 32×32 latent，即 1024 个目标 token |
| RoPE 轴宽 | `[16,56,56]` | 第一轴不是可任意挪作人物标签的空闲轴 |
| 参考理解 | Qwen3-VL，hidden 4096 | pipeline 使用最终 RMSNorm 之前的 hidden |
| Attention | 单流、块因果条件 | 目标图像内部双向，前缀不能读取未来目标 |

原生参考/文本 token 会**交错排列**。Qwen3-VL 标记的一个图像槽位展开成四个 latent token；Transformer 将原生参考 latent 插入这些槽位，目标图像 token 追加在末尾。不能把内部序列一概画成三个连续的 `[text][reference][target]` 块。

目标后缀连续这一点允许使用 `img_shapes[-1]` 确认目标网格，再对 block 输出的最后 `H_lat×W_lat` 个 token 更新。必须验证元数据；不能只凭总 token 长度猜测。

### 3.2 一个 block 实际怎样更新

每个 block 对同一个联合序列做两次残差更新：

```text
LayerNorm → 原生 scale 调制 → 原生 joint attention → tanh(gate) 残差
LayerNorm → 原生 scale 调制 → SwiGLU FFN           → tanh(gate) 残差
```

block 本身不持有一套独立 AdaLN 参数。父 Transformer 一次生成各层需要的 scale/gate，block 切取对应部分。这里的原生调制没有 FLUX 六参数调制中的 shift。

因此，新 adapter 应在**原生 block 完整执行之后**增加自己的目标残差。adapter 内部可以使用 FiLM，但不修改主干的 scale/gate、Q/K/V、attention mask 或 RoPE。

### 3.3 前缀缓存为什么影响设计

`causal_condition=true` 时：

- 文本和参考图像位置使用 timestep 0 的调制；目标使用当前去噪时间。
- 首次 `extract` 计算并保存各层前缀 KV；后续 `cached` 只重算目标。
- 有 true CFG 时，正提示词和负提示词分别持有缓存。
- 第一 RoPE 轴随前面的文本位置变化，图像块冻结到相应位置；高宽轴提供图像网格位置。

只更新目标且不让目标反向进入前缀，可以保留前缀缓存的语义。改写参考/文本调制或主干位置编码，则需要重新证明缓存有效性。新路径在训练初版关闭 KV 缓存，推理阶段专门验证缓存开启/关闭的一致性。

官方模型仓库见 [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1)。本次详细接口证据保存在 [Transformer 快照](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/diffusers/transformer_qwenimage21.py:584) 和 [pipeline 快照](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/diffusers/pipeline_qwenimage21.py:747)。

## 4. 逐项论文与代码审查

### 4.1 InterGen：借鉴共同坐标与同步更新

论文研究双人动作生成，使用共享权重的两个人物去噪器；代码 `InterDenoiser.forward` 每一层都从 `h_a_prev/h_b_prev` 同步更新两人，避免计算顺序偏置。`TransformerBlock` 依次执行人物内 self-attention、跨人物 cross-attention 和 FFN。共同世界坐标及距离/朝向约束针对的是生成的动作变量。

我们已有同类双向模块。新的收益点应是：将大网格交互改为带世界位置的部位节点交互，保留各自的稠密空间状态，避免外观在互相注意力中混合。论文的动作距离损失不能直接套在固定输入 SMPL-X 上训练图像输出。[论文](https://arxiv.org/abs/2304.05684)，[同步更新代码](https://github.com/tr3e/InterGen/blob/b1750df4a859443b8a69329f759ce3a7503ae179/models/nets.py)。

### 4.2 DanceTogether：借鉴绑定，不称为 DiT Mutual Attention

论文基于 StableAnimator/SVD 的视频 UNet。MaskPoseAdapter 共享 PoseNet，单独处理 mask，用门控融合，再通过卷积产生每个人的空间权重，在人物维度 softmax 后融合。这里的 “cross-person soft-attention” 与 InterGen 的 Transformer Q/K/V mutual attention 不同。

我们借鉴逐人物 pose/mask 对应、归一化融合和身份敏感区域监督。用 SMPL-X 的 amodal/visible mask 替代该方法的视频 tracking mask 作为已知几何条件。论文另有 face identity 机制，不能推断仅有 mask 就保证身份。公开仓库当前只有项目资料，没有 Python 实现。[论文 §3.3](https://arxiv.org/html/2505.18078v1#S3.SS3)，[官方仓库](https://github.com/yisuanwang/DanceTog)。

### 4.3 MS-Diffusion：最适合外观路由的参考

代码 `Resampler` 用短语语义及 bbox Fourier embedding 初始化查询，可随机丢弃 grounding 初始化；各参考独立压缩。`MaskedIPAttnProcessor2_0.prepare_attention_mask_qk` 限制人物图像条件的空间读取，并为背景保留 dummy token。

它基于 SDXL 的解耦图像 cross-attention。我们借鉴的是**每个人独立的 appearance bank、目标区域读自己的 bank、空区域不做全掩码 softmax**。具体 attention 在 Qwen 外部控制分支执行，不替换原生 single-stream attention。目标区域用渲染的可见部位权重，比仅 bbox 更适合拥抱遮挡。[论文](https://arxiv.org/abs/2406.07209)，[attention 代码](https://github.com/MS-Diffusion/MS-Diffusion/blob/fefbc7cadabf65f351a48023a7f59ecfcd16754d/msdiffusion/models/attention_processor.py)，[resampler](https://github.com/MS-Diffusion/MS-Diffusion/blob/fefbc7cadabf65f351a48023a7f59ecfcd16754d/msdiffusion/models/projection.py)。

### 4.4 XVerse：借鉴分层条件，避开原生文本调制

论文通过共享 offset 和逐 block offset 改变人物对应的文本调制，同时引入参考 VAE 细节。公开 `norm1_context_forward` 根据 `delta_emb_mask` 选择主体对应 token，重新计算其 shift/scale/gate。论文训练还使用 FLUX LoRA，不是只加一个外部小模块。

新设计只在 adapter 内为人物外观、层编号和去噪时间提供条件调制，再投影成目标残差。原生 Qwen 文本前缀、时间 0 调制保持原计算。分层门控的效果是我们的待验证假设，不能称为复现 XVerse。其跨图像训练结果也提示：源图与监督图必须分离，否则容易形成复制倾向。[论文](https://arxiv.org/abs/2506.21416)，[调制代码](https://github.com/bytedance/XVerse/blob/65e2581aac0faffe4607881a10cc486186f6499d/src/flux/block.py)。

### 4.5 Composing People Together / PeopleComposer：借鉴角色对应

论文在 FLUX 上增加 pose stream，联合生成 pose 图和 RGB，修改 τ 轴以绑定人物文本、bbox、pose/image 区域；其新增路径使用 LoRA。公开代码包含 split input/output、单独 pose modulation 和人物位置索引；训练入口明确说明公共数据 loader 尚缺失。

本项目已提供目标几何，因此不需要同步生成姿态或逐个人重绘 RGB。适合借鉴的是角色对应的显式契约：source person、目标 pose、部位与交互端点保持同一 UID。该绑定在 adapter bank 内实现；Qwen 的 τ/第一轴并非 FLUX 静态图像设置中的空闲轴。[论文 §3.2](https://arxiv.org/html/2605.23178v1#S3.SS2)，[官方代码](https://github.com/Cornell-VAILab/PeopleComposer)，[训练入口](https://github.com/Cornell-VAILab/PeopleComposer/blob/efb836371d48f8d42b629084c89d93b30b6cce9c/train.py)。

### 4.6 CoShMDM：机制与公开实现完整性分开判断

官方项目与作者摘要提出体型条件、法向对齐的接触矩阵、接触/动作 VQ-VAE、SDF 引导的穿透规避策略和 CMC。项目页的 Paper/arXiv 链接返回项目页，出版社全文本轮未能取得；没有声称精读完整公式。

检查公开 44 个 Python 文件及核心调用链，三份核心文件与 InterGen 一致，`CoShMDM` 调用 `InterDiffusion`；未定位完整 VQ-VAE、SDF 策略或 CMC。仅名称搜索不足以证明缺失，但源码调用链和文件校验共同说明当前公开版本不足以直接复现这些模块。我们借鉴接触端点、法向和体型意识，自行实现离线几何检查及边条件，不引用其未核实的模块实现。[官方项目](https://alimanjotho.github.io/coshmdm/)，[公开代码](https://github.com/AliManjotho/coshmdm-code)，[作者摘要与出版信息](https://pubmed.ncbi.nlm.nih.gov/41855059/)。

### 4.7 MPIE-Bench：适合评测，不能代替身份绑定验收

论文提出 2500 例多人编辑及六轴评价。公开 `score_mesh_v3.py` 从生成 RGB 经 Multi-HMR 重建，按预期人数保留 top-k；几何阶段不做身份匹配。`mesh_metrics.py` 对重建失败给零分并保存缺失诊断。接触距离/穿透属于重建代理量，有估计误差。

我们可借鉴 Anatomy/Interaction 分开报告及重建失败记录。还需另做 A/B 身份匹配、交换率和具体部位接触检查；仅两人体表靠近不足以证明握对了手。保留原始检出人数，避免 top-k 隐藏第三人。不要只统计可成功重建的案例。[论文](https://arxiv.org/abs/2607.27616)，[评价代码](https://github.com/AnnLin0628/mpie-bench/blob/227388a2929e7d2453bd23b0ab9963bf52875402/code/eval/score_mesh_v3.py)，[网格指标](https://github.com/AnnLin0628/mpie-bench/blob/227388a2929e7d2453bd23b0ab9963bf52875402/code/eval/mesh_metrics.py)。

### 4.8 VideoX-Fun：Qwen 2.1 接口参考已核实

该仓库真正包含 `qwenimage21_transformer2d_control.py`。默认控制索引是 `range(0,32,2)`，配置列出 16 处。control block 沿并行链执行，`before_proj/after_proj` 零初始化，129 维控制输入包含控制 latent、mask 和 masked-image latent；训练可只开放 `control` 参数。

它复用原生 Qwen block、处理交错图像槽位、块因果规则和 KV 缓存，适合作为工程接口参考。不过其控制输出是联合序列 skip，不天然满足本项目“只改目标”的严格边界；16 个 4096 维完整控制 block 也远大于当前轻量需求。[控制实现](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/models/qwenimage21_transformer2d_control.py)，[训练指南](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/qwenimage21_fun/README_TRAIN.md)。

## 5. 现有代码究竟缺什么

### 5.1 两条路径必须区分

| 路径 | 已有实现 | 主要不足 |
|---|---|---|
| `qwen21/dual_adapter.py` | 512 维；六处 `(3,8,13,18,23,28)`；读取当前 Qwen hidden；A/B 掩码提取源 latent | 每人外观被压成全局 role code，再填入目标 mask；细节空间结构弱；遮挡/接触尚缺显式端点路由 |
| `qwen21_full/qwen_adapter.py` | V6 Body/Face/Hand 接口；1536→4096 输出桥；只改目标 token | 控制核心读取噪声 latent 和 prompt，不读取当前 Qwen block hidden；六个输出重复注入 32 层；双人数据入口未贯通 |

不能用此前轻量双人 300 步结果证明 full 已完成双人训练，更不能证明泛化。

### 5.2 full 已有能力

- `SpatialConditionEncoder` 独立编码 Normal、25 通道 pose、14 类 part，支持 depth，并已有共同场景 depth 归一化。
- `PersonGeometryReasoner` 共享局部/全局编码。
- `BidirectionalCrossPersonReasoner` 用前一层 A/B 同步更新，默认两层。
- `ContactConditionReasoner` 投影 128 通道接触栅格、读取最多八个关系 token。
- `PersonTokenBinder` 用 person/source 槽位绑定几何和八个外观 token。
- full 输出桥已经处理串联两个零投影的死梯度问题，不能把这项保护遗漏后再“新增”。

### 5.3 需要升级的具体位置

1. **状态过早融合。** `DualFeatureFusion` 将 A/B 拼接为 scene geometry，交互由 A、B、差、积、contact 拼接生成。到输出时难以单独约束 A/B 外观。
2. **接触表示不够具体。** 现有关系记录人/part/type/distance，但实际 Hi4D converter 的 type 默认为零；没有端点位置、法向与置信度。不能把已有输入当作高质量动作接触标注。
3. **几何语义有旧约定。** Hi4D 的 `global_26` 与 iPER 约定不同；旧相对朝向用 axis-angle 的 y 分量差近似 yaw，不是一般三维旋转关系。
4. **几何与原生 latent 的语义桥接。** full 用 `64→64` 投影及 pixel shuffle 变成旧 16 通道网格。token 数能对齐，不代表 Qwen latent 与 DeepGen latent 的通道语义已经相同。
5. **缺少逐人物外观读取边界。** appearance、geometry token 在旧核心里共同参与条件注意力；固定 slot embedding 本身不禁止 A 读取 B。
6. **训练数据通路未完成。** `QwenFullDataset` 只支持 iPER/TikTok；`build_condition_bundle` 只填 Person-A。已有 HandConditionStore，但不能据此认为 Hi4D 双人手部数据已接入。
7. **控制预算未验证。** 当前 group 覆盖 `[0..5],[6..10],[11..15],[16..21],[22..26],[27..31]`，组大小为 6/5/5/6/5/5，不能把一次残差强度视为整组总强度。

主要证据：[full 注入](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/src/pose_control/qwen21_full/qwen_adapter.py:577)、[状态融合](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/src/pose_control/qwen21_full/reasoning/fusion.py:47)、[双人轻量路径](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/src/pose_control/qwen21/dual_adapter.py:114)、[数据入口](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/source_snapshot/src/pose_control/qwen21_full/data_runtime.py:122)。

## 6. 三种实施方向

| 方向 | 代价 | 能力保护与局限 | 判断 |
|---|---|---|---|
| 修改旧 full scene fusion，继续 1536 核心和 32 层注入 | 最快 | 仍有旧 latent 语义桥和外观共同注意力；需要额外解释累计控制强度 | 可做局部消融基线 |
| **共享几何状态 + 部位关系图 + Qwen hidden 查询 + 逐人空间路由** | 中等 | 原生前缀/attention/RoPE 不变；单人路径隔离；需验证稀疏控制能力 | **推荐首版** |
| 16 个完整 Qwen ControlNet block，或修改原生文本调制/角色 RoPE | 大 | 适合扩大容量探索；增加缓存、语义和算力回归面 | 暂不作为首版 |

本次“借鉴论文”的具体对应：InterGen → 同步共同坐标关系；DanceTogether → per-person mask/pose 绑定；MS-Diffusion → 各自 bank 与区域读取；PeopleComposer → UID/角色一致；XVerse → adapter 内分层条件；CoShMDM → 形状与接触端点/法向；MPIE → 图像输出重建评价；VideoX → Qwen 原生接口与零残差工程实现。

## 7. 数据与训练边界

首版任务仍是**源图 + 动作文本 + 外部目标几何**。使用目标帧拟合的 SMPL-X、姿态、深度和 mask，表示已知目标几何的控制编辑，不是纯文本动作生成。目标 RGB 只供离线标注、训练监督和独立评测，不进入生成函数。

最新两帧 SMPL-X 试拟合已完成；手部多数缺少足够多视角观测，hug 的手指不能作为可靠真值。参见 [拟合结果](D:/codeplus/pictureedit/reports/Hi4D_SMPLX工具核查与两帧拟合结果_20261011.md)。该试拟合不等于所有 Hi4D 配对都有已完成的 SMPL-X 条件。

固定输入网格上的 contact/SDF loss 对图像 adapter 没有有效梯度。第一版使用 flow matching 监督及几何区域权重；穿透修正在离线拟合阶段处理。对生成图做 DWPose/人脸/网格评价时单独报告缺失率。后续若加入感知/姿态训练损失，必须证明损失经可微图像预测回传到 adapter，不能把 ONNX 推理分数直接当作可训练 loss。

验证先做单样本过拟合，再做未参与训练的配对诊断。只有数据边界、控制切换与能力回归通过后，才提出多样本训练；本轮不自动恢复之前的 2k+8k 全量计划。

## 8. 复核与使用

本地可重跑静态证据核查：

```powershell
cd D:\codeplus\pictureedit
python research/qwen21_dual_upgrade_20261011/audit_evidence.py
```

本轮结果 **10/10**。这是配置/源码审计，不是新版 adapter 的运行测试。[机器可读证据](D:/codeplus/pictureedit/research/qwen21_dual_upgrade_20261011/evidence.json) 包含源码 SHA256、模型配置、repo revision、注入索引和核查范围。

完整审计包、报告和设计也已同步到 `ssh sg:/home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011/`。远端重跑命令：

```bash
cd /home/shangguanrz/project/pic-edit/research/qwen21_dual_upgrade_20261011
/home/shangguanrz/nihaochen/envs/qwen21/bin/python audit_evidence.py
```

远端同样为 **10/10**。执行的是本项目的静态审计脚本，不导入外部仓库模型。

| 仓库 | 固定 revision |
|---|---|
| InterGen | `b1750df4a859443b8a69329f759ce3a7503ae179` |
| DanceTog | `9034b4df5cffcf5c99dabbc77f9da9637559da50` |
| MS-Diffusion | `fefbc7cadabf65f351a48023a7f59ecfcd16754d` |
| XVerse | `65e2581aac0faffe4607881a10cc486186f6499d` |
| PeopleComposer | `efb836371d48f8d42b629084c89d93b30b6cce9c` |
| CoShMDM | `c5116ab3bead55051ffec2ec9a161a38d2ab5116` |
| MPIE-Bench | `227388a2929e7d2453bd23b0ab9963bf52875402` |
| VideoX-Fun | `4b7b6402a1e0f0406bd6801fb66c0a00bd922621` |

远端项目 HEAD 快照为 `3236657b08f0fe51d881a78c114bf57b956e953e`，工作区存在未提交内容，以保存的实际源码 hash 为准。已安装 Diffusers 为 `0.41.0.dev0`；模型配置记录的版本是 `0.37.0.dev0`。研究下载没有执行外部仓库代码或下载其模型权重。

新版使用方式、具体张量协议、测试矩阵和阶段验收已经写入设计规格。它们是待实施接口，不能当作现有 CLI 运行。

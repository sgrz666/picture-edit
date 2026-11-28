# Contact Hand Adapter V1 设计规格

当前代码采用 V3 架构。本文保留 V1 的选型背景、数据盘点和评测方法。 [V3 实现与使用](../../../src/pose_control/contact_hand_adapter/README.md)

独立训练一个 Qwen-Image 2.1 Hand Adapter，依次解决手指解剖、精细手势、双手接触与遮挡。首版覆盖单人单手、同人双手以及双人握手、牵手和有标注的十指相扣。冻结 Qwen DiT、Qwen3-VL 和 VAE。Adapter 从随机初始化开始，只有残差输出头置零。Native 权重、旧 Hand Branch 和 PoseLink 均在本项目的依赖边界之外。

本规格已核对论文、仓库代码和 sg 的数据路径。层位、损失系数、训练预算及通过线是首轮实验配置，效果和峰值显存需通过实际出图验证。

## 1. 论文证据与工程选择

| 证据 | 对 V1 的意义 | 原始来源 |
| --- | --- | --- |
| HanDrawer 以 Mesh GCN 编码 MANO 顶点，将条件对齐到生成区域。HanDiffuser 用手部参数指导扩散生图。 | 单手用 Joint GCN 和可选 Mesh GCN，保留输入拓扑及像素落点。 | [HanDrawer](https://arxiv.org/abs/2503.02127) · [HanDiffuser](https://openaccess.thecvf.com/content/CVPR2024/html/Narasimhaswamy_HanDiffuser_Text-to-Image_Generation_With_Realistic_Hand_Appearances_CVPR_2024_paper.html) |
| FoundHand 在手部图像对上使用空间对齐的 2D 点和掩码，支持换姿态、迁移外观和局部修复。 | 保留 2D 可用路径，用局部真实帧对启动 SFT。现有修复器作为单独推理对照。 | [FoundHand](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_FoundHand_Large-Scale_Domain-Specific_Learning_for_Controllable_Hand_Image_Generation_CVPR_2025_paper.html) |
| InterHandGen 在 MANO 参数空间先采样一手再生成另一手，采样时加防穿透引导。 | 在本项目用有向手对关系连接手部 token。防穿透先用于可靠三维输入的拟合质检，RGB 接触效果用独立评价验证。 | [InterHandGen](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_InterHandGen_Two-Hand_Interaction_Generation_via_Cascaded_Reverse_Diffusion_CVPR_2024_paper.pdf) |
| TCCA 用去噪状态和条件内容调节姿态、外观注入。ControlNet 使用零初始化接入冻结主干。 | 将动态调节迁移到结构和对齐空间两路，分别学习门控。首轮保留固定门控消融。 | [TCCA](https://openaccess.thecvf.com/content/CVPR2026/html/Hao_A_Temporal_and_Content_Co-Awareness_Latent_Diffusion_for_Controllable_Hand_CVPR_2026_paper.html) · [ControlNet](https://arxiv.org/abs/2302.05543) |
| WiLoR 提供单图三维手估计。HandEval 用手部质量排序监督训练评估器。 | WiLoR 可产候选 3D 条件。HandEval 先与人工解剖盲评校准，接触、归属和遮挡仍需独立标注。 | [WiLoR](https://rolpotamias.github.io/WiLoR/) · [HandEval](https://arxiv.org/abs/2510.08978) |
| HandRefiner、HandCraft 和 RHanDS 关注异常手部的局部修复。GraspDiffusion 以显式抓握几何驱动人和物体合成。 | 局部修复用于衡量分辨率和精修上限。V1 的接触关系只包含手和手。 | [HandRefiner](https://arxiv.org/abs/2311.17957) · [HandCraft](https://openaccess.thecvf.com/content/WACV2025/papers/Qin_HandCraft_Anatomically_Correct_Restoration_of_Malformed_Hands_in_Diffusion_Generated_WACV_2025_paper.pdf) · [RHanDS](https://ojs.aaai.org/index.php/AAAI/article/download/32815/34970) · [GraspDiffusion](https://yj7082126.github.io/graspdiffusion/) |

比较三条可实施路径。逐手骨架加局部残差是最低成本的归因基线。结构 token 加空间特征保留关节关系和画布位置，且能加入跨手条件，作为 V1 主方案。联合 ROI 二次生成能增大手部的实际生成像素，但要额外处理耗时、腕部接缝及皮肤外观，作为单独对照。Beach1123 既有 FoundHand 记录中，I2I 三个处理臂各 84 图，SDPose 腕相对手势分为 0.5129、0.5420、0.5313，原图臂为 0.5513。RePaint 的 28 图对照为 0.4941 对 0.5479。这个结果只反映该实验的手势分，解剖与接触仍需盲评。

## 2. 条件契约

- **一个手槽对应一个人和一侧手。** 固定四槽 `[A-L, A-R, B-L, B-R]`，A/B 来自显式人物绑定，L/R 是解剖侧。A-R 和 B-R 可以构成一条握手边。同人 A-L 和 A-R 可以构成双手边。缺槽、归属冲突、不可见、出画分别记录。水平翻转是联合增强，交换解剖侧、重排手槽和交互边，并重算有方向的几何量。
- **最小正向条件**包含目标 Canvas、人物侧别、腕点、21 点 XY、关节可见性及置信度。关节统一为 wrist，随后 thumb、index、middle、ring、little，各从根到尖四点。每个源格式有显式转换表和指节叠图。FoundHand NPZ 文档中的前 21 右手和后 21 左手只在导入层解码。
- **可选条件**含 MANO pose/shape、共同相机下的 Mesh 和投影、逐手实例/可见 mask、每实例 depth/normal、手部 part map、接触图与源手 RGB。所有模态有独立有效位和可信度。局部 3D 相对腕点归一化，画布 XY 独立保留。只有共同相机、尺寸单位和深度尺度均通过核验的手对才使用相对 3D 位移、深度排序或表面距离。独立单图估计只提供各自局部几何。
- **接触标签有三态。** 每个手对的 16×16 表面分区矩阵记录已知接触、已知不接触、未知，并记录证据和覆盖 mask。V1 分区为掌面一个区，每指近节、中段、远节三个区，拇指按 CMC/MCP/IP 定义映射，以版本化顶点索引锁定。交换有向手对时矩阵转置，前后遮挡的方向取反。2D 重叠只用于建立候选边。官方接触标注、可靠多视角几何或人工复核分别记录来源。只有已复核或已完整覆盖的明确负样本才能提供负标签。人工单视角只标看得到的局部以及动作成功，不补未知指面。
- **来源和监督隔离。** 推理读取源 RGB、同一份 prompt/姿态 guide 和已经正式导出的目标几何包。目标 RGB、判分标签、监督 mask 只在离线构建或训练/eval 侧可见。真实几何与从目标参考离线提取的几何分别标为 oracle、estimated 两个评测轨，单独汇报。针对 source 只有不同身份或任务未提供源手的样本，源外观条件置缺失。首个 SFT 使用 Qwen 自带 Source RGB 参考流，预留可选源手 crop 输入做后续外观消融。

模型内按固定槽编排 `joints_xyc[4,21,3]`、`joint_visible[4,21]`、`hand_valid[4]`。每个无序手对只存一份 `contact_state[16,16]` 和同形的可信度，构图时展开有向边。遮挡方向按可见重叠区域保存，未知区域屏蔽；一对手的不同指部可互相遮挡。校验有效位后再计算几何量，失效坐标先置中性填值，避免 NaN 在乘 mask 之前传播。

## 3. 双路径模型

| 模块 | V1 结构与输出 |
| --- | --- |
| Hand Anatomy Encoder | 宽度 512。关节树 21 点，带自环、有效点屏蔽和按有效邻居度归一化，3 层 Joint GCN。可靠 MANO 使用 778 顶点原拓扑，3 层 Mesh GCN，按模板划分为 32 个固定 surface patch，每个接触分区两个 patch。关节/表面 token 加槽位、解剖侧及 part embedding，每手最多 53 个 token。2 层 Intra-hand Transformer 学习不同手指的协同。无合法 Mesh 时屏蔽 32 个表面 token，2D 路径继续工作。 |
| Hand Interaction Transformer | 2 层、8 头。先根据显式接触/非接触、投影接近或重叠构建候选边，或在可靠 3D 中按距离建边。对候选 peer 的有效 token 用相对位置、可用的三维朝向/距离、接触状态与遮挡方向作有向 attention bias。已知接触 bias 只覆盖有证据的位置，未知走中性 bias。关节 token 映射到对应手指分区，使 2D 条件也能承接可用的区域接触标注。孤立手走 Intra-hand 输出，无有效 peer 不执行空 softmax。 |
| Hand Spatial Encoder | 对目标画布上的每手 ROI 或交互连通组的联合 ROI 等比映射到 256×256，记录双向仿射和 pad。共享 CNN/FPN，四级初设 128²×64、64²×128、32²×256、16²×512。输入包含关节热图、分区、可见/实例 mask、可用的法线、每手独立 depth 与接触图，以及有效位。每手保留自身层和仿射，同组共用坐标框。2D 最小路径从可信关节及腕部走廊构建软支持。 |
| Dual-path Fusion | 读取当前层 post-block 的目标 ROI Hidden，从 4096 投影到 512。结构路以 ROI Hidden 查询当前手及交互邻居的结构 token。空间路以相同 ROI Hidden 和对齐后的 FPN 特征做轻量 MLP 融合。FPN 在 ROI 内融合多尺度信息，再按真实空间覆盖采样到目标 token。 |
| Dynamic Residual Controller | 每条路各从当前 ROI Hidden、sigma、层 embedding 和条件置信度得到有界 sigmoid 门控，初始门约 0.5。共享 query、Attention、空间融合和门控主体，每个注入层各有两套零初始化 512→4096 无 bias 输出头。门和投影不能同时归零，确保首轮反传能更新输出头。 |

结构与空间的增量按 hand-slot 写回。直接注入支持包括目标手、腕部和可靠接触像素。重叠处先根据可靠可见归属分配权重。归属未知时把多个有效贡献按和最多为 1 的权重混合，保留各实例的 token 与深度层。最终 residual 每个目标位置只写入一次。顺序改变只改变处理顺序，不改变结果。在乘 canvas coverage 之前核对支持是否已按像素跟 canvas 求了交集，避免重复衰减。

~~~
同一目标手部几何 ─┬─ Joint / Mesh GCN → 单手 Transformer → 跨手 Transformer ─┐
                └─ 256 ROI 条件      → CNN/FPN           → 空间对齐      ─┤
冻结 Qwen DiT 的目标 ROI Hidden ──────────── 两路融合 → 分路门控 → 零初始化残差头
                                                     └─ 散回目标手与腕部
~~~

两路共有 16 个输出头，参数量 33,554,432。4 层 512 宽 Transformer 按 FFN 4× 计约 12.6M，加上共享 Q 投影 2.10M。其余组件以总量 **不超过 60M** 为实现预算，实际 trainable 参数数、激活、单步耗时以 preflight 输出为准。

## 4. Qwen 接口与小手约束

在 sg 的 diffusers 0.41.0.dev0 中，Qwen 图像 latent 为 64 通道，`_pack_latents` 保持 16 px 一个 token 的空间步长。画布尺寸按 32 对齐，四个图像 token 对应一个 VLM 位置。冻结的 DiT 有 32 层，宽度 4096。现有 V7 调用 V6.2 installer，V6.2 在原 block 输出上写目标 token 并显式捕获非重入 checkpoint context。本项目与原生 Qwen 的签名直接适配，借鉴其路由及梯度验证方法。 [官方模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1) · [V7 现有入口](../../../src/pose_control/qwen21_full/native_v7/integration.py) · [底层实现](../../../src/pose_control/qwen21_full/native_v62/integration.py)

接口审计对应本地 `c6de780d`。sg 安装的 `transformer_qwenimage21.py` SHA256 为 `0eb0555e21ca93195e1fe9389113cafbf0e8822f6454c3001cebb3d21521ebd7`，基座 `transformer/config.json` SHA256 为 `56ae3281c4e6c2d1aa3658252d187488071815fd79bef15808bb0205fc1a2241`。正式运行还需绑定实际权重分片和 scheduler。

1. 先用 `img_mask`、`img_shapes` 和 `build_token_metadata()` 校验参考图块和最终目标图块。一个 forward 的目标 grid 在所有注入层保持一致。首版 microbatch B=1，同一画布可有 1–4 个有效手槽。条件 mask 随每个 forward 显式绑定并传给非重入 activation checkpoint，允许反向重算。训练、推理均用关闭 KV cache 的路径，CFG 如启用，两次 forward 按同一条件策略对齐，基线也用相同配置。
2. post-block 注入采用从零编号的 [3, 7, 11, 15, 19, 23, 27, 31]。取 transformer 接收到的归一化 sigma，直接跳过无有效条件或 strength=0 的样本。失效模态降级可观测，错误的 shape、坐标或人物绑定在入口报错。
3. 手部、腕部、contact 和有效画布先在像素空间相交，再用面积覆盖映射 target token。使用仓库共享的 CanvasSpec。ROI 包含腕点和可确定的前臂短段，目标腕锚点用上游目标 pose guide 与手部关键点交叉核对。有确认的近邻接触时按交互连通组合并 ROI。统计 ROI 原生像素、目标画布像素和 token 支持。 [CanvasSpec](../../../src/data/canvas_contract.py)
4. 16 px 步长下，32 px 宽的手在短轴只有约 2 个 token 间隔，256 ROI 只是条件栅格。V1 以可见手框短边小于 32 px 或软面积小于 2 个 token 为全图细节不足标记；训练中可用于手腕/定位压力测试，主精细手势训练优先短边至少 32 px 且软面积至少 2 token。打分按 <32、32–63、≥64 px 分桶。局部 SFT 用 512 输出和 256 条件 ROI。全图试跑从长边 ≤1024 的等比画布起步。提高全图分辨率和 512 联合 ROI 二次生成作为单独分支，记录耗时、接缝和增益。
5. 未来联合使用时，在一个统一回调中从同一 post-block Hidden 计算 Native 与 Hand 残差，保留手腕锚点，按 hand support 协调 Native 局部 hand 分量及重叠 RMS。保留两者单独旁路。现有 V6.2 installer 明确拒绝其他控制器堆叠，联合阶段需单独实现和验证统一回调。

## 5. 数据清单、现有资产与 SFT

每条样本有可验证的 source/target RGB SHA、几何和 mask SHA、Canvas 指纹、数据源版本、原始人物/视频/相机/帧 ID、动作、每手所属人物和解剖侧，以及有效性/拒绝原因。训练清单与推理条件包分开，cache key 绑定基座、源参考顺序、prompt、Canvas、几何映射和标注版本。pair 按同人/同人对的真实帧对建立，局部 crop 保留同一人物和手侧，源帧与目标帧分开。全图 SFT 使用同场景和背景稳定配对。切分先分割人物或人物对的连通群组，再分序列、原始视频和近重复组。FoundHand 的 InterHand2.6M / Re:InterHand 子集和原始数据共享 lineage，所有相机、重光版本和转码留在同一 split。HIA 现有视频 split 可用于冒烟，身份隔离未证实，不能直接作为独立泛化盲测。

| 数据 | 盘点与用途 |
| --- | --- |
| HIA strict | sg 的 /data/shangguanrz/HIA/release/strict 三个 pair 文件有 train 14,366、val 1,836、test 1,793 条，逐条核对源图、目标图、两份 pose 路径均存在。现有清单是 2D 条件。Train 清单 SHA256 `6e680626ef01e3942bb4e3ad4f39439234cb10cf9aa4a2cee3e96234611d5f07`。按下方用途核验决定是否纳入。 |
| HIA 96 帧试集 | /data/shangguanrz/pic-edit-datasets-test/HIA_face_hand_v1_20261026/video_split_v1 的 READY 登记 train/val/test 各 57/2/22 对。清单原前缀 /data/shangguanrz/pic-edit-datasets 已不可解析，按 /data/shangguanrz/pic-edit-datasets-test 作只读根映射后，81 对的四个输入文件均可定位。目录信息可用于检验路径解析规则，像素和 pose 资产的使用按下方用途核验。 |
| Hi4D | 同一新根下有 32 个原始压缩包和 Hi4D_pilot_v1。Pilot 的已有帧对可先用于双人腕部/画布审计，旧 Pilot 是缺手指拓扑的 SMPL，不能给 16×16 接触矩阵提供真值。原包有顶点级接触信息，且仍需按手部区域、配准残差、像素可见性筛查。原始接触标签和新手部重建 SHA 写入清单。 [Hi4D 项目页](https://yifeiyin04.github.io/Hi4D/) |
| Bilibili / 小红书 | 已看到数据根目录及现有流程，手专项合格规模待审计。从同场景 SFT pairs 筛清晰单手、同人双手和双人互动作，人工补核握手、牵手、相扣及近距离未接触。按拍摄者/人物、序列和重复簇防泄漏。 |
| FoundHand-10M | 官方仓库提供 Globus 和源数据谱系，标注主要是 42 点和二值手 mask。sg 核查到 FoundHand 代码、推理权重及 demo，已扫描范围内未定位到完整训练库。按可用许可选同身份帧对，供 2D 姿态和局部外观预训练。 [官方数据说明](https://github.com/arthurchen0518/FoundHand#foundhand-10m-dataset) |
| InterHand2.6M / Re:InterHand | 优先补同人双手共相机的相对 3D、MANO 和遮挡样本。用可靠 fit 推导表面距离只作为带置信度的派生标注，记录对齐和误差，人工复核可靠负例。已扫描范围内未定位到正式库。 [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) · [Re:InterHand](https://mks0601.github.io/ReInterHand/) |
| CHI3D | 用官方 contact signature 对齐的关键帧扩充跨人握手/牵手。第二位人物 Mesh 是伪真值，signature 集中在每段单个关键时间戳，标注接触区以外仍有未知。多视角按同一次交互计数。已扫描范围内未定位到正式库。 [CHI3D 标注说明](https://ci3d.imar.ro/chi3d) |

HIA 上游数据卡正文附有禁商用和下载后 24 小时保留限制，现存衍生数据的持续训练授权在本次盘点中未核实。HIA 统计保留为只读库存证据，训练清单只接纳有独立授权记录的样本。首轮可先用合成张量验证 Canvas、路由和梯度，再用有训练授权的自采帧对或可用许可下的 FoundHand-10M 子集做真实图过拟合。Qwen、FoundHand、原始子数据、MANO 权重各自记录适用条款。阶段 2 的 3D 和接触子池按各自覆盖与验收启用。 [HIA 上游数据卡](https://huggingface.co/datasets/gulucaptain/Human-Image-Animation-Data) · [Qwen 许可证](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) · [FoundHand 数据许可证](https://github.com/arthurchen0518/FoundHand#license)

| 阶段 | 配样与训练目的 | 通过条件 |
| --- | --- | --- |
| Stage 0 数据/梯度冒烟 | 先用合成条件覆盖左右手、非方形画布和缺测，检查导入、叠图、路由和 1 步有限梯度。再从准入数据固定 8–16 个真实帧对过拟合。 | bypass 和零初始化与基座一致、可重复严格重载、输出头在首步有梯度、之后两路编码器能获得梯度，条件交换时手势有方向性变化。 |
| Stage 1 局部 SFT | 100% 局部。以通过用途核验的 2D 手部专项帧对训练解剖与手势。手框按人物侧别配对，源/目标独立画布。Mesh/接触缺失用有效位。 | 同 Source/seed 下提高目标侧手势分，同时人工解剖错误率低于同条件 Qwen，过线后才扩大。 |
| Stage 2 接触和遮挡 SFT | 70% 局部、30% 全图。用有共相机 3D 的同人双手和实拍跨人双手，保留低遮挡单手与可确认的近距离未接触。优先握手、牵手；十指相扣有独立合格子集时单列。 | 接触训练和盲测均有独立标注样本及硬负例。每类记录独立人对/会话数和未知率，稀缺类别只报告覆盖和样例。 |
| Stage 3 全图 SFT | 30% 局部、70% 全图，覆盖小手、腕部和前景背景。沿用同一 Adapter 结构。 | 全图 ODE 通过独立保留集，手部增益和非编辑区域保持同时达标。 |

固定采样配额以优化步内有效样本数计，记录每路实际曝光；Stage 2/3 不用重复样本填满缺失的动作类别。首轮 AdamW、learning rate 1e-4、microbatch 1、8 步梯度累积、grad clip 1.0、主干 BF16、Adapter FP32。Stage 0 的过拟合预算 100 更新，Stage 1 初次对照每臂 1,000 更新且匹配实际样本和 seen-token 数。先跑本机预检和短跑测峰值，再根据耗时决定扩容。大阶段从当前架构最新验收 checkpoint 严格接续，模型、缓存、拆分和 evaluator SHA 随运行固化。

## 6. 损失和归因

$$
z_\sigma=(1-\sigma)z_0+\sigma\epsilon,\quad v^*=\epsilon-z_0,\quad
\widehat z_0=z_\sigma-\sigma\widehat v.
$$

首轮仅用区域 Flow Matching。对每图在像素空间按优先级得到互斥的 contact 4.0、普通手 3.0、腕部过渡 1.5、其他有效区域 0.5，再池化至实际网格。每区先对通道求平方误差均值，再以软面积作归一。每手和每对的区域均值先在图内等权合并，同家族只对累积窗口中有资格的图平均。面积下限首测设 0.5 个等效 token，分母为实际面积与下限的最大值。缺失家族权重不重分配。接触未知略过 contact 项，其像素可进入普通手区。padding 权重为 0。后续 DiT 层的 Self-Attention 可以传播手区改动，全图有效区监督和输出保真门禁均需保留。

若后续要验证 Joint、Part、Contact 或 Occlusion 额外监督，使用从 $\widehat z_0$ 解码的图像和独立标注，首先验证解码及评估头到 Adapter 有限梯度；拟合输入 Mesh 或复述输入 contact 矩阵只作条件诊断。Mesh 碰撞检查只在共同相机和可靠拟合上启用。可微输出解剖监督和 HandEval 奖励待 SFT 证据成立后做独立实验，V1 不预设 RL 权重。

| 对照 | 要验证的变化 |
| --- | --- |
| Q0 原生 Qwen | 与 Adapter 各臂共享 source、prompt、seed、画布、采样器和 CFG。全图使用同一全身 pose guide，局部使用相同 Source crop 和 prompt。另记无 guide 的全图原生 Qwen 作为产品参照。 |
| H1 空间路径 | 使用相同目标几何的 raster 通道，不建跨手边。 |
| H2 双路单手 | H1 加结构路和单手 GCN/Transformer，隔离 token 路带来的收益。以同维 MLP 替换 GCN 做拓扑消融。 |
| H3 跨手几何 | H2 加跨手 attention 和相对几何，关掉显式接触 bias 和热图。 |
| H4 完整 | H3 加接触 state/bias、空间热图和可靠可见层。H3/H4 共享数据与 FM 监督区域，以隔离条件差异。额外对照固定/动态门控和 4/8 层注入。 |
| R1 外部精修 | Q0 加 FoundHand，并列等分辨率联合 ROI 结果，记录额外推理预算。 |
| N1 / N2 后续联合 | Qwen+Native 与 Qwen+Native+Hand。同一 Qwen base、参考图和有验收记录的 Native checkpoint，Hand 用独立验收产物。 |

训练消融按相同 split、seed、数据顺序、有效优化步及初始化协议运行，共享参数从同一份初始化或已验收父权重复制。新模块按稳定子模块种子生成，参数预算差异公开。对一类数据只切 Contact 条件时监督区域清单保持一致。另在同一权重上做病理条件测试，包括缺 3D、交换 A/B、去掉接触、可见区域内增噪。单张生成样例不能代替成对留出结论。

## 7. 怎么测试与使用

**设计复核。** 先看 §2 的槽位和 unknown 规则，再沿 §4 检查 Qwen token mask 和非方形画布。对 sg 的 HIA 和 Hi4D 路径可查实际文件，HIA 旧路径应由导入器的显式 root mapping 处理。运行产物使用 /data/shangguanrz/pic-edit-experiments/ 下的独立实验目录。正式开跑报告权重路径、SHA256、来源与父链。新 Adapter 首轮父来源为同一份冻结 Qwen 配置与初始化种子，Stage 2/3 的 Adapter checkpoint 另记自身父 SHA。

**实现验收。** 计划入口在 scripts/train/、scripts/run/、scripts/eval/。核心放在 src/pose_control/contact_hand_adapter/。训练入口先验证 data manifest 和单步前后向，推理入口读取 ordered source refs、prompt、目标手几何包、seed 和独立 checkpoint，输出全图及手/腕/接触叠图。首个测试集覆盖：

- B=1 有零手、单手、两人右手接触与四手，以及缺 Mesh、深度和接触。所有有效张量有限。缺条件和 strength=0 下，预测与未安装基线逐元素一致。零初始化有效条件也应与基线预测值相同，首步两路残差头梯度非零。后续一步核对编码器和动态门获得梯度，冻结参数无梯度或更新。
- 两种非方形画布、有/无 padding、有一个或多个 Qwen 参考图。按目标 Canvas 画叠图，text、reference 和目标 ROI 外的**直接**残差为零。逆映射误差不超过 1 画布 px。重叠区权重和 ≤1，翻转场景的侧别、前后关系和接触边保持一致。
- 同一固定 forward 比较开启/关闭 checkpointing 的 loss 与 Adapter 梯度。模拟失败、重试和恢复后条件不会串行残留。独立 checkpoint 严格加载、核查 shape/版本与 base 指纹，续跑保持相同 split、sigma、噪声/采样顺序。

**效果验收。** 同一任务、seed 42/43/44 运行配对 ODE，候选图和评测身份记录 SHA。SDPose-Wholebody + YOLO11x 调共享 load_scoring_runtime() / score_people()，几何 134 槽、关节阈值 ≥0.30，固定目标侧点、腕和框尺度并逐手等权。报告 Hands42 位置分、PCK、腕相对手势分、检出率和双人检出率。目标合格而生成漏检计零，评测器异常中止该批。局部手裁图常缺人框，单独报告此口径的覆盖并以全图姿态和独立人工局部盲评为主，禁止切换姿态评分后端。 [SDPose 规范](../../SDPOSE_POSE_EVAL_SPEC.md)

人工盲评按单手、同人双手、握手、牵手、相扣和近邻未接触分层，评主体侧别、增/缺/粘连指、接触动作正确且无视觉穿透、遮挡顺序和腕部接缝；两位评审独立打标，分歧复核。报告有独立几何/人工证据的 part-contact precision/recall/F1，未知单列覆盖，Mesh penetration 仅对可可信重建的输出报告。对全图跟踪 Body17、glintr100 和共同可靠背景区 MAE，背景从 Source A 取参考，排除原/目标前景及边界。对同图不同 seed 按原始人物对/视频分组做 paired bootstrap，而非把多个种子和多视角当独立样本。预设扩大线为同条件腕相对手势 +0.03、人工解剖严重错误相对下降 ≥20%，接触阶段动作成功率相对 Q0 +10 个百分点且误接触率劣化 ≤3 个百分点。全图 Body17 / 双人检出率下滑 ≤2 个百分点，glintr100 绝对下降 ≤0.02，可靠背景 MAE 增量 ≤2/255。首次采样需按覆盖与置信区间判断可否定论，样本不足保持试验结论，正式验收前在 dev 集锁定阈值，测试集只验一次。

V1 的交付顺序为独立 SFT Adapter、运行清单、同条件生成与评测，再决定是否引入局部精修、Hand-specific LoRA 或 RL。RL 若启用，从通过固定 ODE 门禁的 SFT 权重启动，以人工排序校准解剖与接触分，分开记录首次成功、接触覆盖、误接触、组内方差和保真。联合 Native 作为后续独立实验。

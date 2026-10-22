# Full 方案 C 研究评审：条件编码、Qwen 主干适配、Depth

研究日期：2026-10-22，Asia/Shanghai。状态：**已完成代码与文献研究，候选结构尚未实现或验证**。用户选择现有 Full 作为基准，明确不再使用 Union 控制权重、后续不派 subagent。本次没有修改模型源码、训练配置、缓存、权重或远端文件。

## 1. 推荐方向与边界

推荐沿用户的方案 C 做 Full 的原生化迁移：保留已经训练过的几何前端，让轻量控制模块读取 Qwen 主干当层目标 hidden，再产生目标图残差。优先复用 Full 自己的知识、数据和监督；不再以 Pro/Union 作为主平台。

需要同时说明：保留前端不等于保留 Full 全部控制能力。旧1536维控制核、源 latent 条件与输出桥也承载了学习结果，拿掉它们需要重新训练映射。现有 Full 可作为迁移教师或固定对照，教师不会成为新版本的推理依赖。

三个实现选择都不依赖 Union：

| 选择 | 复用程度与代价 | 判断 |
|---|---|---|
| Full 前端＋主干实时512维控制模块 | 保留几何编码/推理；新接口需要训练，Body可去掉旧SD3控制核 | 推荐的最终结构 |
| 先在旧Full旁加实时反馈修正，再迁移旧控制输出 | 初始更容易保持现有输出，但过渡期保留两条控制路径，归因更复杂 | 新接口明显回退时的备选，不作为默认长期结构 |
| 几何条件token加入Qwen原生序列并训练LoRA | 能直接使用原生attention，但增加序列、位置与因果mask设计，也改变主干 | 后续备选；不是第一版 |

本次先聚焦 Body。原有 Face/Hand 接口和配置保留，迁移Body时不同时重做脸手网络。保留旧Face/Hand时，整套系统仍可能带有1536维局部残差路径，不能宣称整个Full已经彻底去掉所有旧维度。

## 2. 两份来稿：采纳、修正、暂缓

| 意见 | 评审 |
|---|---|
| 保留Body25，不直接换成DWPose RGB | 采纳。已有训练输入是25通道热图；RGB骨架不是显式3D。全身更密集关键点可以作为增量，不必替换已训练Body分支。 |
| 去掉空间来回采样、直接使用512维reasoner特征 | 采纳方向，但旧高分辨率支路不能丢。现有bridge先融合高低分辨率，再压到128；新路径可复用前半段的512维融合。 |
| 1536→4096线性投影说明“不适配” | 修正。跨维投影是正常做法，低秩也不必然有害。需要验证的是特征分布、生成状态反馈、条件有效性和注入强度，不能仅凭维度断言根因。 |
| 新模块读取Qwen当前层hidden | 采纳为主要假设。它改变控制模块的信息来源，值得与无实时hidden但容量相同的模块对照。 |
| Depth固定升级为7通道 | 暂缓。梯度/边缘是派生信息，置信度需真实来源；逆深度不能从中心化后可正可负的depth求倒数。先做depth+valid最小对照。 |
| 四模态逐token Softmax 必然优于concat | 不成立。旧concat后有卷积与非线性，并非不能学习融合；Softmax还会让互补信息竞争。先保留旧融合，加独立Depth增量。 |
| Depth-Normal联合编码和部位前后关系 | 有价值，但几何关系需要相同相机、坐标系、可见性和可靠原始depth；不能先堆网络再补这些约定。 |
| 由输入geometry附近的辅助head预测depth即可保证控制 | 不成立。head可能直接重建输入。只能证明表征可读，不能证明最终RGB遵循几何。 |
| 使用A体型+B姿态渲染条件 | 对重定向推理合理；训练中不能直接继续把原B的RGB当严格像素GT。 |
| 在Pro内修改并保留Union | 不采纳。与本次用户选定方向冲突。独立几何工具可复用，但不引入Union模型依赖。 |

## 3. 当前磁盘证据

远端根 `/home/shangguanrz/project/pic-edit`，HEAD `2ebd026b8568737d2f0e13c4ebea6a0634c54fff`，本次读前和读后 `git status --short` 均为空。以下行号来自实际远端文件；前端相关文件与本地对应源码已核对SHA，iPER dataset两侧不同，因此以远端行号为准。

| 代码位置（相对远端根） | 实际行为与影响 |
|---|---|
| `src/pose_control/qwen21_full/geometry_encoder.py:127–151` | Normal/Pose/Part各输出128，concat384→256；启用Depth后fusion变成512→256，直接改变checkpoint形状。 |
| 同文件 `:167–183,186–236` | 条件先乘human mask；缺depth可变全零；归一化按场景min-max到[-1,1]，没有在此恢复物理尺度。缺失有效性与数值0要分开。 |
| `reasoning/person_geometry.py:27–40,65–76` | 256→512局部推理，高分辨率再降采样到低分辨率，已有global/task cross-attention。不是需要从零重建的纯CNN。 |
| `condition_bridge.py:31–55` | `geometry_feature + geometry_high_downsample(geometry_highres)`仍为512维，然后才压128；前面的高分辨率融合权重可以继承。 |
| `deepgen_adapter.py:429–451` | geometry被插值到legacy网格，与source latent拼144通道，再交旧patch embedding。条件画布与目标画布相同时存在1/16→1/8→1/16路径；不同画布必须以实际变换核实。 |
| `qwen_adapter.py:124–175,318–340,578–623` | latent64转legacy16；控制残差由独立控制核产生；6组输出重复写32层的target尾段。输出桥根据继承head状态避免两道全零映射。 |
| `control_core.py:374–401` | 旧CrossNorm对齐的是旧控制核自己嵌入的noisy target，非当前Qwen层hidden。已有CrossNorm不等于已解决两种网络的语义对齐。 |
| `src/data/iper_dataset.py:183–204` | 优先灰度`depth_vis_512`；无图时读npy并min-max；再无则全零。不能将该字段整体视作原始米制Z。 |
| `src/data/tiktok_dataset.py:750–753,852` | 读`target_depth_vis`，缺失填零；未在这些路径保存米制depth契约。 |
| `src/controls/generator.py:204–214` | 注释有“近0远1”，实际输出`1-(Z-Zmin)/range`，近大远小。这证明代码中有不同表示，尚未证明每份既有资产都来自此函数。 |
| `src/pose_control/qwen21_pro/geometry.py:397–436,958–1025` | 已有独立的体型重定向、raw `depth_m`、camera CV normal、owner与相机元数据；参数改动会拒绝原RGB作为严格GT。可借用几何工具，不用Pro网络。 |

现有默认评测入口 `scripts/eval/eval_iper_qwen21_full.py:35–36` 指向 `/home/shangguanrz/nihaochen/qwen21_full/body_512_100/checkpoint_last.pt` 和 `face_512_100/checkpoint_last.pt`；本次已确认文件存在，未加载或重新哈希GB级权重。后续运行应锁定实际使用checkpoint的SHA，不把“现有Full”泛化为每一个历史训练版本。

另一个已有运行问题：`execution.py:30–31`的DeepGen初始继承旧路径不存在；当前评测仍会先构造旧adapter再load Full。实现前应明确构造/恢复入口，但本次不借研究任务修改旧代码。

## 4. 论文与官方实现：借方法，不搬整套模型

下表均核对原论文、作者项目页或官方仓库。对本项目的使用方式是推导，论文没有直接验证这里提出的Qwen组合。

| 来源 | 可借鉴的具体方法 | 适用边界 |
|---|---|---|
| [T2I-Adapter](https://arxiv.org/abs/2302.08453) | 小型条件编码器适配冻结生成器，复用已有生成能力 | 原实验主要是latent diffusion/U-Net；不直接保证Qwen单流控制有效。 |
| [ControlNet-XS](https://vislearn.github.io/ControlNet-XS/) | 让控制模块收到生成主干的中间反馈，再及时修正主干 | 最贴近实时hidden方向；论文实验为SD系列U-Net，本项目需重新设计DiT接口。 |
| [ControlNeXt](https://arxiv.org/html/2408.06070v3) | 轻量卷积控制与特征分布对齐；观察控制特征/主干特征的尺度关系 | 它也选择性微调主干，不能引用为完全冻结Qwen方案的直接成功证据。第一版保留单一zero输出边界，CN仅作后续消融。 |
| [CHAMP](https://arxiv.org/html/2403.14781v2) | 同一人体模型渲染Depth/Normal/语义/Skeleton，先做shape alignment，再用轻量多路编码 | 多条件最后相加也是其实现的一部分，所以“concat或相加必然不行”没有论文依据。Full已存在CHAMP stem可选包装，默认仍是native stem。 |
| [Ctrl-Adapter](https://arxiv.org/html/2404.09967v2) | 不同骨干的特征适配、DiT等尺度层映射、patch级多条件融合 | 原方案依赖预训练ControlNet；这里只参考融合/对齐。其latent-skipping结果也提醒：加入生成状态可能诱发忽略控制条件，必须做条件替换测试。 |
| [OminiControl](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_OminiControl_Minimal_and_Universal_Control_for_Diffusion_Transformer_ICCV_2025_paper.html) | 利用原生DiT处理条件、明确空间对齐与非对齐的位置信息 | 作者实现主要在FLUX，使用条件token和LoRA；不是本提案的target残差模块。向Qwen插token会触及block-causal布局及额外计算。 |
| [Metric3D v2](https://arxiv.org/html/2404.15506v4)、[GeoWizard](https://arxiv.org/html/2403.12013v1) | 相机与深度尺度建模；Depth/Normal互补和联合表达 | 任务是几何估计。这里借几何约定与融合思想，不加载完整估计模型。 |
| [ControlNet++](https://arxiv.org/abs/2404.07987) | 从生成图提取条件进行一致性反馈，使用单步去噪降低成本 | 需要可信、可微的提取器。不能把输入条件自洽检查或辅助重建head等同于生成图控制损失。 |

Qwen接口依据[官方仓库](https://github.com/QwenLM/Qwen-Image-2.1)和服务器已安装源码：32层单流、4096 hidden、64通道VAE、16倍压缩、图像块内部双向、序列block-causal、prefix使用t=0。保留这些约定，不把Qwen-Image早期双流或FLUX接口直接套用。

## 5. 推荐的 Full Native Body 接口

### 5.1 保留前端，明确移除Body中的哪段兼容路径

第一版直接复用已训练的normal/pose/part stems、原384→256 fusion、RoleFiLM、global/task tokens、512维reasoner与高分辨率downsample融合权重。先不开Depth或密集关节支路，确认仅换控制接口的收益与代价。

取出：

`G0 = geometry_feature + geometry_high_downsample(geometry_highres)`

G0仍为512维。按实际目标latent网格对齐后展平为 `[B,N,512]`。若条件画布和目标画布已一致，避免旧路径再升到legacy网格后重新patchify；若不一致，必须使用记录的画布变换，不能把任意相同尺寸当对齐。

Body新路径不再需要128维geometry projection、144通道condition拼接、legacy patch2/16通道去噪输入、SD3 recurrent block及1536→4096 Body桥。旧模块保留给legacy实现和迁移教师；不能通过改旧checkpoint键名假装原样兼容。

### 5.2 当前主干hidden驱动控制

设 `H_l` 为原生Qwen第l个block输出的目标token，N为目标latent格点数：

\[
Q_l=W_d\,\mathrm{Norm}(H_l),\qquad Q_l,G\in\mathbb R^{B\times N\times512}
\]

\[
F_l=\Phi(Q_l,G,e_l,e_\sigma),\qquad
H_l'=H_l+\lambda_l M_{valid}\odot W_{up,g(l)}F_l.
\]

`Phi`第一版使用轻量的逐位置融合、局部卷积和MLP，包含显式geometry项及geometry对query的调制；不再另造一个全网格N×N自注意力Transformer。层号与真实sigma分别编码，避免把layer stage和采样进度混为一谈。控制不应只学会从Q预测噪声而忽略G，需条件打乱/关闭对照。

Norm仅对新分支输入归一化，不改原生Qwen归一化或调制参数。共享down projection与融合模块、分组独立up projection；最终up初始化为0，上游正常初始化，固定控制强度初始1。不要再串一个初始全零乘法gate。

target-only表示只写目标图token，**不等于只写人体内部**。移走人物后原位置的背景也需修复。初版用有效画布mask屏蔽padding；Depth输入仍只在有效几何支持区，人体mask用于权重/条件而非无条件切断所有背景更新。

保留源图→Qwen3-VL/VAE原生双路径。旧Full控制核里直接拼source latent与appearance token的通路也会被替换，这是需要迁移验证的能力，不能宣称只保留几何前端便无损继承了源外观控制。首版不新建Pro式DINO大bank；如源外观回退，再独立比较继承旧source token编码或原生source latent小token池，不能同时更换身份体系掩盖问题。

### 5.3 层数是单独实验变量

为了不把“新接口效果”与“32次控制减成6次”混在一起，建议先保留32个注入位置、6组head共享，但每一层都重新读取自己的H_l计算增量；这与旧的6份静态组残差重复写入不同。它可隔离接口变化，代价是轻量模块仍执行32次。

随后比较稀疏6位置 `(0,6,11,16,22,27)` 和8位置 `(0,4,8,12,16,20,24,28)`，均为0起始block编号，只是初始实验设置，不是已知最优点。稀疏版需要匹配的训练/微调，不能直接把一份残差乘5声称等价32层。

一个共享4096×512 down与6个512×4096 up仅线性权重有 **14,680,064** 个元素；不含bias、fusion、编码前端、Face/Hand，也不是显存或速度测量。稀疏注入减计算而不必按同一比例减参数；总参数、峰值显存、延迟需实测。

### 5.4 Qwen运行约定

保留原生图文交错序列、四latent/image-slot展开、RoPE、time modulation、flow scheduler和CFG含义。目标mask来自本次真实token metadata，不能写死文本长度/图像数量。512图像是1024 tokens；1024图像是4096 tokens。4096个token与4096维hidden是两回事。

新的动态条件应显式进入block调用/gradient-checkpoint重算参数，不能依赖训练前向结束就清空的可变hook上下文。冻结Qwen参数仍需要保留反向图，才能训练控制器。正负CFG各自重建布局。

目标侧新增残差在理论上不改变block-causal prefix，因而值得保留prefix cache能力；首轮先禁用cache建立数值对照，再验证cached/uncached等价后开放。不能仅凭target-only就宣称现有实现自动支持缓存。

## 6. 编码与Depth改造

### 6.1 先稳定原有三路编码

保留Body25热图通道顺序、关键点定义、heatmap sigma及坐标变换；Normal保留原训练约定，Part保留14类映射，避免只拷权重却变输入分布。

后续增加手/脸/足关键点时，用独立增量编码或紧凑3D关节token，不替换Body25第一层。3D token要带拓扑、相机空间/根相对坐标、投影坐标及有效性；“3D DWPose PNG”只有投影，不能当显式Z。不能直接把25通道扩成全身通道后继续strict加载原权重。

### 6.2 Depth首先是数据契约

新条件至少记录：原始depth路径/单位、near方向、轴定义、相机类型与K/extrinsics、normal基、画布变换、人物顺序、valid、来源、归一化中心与尺度。confidence有可靠拟合指标才填写，未提供不能伪装成测得的概率。

优先使用同一个SMPL-X/SMPL网格、同一相机渲染的raw camera-Z与normal/part/visibility；家族、拓扑不匹配必须拒绝。可复用现有几何渲染工具，不需要Pro网络或Union模型。只有灰度可视化图时，可建立明确的legacy ordinal profile，但不能反推米制Z，也不能执行需要真实K/Z的损失。

建议保存raw Z，另构造稳定相对表示 `(Z-z_center)/s`。优先采用记录在样本/序列契约中的统一场景中心与尺度；也可以用稳健分位差，但必须记录，避免每帧尺度跳变。双人共享尺度与中心，不各自min-max后比较前后。超出范围的处理和裁剪要固定，不能随机改。

若做逆深度，仅对有效正的raw Z求倒数后单独规范化；不对已中心化的相对depth或灰度近远图求倒数。它与depth有确定关系，不先假设两者一起必然更好。

### 6.3 Depth支路分两步做

**最小版本**：输入 `[relative_depth, valid]`，经过局部CNN，直接输出目标latent网格的512维增量。旧三路fusion仍为384→256，Depth通过独立支路与G0融合，保留旧权重形状。

**增强版本**：在同一条件/预算下增加有效邻域上的dx、dy、discontinuity，以及有来源的quality/visibility；再比较与Normal特征的联合局部编码。不固定“必须7通道”，用对照决定派生通道是否值得。

梯度不能跨背景0做差；平滑表面normal估计还需避开遮挡边界。遮挡边界应单独编码，不能因为避开差分就把它删除。缩到1/16网格时采用mask-aware聚合，记录覆盖率/可见归属，避免前景与零背景、前后两层直接平均出虚假表面；具体nearest/frontmost与masked-average选择也应作为固定codec并用边界样例验证。Normal缩放后重新归一化，Part/owner不做普通连续插值。

Depth不可用时必须走确切的“无Depth”路径，不能仅靠输入全零，因为CNN bias/归一化仍可能产生输出。新接口与Depth同时从零训练时，只保留最终主干输出一道全零边界；若在已训练接口上后加Depth零输出支路，应单独检查两次更新后的上游梯度。

### 6.4 融合与前后关系

先用残差式补充 `G=G0+P_D(F_D)`，保住旧融合。随后可比较由Q、G、depth-valid决定的逐token sigmoid增益；不强制Pose、Depth、Normal、Part的权重和为1，避免互补模态相互排斥。gate图只能辅助诊断，不能当作因果解释或“已理解几何”的证明。

对交叉手臂、手遮躯干、双人遮挡，单张可见Z-buffer看不到被遮挡层。可从同一三维网格按部位单独渲染/投影，**只在投影重叠、深度差可靠的局部区域**建立前后关系，编码关系端点、局部支持和置信度。整条前臂平均Z小于躯干平均Z并不能证明接触处手臂在前；没有重叠也不能当遮挡关系。初版不创建全身部位两两N²关系表。

已知透视K时，可通过 `P=Z K^{-1}[u,v,1]^T` 导出normal；正交/弱透视模型要使用对应公式。先把Depth–Normal一致性当数据QA；不是对输入计算一个loss就能训练生成器。

### 6.5 A体型+B姿态和训练GT

跨人重定向推理采用 `Render(beta_A, theta_B, camera_target)`，同时渲染Normal/Part/Depth；相机、根平移、地面接触和画布对齐仍需核对，不是换beta即可保证正确。

训练最初使用具有配对RGB的同身份条件，并保持和GT对应的几何。跨身份换beta后原B图通常不是新几何的真值，不能继续使用其全图像素/flow监督。可另建可靠合成配对或几何/身份弱监督任务，但与配对训练分开记录。现有`require_exact_gt_pairing()`原则可复用。

## 7. 保住Full能力的迁移和验证顺序

1. 锁现有Full Body基准：实际checkpoint/代码SHA、输入、prompt、source refs、分辨率、sigma schedule、seed、步数。先复现完整采样，确认我们比较的正是用户现有版本。
2. 新接口先只接原Body25/Normal/Part前端。Qwen冻结，前端先冻结，新接口训练；再用较小LR解冻前端。继承清单只包含shape/语义一致的模块，旧SD3权重不强行投影成Qwen权重。
3. 可用冻结旧Full对相同noisy latent/sigma/条件输出的velocity作早期蒸馏，辅助保持行为；主目标仍为配对flow。蒸馏只用于教师可信的同域样例并逐渐减弱，不逼新Depth方案复制旧模型的遮挡错误。教师是训练工具，不是新推理依赖。
4. 在接口已经可控后加入Depth+valid，再逐项加入边缘/Normal融合、局部排序或密集关键点。不把数个改变捆成一次实验。
5. 如新接口有稳定条件梯度但冻结主干明显限制效果，再独立研究少量Qwen LoRA；这是备用实验，首轮不同时改主干。

建议的可归因对照：

| 编号 | 改变 | 要回答的问题 |
|---|---|---|
| E0 | 当前Full，配置锁定 | 原有控制/外观基准 |
| E1 | Full前端＋新接口，保持32位置/6组head，Depth关 | 旧控制核到实时接口的迁移能否保住效果 |
| E1-static | 与E1同容量，query换成可学习layer/time输入，无实时H | 收益是否确实来自主干反馈；不是只比较参数量不同的模块 |
| E2 | E1改为6或8位置，匹配训练 | 稀疏注入的质量/成本权衡 |
| E3 | 选定E1/E2＋Depth/valid | Depth本身是否有增量 |
| E4 | E3＋梯度/边缘，再独立比较DN融合 | 空间增强和Normal互补是否有效 |
| E5 | E4＋局部遮挡顺序token | 对交叉肢体是否改善 |
| E6 | 单独新增密集关节分支 | Body之外的手/脸/足信息是否有效 |

另外用“旧Full＋兼容Depth支路”作为旁路对照，可分离Depth收益与接口重构收益；它不是重新走Union。相同update数外还记录样本数、训练时长和有效监督覆盖，避免轻量方案与大模型只比较步数。

训练损失初版保留原flow定义与已验证区域权重，不同时加入五种新辅助loss。Pose/Depth图像一致性后续参考ControlNet++：从预测x0/生成图重新提取，用可微冻结网络让梯度到Adapter；ONNX/argmax检测和`no_grad`提取不能冒充可微监督。几何辅助head可作诊断，但不足以证明RGB控制。

## 8. 未来实施验收（本次未执行）

- 新Body分支关闭/零输出时恢复同配置native Qwen；若保留旧Face/Hand则恢复同样局部分支基准。不能声称零新模块自动等于已训练Full。
- target mask在不同参考数量、宽高、CFG文本长度时正确；prefix没有新增写入；padding、invalid depth不生效。
- 检查first step输出head梯度，更新后前端/融合梯度；启用checkpoint重算时与非checkpoint结果对齐，不串样本条件。
- Depth符号、单位、正交/透视相机、重投影/letterbox、Part/Normal方向一致；缺失Depth和“有效但相对值0”严格区分。
- 深度打乱、反向、关闭，以及Pose/source替换时观察响应；若关闭与替换都不改变结果，不能把loss下降视作学会条件。反向depth为压力测试，不作为自然分布质量评价。
- 固定验证集完整多步出图：Body PCK/NME、可见关节漏检、源脸ID及覆盖率、服饰、体型、背景、手臂交叉前后关系、伪影；指标口径与人工盲审并行。MPJPE仅在确有可信3D估计/标注时报告。
- 判别器/姿态检测器可能对画出的骨架给高分；检查结果是否出现骨架纹理，避免用自洽指标替代图像质量。
- 对旧Full允许的非劣界限应在正式比较前依据基线多seed波动固定，而非看到新结果后改阈值。CPU单测、数值等价、训练loss和图像效果分别验收。
- 版本化条件codec、checkpoint及缓存；旧checkpoint→新架构是warm-start，不是exact resume。新run使用新name，不覆盖既有产物。

## 9. 落到仓库的候选边界

以下均为未来实现建议，不表示文件已存在：

| 位置 | 处理 |
|---|---|
| `src/pose_control/qwen21_full/geometry_encoder.py`、`reasoning/` | 继承当前前端，避免改旧默认行为；以显式新模式输出原生网格特征 |
| 新`qwen21_full/native_geometry.py` | 512维特征出口与空间/有效性契约，复用高低尺度融合 |
| 新`qwen21_full/native_body.py`、`native_integration.py` | 主干实时反馈模块、target写入、显式checkpoint执行上下文 |
| 新`qwen21_full/depth_features.py` | 原始深度codec、有效差分、边缘及DN增量编码 |
| 数据层/几何准备工具 | 提供raw depth/相机/画布/有效性；不覆盖旧可视化depth资产 |
| `execution.py`与评测入口 | 显式分派legacy/new架构、权重继承清单与新版本checkpoint；旧模型继续可复现 |
| `tests/` | 新接口、几何契约、条件响应、梯度及恢复测试；通过后才连接真实训练 |

Pro中可复用的是独立的几何数学、有效性处理和运行边界经验；无需加载Pro checkpoint或Union权重。旧 `union-adapter/fulladapter` 是参考快照，不能在那里改代码后期待正式入口改变。

最终建议是先实现**保留Full几何前端、读取Qwen主干当前hidden的Body接口**，再用独立Depth增量改善遮挡。现有证据支持这是一条值得验证的路线，没有证据能承诺512维、6个注入点或7个Depth通道一定比原Full更好。

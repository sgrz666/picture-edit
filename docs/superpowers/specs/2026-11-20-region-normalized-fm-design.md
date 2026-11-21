# Region-Normalized FM v1：身体、关节、逐手监督设计

本次先完成新目标的定义和可检查参考，再开展「无空间加权 vs 新区域加权」训练对照。用户优先身体和手部姿态，允许背景小幅退化；不使用 subagent。本文件不是已接入 Trainer 的交付声明。

## 1. 目标范围

保留当前 RF 概率路径、Qwen 输入、模型结构和条件资格。只改变 Target velocity 的监督分配。`z_sigma=(1-sigma)*z_GT+sigma*noise`，`v*=noise-z_GT`，`e_n=mean_c((v_pred-v*)²)`。第一版不增加 Contact、ID、Perceptual、sigma 调度或新解冻策略；已有辅助项在空间目标外独立配置。

本版用于现有同场景可信配对。跨背景且要求保留 Source 场景的任务，需要另设一致的目标/路径协议，不能直接套本预设来监督整张 Target GT 背景。

提供两个明确模式：

- `uniform`：全部有效 Target Token 等权，padding=0，数值和梯度复现旧 `native_velocity_loss`（当前 B1）。
- `region_balanced_v1`：Global / Person / Joint / Hands / Background 五类目标。它是新的主训练目标，**替换**旧全图/像素空间 velocity 目标，不是默认在旧目标上追加第二份 FM。

未来 U/R 对照只有本模式及其空间系数不同；架构、数据、区域缓存、分支开关、初始化和训练日程相同。旧 cap=2.5 仅保留为可选额外消融，不再作为用户要求的主基线。

## 2. 当前实现缺口

`native_v5/spatial_masks.py::regional_maps` 对两只手执行 `max`，旧 NPZ 仅保存 Body/Joint/Hand 合并图。合并结果不能恢复左右手，不能用连通域猜人物/左右归属。

`NativeTrainingSupervisionV5.weights` 已有 `[Face A, Face B, A-left, A-right, B-left, B-right]` 六槽，但依赖旧严格资格，与 v2 粗可见手区不是同一覆盖范围。

V7 已有逐 Person/Contact 均值，却没有逐 Hand 目标；其附加区域项当前除以 `counts['velocity']`。新模式启用时，原空间 FM、旧 Face/Hand velocity，以及 V7 `person_fm/contact_fm` 必须显式为 0，否则配置拒绝。ID/Perceptual/对比项由外层处理，不由新模式偷偷开启。

## 3. 区域监督 sidecar

新增 `native_region_supervision_v1`，保留旧 index/NPZ。每个 Target frame + Canvas 一份 sidecar，不重复存储模型权重或 Qwen features。通过稳定人物 ID、Target RGB SHA、Canvas 指纹绑定；同帧相同 Canvas 的多个 pair 可复用。它只用于训练/评价，不能拼进推理条件。

像素支持必须在真实 Target Canvas 上：

| 分量 | 槽位 | 来源和定义 |
|---|---|---|
| valid | `[1,H,W]` | CanvasSpec 的原有效覆盖，不新增几何裁剪 |
| person | `[2,H,W]` | GT 可见人体支持；单人只启用槽 0，双人按稳定 ID。已有可见几何 owner 可作明确标记的替代来源，不能用完整投影覆盖被遮挡者 |
| joint | `[2,H,W]` | 各人物身体关节邻域，保留人物槽。第一版沿用现有 `BODY_JOINTS=(2..13)`，不误称为 17 点监督；PCK 仍可评估 Body17 |
| hand | `[4,H,W]` | 固定顺序 `A-left,A-right,B-left,B-right`，每只手独立保存，禁止由合并图反推 |
| background | `[1,H,W]` | `valid - max(person)`，在像素空间计算，再单独池化；不是 Source 人体原位置必须保持不变的声明 |

metadata 至少记录：schema、Target RGB SHA、Canvas metadata/fingerprint、原始标注 SHA、person_ids、hand_slot_order、区域生成版本、每 ROI 来源/拒绝原因、质量系数、软面积、实际 Token grid。未配对 RGB 或 counterfactual geometry 无合法 GT 时拒绝 FM 监督。

左右手指人物的解剖侧别，必须沿原始 keypoint/SMPL-X 标注的显式映射；A/B 是稳定人物 ID，不按图像左右或 bbox 排序重建身份。

### 来源与可靠性

- Person：合格可见 mask 的 `q=1`，缺失/不可信为 0。由来源与资格判定，不用网络预测的 mask 调节自身 loss。
- Joint：只用可见、分数不低于 .3 的现有身体点；Gaussian 尺寸沿用现有 v2 的人物尺度规则（8–24 Canvas px）。shape 使用单位峰值，`q` 为本人物接纳点分数均值；没有接纳点时 q=0。保留每点分数与数量，q 只是工程信任系数，不解释为校准概率。
- Hand：优先合格且可见的旧 fine 区域，q=1；否则使用现有可见粗手策略，q=.5。明确遮挡、出画、不存在或归属未知时 q=0。fine 和 coarse 不同时算成两只手。
- 旧粗手生成器包含 `.65` 幅度。本 sidecar 复用其形状时，只在生成版本、gain metadata 一致且已核对时除去该已知幅度，再使用新的 q；不能把 `.65` 当作检测正确率，也不能对任意 mask 按峰值归一化。
- 粗手区允许 RGB/velocity 区域监督，但不制造精细手指坐标或把无效 Hand 条件改成有效。

**所有裁剪/归属交集先在像素空间完成。** 原本已带 coverage 的支持采用集合相交语义，避免反复相乘；随后各区域通过 area pooling 映射到实际 Target Token grid。不能以 `min(pool(A),pool(B))` 冒充 `pool(A∩B)`：同一 Token 内左右两个完全不相交的半区，前者仍能产生 0.5 的虚假交集。

新 sidecar 的完整性失败即停止；某只手本来缺失则记录并跳过该项。额外区域资格不参与原训练 sampler 的 route/source/index 筛选，避免 loss 对照同时改变样本分布。

## 4. 每 ROI 的数学定义

对有效 ROI r，已池化支持为 `m_r,n∈[0,coverage_n]`，`A_r=sum_n(m_r,n)`：

`ell_r = q_r * sum_n(m_r,n*e_n) / max(A_r,tau_r)`

- 支持和 q 均 detach，不允许 ROI 向模型反传梯度。
- A=0 或 q=0 时该 ROI 不进入有效计数，返回可微零。
- q 位于区域均值外面；不除以 `sum(q)`，避免把整体低可信程度约掉。
- tau 是完整 Token 的等效面积下限：Person=4、Joint=1、Hand=.5、Background=16。它们是第一版待检验常量，不是论文最优参数。
- A>=tau 时区域均值与面积无关；A<tau 时有意衰减，限制极小 ROI 的单 Token 梯度。使用软面积，不将若干极小非零格子当作足够支持。
- soft mask 只出现一次，即 `m*error²`，不是 `(m*error)²`；Token coverage 不再次相乘。

同一人物的 Hand/Joint 可以与 Person 重叠；这是显式细节增强。两个 Hand 的局部交叠也不新增槽数，Hand 家族总系数仍由一次家族均值控制。跨人物可见归属不明的像素排除出对应 ROI。

## 5. 每图及累积窗口的顺序

每个 micro-step 是 B1，但其 Token 数可因高宽比而不同。不能按整窗口 Token 面积合并样本。

1. 每图对有效 Person ROI 等权平均；Joint 按有效人物平均；Hands 按有效手槽平均；Background 单区域。
2. 每一类 F 在窗口内只对具有该类有效 ROI 的图求均值：`L_F=sum_i(mean_r(ell_i,r))/K_F`。
3. Global 继续按窗口内全部有效图平均。`K_F=0` 时对应家族为可微零，不填假区域，不把分母改为总 micro 数。

例如窗口有 4 张图，仅一张含有效手区，则 Hand 项除以 1；一图一只手与一图四只手仍先形成各自的图内 Hand 均值。q 是质量衰减，不充当计数权重。

**空家族的系数不自动转给背景或 Global，也不重归一化剩余系数。** 因此总目标尺度可随有效家族变化；记录 `K_F`、有效 ROI 数、q、A/tau 衰减、active coefficient sum 及裁剪比例。使用固定规则的梯度预检判断是否稳定，不在训练中自动提高 LR。

## 6. 第一版预设

| 家族 | uniform | region_balanced_v1 | 目的 |
|---|---:|---:|---|
| Global | 1.00 | .10 | 完整场景与旧目标锚定 |
| Person | 0 | .30 | 身体整体与两人体型尺度平衡 |
| Joint | 0 | .15 | 身体关节邻域，避免 Person 项主要关注服饰纹理 |
| Hands | 0 | .35 | 每只手独立监督 |
| Background | 0 | .10 | 以小面积无关系数约束背景 |

这组值延续已有研究起点。系数和为 1 不保证参数梯度相同；背景也会从 Global 项得到监督，不能称作只分到 10%。先冻结常量，不加 sigma/route 自适应：每个实际启用的训练 route 使用相同目标定义，缺失条件不被强制打开，按 route 记录各家族梯度和资格。

评估与日志始终额外报告相同定义的 unweighted Global/Person/Joint/Hand 原始误差。训练 objective 与共同评价指标使用不同字段，不能把新旧 objective 数值当作质量排序。

## 7. 接入位置与验收

后续正式实现建议分为以下职责，当前只交付设计与 CPU 参考：

- `src/pose_control/qwen21_full/region_objective.py`：共享纯 loss/窗口归一化；不加载模型、执行采样或 decode。
- 独立 sidecar builder + typed reader：绑定标注、人物 ID、Canvas、SHA，逐手生成及实际资格审计。
- V5/V6/V6.2/V7 config/backend 适配：模式选择、旧空间项互斥、窗口资格传递。默认不改变旧运行签名；开启新模式必须有新的配置指纹。
- 监测：共同无加权指标、各家族统计、固定参数子集梯度范数与余弦。不把不同模块范数直接当作梯度份额。

数值验收需覆盖：uniform 值与梯度精确回退；逐手面积变化；单/双人；不同 Token 数；部分 padding；像素交集；低置信衰减；极小/空 ROI；全窗口无手；一次手/四次手窗口；mask 无梯度；纯 loss 不改变 RNG；source/Canvas/人物错绑定拒绝。

CPU 参考只证明目标定义与这些数值性质。正式 data/grad preflight 还必须证明新 sidecar 在真实单人/双人样本上的覆盖、绑定和前反向。主干/条件/损失签名、半窗口恢复与优化器状态仍服从现有严格门禁，不以跳过门禁完成接入。

## 8. 与后续对照实验的连接

新设计冻结并完成正式接入后，主对照是 `uniform` vs `region_balanced_v1`。两臂都能读取同一 sidecar，只有 loss 模式不同，且不混入 V7 默认 Person/Contact 附加项。

若目标是“空间加权的总训练收益”，从空间加权训练前的共同初始化开始；已经加权训练的 S0_10k 只能用于检验续训增量。已核对原 V5 S0 配置引用的预空间初始化为 `native_v5_init_v2.pt`，SHA `f68a2a881e57a9537ae68d69b51c660fa503a3b48cbf9f6b4ad88f29fa259e45`；跨到另一架构须显式、共同迁移，不跨版本直接 resume。

阶段预算、三组配对训练 seed、共同评价口径和背景回退门槛在正式实验方案中另行冻结。当前不把旧 weighted-vs-bg035 结果冒充无加权对照，不启动长训。

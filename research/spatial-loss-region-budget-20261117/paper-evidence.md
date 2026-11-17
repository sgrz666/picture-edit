# 原论文核对与可迁移范围

核对日期：2026-11-17。以下为原论文/作者发布渠道；论文自己的实验收益只归属于其模型、数据与协议，不外推为本项目的收益。

| 论文与原文 | 已核对内容 | 对我们有用的部分与限制 |
|---|---|---|
| [Break-A-Scene, 2023](https://arxiv.org/abs/2305.16311)、[作者 PDF](https://omriavrahami.com/break-a-scene/static/paper/Break-A-Scene.pdf) | §3：masked reconstruction、cross-attention loss、union sampling | 区域重建之外仍需人物/概念归属约束；不提供四只手独立归一化的直接证据 |
| [MoA, 2024](https://arxiv.org/html/2404.11565v1#S3.SS3) | §3.3：前景重建 + 背景 prior router + object attention 项 | 前景优化依赖冻结 prior 与显式双路径路由，不等于冻结 Qwen 后任意 Adapter 都能删除背景约束 |
| [Edit Where You Mean, 2026, v2](https://arxiv.org/html/2604.23763v2#S3.SS5) | §3.5 式(7)：`sum(w*||error||²)/sum(w)`，`w=1+alpha*M`；alpha=2。附录 I 讨论更大的 alpha | 与现有 weighted FM 属于同一数学形式；不是独立 ROI 归一化的验证。来源基于 Qwen-Image-Edit，不能把其 alpha=2 当作 Qwen-Image-2.1 姿态任务最优参数 |
| [MaskFlow, 2026](https://arxiv.org/html/2608.06929v1#S4.SS1) | §4.1 式(5)：mask 同时进入 target/source 的概率路径；式(6) 中面积因子位于平方范数内部；§4.2 调整预测场和采样 | 能指导跨背景监督的一致性设计；不是替换一个 loss 就完成的方法。其面积因子平方后的效果与简单除以 ROI 面积不同 |
| [Foreground Focus, 2025](https://arxiv.org/html/2504.02180v1#S3.SS5) | §III-E 式(12)-(15)：`w=1/(alpha+r)`；alpha=.125。w 在误差平方范数内部 | 面积依赖权重有依据。按论文公式，w 的上界 8 对应平方误差系数上界 64；不能按 8 倍 MSE 来复述。任务是伪装图像生成，不是手部姿态迁移 |
| [HumanSD, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Ju_HumanSD_A_Native_Skeleton-Guided_Diffusion_Model_for_Human_Image_Generation_ICCV_2023_paper.pdf)、[作者实现](https://github.com/IDEA-Research/HumanSD) | §4.2、附录 A.4：解码误差的姿态热图用于加权；Table 2 单独 loss 消融 AP 30.63→32.66 | 比普通前景权重更接近“身体姿态精确”。其 heatmap 不是简单 GT 高斯关节图；原方法也改变 SD 训练方式，不能宣称只改 Qwen loss 可复现同等效果 |
| [DanceTogether, arXiv v1, 2025](https://arxiv.org/html/2505.18078v1#S3.SS3)、[作者项目](https://dancetog.github.io/) | §3.3 式(3) 对每个人使用 `(1+M_body+2*M_face)` 的加权重建并求和 | 逐人物区域监督是相关启发；式(3) 没有逐人物面积分母，不是 equal-person ROI mean 的原始公式。作者项目标注 ICLR 2026；本轮公式核对以可读取的 arXiv v1 为准 |
| [FastComposer, 2023](https://arxiv.org/abs/2305.10431) | 训练期 cross-attention localization 与延迟 subject conditioning | 与人物归属和身份混合相关，不直接解决全图分母中的手部面积稀释 |
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | 调节多任务权重以平衡共享参数梯度及学习速度 | 可作为区域多目标的后续权重方法；选择同一共享参数子集测量，先验证是否存在失衡，不默认运行到整个 7B 主干 |
| [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | 检测不同目标的负梯度余弦并作投影 | 用于持续的梯度方向冲突，和改变区域面积分母是两个问题；额外求梯度有计算与显存代价 |
| [Min-SNR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html) | 按 timestep/SNR 重加权以缓解不同时间步的优化冲突 | 支持研究 sigma 维度；不是空间权重，也不能直接把 DDPM 的参数化公式套到 RF velocity |

## 对来稿的四处具体修正

1. **Edit Where You Mean 的归一化不能省略。** 原式除以 `sum(w)`，与当前总权重归一化更接近；并非仅仅对未归一化 MSE 乘 `(1+alpha*M)`。来稿中的表达可作示意，不宜直接作为实现公式。
2. **权重放在平方内外不同。** Foreground Focus、MaskFlow、DanceTogether 的上述原式都要留意平方范数。对于二值 mask，`M²=M`；对于我们的软 mask，`M²` 会进一步削弱边界/低置信区域。实现必须明确采用一次 coverage 和哪一种软权重含义。
3. **逐人物加权不等于逐人物等权。** DanceTogether 式(3) 的每个人项仍受区域面积影响；V7 的逐人均值是本项目的改编。我们可以提出它，但不能把“独立归一化优于现有目标”写成论文已经为本项目证明的结论。
4. **背景不变也不等于训练 velocity 为 0。** 本项目的 noisy latent 来自 noise 与 GT 插值，即便 Source/GT 的背景一致，`noise-z_GT` 在背景一般也不为 0。降低背景监督的依据应是目标任务与区域分配，而不是把背景当成没有去噪目标。

## 原文与实现的验证边界

本轮读取了上述论文的相关方法段落和公式；HumanSD 额外检查了 loss 消融表，Edit Where You Mean 检查了 alpha 讨论。没有复现这些论文的训练，也未将其 benchmark 数字作为我们模型的效果。

DanceTogether 的 OpenReview PDF 链接在本轮返回浏览器验证页，因此没有绕过访问检查；相关公式使用可正常读取的作者 arXiv 版本。Foreground Focus 与 Edit Where You Mean 的本轮检索没有确认可用的官方训练代码，公式解读以原文为准，不声称完成实现级复现。

区域面积归一化、置信度必须位于区域均值外侧、逐手平均和有效窗口计数等建议，是针对当前代码与数据条件形成的设计。其数学性质见 `output/checked-v1/checks.json`，生成效果仍须同架构、同预算实验。

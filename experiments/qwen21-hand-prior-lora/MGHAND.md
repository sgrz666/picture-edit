# MGHanD 可复用组件

MGHanD 在 Stable Diffusion v1.4 的去噪过程中加入两类引导。视觉分支借用 StyleGAN-T 判别器结构，用 DINOv2 ViT-S/14 编码预测的干净 RGB，用 CLIP 编码文字并训练 real/fake 打分。文本分支用 Concept Sliders 在 SD v1.4 上训练 rank 4 LoRA。后段去噪再把两种修正限制到跨步累积的手部掩码。论文报告直接全局启用 LoRA 有时会模糊背景，去掉累积掩码也会出现颜色与物体细节变化。 [论文方法与消融](https://arxiv.org/html/2503.08133v1#S3)

| 组件 | 复用状态 |
| --- | --- |
| MGHanD 手部判别器 | 论文描述的是 RGB 与 prompt 接口，适合复用为 Qwen 的候选图打分器。查阅论文和 GitHub、Hugging Face 的公开项目索引后，未找到作者发布的判别头、专用训练集或 checkpoint。 |
| StyleGAN-T 判别器代码 | 已固定到 `sg:/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora/source/stylegan-t` 的 commit `36ab80ce76237fefe03e65e9b3161c040ae888e3`。`networks/discriminator.py` 展示冻结 DINO、多层 token 接判别头和文本投影，原实现用 DINO ViT-S/16。该仓库提供架构参考，作者文档说明没有发布预训练 checkpoint。 [官方实现](https://github.com/autonomousvision/stylegan-t/tree/36ab80ce76237fefe03e65e9b3161c040ae888e3) |
| DINOv2 ViT-S/14 | 已下载 Meta 的 `facebook/dinov2-small` revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` 到 `sg:/data/shangguanrz/LoRA/experiments/qwen21-hand-prior-lora/source/dinov2-small-ed25f3a31f01`。Safetensors SHA256 为 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`。真实 InterHand RGB 的 CPU 前向已得到有限的 384 维特征。 [Meta 模型卡](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) |
| MGHanD 文本 LoRA | 参数绑定 SD v1.4 的 UNet。Qwen 的 32 层单流 DiT 需在自身基座上训练对应的低秩方向。 [论文实现](https://arxiv.org/html/2503.08133v1#S4.SS2) |
| FoundHand / HandRefiner | `sg:/data/shangguanrz/models/FoundHand` 已有公开生成器、SAM 和 VAE 权重，可给手部图像扩增提供独立 RGB 候选。采用其合成图前须核对原始 InterHand lineage，避免与留出集重叠。HandRefiner 有公开代码与权重，可接在 Qwen 出图后做局部修复对照。 [FoundHand](https://github.com/arthurchen0518/FoundHand) · [HandRefiner](https://github.com/wenquanlu/HandRefiner) |

## 在 Qwen 上做方法对照

第一项迁移是对 Flow Matching 的 target velocity 残差做掩码。固定参考图、prompt、scheduler 和逐例 seed，在同一目标 latents 上计算基座速度 `v_base` 与 LoRA 速度 `v_lora`，使用 `v_base + λ(σ) · M_t · (v_lora - v_base)` 更新目标画布。累积掩码可从较可信的手部候选区域跨步取并集，限制最大覆盖面积并记录每步覆盖率；首个对照同时保留固定 mask，用来判断收益来自空间限制还是跨步累积。该公式在输出速度处融合，能直接用现在的官方全局 LoRA 权重。

第二项迁移是复用 DINOv2 骨干，先用同 Source、同 prompt 的 Qwen 候选做手部 RGB 排序。人工标注正常、少指、多指、粘连与遮挡不确定，把主体和源图留在同一 split。先看独立盲评的一致性，再比较候选重排与掩码引导；这能区分判别器质量和推理期梯度引导的效果。若之后加梯度引导，按当前训练约定从 `x_t - σ · v` 求预测的干净 latent，再经冻结 VAE 解码送入判别器。

基座、全局 LoRA、固定掩码 LoRA 与累积掩码 LoRA 用同一留出集比较。记录人工解剖错误率、SDPose 手势指标和掩码外的保真差异。全身图与局部手部图分别统计；论文的检测分数和用户偏好属于其 SD v1.4 设置。

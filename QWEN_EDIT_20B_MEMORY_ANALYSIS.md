# Qwen-Image-Edit 20B 在第三阶段姿态编辑中的显存与适配分析

## 1. 结论

如果将 DeepGen 2.47B 主干替换为官方 `Qwen/Qwen-Image-Edit-2511` 20B，在当前 RTX 6000D 84GB 服务器上，**单卡 BF16 原生推理和“冻结主干 + 条件缓存 + 外挂适配器 + 梯度检查点”训练都有较大概率可运行**。1024 分辨率的外挂适配器训练预计需要约 **52–62 GiB**，完整生命周期（包含条件生成和独立推理）预计峰值约 **60–72 GiB**。

这些是依据官方权重大小、公开的 1024 推理实测、Qwen-Image 训练框架数据及本项目 DeepGen 实测建立的区间估算，并不是本机实测值。在没有下载权重并运行相同测量脚本前，不能把区间上限写成确定峰值。

Qwen-Image-Edit-2511 有更强的人物一致性和多人图像编辑先验，作为效果基座比 DeepGen 更符合任务书中的 Qwen-Edit 路线。但它仍不原生接收 SMPL-X、骨架、深度、法线或 contact map，外挂控制器仍然需要训练。模型更大也不保证指定人物绑定和精确接触自动解决。

还有一个硬约束：官方 Qwen-Image-Edit 主干为 20B，明显超过项目当前“生成主干低于 6B”的要求。若该要求不可修改，20B 只能作为质量上界和教师/对照模型，不能成为最终交付主干。

## 2. “20B”实际包含什么

官方模型卡将 Qwen-Image-Edit-2511 标为 20B、BF16，并提供 `QwenImageEditPlusPipeline`。官方仓库把 Qwen-Image 描述为 20B MMDiT 图像基础模型。

根据官方 Hugging Face 仓库文件元数据：

| 组件 | BF16 文件体积 | 由文件体积折算的张量规模 |
|---|---:|---:|
| MMDiT transformer | 38.06 GiB | 约 20.43B |
| Qwen2.5-VL text encoder | 15.45 GiB | 约 8.29B |
| VAE | 0.24 GiB | 约 0.13B |
| 合计 | 53.74 GiB | 约 28.85B |

因此，“20B”主要是生成 Transformer，不是整套 pipeline 的总参数。完整仓库页面显示约 57.7GB 文件；加载到 GPU 后，仅 BF16 权重下限就约 53.74GiB，还要增加 CUDA context、临时工作区、输入图像 token、latent 和注意力激活。

Qwen Transformer 配置为 60 层、24 个注意力头、每头 128 维；Qwen2.5-VL 编码器为 28 层、hidden size 3584。当前 DeepGen DiT 为 24 层、hidden width 1536。原有 1536 维残差适配器权重不能直接加载到 Qwen；必须重新确定注入层和通道维度并重新训练。

来源：

- 官方模型卡：https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- 官方模型集合：https://huggingface.co/collections/Qwen/qwen-image
- 官方代码仓库：https://github.com/QwenLM/Qwen-Image
- Diffusers Qwen 编辑 pipeline：https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage

## 3. 显存估算

### 3.1 BF16 原生推理

vLLM-Omni 的 Qwen-Image-Edit 单卡基线在 H200 上对 1024×1024、50 步任务实测约 59.66GiB。Qwen-Image-Edit-2511 的主干形状和权重体积处于同一级别，但 Plus pipeline 支持多图输入；输入图片数量增加会增加视觉 token 和编辑条件开销。

| 模式 | 512 | 768 | 1024 | 当前 84GB 卡判断 |
|---|---:|---:|---:|---|
| 单图 BF16 推理 | 56–62 GiB | 58–64 GiB | 60–66 GiB | 预计可运行 |
| 双图 BF16 推理 | 59–66 GiB | 62–69 GiB | 64–72 GiB | 预计可运行，余量收窄 |

由于权重本身占主导，降低分辨率不能像小模型一样大幅降低总显存。不同框架、Flash/SDPA kernel、attention 实现和显存碎片可能造成数 GiB 差异。

公开的 1024 推理实测：https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image-Edit.md

### 3.2 冻结主干并训练外挂适配器

估算沿用本项目已验收口径：

- MMDiT、Qwen2.5-VL 和 VAE 冻结为 BF16；
- 外挂适配器参数、梯度和 AdamW 状态使用 FP32；
- micro-batch=1，梯度累积=1；
- 源图语义条件和 VAE latent 预缓存；
- 开启梯度检查点和高效注意力；
- 主干仍保留到适配器残差的梯度，不能整体 `no_grad()`。

当前 130.28M 适配器的 FP32 参数、梯度和 Adam 两组状态约占 1.94GiB。Qwen 的内部宽度和层数更大，适配器改造后预计需要预留约 2–3GiB 持久状态。

先用 DeepGen 同配置实测峰值，加上两套 DiT BF16 权重差约 33.45GiB，可以得到不计额外架构激活的下限：512 约 44.0GiB、768 约 46.9GiB、1024 约 51.5GiB。再结合 Qwen 60 层结构、编辑参考图 token 和适配器加宽，得到以下运行区间：

| 训练配置 | 512 | 768 | 1024 | 84GB 判断 |
|---|---:|---:|---:|---|
| 缓存 + 检查点 | 45–51 GiB | 48–56 GiB | 52–62 GiB | 推荐，预计可运行 |
| 在线编码 + 检查点 | 62–70 GiB | 66–77 GiB | 72–84+ GiB | 1024 有 OOM 风险 |
| 缓存 + 无检查点 | 60–72 GiB | 72–88 GiB | 85–110 GiB | 768 起风险高，1024 不建议 |
| 在线 + 无检查点 | 75 GiB 以上 | 90 GiB 以上 | 110 GiB 以上 | 单卡不建议 |

Musubi Tuner 对 Qwen-Image 1024、batch=1、BF16、梯度检查点和 xFormers 给出的 LoRA 训练参考值约为 42GB；Qwen-Image-Edit 因控制图需要额外显存。该数字支持缓存加检查点区间的量级判断，但它是 LoRA/T2I 训练数据，不能替代本项目外挂适配器实测。

训练参考：https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md

### 3.3 量化与 CPU offload

FP8 冻结主干可以把 20B DiT 权重从约 38GiB 降到约 19GiB。Musubi Tuner 的公开参考中，1024 LoRA 训练从 BF16 约 42GB 降至 FP8 约 30GB；交换 16 个 block 后约 24GB，交换 45 个 block 后约 12GB。Qwen-Image-Edit 仍会因参考/控制图增加开销。

对本项目建议先做 BF16 基准，再单独测 FP8。FP8 和 block swap 会改变速度、CPU 内存占用及数值误差，不能把低显存结果与 BF16 质量结论混在一起。Diffusers 官方也支持模型 CPU offload 和顺序 CPU offload，但后者可能很慢。

当前服务器有约 125GiB 系统内存、112GiB 可用内存和约 1.2TiB 可用磁盘，足够下载 57.7GB 模型并执行普通缓存实验。大量 block swap 或并行多进程时仍需控制主存峰值。

内存优化说明：https://huggingface.co/docs/diffusers/en/optimization/memory

## 4. 对编辑效果的预期

Qwen-Image-Edit-2511 官方说明包含减少图像漂移、提高人物一致性、改善多人一致性和增强几何推理。它更适合复测目前 DeepGen 暴露的衣服变化、人物新增和身份替换问题。

合理预期是：

1. 不变编辑、配饰和服装保持有机会明显优于 DeepGen。
2. 多人输入及人物融合能力更强，错误新增人物可能减少。
3. 对自然语言中“谁和谁交互”的理解更好，但多人单图中的实例绑定仍需实际验证。
4. SMPL-X、contact map 和目标骨架仍需外挂控制器。官方原生 pipeline 没有这些结构条件接口。
5. 20B 主干更深、更宽；六个 DeepGen 残差头直接平移可能控制不足，需要比较 6/12 个注入层或分层缩放。

官方 2511 模型卡：https://huggingface.co/Qwen/Qwen-Image-Edit-2511

## 5. 与当前项目约束的关系

| 项目要求 | Qwen-Image-Edit-2511 判断 |
|---|---|
| Qwen-Edit 类模型 | 完全符合 |
| 生成主干低于 6B | 不符合，MMDiT 为 20B |
| 不微调生成主干 | 可以，冻结主干训练外挂适配器 |
| 支持人物编辑 | 原生支持，官方强调一致性改进 |
| 支持 SMPL-X/contact 控制 | 不原生支持，需要新适配器 |
| 单张 RTX 6000D 84GB | BF16 推理与缓存训练预计可行 |

如果低于 6B 是华为项目的硬性验收条件，建议把 20B 定位为：

- 质量上界和教师模型；
- 用于生成或筛选姿态编辑伪配对数据；
- 与 DeepGen/小型蒸馏模型做同任务效果对照；
- 用于判断失败来自小模型容量还是控制器/数据问题。

如果允许取消 6B 限制，2511 可以成为主实验基座，但工程成本和每次试验时间都会明显增加。

## 6. 建议的本机实测顺序

下载模型后应复用现有 50ms NVML、PyTorch allocated/reserved 和阶段计时，不直接把估算区间写入最终报告。

第一批只做原生推理：

- 512/768/1024；
- S0、S2、D0、D1 四个任务；
- 种子 42；
- 单图输入；
- 1024 追加一次双图输入；
- 每个配置独立进程。

第二批修复适配器兼容性后做训练冒烟：

- 统计新适配器参数量；
- 512 缓存 + 检查点执行 3 次实际更新；
- 验收冻结哈希、零初始化等价性、六/十二个注入点梯度和三次更新后的主体梯度；
- 通过后再运行 768、1024 的 5 次预热 + 20 次测量；
- 最后对最低显存有效配置运行 100 次更新。

停止条件：1024 峰值超过 78GiB 时停止扩大 token 数和图片数，先测 FP8 冻结主干或 block swap；OOM 不通过缩短语义序列、减少参考图或改变训练精度伪装为可行。

## 7. 最终建议

当前 84GB 服务器具备开展 Qwen-Image-Edit-2511 单卡实验的硬件条件。若目标是尽快得到真实显存值，优先级应为：**BF16 单图推理 → 缓存条件 → 新适配器 512 冒烟 → 768/1024 显存矩阵**。预计正式显存结论会落在 1024 缓存训练 52–62GiB、完整生命周期 60–72GiB附近。

但在项目决策上，先确认“低于 6B”能否为对照实验放宽。若不能放宽，不应投入完整 20B 效果训练；应把它用于质量上界、教师数据和基线对照，并继续寻找 Qwen-Edit 蒸馏或更小的可控编辑主干。

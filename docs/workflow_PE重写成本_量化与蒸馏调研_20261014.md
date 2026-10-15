# PE-I2I 重写成本、量化与蒸馏可行性调研（2026-10-14）

对象：`sg:/home/shangguanrz/project/pic-edit/workflow`（单人）的 `prompt` 阶段，
模型 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I`。

问题：每来一对新图就要跑一次 PE 重写提示词，成本高。想法两条——**量化 I2I**，或**把它蒸馏进"我们的 qwen vl 3"并给它加一条任务分支**（一边理解图一边写出适合编辑的提示词）。
本文给：成本到底在哪、两条路的实测难度、社区方案对照、建议路线。

**标注约定**：`[实测]` = 本文当天在 sg 盘上量出来/读出来的，路径可查；`[推断]` = 由架构与算式推出、还没跑；`[未证实]` = 依赖社区/外部信息，本机没有条件验证。**没跑过的实验一律写成"未做"，不当结论用。**

---

## 0. 一句话结论

- **PE 就是单人流水线的时间瓶颈**：一次完整 run 里 PE 中位 **242.1 s**，出图中位 **47.6 s**，几何 **31.5 s**——**PE 占三段合计的 75.4%** `[实测]`。
- **成本不在模型大小，在 token 数**：一次调用生成 **4479 token、243.7 s**，最后真正给生成器用的只有 **310 词** `[实测]`。≈ 93% 的输出是 PE 自己的推理脚手架。
- **量化的运行时支持是有的，但收益被架构卡住** `[实测 + 社区主源]`：vLLM / SGLang / TensorRT-LLM 都已原生支持 Qwen3-Next 系的 Gated DeltaNet 混合架构（vLLM 直接吃 Flash-Linear-Attention 的 Triton 内核），**`qwen3_5` 不是"没工具认它"**。
  真正的坑是：**官方 llm-compressor 的 Qwen3.5 recipe 把 `linear_attn` 和 `visual` 整个排除在量化之外**（ignore list = `lm_head + visual + linear_attn`），而 32 层里 24 层是线性注意力 → **名义 4×/2× 的压缩实际落到权重上远小于这个数**。
  递归状态（`mamba_ssm_dtype: float32`）**今天没有任何 off-the-shelf 量化支持**（vLLM 的 `--mamba-cache-dtype` 只允许 fp32/fp16/bf16）。
  社区同尺寸实测天花板：Qwen3-8B/H20 上 **FP8 1.49×、AWQ-4bit 1.29×** decode。而我们现在 18.4 tok/s，远低于同尺寸应有水平——**差的是内核，不是位宽**。
- **蒸馏的正确目标是"压 token"，不是"换小模型"**：我们的 `text_encoder` 是 **8.8B 的 Qwen3-VL**，PE-I2I 是 **9.4B 的 Qwen3.5-VL**——**参数量几乎同级** `[实测]`。所以蒸馏的收益不来自尺寸，而来自 (a) 学生不写 4.5k 推理 token、(b) 换到生态成熟的 `qwen3_vl` 架构（量化/服务化全都支持）。
- **用户提的"任务分支"有一个安全实现形状**：同一个 `text_encoder` 权重 + 一条 **PE LoRA**，PE 时 `enable_adapter`、条件编码前 `disable_adapters()`。LoRA 关闭时主干数学上不变，**可一次性硬校验**（`prompt_embeds` 逐位一致）。全量微调条件编码器则不可取。
- **先做不要钱的三件事**：把 PE 变成**常驻/可并发**的服务（省的是每阶段重起进程与串行等待，冷加载本身只有 5.8–8.5 s；**同窗实测 K=3 只有 ≈1.36×**，§5-T0）、装 `fla` 内核（**要先解决 fp32 状态与 FLA 要求 bf16 的冲突**，§1.4）、以及**离线批量时按输入哈希复用已算好的提示词**（缓存机制已经在了）。
  **"关 thinking"这条社区首选在本机被实测否定**（§1.3：token 数不降，还多出 JSON 契约违约），别按社区默认去做。
- **还有一个前提没被证明**：同一份几何下**手写提示词 body PCK 0.667 > PE 版 0.389** `[实测，见 workflow README]`。PE 值得便宜，但"PE 比手写强"这件事本身还没在成对对照上立住。

---

## 1. 现状与成本解剖 `[实测]`

### 1.1 四阶段耗时（单人线，`workflow/runs/**/*/stage.json` 的 `metadata.seconds`）

| 阶段 | 样本数 | 中位秒 | 均值 | 最大 | 备注 |
| --- | --- | --- | --- | --- | --- |
| `geometry`（SMPLest-X + DWPose） | 26 | 31.5 | 33.2 | 44.3 | 走 `smplestx_bw` 环境 |
| `prompt`（**PE-I2I**） | 24 | **242.1** | 251.5 | 400.7 | 走 `qwen21` 环境 |
| `generation`（Qwen-Image-2.1，768/30 步） | 29 | 47.6 | 53.4 | 100.0 | 峰值 33.97 GiB |
| `evaluation` | 18 | 30.4 | 31.6 | 37.3 | `evaluate.py`/`evaluate2.py` |

PE 占前三段合计 **75.4%**；含评测占整条 run **68.9%**。
双人线同样吃 PE：`workflow-multiple/runs/**/prompt/stage.json` 已记 **87 次**调用、中位 **200.3 s**（累计 ≈ 4.8 GPU·h）。
好处是矩阵重试用**缓存**跳过 PE（只重付生成 2–8 分钟），但**新格、改代码、换 seed 触发上游失效时都要重付 242 s**。

### 1.2 一次 PE 调用的构成（run `wp1_accept_hard02`，A 640×928 / B 855×1280 / 骨架 640×928）

| 项 | 实测值 | 来源 |
| --- | --- | --- |
| 权重冷加载 | **5.8–8.5 s** | 各次 `PE_LOADED seconds=` 日志 |
| prefill token | **6803**（视觉 2240 + 文本 4563） | `apply_chat_template` CPU 侧量化 |
| 视觉 token | 3 图 `image_grid_thw`=[[1,58,40],[1,80,54],[1,58,40]] → merge=2 后 **2240** | 同上 |
| system prompt | 官方 18,344 B + 本流程角色后缀 → **占了 4563 文本 token 的绝大部分** | `system_prompt.txt` 体积 |
| decode token | **4479** | `prompt/stage.json.metadata` |
| 整段耗时 | **243.66 s** → **18.4 tok/s** | 同上（扣掉 8.5 s 加载 ≈ 19.0 tok/s） |
| 峰值显存 | **19.58–19.65 GiB** | 各记录 `peak_gpu_memory_gib` |
| 最终给生成器的提示词 | **310 词**（编译后） | `pe_result.json.compiled_prompt` |

**吞吐分布**（同一台卡、不同时段/配置，n=1~6，只作量级参考）：**12.8 / 16.3 / 21.8 / 22.5 / 28.3 / 31.6 tok/s`**——跨度 2.5 倍，主因是**同卡他人负载**，不是配置。
所以绝对秒数不可跨时段比，只认同批同环境的相对值（这条与 `tiktok_pe_prompt_rewrite_20261014` 的门控原则一致）。

### 1.3 关键否定：关 thinking **不省**任何东西 `[实测]`

`experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014/ablation*` 里：

| 臂 | n(complete) | thinking | mean token | mean 秒 | tok/s |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 6 | True | 4615 | 283.2 | 16.3 |
| `notk_native` | 3 | **False** | **4674** | 207.3 | 22.5 |
| `C_think_native_k3` | 6 | True | 4222 | 328.9（3 并发） | 12.8 |

- **token 数没降**（4674 vs 4615）。原因读模板就明白：`enable_thinking=False` 时模板只注入**空的** `` 块（prompt 尾部实测 `…assistant\n\n\n`），**模型照样自由发挥散文**；模板里 **没有 `budget` 相关字段**，即这套 checkpoint 不提供思考预算旋钮。
- 更糟：`notk_native` 6 对里 **1 对两次都没落在 JSON 契约上**（`PE failed the JSON/image-role contract twice`，白花 1023 s）；单人线 `hard_05` 早期也出现过 **18,014 词**散文、`</think>` 缺失的失控样本。
- 结论：**"社区首选的 thinking 开关"在这个 PE 模型上不成立**，别把它当省钱手段。

### 1.4 真正的慢在哪：内核回退 `[实测，源码级]`

`transformers/models/qwen3_5/modeling_qwen3_5.py`：

```
249:@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
269:@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
300:@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
437:@use_kernel_func_from_hub_with_fallback("fused_recurrent_gated_delta_rule", "fla")
479:    # Loop over each token and update the recurrent state
```

`qwen3_5` 的 32 层里 **24 层是 Gated DeltaNet 线性注意力**（`linear_attn.in_proj_qkv/z/b/a` + 深度可分离 `conv1d` + `out_proj`），只有 8 层是常规 full attention（`full_attention_interval=4`）。
装了 `fla` / `causal_conv1d` 就走融合内核；**没装就退到第 479 行那种逐 token Python 循环更新递归状态**——这就是 18 tok/s 的直接来源。

**但装内核不是免费的**，这里有一个真实的 dtype 冲突面要 A/B：
FLA 的 `ChunkGatedDeltaRuleFunction` **拒绝 float32**（`AssertionError: … does not support float32. Please use bfloat16`，[transformers#41732](https://github.com/huggingface/transformers/issues/41732)），
而我们的 config 写的是 `mamba_ssm_dtype: float32`；Qwen 自己在 397B 部署文档里也用过 `--mamba-ssm-cache-dtype float32`，因为 `auto` 解析成 bf16 会**降质量**（[HF discussion](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/discussions/2/files)）。
→ **两条路都有代价**：保留 fp32 状态 = 融合内核可能根本不调用（回到回退路径）；改 bf16 状态 = 可能掉质量。
**必须先在同 seed 同输入上对拍 20 对**（比 `rewritten_prompt` 词级差异 + 契约通过率），别把"装上就快"当默认。

**本机实测可安装性**（只做 `pip install --dry-run`，**没有装**，不碰红线 7）：

| 包 | 结果 |
| --- | --- |
| `flash-linear-attention` 0.5.2 | **`py3-none-any` 纯 wheel，可直接装** |
| `fla-core` 0.5.2 | 同上 |
| `causal_conv1d` 1.7.0 | **只有 sdist（tar.gz），需本机 nvcc 编译** |
| `kernels` 0.17.2 | 纯 wheel（transformers 的 from-hub 内核路径可能用它） |
| `torchao` 0.18.0 | 纯 wheel 可解析 |

`qwen21` 环境实况：**`torch 2.8.0+cu128` / `triton 3.4.0` / `transformers 5.17.0` / `diffusers 0.41.0.dev0`**；
**缺**：`vllm`、`sglang`、`flash_attn`、`causal_conv1d`、`fla`、`torchao`、`optimum`、`autoawq`、`auto_gptq`、`gptqmodel`、`compressed_tensors`、`xgrammar`、`outlines`、`llmc`（有 `peft 0.20.0`）。

### 1.5 一条必须更正的既有记录 `[实测]`

**sg（node01）有外网**：`https://pypi.org/simple/` → 200、`https://huggingface.co` → 200、`https://github.com` → 200、`ping 8.8.8.8` 通、`git ls-remote origin` 成功返回 refs、pip 走 `mirrors.aliyun.com` 实际下载到了 wheel。
此前文档与记忆里"sg 无外网，只能 Windows `pip download` + `scp`"不成立（`docs/workflow_大方向优化调研_20261014.md` §10 那句需要改）。
**这直接改变了可行性判断**：vLLM / torchao / ms-swift 这类工具可以**在 sg 上另建新环境**安装，不必手工搬轮子；但**红线 7 仍然有效**——不要塞进 `qwen21`，新实验请开新 env。

小噪声：这台卡的 `nvidia-smi` 现在报 `Driver/library version mismatch (NVML 580.178)`，但 `torch.cuda` 正常（`1× NVIDIA RTX 6000D, 83.05 GiB`）。显存占用只能靠 `torch` 侧或 `gpu_gate.sh`，不能靠 `nvidia-smi`。

---

## 2. 两个模型的真实差别 `[实测，读 config]`

| | **PE-I2I**（现用重写器） | **Qwen-Image-2.1 的 `text_encoder`**（"我们的 qwen vl 3"） |
| --- | --- | --- |
| `architectures` | `Qwen3_5ForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| `model_type` | `qwen3_5` / `qwen3_5_text` | `qwen3_vl` / `qwen3_vl_text` |
| 层结构 | 32 层，**24 层线性注意力(Gated DeltaNet+depthwise conv1d) + 8 层 full**，GQA 16/4 头，`head_dim 256`，`attn_output_gate`，`mamba_ssm_dtype float32` | 36 层**全 full attention**，GQA 32/8 头，`head_dim 128` |
| 隐藏/中间 | 4096 / 12288 | 4096 / 12288 |
| 词表 | 248,320（tokenizer 248,077） | 151,936 |
| 视觉塔 | depth 27，hidden 1152，**无 deepstack** | depth 27，hidden 1152，**deepstack [8,16,24]**，`patch 16`，`spatial_merge 2` |
| RoPE | mrope_interleaved, section [11,11,10], `partial_rotary_factor 0.25`, theta 1e7 | mrope_interleaved, section [24,20,20], theta 5e6 |
| 体积 | **18.8 GB** BF16 ≈ **9.4B 参数** | **17.5 GB** BF16 ≈ **8.8B 参数** |
| 能否自回归解码 | 是 | **是**（权重表里有 `lm_head.weight`，398 个 `model.language_model.*` + 351 个 `model.visual.*` 张量） |
| 运行时生态 | 新架构：缺内核、服务化/量化工具**未证实支持** | 成熟：vLLM/量化/社区微调脚本普遍支持 Qwen-VL 系 |

三点要紧的：

1. **两者参数几乎同级**（9.4B vs 8.8B）。**蒸馏不会带来"模型变小"的带宽收益**；收益只能来自 token 数与生态。
2. **词表不同**（248k vs 152k）。teacher 的 token 数不能直接当学生的成本估算，标签要用**文本**传递、由学生自己的 tokenizer 重新切分。
3. `pipeline_qwenimage21.py` 的 `encode_prompt` 只做**前向取 `hidden_states[-1]`**（`:297-310`），并**丢掉 `_drop_idx` 之前的 system token**，输入是"模板 + 文本 +（可选）参考图 `pixel_values`" `[实测]`。也就是说这台机器上**同一个权重**既能被当嵌入器用、也能被 `.generate()` 用——用户设想的"一条分支"在这套代码里是有落点的。

---

## 3. 量化：难度与收益 `[实测架构 + 社区主源]`

### 3.1 先算清量化能买到什么

decode 的每步成本 ≈ 读一遍权重（带宽受限）。
- **BF16 → INT8**：读 18.8 GB → 9.4 GB，**上限约 2×**；
- **BF16 → INT4**：**上限约 3.5–4×**；
- 但**现在的问题是第 479 行那条逐 token Python 循环**（kernel 开销 / 大量小算子），不是权重带宽。`[推断]` **在内核回退没解决之前，量化大概率只能拿到远小于 2× 的收益，而且没法干净归因**。

**结构上是否可量化**（`[实测]` 从模块表看）：`linear_attn` 的 4 个投影 + `out_proj`、`self_attn` 的 q/k/v/o、MLP 的 gate/up/down、视觉塔的 fc1/fc2/qkv/proj **全是 `nn.Linear`** → 权重级 per-channel/int8/int4 方法在**结构上没有障碍**；
唯独 **`conv1d` 是 depthwise 小张量**（`groups=conv_dim`），一般量化器不碰（参数量小，不碰也不亏）；
而**递归状态 `mamba_ssm_dtype=float32`** 必须留在 fp32，量化不动它——这是对的，但也意味着**量化省不了状态本身的计算/带宽**。

### 3.2 运行时支持实况 `[社区主源，本机未装]`

| 运行时 | 对这个混合架构的支持 | 关键约束 |
| --- | --- | --- |
| **vLLM** | **原生支持 Qwen3-Next 系 GDN**，"uses Triton kernels from Flash Linear Attention"（[vLLM blog 2025-09-11](https://vllm.ai/blog/2025-09-11-qwen3-next)）；Qwen3.5 有官方 recipe（FP8 checkpoint、`--reasoning-parser qwen3`、MTP）（[vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)） | 多模态 Qwen3.5 要 **main/nightly**；纯文本可 `--language-model-only` 跳过视觉塔（我们要看图，用不上） |
| **官方 PE 的 vLLM 路径** | **存在**：`prompt_rewrite/serve.sh` + `run_vllm.py`，且 `serve.sh` 里已经有 `--quantization`、`--gpu-memory-utilization`、`--reasoning-parser qwen3`、`--enforce-eager` 兜底（[prompt_rewrite/README](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/README.md)） | 官方钉 **`transformers==5.4.0` + `vllm==0.19.1`**；本机是 5.17.0 且**没装 vllm** → 要新建 env。**"flag 存在"≠"官方测过某个量化方案"**，文档里找不到被测过的量化配置 |
| **SGLang** | 支持 hybrid GDN（Mamba pool + KV pool 双内存池）（[PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)）、FP8 与 EAGLE（[cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next)） | 已知坑：CPU offload 与 hybrid 不兼容（[#20774](https://github.com/sgl-project/sglang/issues/20774)）；**llm-compressor NVFP4 权重的 `linear_attn` 被静默丢弃 → 输出乱码/首个 multinomial CUDA assert**（[#22287](https://github.com/sgl-project/sglang/issues/22287)） |
| **llama.cpp / GGUF** | `qwen35` 可跑；hybrid 张量（`ssm_conv1d`、`in_proj_a/b`）转换器布局 bug 于 2026-08 由 PR #27132 修复（[#27019](https://github.com/ggml-org/llama.cpp/issues/27019)）；MTP GGUF 已合并 | **视觉可用但脆**，社区常见 "image input is not supported … provide the mmproj"（anecdote）。骨架图这种 OOD 输入风险更高 |
| **TensorRT-LLM** | supported-models 页称 Qwen3.5-hybrid 文本塔跑 BF16，列出 FP8 block-scale / NVFP4 条目 | **此页由摘要读取、未逐字核对 → 按未证实处理** |
| **ExecuTion** | 找不到 `qwen3_5`/GDN 支持证据 | 视为不支持 |

### 3.3 现成量化权重：**有，但每一条都锁死一个运行时** `[社区主源]`

| 现成产物 | 体积 | 能不能给我们的流水线用 |
| --- | --- | --- |
| Comfy-Org `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` | **9.47 GB** | **不能**。`int8_convrot` = **Hadamard 旋转（`--convrot-group-size 256`）后只对 LM 的 transformer 层做 INT8**，**视觉塔 / embedding / 首末层保持 BF16**；由 ComfyUI 原生 `CLIPLoader` + `comfy_kitchen` 反量化路径加载（[格式说明](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/quantization/int8_w8a8.md)、[权重来源](https://huggingface.co/Winnougan/Qwen-3.5-INT8-Convrot-Comfy)）。**`convrot` 与 `conv1d` 无关**，别被名字骗了。transformers / vLLM 都读不了这个非标准格式 |
| Comfy-Org `pe_i2i_w4a8.safetensors` | 6.72 GB | 同上，锁 ComfyUI |
| `prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF`（Q3/Q4/Q5，**Q4_K_M 5.63 GB** + `mmproj-bf16.gguf` 922 MB） | 5.6 GB | 走 **llama.cpp** 生态，与我们的 `Qwen3_5ForConditionalGeneration` + HF 预处理链不同源；换过去要重验 chat template / mrope / 骨架图路径 |
| RedHatAI 预量化 **Qwen3.5-9B**（FP8-dynamic / W4A16 / W8A8 / NVFP4，[llm-compressor 文档](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/)） | — | 这是**语言模型**的，不是 PE-I2I 的；说明的是"这条工具链认这个架构"，我们要自己压 PE 权重 |
| **官方有没有更小的 PE** | — | **没有**。PE-T2I / PE-I2I 都是 Qwen3.5-VL 9B 微调（[模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)）。想要 4B/2B/0.8B 只能自己做学生（§4） |

**量化方法侧的两条硬事实**（决定收益上限）：
1. **recipe 的 ignore list 是 `lm_head + visual + linear_attn`**（[llm-compressor#3108](https://github.com/vllm-project/llm-compressor/issues/3108)）→ 我们 32 层里那 24 层线性注意力的大投影（`in_proj_qkv/z/b/a`、`out_proj`）**留在 BF16**，可压的只剩 8 层 full-attn 投影 + MLP + 视觉塔 → **实际压缩比远小于名义值**。
2. **状态量化今天没有 off-the-shelf 支持**：vLLM `--mamba-cache-dtype` 不含 fp8/int8；研究侧 LeapQuant 声称 8-bit 状态可达 fp32 精度、kernel 级 2.05–3.70×（[arXiv 2609.38166](https://arxiv.org/html/2609.38166v1)），但**没进主线**。Minima 把 GDN gate 也压进 NVFP4 W4A4 并报告 5-task 均值 −0.52（≈种子噪声）、decode 1154 vs 621 tok/s（[arXiv 2609.04098](https://arxiv.org/html/2609.04098v1)），但**未发布为通用工具链、且非 VL**。
3. 现成工具链还有崩溃面：Qwen3.5-9B BF16 与 AWQ-4bit 在 WSL2/3090Ti 报 illegal memory access（归因驱动/MTP，workaround 关 spec decode）（[vllm#36408](https://github.com/vllm-project/vllm/issues/36408)）。

**投机解码（MTP）对本 checkpoint 不可用** `[实测 config]`：PE-I2I 的 `config.json` 里 **`mtp_num_hidden_layers: null`** → 权重表里没有 MTP head。社区靠**外挂 MTP head** 拿到 ComfyUI 内 2.0×(t2i)–3.5×(edit)（[mozophe 节点](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP)，anecdote），那是移植工作，不是开关。

### 3.4 难度评级（本机视角，一句话每条）

| 方案 | 工程难度 | 现实收益 | 判断 |
| --- | --- | --- | --- |
| 装 `fla`（纯 wheel） | **低**（安装）/ **中**（state dtype 要 A/B，§1.4） | 可能是数倍，**本机无数字** | **唯一值得先试的一条**，且它不是量化 |
| `torchao` int8 weight-only（transformers 原地） | **低—中** | ≤2×，但 `linear_attn` 能不能一起压要自己验；回退内核下大概率 <1.3× | 二阶 |
| vLLM + llm-compressor W8A8/FP8（ignore `linear_attn`+`visual`） | **中—高**（新 env、vllm nightly、服务化改造） | **~1.2–1.5×**，但**服务化本身的并发收益更大** | 若走 vLLM，**先为常驻/并发上它，量化顺带** |
| AWQ/GPTQ 4bit | **中—高** | ≤1.5×（线性注意力不压） | 不值 |
| GGUF Q4_K_M + mmproj（llama.cpp） | **高** | 带宽收益实、**视觉路径脆** | 换运行时代价大于收益 |
| Comfy int8_convrot / w4a8 | 表面最低 | 锁死 ComfyUI | **对本流水线是死路** |
| 状态 int8/fp8 | — | 研究级 | **今天不可用** |
| MTP 投机解码 | **高**（要外挂 head） | 1.5–2×（社区） | 不值 |

### 3.5 一句话判断

**量化不是这件事的第一优先。** 它面对的是"工具认这个架构、但把最占比例的那 24 层线性注意力留在 BF16"，
于是**收益天花板 ≈1.5×**（社区同尺寸数字），而"少生成 4000 个 token"是 **≈11×** `[实测 token 构成]`。
**唯一该现在做的量化前提是先把内核/服务化解决**，让带宽真的成为瓶颈，否则测出来也说不清是谁的功劳。


---

## 4. 蒸馏：难度、数据与验收 `[实测可行性 + 部分未证实]`

### 4.1 为什么"蒸馏"在这里是真收益

成本 ≈ **decode 步数 × 每步读权重** + prefill。
学生只要**不写那 4.5k 推理 token**，就直接拿到 **≈ 11× 的 decode 削减**（4479 → 约 400 token 对应 310 词答案）`[推断，基于 §1.2 实测 token 构成]`。
另外，18 KB 的官方 system prompt（4.5k prefill token 的主体）可以被**蒸进权重**：学生不需要每次重读契约 `[推断]`，prefill 也能砍一大截。

**这两项之和，比任何量化都大一个数量级。**

### 4.2 目标有多"窄"——这是蒸馏能不能成的关键 `[实测]`

对 `tiktok_pe_prompt_rewrite_20261014` 里 **24 条**已完成输出做统计：

- 全部输出合计只有 **752 个不同词**；
- **52.1%** 的 token 落在文档频率 **top-50** 词里；
- 编译后提示词长度 **272 / 295 / 366 词**（min/median/max）——很紧；
- 反复出现的固定句式（各 3 次）：
  `treat the target joint arrangement described in this instruc…` /
  `preserve from the source image the woman's facial identity, …` /
  `transfer only the pose from the target joint arrangement …`

也就是说：**PE 的输出高度模板化，真正的信息只有两块**——(a) 从骨架图读出的逐肢体动作描述，(b) 从 A 读出的身份/服装/发型/背景清单。
**窄、格式固定、模板可复现**——这是**最适合蒸馏**的任务形状 `[推断，基于上述实测分布]`。
（顺带：既然 52% 是固定词，**"模板 + 小模型只填两个槽"** 也是一个 cheaper 的备胎方案，天花板更低但几乎零训练。）

### 4.3 数据 `[实测有货 + 未证实需求量]`

- **标签已经在产**：每条记录都带 `rewritten_prompt`、`compiled_prompt`、`compiled_prompt_training`、`compiled_prompt_inference` 四种形式，含 SHA256 —— **蒸馏标签格式不用重新设计**，直接用编译后的 training/inference 两版之一做目标串。
- **teacher 成本**：本机 PE 中位 242 s/对 `[实测]` → 1500 对 = **101 GPU·h**（串行）；若 §5 的内核 + K=3 并发落地，**同一批大概 25–35 GPU·h** `[推断]`。
  **注意量级**：10,850 对全量（78.7–91.4 h/1000 → **854–992 GPU·h ≈ 35–41 天单卡** `[实测统计，来自 stats.json]`）
  **不是蒸馏的前置条件**——训练用的学生只需要 workflow 任务口径的几百到几千对，而 TikTok 全量 bank 是另一件事（训练数据），别把两件事绑死。
- **单人线的图与 TikTok 的图不是一批**：workflow 是 `A_reference + B_rgb + B_3d_dwpose`（骨架是 SMPL-X **投影渲染**的黑底彩线），TikTok bank 的骨架来自检测器渲染。**分布不同，学生必须在 workflow 自己的骨架图上训练**，否则第一条就踩坑 `[实测：README 已明确两侧骨架"同名同点数但不能互换"]`。
- **社区需求量** `[未证实]`：ms-swift/LLaMA-Factory 对 Qwen-VL 窄任务常用 **2k–15k** 对；我没找到"固定 JSON 的 image→prompt"任务的权威数字。**我的建议是从 300–500 对起步做可行性探针**，因为 §4.2 显示目标分布很窄，2k 可能是浪费。

### 4.4 先选底座：这是一个真正的分叉 `[实测 + 社区]`

**官方没有更小的 PE**（PE-T2I / PE-I2I 都是 Qwen3.5-VL **9B** 微调），所以学生底座只有两类候选，它们的取舍完全不同：

| | **A. 同家族小模型**（Qwen3.5-4B/2B/0.8B，本机**没有**，要下） | **B. 我们的 Qwen3-VL-8B**（`text_encoder`，本机现成 17.5 GB） |
| --- | --- | --- |
| 参数量收益 | **有**（4B≈9.4B 的 43%，0.8B 是 1/12） | **没有**（8.8B vs 9.4B，同级） |
| tokenizer/词表 | **与 teacher 相同**（同家族）→ 契约、`<image1>` 记号、system 模板可原样继承 | **不同**（152k vs 248k）→ 标签必须按文本重切，模板要重写一遍并对拍 |
| 视觉塔 | 同家族（无 deepstack 差异风险） | Qwen3-VL 带 **deepstack [8,16,24]**，与 teacher 的视觉编码不同源 → 骨架图读数可能变 |
| 运行时生态 | 同样是混合线性注意力 → **量化的 `linear_attn` 限制照旧**（§3.3），但 vLLM/FLA 认它 | **成熟 dense**：vLLM/GPTQ/AWQ/torchao/llama.cpp 全都认，**量化在这里才真的容易** |
| 额外显存 | 需要再驻一个模型（0.8B/2B 极小，4B ≈ 8 GB） | **V2 形状下零额外权重**（§4.5） |
| 适配工作 | 换 config/processor，代码改动小 | 要写"同一个权重两用"的 adapter 切换与一致性校验 |

**我的判断**：
- **如果目标是"省钱"**（用户原话的成本浪费）→ **选 A，4B 或 2B**。
  成本 ≈ 步数 × 每步读权重，A 同时在**两个因子上都赢**（token 数 4479→~400，权重 18.8→8/4 GB），
  量级是 **10×（token）× 2–4×（权重）≈ 20–40×**，单对 PE 有希望从 242 s 降到 **秒级到十几秒** `[推断]`。
- **如果目标是"省显存/少一个模型"**（这台卡确实常只剩 9.8–21 GiB）→ **选 B 的 V2 形状**。
  它省的是 18.8 GB 与一次加载，**不省时间**。
- **B 的另一个真实价值**：dense 架构让**量化立刻变成简单事**——
  也就是说"蒸馏到 qwen vl 3"这条路的隐藏红利是**把量化从'架构未证实支持'搬到'工具链现成'**，
  而不是"模型变小"。这一点值得在决策时和用户原来的直觉区分开。

### 4.5 两种实现形状

**V1（推荐）：兄弟小模型 + 任务标签 LoRA**
- 底座：`Qwen3-VL`（先用现成的 8.8B `text_encoder` 权重副本，或另下 4B/2B——**4B/2B 本机没有**，要新下）。
- 单 LoRA（`r=8–16`, `target=all-linear`, lr 1e-4, 2–3 epoch），system 里放**硬任务标签**（`TASK=pose_edit_prompt_compile`），训练时掺**少量通用图说/理解样本回放**防灾难性遗忘。
- 输出：**直接吐编译好的提示词**（或 3 键扁平 JSON + 约束解码），**不吐思考**。
- 部署：PE 用独立进程/服务，生成阶段仍用原封 `text_encoder` → **条件通路零改动**。
- 显存：学生 17.5 GB（8.8B BF16）≈ 与现在 PE 一样，**不省显存**，只省时间。

**V2（用户设想原样：一条任务分支挂在同一个 qwen vl 3 上）**
- `text_encoder` 主干**冻结**，只挂一个 `pe` adapter；
- PE 模式：`pipe.text_encoder.set_adapter("pe")` + `.generate(...)`（权重里**已有 `lm_head`**，能解码 `[实测]`）；
- 出条件：`disable_adapters()`（或 `set_adapter(None)`）后走原 `encode_prompt`。
- **收益**：**零额外权重**（省 18.8 GB 与一次加载）。现在生成段峰值 **33.97 GiB** + PE **19.65 GiB** = 53.6 GiB `[实测]`，而这台卡常被同机任务吃到只剩 9.8–21 GiB —— **同权重两用对这个环境是真痛点解药**。
- **风险与它的可验证性**：LoRA 关闭时 delta 不参与计算 → 主干数学等价，**验收是一次硬校验**：同一 prompt 下，挂/不挂 `pe` adapter 的 `prompt_embeds` 应逐位一致或 cos = 1.0000。**校不上就退回 V1**。
- 注意：这**不是**"微调条件编码器"。**全量微调 `text_encoder` 会永久改变条件通路**，与官方 checkpoint 失去可比性，`[未证实但社区共识]` Qwen-Image 侧是把 PE 做成**独立 VLM** 的，不要对抗这个设计。

**不建议的形状** `[推断]`：MoE-of-LoRA / 多专家路由。本任务只有一个下游消费者，路由机制只是把干扰风险换成复杂度。

### 4.6 蒸馏难度总评

| 子步骤 | 难度 | 依据 |
| --- | --- | --- |
| 标签与格式 | **低** | 字段已在盘上，含哈希（§4.3） |
| 造 300–500 对训练数据 | **低—中** | 纯 GPU 时间，20–35 h（内核+并发后）`[推断]` |
| 训练环境 | **中** | 需新建 env 装 `ms-swift`/`peft`/`trl`（sg 有网 `[实测]`），不碰 `qwen21` |
| LoRA 训练本身 | **中** | 单卡 83 GiB 跑 8.8B LoRA 显存够（权重 17.5 GB + LoRA 优化器）`[推断]`；4B 更宽松 |
| **让骨架图不 OOD** | **中—高** | 学生必须见过 workflow 的 SMPL-X 投影骨架（§4.3） |
| **图像引用不漂移**（不许出现 `<image2>`/把 A 的服装安到 B） | **高** | 这是 PE 今天就在失败的模式（18k 词散文那次就是引用违约）`[实测]`；需要 verifier + 约束解码兜底 |
| 输出坍缩成模板（复读机） | **中** | 监控 distinct-n 与 `rewritten_prompt` 多样性；模板化本身是目标，但**动作描述必须逐对变化** |
| 端到端验收 | **中** | 工具齐了：`evaluate.py`/`evaluate2.py`/`scorecard.py`/`human_eval/`，同几何同 seed 成对比 PCK/torsoG/idcos `[实测存在]` |

---

## 5. 建议路线（按"本周就能做"排）

**T0 免费/低风险，先吃**（不碰缓存指纹的 CODE 列表，不影响在跑的 85 格矩阵）
1. **PE 常驻化**：`pe_worker.py` 每阶段新起进程、冷加载 18.8 GB（`[实测]` 5.8–8.5 s/次）。批量时改成一个常驻进程/本地服务，按请求喂 3 张图。**顺带才能谈并发**。
2. **并发 K=3**：`[实测，只认同窗同配的相对值]` 同一批两对（`train_0004176`、`train_0004238`）在 `ablation2/A_think_native_k1`（串行）均值 **163.0 s/对**，在 `C_think_native_k3`（3 进程共存）单对均值涨到 **360.3 s** ⇒ 有效 **120.1 s/对**，**净收益只有 ≈1.36×**（不是 3×；带宽与算力都被同卡任务摊薄）。
   **不要用 baseline 的 283.2 s 去比 K3 的 328.9 s 得出 2.2×**——那是跨时段跨批次，同卡他人负载不同，违反本实验自己的"绝对秒数不可跨时段比较"原则。
   显存 3×19.6 GiB ≈ 59 GiB `[实测峰值]`，与生成段（33.97 GiB）不能同时全速——用 `gpu_gate.sh` 分窗。
   **结论：并发是有的，但很小；真正的倍率要等内核（§1.4）。**

3. **装内核 / 换服务栈**：新 env 装 **`flash-linear-attention`（纯 wheel，实测可装）**；`causal_conv1d` 只有 sdist，需要 nvcc，**先不碰**（`fla` 已覆盖 `chunk_gated_delta_rule`/`fused_recurrent_gated_delta_rule` 两个最热函数，源码行 300/437）。
   **前置门禁（§1.4 的 dtype 冲突）**：装完先在同 seed 同输入上对拍 **20 对**——同时记录 `tok/s`、契约通过率、`rewritten_prompt` 词级差异；**只要内容变了就不是"纯加速"**，要人来判。
   另一条并行的路是**官方自己的 vLLM 服务化**：`prompt_rewrite/serve.sh` 已在仓库里，钉 `transformers==5.4.0 + vllm==0.19.1`，**这条路同时解决冷加载、并发、以及后面所有量化选项的可用性**（代价：一个独立 env + worker 改造成客户端）。
   **红线 7：不往 `qwen21` 塞包**；要接进 workflow 需人在场决定 env 切换。

4. **别做**：`--no-thinking`（§1.3 实测无效且增加契约违约）。

**T1 拿归因清楚的数字**（一次实验回答"PE 到底值不值"）
5. 在**同一几何、同一 seed** 下做三臂成对：`PE 版` / `手写版` / `纯模板版`（用 §4.2 的固定句式 + 槽位填充），
   判据 `body PCK@0.2 / torsoG / idcos / bgMAE`。
   理由：现有证据是"手写 0.667 > PE 0.389"`[实测，但 n=1 且是 hard_05 一对]`，**统计上不算结论**。这个实验同时也是**纯模板方案的可行性下界**，而它可能比蒸馏便宜 10 倍。

**T2 蒸馏探针**（T0+T1 出数后）
6. 300–500 对 workflow 口径数据 → 按 §4.4 选底座（**省钱走 A：Qwen3.5-4B/2B；省显存走 B：本地 Qwen3-VL-8B 的 V2 形状**）→ LoRA → 目标串用 `compiled_prompt_inference`；
   验收：契约通过率 ≥99%、**同 seed 成对 PCK 不低于 PE 版**、tok/s 与单对总秒数记录在案；
   过了再考虑更小底座。**注意学生若仍是 `qwen3_5` 架构，量化的 `linear_attn` 限制照旧；换成 `qwen3_vl` 才把量化变成现成工具链的事。**


**T3 二阶**：int8/int4 只作为 T2 之后的加分项（学生若是 `qwen3_vl` 架构，量化生态现成，难度立刻降一档——**这也是"蒸馏换架构顺手解决量化"的核心论点** `[推断]`）。

---

## 6. 明确不建议

1. **不要把量化当省钱主路**：4479 token 的 decode 步数不会因为 int8 而变少；且官方 recipe 把 `linear_attn`+`visual` 留在 BF16，**同尺寸社区的天花板就是 ~1.5×**（FP8 1.49× / AWQ 1.29×），远小于"少生成 4000 token"的 ≈11×。
2. **不要全量微调 `Qwen-Image-2.1` 的 `text_encoder`**：它在条件通路上，改了就与官方基座失去可比性；V2 只允许**可关闭的 adapter** + 一致性校验。
3. **不要在 `qwen21` 里装内核**（红线 7）。
4. **不要把 TikTok 10,850 对全量重写当成蒸馏前置**：两件事，别绑一起（§4.3）。
5. **不要用学生直接替换生产 PE 而不做同 seed 成对**：代码哈希在指纹里，换 PE 会让全部既有 run 变 stale，数字必须成对可比。
6. **不要指望 thinking-budget**：这套 PE checkpoint 的模板里没有预算旋钮（§1.3 实测 `budget` 出现 0 次）。
7. **不要把 Comfy-Org 的 `int8_convrot` / `w4a8` PE 权重当"现成量化模型"拿来就用** `[社区主源]`：那是 ComfyUI 私有格式（只压 LM transformer 层，视觉塔/embedding/首末层保持 BF16），transformers 与 vLLM 都读不了；`convrot` 是 **Hadamard 旋转**，跟 `conv1d` 没有任何关系。
8. **许可证这条要人过目** `[实测，盘上 LICENSE 原文]`：`Qwen-Image-2.1` 与 `Qwen-Image-2.1-PE-I2I` 都是 **Qwen RESEARCH LICENSE AGREEMENT**（2026-09-20），正文含
   *"You shall not use the Materials for any commercial purpose without obtaining a separate …"*。
   合同交付里写了"支持华为方复现"，**用 PE 产物训练学生模型（蒸馏）会牵涉派生权重与交付口径**——这不是技术问题，请在动手前让责任人确认授权路径。


---

## 7. 待验证清单（本文没下结论的）

**本轮已由主源回答掉的**（不再列为待验证）：vLLM/SGLang/llama.cpp 是否认这个混合架构 → **认**（§3.2）；
Comfy `int8_convrot` 是什么、能否给 transformers 用 → **不能**（§3.3）；有没有更小的官方 PE → **没有**（§4.4）。

**仍然打开的**：
- [ ] **装 `fla` 后的真实倍率，以及 state dtype 被迫改 bf16 是否掉质量**（**本机尚未安装，没有任何倍率数据**；§1.4 的 20 对 A/B 是第一个要跑的）。
- [ ] vLLM 里 **PE-I2I 这个 VL checkpoint** 能否跑通（官方 recipe 认 Qwen3.5 家族，但"带视觉塔的 PE 微调权重 + 多模态输入"这条组合**没人验过**；需 `main/nightly`，本机未装）。
- [ ] `torchao` int8 weight-only 在 `qwen3_5` 上是否真能包住 `linear_attn` 的 4 个投影（结构上是 `nn.Linear`，工具是否跳过未知）。
- [ ] TensorRT-LLM 对该架构 VL 的量化范围（摘要读取，**未逐字核对**）。
- [ ] **学生模型在"从骨架图读出动作"上的正确率**（蒸馏最容易失败处）。
- [ ] **GGUF Q4 对 PE 输出 JSON 合规率的影响**（无现成数据）。
- [ ] `--condition none`（纯 Qwen-Edit、不给骨架）参照臂仍未实现——**"PE 到底带来多少增益"这个前提问题的严格对照还缺**。
- [ ] 蒸馏数据的**需求量**：社区 2k–15k 是宽口径经验，本任务目标分布很窄（§4.2 实测 752 词/52% top-50），**300–500 够不够只能靠探针实验回答**。


---

## 8. 参考

**本机证据（全部 `[实测]`）**
- 阶段耗时：`workflow/runs/**/*/stage.json`（`metadata.seconds`）、`workflow/README.md` 实况表
- 一次调用解剖：`workflow/runs/wp1_accept_hard02/prompt/pe_result.json`、CPU 侧 `apply_chat_template` 计数（本文 §1.2）
- thinking 对照：`experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014/ablation*/{stats.json,records/}`
- 模板性统计：同一目录 24 条 `compiled_prompt_training`
- 架构与权重：`/data/.../Qwen-Image-2.1-PE-I2I/config.json`、`model.safetensors.index.json`；
  `/data/.../Qwen-Image-2.1/text_encoder/config.json`、`model_index.json`
- 内核回退：`site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:249/269/300/437/479`
- 条件通路取嵌入方式：`site-packages/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:207-333`
- 可安装包：`pip install --dry-run`（`fla`/`fla-core` wheel、`causal_conv1d` sdist、`torchao`、`kernels`）
- 网络可达性：`curl` pypi/HF/GitHub、`git ls-remote origin`

**外部（社区/官方主源；标 anecdote 的是社区经验，不作结论）**
- 运行时与架构支持：[vLLM Qwen3-Next GDN 博客](https://vllm.ai/blog/2025-09-11-qwen3-next)、[vLLM Recipes Qwen3.5](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)、[Qwen3.5-9B 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[SGLang hybrid 模型支持](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)、[SGLang Qwen3-Next cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next)、[TensorRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html)（**该页未逐字核对**）
- 官方 PE 用法：[Qwen-Image-2.1 `prompt_rewrite/README`](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/README.md)、[`serve.sh`](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/serve.sh)（钉 `transformers==5.4.0 / vllm==0.19.1`，含 `--quantization` 与 `--reasoning-parser qwen3`）、[PE-I2I 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)、[Qwen-Image 技术报告](https://arxiv.org/html/2508.02324)
- 量化：[llm-compressor Qwen3.5 recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/)、[ignore list = `lm_head+visual+linear_attn`（#3108）](https://github.com/vllm-project/llm-compressor/issues/3108)、[comfy-quants int8_w8a8 格式说明](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/quantization/int8_w8a8.md)、[Comfy-Org/Qwen-Image-2.1 text_encoders](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders)、[PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)、[Qwen3-8B/H20 速度基准（FP8 1.49× / AWQ 1.29×）](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- 混合模型的 dtype/状态约束：[FLA 拒绝 fp32（transformers#41732）](https://github.com/huggingface/transformers/issues/41732)、[vLLM `--mamba-cache-dtype` 无 fp8/int8（#55196）](https://github.com/vllm-project/vllm/issues/55196)、[Qwen 397B 部署用 fp32 mamba cache](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/discussions/2/files)、[llama.cpp hybrid 张量布局 bug #27019](https://github.com/ggml-org/llama.cpp/issues/27019)
- 状态量化研究（**未进主线工具链**）：[LeapQuant 8-bit 状态，kernel 2.05–3.70×（arXiv 2609.38166）](https://arxiv.org/html/2609.38166v1)、[Minima NVFP4 W4A4 含 GDN gate（arXiv 2609.04098）](https://arxiv.org/html/2609.04098v1)
- 蒸馏方法：**Distilling Step-by-Step**（推理当辅助监督而非输出；[ACL Findings 2023](https://aclanthology.org/2023.findings-acl.507.pdf)、[Google blog](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)）、[Rejection Sampling fine-tuning](https://rlhfbook.com/c/09-rejection-sampling)、[Student-in-the-Loop CoT 蒸馏](https://arxiv.org/html/2604.02819v1)、[LLaVA-KD（4B→0.5B/1.8B，1–2M 通用样本）](https://arxiv.org/html/2410.16236v2)
- 训练与格式约束：**ms-swift《Qwen3-VL Best Practice》**（dense → LoRA `r=8`、`target=all-linear`、lr 1e-4、`IMAGE_MAX_TOKEN_NUM=1024`；[docs](https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3-VL-Best-Practice.html)、[repo](https://github.com/modelscope/ms-swift)）、**Qwen3-VL 同数据 token 数显著高于 Qwen2.5-VL**（[issue #6179](https://github.com/modelscope/ms-swift/issues/6179)）→ 换底座要重算序列长度、**teacher 的 token 成本不能直接换算给学生**、[vLLM structured outputs / xgrammar](https://docs.vllm.ai/en/latest/features/structured_outputs/)（保 JSON 契约的正规做法）
- "一个权重两用"的反例线索：[Janus（刻意解耦视觉理解/生成通路）](https://arxiv.org/html/2410.13848v1)、[BLIP3-o](https://openreview.net/pdf/f4f1fc7229ab3e55089e1902444bd87f961a3d20.pdf)；LoRA 是否防灾难性遗忘**证据不一致**（[支持](https://ai-tldr.dev/learn/fine-tuning/distillation-and-tools/catastrophic-forgetting/) vs [反对](https://www.legionintel.com/blog/navigating-the-challenges-of-fine-tuning-and-catastrophic-forgetting)）→ **要测，不要假设**
- 提示词质量确实影响生成（方向性证据，非本机口径）：[DALL-E 3 训练了 captioner + prompt recommender](https://cdn.openai.com/papers/dall-e-3.pdf)、[RECAP 重标注 10M 图后 CLIP score 92.78→93.80](https://arxiv.org/html/2310.16656v1)
- 官方警告"不用提示词重写结果可能不稳定"（[Qwen-Image README](https://github.com/QwenLM/Qwen-Image)）、阿里云 `prompt_extend` 默认 true（[docs](https://help.aliyun.com/en/model-studio/qwen-image-edit-api)）——**但公开检索不到 Qwen-Image-Edit 的 PE 开/关消融数字**（[Step1X-Edit/GEdit-Bench](https://arxiv.org/html/2504.17761v1)、[X2Edit](https://ojs.aaai.org/index.php/AAAI/article/view/37719/41681) 都没有 isolate PE 的 delta）→ **§5-T1 那个三臂对照是我们自己必须补的**
- 服务化吞吐参考（**非本任务实测**）：[Anyscale 连续批处理](https://www.anyscale.com/blog/continuous-batching-llm-inference)、[vLLM 视觉编码器数据并行](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html)、Qwen 官方提示"用 FLA 与 causal-conv1d 可获更好效率"（[Qwen3-Next blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)）、[Axolotl Qwen3-Next 文档"装 FLA"](https://docs.axolotl.ai/docs/models/qwen3-next.html)
- ComfyUI 侧节点：[benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer](https://github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer/)（`CLIPLoader` 加载、thinking + 24000 token）、[mozophe 的外挂 MTP head 节点](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP)（anecdote：4090 Laptop 上 t2i 2.0× / edit 3.5×）
- 未找到：`fla` 安装 vs int8/int4 量化的**同机受控对比**；官方 `--quantization` 在 PE-I2I 上的已验证配置；ExecuTion 对本架构的任何支持；MTP GGUF 之外可直接用于 HF 栈的投机解码方案


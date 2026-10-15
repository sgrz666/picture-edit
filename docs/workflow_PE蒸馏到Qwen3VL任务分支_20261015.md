# 单人 workflow：将 PE-I2I 蒸馏为 Qwen3-VL 提示词任务分支

审计时间：2026-10-15 00:26（Asia/Shanghai）。对象：`sg:/home/shangguanrz/project/pic-edit`，远端 HEAD `17991d3`；工作树有未提交改动，本文以实际文件内容为准。证据：[audit.json](../research/pe_i2i_qwen3vl_distillation_20261015/audit.json)，复核脚本及说明：[README](../research/pe_i2i_qwen3vl_distillation_20261015/README.md)。

**00:52 补充判断：如果首要目标是降低当前单人 workflow 的成本，应先完成已下载 PE-I2I Q8_0 + MTP 的公平评测，而不是优先训练共享 8B Prompt LoRA。** LoRA 本身不减少两阶段计算，也不保证降低当前顺序流水线的峰值显存；短答案蒸馏与共享模型是不同收益来源。Pocket-2B 的具体 checkpoint 是纯文本模型，不能直接读取三图。详见第 11 节。

本次完成 SSH 源码审查、safetensors 文件头审计、既有日志统计、CPU tokenizer 计数及社区主源核对；没有运行学生推理、训练或新一轮出图。下文区分“磁盘事实”“社区作者报告”和“建议/估计”。本地与远端 `workflow/run.py` 的 SHA256 不同，编排结论以远端为准；另外四个核心文件的 SHA256 一致。

## 1. 判断：任务分支可行，但降本先验证现有量化方案

**让现有 Qwen3-VL 增加“根据三图与任务生成编辑提示词”的能力，技术可行性高；做出能运行的版本是中等难度，证明它便宜且不损伤最终编辑质量是中高难度。**

不需要新造语言生成头：盘上的 Qwen-Image-2.1 `text_encoder` 已是 `Qwen3VLForConditionalGeneration`，权重含完整 `lm_head.weight`，形状 `[151936, 4096]`。这证明结构支持文本生成，但还不证明这份具体 checkpoint 的零训练提示词质量。它是否与独立发布的 Instruct 权重逐字节一致，本次未核验，不能直接替换。

若决定开展共享任务分支实验，推荐冻结这份 VL 的原始参数，训练一条 Prompt LoRA；写提示词时启用，给原 DiT 编码条件时关闭。它是同一个模型的两种工作模式，无需增加 MoE、复杂路由或全量联合训练。但若优先解决当前推理成本，应先按第 11 节完成已有量化 PE 的公平对照，而不是先训练。

任务范围限定为当前单人姿态迁移：保留 A 的身份、服装、场景和已有物体，迁移 B 的动作、头朝向、可见手势与表情。暂不蒸馏 PE 的通用图片编辑全集。

## 2. 当前单人线实际怎样运行

```text
A 原图 ── prepare/letterbox ────────────────── A_reference ────────┐
B 原图 ── SMPLest-X / 投影 / 对齐 ─────────── B_3d_dwpose ───────┤
                                                                  │
PE-I2I：看 A_reference + B_rgb + B_3d_dwpose → 三字段 JSON         │
                   ↓ compile_prompt：<image3> → <image2>          │
              generation_prompt.txt                              │
                   ↓                                              │
Qwen3-VL：看 A_reference + B_3d_dwpose + 提示词 ←──────────────────┘
                   ↓ 原生条件编码
Qwen-Image-2.1 DiT + VAE → 生成图
```

磁盘事实：

- `workflow/config.json` 的 `adapter=false`、`lora=false`，`generate_worker.py` 还会拒绝检测到 PEFT 的组件；当前单人线没有接入材料中提到的 Global Adapter。`src/pose_control/` 下的适配器训练路线与这里是不同入口。
- PE 读取三图，而生成器读取两图。B 原图的头部方向、手势等信息，需要经过文字传入生成器。只让学生看 A 和骨架会减少信息，属于另一项实验变量。
- `compile_prompt()` 拒绝教师最终答案引用输入 `<image2>`（B RGB），并将 `<image3>`（骨架）改为生成器的 `<image2>`。学生不能随意混用“编译前”与“编译后”的监督标签。
- `pe_worker.py` 使用 `enable_thinking=True`，最多两次尝试；采样上限 24,000 token。长推理、重试都可能增加时间。
- 远端 `run.py` 通过 `subprocess.run()` 顺序启动独立 PE worker 和 generation worker。两者不是同时常驻。现在的单任务峰值不能用两阶段显存直接相加。
- 相同 run 内，阶段指纹相符即可缓存；换生成 seed 不必重写 PE。不同 run 的相同输入仍可能重复计算，跨 run 内容缓存有优化空间。
- `--prompt-file` 在 prompt 阶段完成后选择实际生成提示词。它在新 run 中不会自动跳过 PE，所以不能直接拿它测“首次完全不调用 PE”的端到端速度。

## 3. 成本确实在 PE，但收益要分开计算

重新扫描 `workflow/runs/**/stage.json`，结果是**历史记录的重新统计**，不是本次 GPU 性能实测；包含不同运行条件及可能复用的产物：

| 阶段 | 有耗时字段的记录数 | 中位耗时 |
|---|---:|---:|
| 几何 | 27 | 31.45 s |
| PE | 24 | 238.15 s |
| 生成 | 29 | 47.57 s |
| 评测 | 18 | 30.32 s |

在同时具有 geometry、prompt、generation 耗时的 **23 个 run** 中，逐 run 计算 PE 占三阶段总时长的比例，再取中位数，得到 **73.93%**。这比把三个不同集合的中位数直接相加更适合描述这批完整记录。24 条 PE 结果按三张输入图哈希去重后只有 **20 组三图**，不够当成泛化训练集。

`wp1_accept_hard02` 的详细计数：

| 项目 | 本次核验结果 |
|---|---:|
| 日志记录生成长度 | 4,479 token |
| 将 decoded raw 用教师 tokenizer 重新编码 | 4,478 token |
| 最终 JSON，用同一个教师 tokenizer 计数 | **320 token** |
| JSON 内 rewritten_prompt | 299 token |
| 编译后完整提示词，用学生 tokenizer 计数 | **400 token** |
| 编译后提示词英文词数 | 310 词 |
| PE 阶段耗时，含加载等开销 | 243.66 s |

特殊 token 被解码跳过及重新分词会使 raw 计数与 generate token 数略不同。同一教师 tokenizer 下，最终 JSON 仅占 decoded raw 的约 **7.15%**；约 **92.85%** 是最终 JSON 之外的内容。这是一个样本，不能推成全数据集固定比例。

因此，学生即使仍是 8B 级别，也有可能通过“直接输出答案、缩短任务说明”节约大量时间。**共享 8B 不等于不会加速；只是加速主要来自少生成 token，而非参数量明显缩小。** 教师实际 9.410B，学生 8.767B，尺寸差距很小。

同样，4479→400 不代表端到端自动快 11 倍：两种 tokenizer 不同，每 token 成本不同，还有视觉编码、prefill、第二次条件编码、几何和出图。以这个具体 run 的记录作情景计算：

| 假设学生重写阶段总耗时 | 假设其余阶段不变的三阶段总时长 | 相对原 327.01 s |
|---|---:|---:|
| 15 s | 98.35 s | 3.32× |
| 30 s | 113.35 s | 2.89× |
| 60 s | 143.35 s | 2.28× |

这是预算情景，不是预测或实测。共享进程还会改变加载开销，应重新分项计时。

显存也应准确表述：当前顺序执行的这条样本生成峰值约 33.97 GiB，PE 约 19.6 GiB；不能宣称目前峰值 53.6 GiB，或接入共享分支后必省 19.6 GiB 峰值。共享方案能消除独立 PE 权重的部署需求，但需要保留 LoRA、瞬时激活和 KV cache，最终峰值须实测。

## 4. 推荐的任务分支与工程接入

```text
一次加载现有 Qwen3-VL
        │
        ├─ Prompt 模式：开启 Prompt LoRA
        │    A + B + 骨架 + 短任务说明 → 最终 JSON，不输出长推理
        │    → 原 parse_pe_answer / compile_prompt
        │
        └─ Condition 模式：关闭 Prompt LoRA，释放上一任务生成 cache
             A + 骨架 + 编译后提示词 → 原 encode_prompt → 原 DiT
```

**第一阶段先保持现有三字段 JSON 和编译规则。** 训练 `official_result` 中经检查的最终答案，不监督 `<think>` 内容；视觉 token、输入说明和 padding 不参与语言生成 loss。Teacher 与 Student 的 tokenizer、视觉塔及架构不同，采用文本级 response distillation / 多模态 SFT，无需对齐两者 logits，也不能直接搬教师 LoRA。

建议起始配置，均为待实验超参：

- 基座读取 Qwen-Image-2.1 实际 `text_encoder/`；processor/tokenizer 读取同模型的 `processor/`。两者在磁盘上分开放置，训练工具可能需要独立包装目录或显式 processor 路径，不能直接套标准 Hugging Face 仓库布局。
- 冻结 vision tower、merger/DeepStack、语言主干、embedding、norm 与原 `lm_head`。不新增词表 token，以普通任务字符串区分模式。
- 首轮 rank 16、alpha 32、dropout 0～0.05，学习率 5e-5～1e-4；microbatch 1、gradient checkpointing、BF16。先限制单图视觉预算并检查手部/骨架可读性，再决定序列上限。
- 精确限定语言层 Attention/MLP 投影，不盲目对全模型用 `all-linear`。按当前 checkpoint 文件头计算，36 层的 q/k/v/o/gate/up/down 全部使用 rank 16，共 **43,646,976 个 LoRA 参数**，BF16 参数约 **83.25 MiB**；实际保存可能用 FP32，训练优化器与激活还会另占显存。
- 不训练共享 bias，不把 embedding 或原 lm_head 加入 `modules_to_save`。保持增量未 merge，才能在条件模式真正回到原始主干。若只用 Attention LoRA，参数量另算。
- 图像条件分支关闭 LoRA 即可保留原模型能力；只有要求“LoRA 开启时也能做通用 VQA”，才另加理解任务回放/评测，不必为条件编码机械追加通用数据。

训练工具可选 [ms-swift 的 Qwen3-VL 路线](https://swift.readthedocs.io/en/v3.12/BestPractices/Qwen3-VL-Best-Practice.html)，其文档提供多图数据格式、LoRA、冻结视觉塔与 aligner、gradient checkpointing。现有环境有 `peft 0.20.0`，没有 ms-swift/TRL/Unsloth。新建训练环境并固定版本，不向生产 `qwen21` 装包。

真正需要改的工程位置有四处：

1. **重写 worker 接口**：增加学生后端，保留三图顺序、JSON 校验、重试及 provenance。首轮单独进程验证质量即可，不宣称已经共享显存。
2. **进程生命周期**：通过同一 worker/服务对象顺序完成重写与条件编码，复用同一个 `text_encoder` 实例。仅让两个进程指向同一路径，仍会加载两份模型。
3. **原生 guard**：当前 generation worker 一见 `peft_config` 就拒绝；实验实现需明确允许“文本侧任务 LoRA 已挂载但条件编码时禁用”，并继续禁止误加载 DiT LoRA。保留所有原始模式检查。
4. **缓存身份**：记录 student checkpoint、adapter 哈希、processor、任务模板、图像预算、输出模式、采样与 compiler 版本。Prompt LoRA 是独立配置，不复用当前表示生成器 LoRA 的 `lora` 布尔值。

部署时优先保留 Transformers 原生模型类，使用其 adapter 集成；[官方接口](https://huggingface.co/docs/transformers/peft)提供 `load_adapter()`、`enable_adapters()`、`disable_adapters()`。`PeftModel.disable_adapter()` 则是另一个包装器的上下文管理接口，二者不要混写。不要未经验证使用 `set_adapter(None)`。

本机 Diffusers `_get_qwen_prompt_embeds()` 会访问 `text_encoder.model.language_model`，临时 hook 最后 RMSNorm 以提取 **norm 之前**的特征，随后移除 hook；还会处理 `encoder_attention_mask` 与 `image_pad_mask`。随手把整个对象包一层 PeftModel 可能改变该属性路径，应专门核验。重写 generate 应使用正常 norm，不能沿用条件编码的 bypass。

共享实例必须串行切换模式；多请求并发时要锁住 adapter 状态和临时 hook。开启 LoRA 生成时的 KV cache，不可带入关闭 LoRA 后的条件编码。

## 5. “理解同时写提示词”能共享多少计算

需要区分三层目标：

| 目标 | 可行性与主要工作 |
|---|---|
| 同一份权重承担两个任务 | 高；现有生成头 + 可关闭 LoRA |
| 两任务复用部分视觉特征 | 可研究；视觉模块冻结、同图预处理一致时，缓存 vision/merger/DeepStack 特征 |
| 一次 VL 前向，同时得到完整提示词和原 DiT 所需条件 | 不能直接实现；提示词需要自回归生成，DiT 条件又依赖生成后的文字 |

两阶段不仅 LoRA 状态不同，图像集合还是三图与两图、任务模板也不同，不能把重写的 hidden states/KV cache 原样当作 DiT 条件。视觉特征复用需同时保存 DeepStack 各层特征和 grid/位置等信息；当前管线没有一个开关自动完成，建议在功能验证之后再做。

若最终目标是完全不生成文字，可以训练固定 query/语义连接器，用教师提示词产生的条件及冻结 DiT 的行为监督学生条件。但变长 token、图像 mask、位置编码、DiT 接受的特征分布都要处理，不能用长短序列逐 token MSE 草草对齐。这应是后续研究课题。

[ELLA](https://github.com/TencentQQGYLab/ELLA)可参考语言语义接入扩散模型的连接器思路，但不是本项目现成蒸馏器。[DICE](https://ojs.aaai.org/index.php/AAAI/article/view/38397)蒸馏的是 CFG 引导行为，实验包括 SD1.5、SDXL、PixArt；当前 workflow `true_cfg_scale=1.0`，它宣称的去 CFG 计算收益不能直接套用到我们的 PE 成本。

## 6. 社区有哪些真正可借用的方案

截至本次检索，没有找到可核验的“PE-I2I → Qwen-Image-2.1 内置 Qwen3-VL、支持本项目三图契约”的开源即用权重。这个结论仅限本次检索，不表示绝对不存在。

| 社区方案 | 已公开的能力/证据 | 对我们的用途与限制 |
|---|---|---|
| [PE-T2I-Pocket-2B](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-2B) | 将 PE-T2I 的最终答案蒸馏到 2B；1,776 条过滤训练对；作者报告 300 条评测 JSON 合法率 100%，生成均值约 483 token | 最接近的实践先例；**纯文本 T2I**，不能看 A/B/骨架，也不是共享 VL 的 LoRA；格式合规不等于 I2I 编辑质量达标 |
| [对应数据与评测材料](https://huggingface.co/datasets/ML-Intern-lab/Qwen-Image-2.1-rewriter-distill) | 教师标签、过滤标记及评测材料公开 | 可借用数据筛选和答案级蒸馏思路，不直接当作我们多图任务的训练集 |
| [Qwen3-VL 官方 fine-tune](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md?plain=1) | 官方训练代码及 LoRA 路径 | 训练基础设施，不提供 PE 教师标签或本任务质量保证；应打印实际可训练参数验证冻结范围 |
| [ms-swift](https://swift.readthedocs.io/en/v3.12/BestPractices/Qwen3-VL-Best-Practice.html) | 多图 SFT、LoRA、冻结视觉模块 | 第一选择，便于自定义数据；实际内置 checkpoint 的 processor 布局仍需适配 |
| [Unsloth Vision](https://unsloth.ai/docs/basics/vision-fine-tuning) | Qwen3-VL 8B 示例、多图 collator、选择训练语言/视觉层 | 显存不足时的候选；示例含可训练 embedding/lm_head 等选项，不能照抄进共享主干方案 |
| [PE-I2I llama.cpp + MTP](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP) | 作者 i2i 测试从 19.4 tok/s 提到 Q8+MTP 的 67.8 tok/s | 保留教师能力的独立降成本路线；不是蒸馏，不共享 Qwen3-VL，作者设备/两图输入的数字不能当本机三图基准 |

Pocket 的主要启示是可以把长 system prompt 的任务规则和答案格式学进学生，同时不训练输出长推理。该卡片的 100% JSON 是 300 条有限样本，且部分语义指标仍与教师有差距，不能推成“蒸馏无损”。

## 7. 数据怎样做，难度究竟在哪

**训练可运行与编辑质量可靠，是两个不同里程碑。** 这里最难的是几何/语言一致性、教师错误过滤与跨身份泛化，不是写 LoRA 代码。

建议先保留原教师接口训练：输入严格为 `A_reference, B_rgb, B_3d_dwpose` 和短任务说明；输出为编译前最终 JSON，其中骨架仍叫 `<image3>`，B 不出现在最终引用中。训练完成后继续调用原编译器。若以后改成直接输出编译后的提示词，需另设输出 schema，不能再次套原 `compile_prompt()`。

样本必须包含：三图路径与 SHA256、角色映射、画布/对齐方式、原始身份与视频来源、教师版本和采样、最终答案、重试/格式状态、人工动作审查结果。只存“图片路径 + 一段话”无法定位后续失败原因。

质量过滤优先检查：viewer image-left/right、头朝向、手是否空着、源物体是否保留、是否从 B 偷带衣物/道具、动作是否与实际缩放后的骨架冲突、不可见肢体是否被臆测。正则合法并不代表这些语义正确。可将数值关节关系用于辅助检查，不能把二维骨架当作完整语义真值。

TikTok prompt bank 在盘上已存在，抽样记录确有 `rewritten_prompt`、`compiled_prompt_training`、`compiled_prompt_inference`。但其骨架来自 `TikTok_v3/dwpose_v1`，与单人 workflow 的 SMPL-X 投影/对齐图不同；训练版还去除了图片引用。不能不经区分就混用。已有 `records/test/` 继续留作测试，不能纳入学生训练。

分割数据按人物/源视频/近重复图组进行，并检查 A 与 B 两侧跨集合重叠；不能把同一视频相邻帧随机分到 train/test。首轮 300～500 条高质量训练对只是学习曲线起点，不是充分样本量承诺。验证稳定后再增加到 1,500～3,000 条，优先补新人物、服装、遮挡、侧背身、坐蹲和手部姿态。

另一条值得做的对照是“固定保留模板 + 学生只生成动作/头向/手势槽位”。身份、背景和画幅规则交给程序固定，可能进一步缩短文本并减少身份清单幻觉。第一轮先用完整答案 SFT 建立基线，再比较槽位方案，避免一次改变太多变量。

## 8. 小规模验证顺序、难度和成本

以下是工程判断，按熟悉 PyTorch/Transformers 的一名工程师估算；不包含排队和额外数据授权时间。

| 目标 | 主观难度 | 时间预期与验收重点 |
|---|---|---|
| 原 checkpoint 零训练重写 + 模板对照 | 低，2～3/10 | 1～2 个工作日搭探针；先确认本 checkpoint 的 generate、三图和输出质量 |
| 冻结 VL + Prompt LoRA，训练/推理跑通 | 中，5/10 | 数据就绪后约 3～5 个工作日；不能以 loss 下降代替质量验收 |
| 同一实例接入两任务，稳定替代 PE | 中高，6～7/10 | 含数据与回归通常需 2～4 周；切换、缓存、接口和端到端质量是主要工作 |
| 完全取消文字生成，直接蒸馏条件 | 高，8～9/10 | 按月计的研究探索，是否优于显式提示词尚不确定 |

具体测试顺序：

1. **零训练基线**：当前 PE、固定模板、原 Qwen3-VL 直接写提示词，使用一批未参与训练的三图输入。先人工筛查 20～50 对动作与角色，再出图。此步可能发现不必训练或只需很少训练。
2. **LoRA 探针**：准备 300～500 条过滤后的训练对，独立验证集挑选 checkpoint；仅训练学生，不把 DiT/PE 一起常驻 GPU。单卡 80GB 级在控制序列/图像预算后有实现空间，但必须 smoke 测量峰值，不能只用权重大小判断。
3. **原始条件等价检查**：同 prompt、同图、同 processor、同 dtype/attention 实现，比较未挂 LoRA 与挂载但关闭 LoRA 的 `prompt_embeds`、两个 mask 和 shape。mask/shape 必须相等；数值先看逐元素最大误差及可重复基线噪声，cos≈1 不能代替逐元素核验。条件比较固定同一条提示词，不能混入学生新提示词造成的正常差别。
4. **出图成对测试**：至少 50 对独立留出图，每对 2 个相同 seed，比 PE/模板/原 VL/蒸馏 VL；保持几何 SHA256、画幅、步数、CFG 相同。记录 body/hand strict PCK、头向、身份、服装/场景、目标道具误引入，并做盲看。使用配对差值和置信区间，阈值先定，不用单一平均分掩盖手部失败。
5. **延迟测试**：同一时窗交错执行教师/学生；分别记录冷加载、prefill、decode、最终答案 token、重试、fallback、条件编码、端到端时间和峰值显存。重复 seed 的生成不再次计入 PE 缓存收益。若保留教师 fallback，最终统计必须包含回退成本。

建议验收目标（不是已达到）：候选 JSON/角色契约通过率 ≥99%，进入生成器的最终提示词必须全部通过校验；在有足够留出样本的前提下，预设姿态非劣界例如 PCK 下降不超过 0.02，且身份/衣服/道具审查不能退化；重写延迟中位数至少降低 3 倍。约 50 对只能做 smoke，不能证明总体 99% 稳定率，正式门槛还需扩大样本。

标签成本不可忽略：按本批历史中位 238.15 s/对，500 条全新教师标签约 **33.1 小时串行墙钟**，1,000 条约 66.2 小时；重试与标签筛掉会增加成本。若已有合格标签可复用，成本相应降低；换教师服务后需重新测吞吐。

回本公式为 `(制标签 + 训练 + 验证总成本) / (每次教师重写成本 - 每次学生重写成本)`。举例仅计上述 500 条标签、假设学生 30 s/对，需要约 **572 次新输入重写**才能抵回制标签时间，训练和出图验证另算。现有 run 的缓存命中不能算成蒸馏节省。长期批量工作更有价值，少量偶发任务可能优先改 runtime/缓存。

## 9. 对提供材料及旧结论的修正

材料总体方向正确，但有几处必须改成当前项目事实：

- 当前 workflow 无 Global Adapter；蒸馏分支先服务原生 Qwen 生成链路。
- `lm_head` 实际在位，不用先重建一个文字头；但生成质量仍须零训练基线验证。
- 不能把目前顺序运行的 PE 与生成显存相加，也不能把共享权重写成“零额外权重”：rank16 分支约 43.65M 参数，且有激活/cache。
- 共享 VL 仍可能因短答案获得明显速度收益；反过来，也不能直接把输出 token 比例当成整体加速倍数。
- 以本次脚本口径重算，PE 中位 238.15 s，完整三阶段记录的 PE 占比中位 73.93%；保留历史报告的原始数字和时间截面，不静默覆盖旧结论。
- ELLA/DICE 可当研究参考，不能当作已经解决本项目双任务蒸馏的社区方案。
- LoRA 关闭仅在原主干、embedding、bias 等都未被改写、配置一致的前提下可恢复原条件；“cos=1.0000”不足以单独证明数值等价。

许可证另有实际约束：[PE-I2I 原文](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/main/LICENSE)的第 2 节将许可限定为研究/评估，商业使用要求另行许可；第 4(b) 节还提到用模型输出训练并分发模型时的标识要求。学生换成 Qwen3-VL 并不自动消除教师使用条件，交付前需按实际授权核对。本次仅做技术调研与只读审计。

## 10. 本次交付怎样使用

先阅读第 4、7、8 节决定实验范围；完整统计与逐文件 SHA256 在附带 `audit.json`。本次只新增本地报告和审计包，没有修改生产 workflow、模型权重、图片、运行产物或环境依赖，也没有启动训练。

审计脚本已用 `scp` 投递到 `/data/shangguanrz/pe_i2i_qwen3vl_audit_20261015_002628/audit_readonly.py`，已通过 `py_compile` 并成功执行。需要重新核对现状时：

```bash
ssh sg
/home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  /data/shangguanrz/pe_i2i_qwen3vl_audit_20261015_002628/audit_readonly.py \
  --project /home/shangguanrz/project/pic-edit
```

该命令只读模型文件头与既有记录，CPU 分词后将 JSON 输出到终端；如保存结果，请使用新的时间戳文件名。共享任务分支尚未实现，所以当前没有“开启学生分支”的生产命令；第 8 节列的是下一阶段可执行的实验验收路线。

## 11. 追查 Pocket-2B 与 /data 上量化 I2I 的实际结果

本节取证时间为 2026-10-15 00:52，进度更新至 00:55（Asia/Shanghai）。本次只读取正在运行的验证任务，没有改其参数、停止进程或新开推理抢占 GPU。

### Pocket-2B 能用在哪里

除模型卡外，已直接读取 Hugging Face 的 [config.json](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-2B/blob/main/config.json)：`architectures=[Qwen3_5ForCausalLM]`，`model_type=qwen3_5_text`，hidden_size 2048，24 层，没有 `vision_config`。该发布权重是纯文本模型；原始家族支持视觉，并不意味着这个导出 checkpoint 仍有视觉能力。

它可把已经写好的文字需求扩写成 T2I prompt；若上游已经提供可靠的动作/头向/手势文字或结构化描述，可作为文本组织器做实验，但还需验证其 T2I 训练是否适合“保留 A、迁移 B”的编辑语义。它无法直接完成当前 PE 的三图理解与角色绑定。

若为了使用 Pocket，另让 8B VL 先描述三图，再交给 Pocket，最后再让 VL 编码提示词，增加的是一个处理环节，不能预设比量化 PE 便宜。窄任务已有足够结构化信息时，应把确定性模板也放入对照。

### 盘上的量化模型确实在运行

目录：`/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF/`。

| 文件 | 磁盘字节数 |
|---|---:|
| `qwen3.5_9b_qwen_image_2.1_pe_i2i.mtp.Q8_0.gguf` | 9,786,060,352 |
| `qwen3.5_9b_qwen_image_2.1_pe_i2i.mmproj.bf16.gguf` | 921,704,640 |

合计约 **10.71 GB**，相对原 PE safetensors 的约 **18.82 GB** 减少约 **43.1% 磁盘权重体积**，不是 GPU 峰值显存实测。量化版还包含额外 MTP 头，二者并非逐模块完全同构的体积对照。

运行时为 llama.cpp b11617 CUDA 12.8。已有输出在 `/data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/output/`，本次保存了小型取证副本到本报告对应 research 目录。

| 已完成观测 | 结果及口径 |
|---|---|
| gate0，一条 hard_02 三图输入 | HTTP 调用约 248.4 s，输出 4,821 token |
| gate0 服务器 prefill | 9.23 s，7,358 token |
| gate0 服务器 decode | 239.07 s，20.16 token/s |
| gate0 含 prefill 的输出吞吐 | 19.4 token/s；不同于纯 decode 速度 |
| MTP | 确实启用，draft acceptance 0.644，mean accepted length 2.29 |
| 最终 JSON | 可以解析，画幅字段合法 |
| workflow 角色契约 | **拒绝**：答案引用了下游生成器看不到的 B RGB `<image2>` |
| 新 gate1，hard_02 第一次尝试 | 约 246.13 s，仍为 4,821 token，同样被拒绝 |
| gate1，hard_02 第二次尝试 | HTTP 耗时 325.4 s，6,443 token，编译后 335 词，通过文本角色契约 |
| gate1，hard_02 两次合计 | 571.7 s；该记录中的 BF16 基线为 316.8 s、4,479 token，当前观测没有显示提速 |

两次第一次尝试的最终回答 SHA256 完全相同：`e287ea856ad52e7fc7e134eb2031e68908012ef68e82766709c78ca13beba182`。这是相同输入/seed 的重现，**不是两个独立质量样本**。

00:55 时 gate1 的 hard_02 已完成：第一次拒绝，第二次通过文本校验，第二个 case 尚未产出完整记录。571.7 s 包含两次重写请求，但不含服务冷启动或后续出图；不是整条 workflow 的耗时。该样本的 Q8 总耗时约为记录中 BF16 基线的 1.80 倍，但输入/采样/负载没有充分对齐，不能据此认定 Q8 固有地更慢。不能报告“两例通过率”，也不能只报成功的第二次而隐藏第一次失败成本。

`workflow/runs/q8s_hard_02` 与 `q8s_hard_05` 的 `pe_model` 指向原 BF16 PE，`requested_until=prompt`，当时没有 generation 产物。**这两个目录提供的是新鲜 BF16 基线输入，不能根据 q8s 名字把其中提示词当量化结果。**量化结果位于独立实验的 `output/gate1/records` 和 `prompts`，本地已保存 hard_02 的记录与编译后提示词副本。

### 已能看到的内容问题

直接查看了 A（持麦克风的人物）和 B（双臂交叉的人物），并对读 BF16 与量化回答：量化回答能描述服饰、场景及交叉手臂，但同时要求“双方前臂在胸前交叉”和“持麦克风的手仍按原位置举在嘴边”，存在动作约束冲突。BF16 原版回答也包含类似冲突，因此不能据此判定是 Q8 造成的语义退化。

正确的语义区别是“保留源物体”与“冻结源持物手的位置”不同。当前文字把二者混在一起；这是适合加入 teacher 标签筛查和提示词对照的失败例。仅用删除 `<image2>` 字符的方式让正则通过，并不能修正动作语义。

通过校验的 hard_02 第二次回答仍要求交叉双臂，同时保持麦克风在嘴下方，说明文本契约通过不等于目标姿态正确；是否破坏最终编辑效果还需要出图验证。

### 现在还不能公平评判速度或量化损失

1. **输入顺序不同。** 原 `pe_worker.py` 是 `[A, B, 骨架, instruction]`，量化脚本的消息 content 是 `[instruction, A, B, 骨架]`。图与文本内容相同不代表 token 序列相同；模型位置上下文也变化了。
2. **采样默认值不同。** 通过运行中服务的只读 `/props` 接口，实测 `min_p=0.05`。原 HF 脚本及模型 generation_config 未启用此项。下一轮需明确关闭 min-p，并核验其余采样项；同 seed 跨 backend/MTP 本来也不保证逐字一致。
3. **视觉路径并非全部在 GPU。** 两个脚本均带 `--no-mmproj-offload`；本机二进制 `--help` 说明该参数关闭多模态 projector 的 GPU offload。旧记录“全 GPU 驻留”的表述不适用于视觉部分。应单独测量视觉侧时间与峰值，再评估是否改回 GPU。
4. **同卡有其他任务。** 两个 `pe_rewrite_tiktok.py` shard 同时在运行，量化验证还在执行。争用是干扰因素，但不能据此断言“慢一定是争用造成”；需要同窗、相同负载或独占窗口的成对对照。
5. **视觉 token 口径未对齐。** llama.cpp 记 7,358 个 prompt token，旧 HF 样本记 6,803；需分开核对视觉预算、模板/位置与记账方式，不能仅凭差值断言图片内容发生了怎样的变化。

此外，现有 gate1 脚本两次均失败时，记录里没有 `tok_per_s`，随后打印摘要却直接索引该字段，可能因 KeyError 中断后续 case。本次未改正在运行的脚本；它会影响失败样本是否完整进入统计，需要下次批测前修正。

### 调整后的优先级与测试方法

若目标是当前降本：先评测现有 Q8+MTP，之后再决定训练。公平对照应先对齐图文顺序、采样、视觉预算，明确 mmproj 设备；同一时窗跑 BF16、Q8 MTP off、Q8 MTP on，统计含所有重试的耗时。合法候选再做同几何/同 seed 出图对照，才能回答编辑效果是否保持。

若这条路线仍太贵，再考虑小型 **VLM** 的短答案蒸馏，或更窄的动作槽位预测。共享 8B Prompt LoRA 可继续作为能力复用实验，但不应仅凭参数共享就列为成本最优方案：当前流水线峰值由生成阶段主导，新增分支并不降低该峰值，而且两阶段计算仍存在。其可能的速度收益来自短答案训练，而非 LoRA 这个形式本身。

当前任务的只读检查命令：

```bash
ssh sg
tail -n 30 /data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/logs/gate1.log
find /data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/output/gate1/records -maxdepth 1 -type f -name '*.json' -print
```

以上只显示已存在的进度和记录，不会重跑或覆盖产物。当前快照能回答“模型在位、三图可推理、MTP 生效、hard_02 首答失败后重试通过文本契约”；最终出图质量、稳定通过率、净提速与峰值显存节省尚未证明。

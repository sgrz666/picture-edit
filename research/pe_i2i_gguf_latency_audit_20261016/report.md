# GGUF 为什么仍然慢，以及 PE 热请求 20 秒目标怎样验证

目标按用户本次确认：**模型常驻之后，从提交三图重写请求到拿到可用提示词，每次 ≤20 秒**，不计服务首次加载。本文是只读审计，没有修改生产流程或运行新的 GPU 推理。

采集时间：2026-10-16 12:12（Asia/Shanghai）。原始数据为 [audit.json](audit.json)，从 sg 复制的 21 份原始脚本、记录与日志均逐文件核对 SHA256。历史日志写于 10 月 14～15 日；它们不等于今天独占 GPU 重测的结果。

## 1. 项目里实际做过哪些 GGUF 实验

| 内容 | 实际做了什么 | 已有证据与边界 |
|---|---|---|
| 社区 GGUF 文件头选型 | 对 prithivMLmods plain Q4 与 mozophe MTP Q8/Q4 查结构、敏感层位宽、视觉塔与模板 | `experiments/pe_i2i_gguf_selection_20261014/README.md`、`scripts/gguf_probe.py`；文件头与作者 KLD 表不能代替本机图像编辑质量测试 |
| runtime 与模型下载 | llama.cpp b11617 / commit 14310190c，CUDA 12.8；下载 i2i MTP Q8_0 与 BF16 mmproj | sg 的 `/data/shangguanrz/tools/llama-cpp/b11617-cuda12.8/`；Q8 权重 9,786,060,352 B，mmproj 921,704,640 B |
| Gate 0 | 1 个真实三图 case，Q8_0＋MTP，调用格式与 CUDA 可用性检查 | [gate0 JSON](source/output/gate0_20261014T231113.json)；248.4 秒、4,821 输出 token；被角色契约拒绝，不能当成功样本 |
| Gate 1 | hard_02 / hard_05 两对，Q8_0＋MTP，每对最多两次完整生成 | [stats.json](source/output/gate1/stats.json)、[逐次记录](source/output/gate1/records/q8s_hard_02.json)；两对最终通过，首答均失败，n=2 不足以比较量化后的普遍质量 |
| 下游出图旁路比较 | 同几何、同 seed，HF 文本与 GGUF 文本分别给生成器 | 旧选型报告第 9 节保留结果；两对姿态/身份指标方向不一致，并受人体尺度影响，不能据此宣布量化好或坏 |
| TikTok PE 吞吐消融 | BF16 的 thinking、512 视觉输入、并发等对照 | 上游 `tiktok_pe_prompt_rewrite_20261014`；属于 HF 对照，不是 Q4 或 GGUF 实测 |
| Qwen3-VL 任务分支蒸馏 | 10 月 15 日做结构、token 长度与成本审计 | `research/pe_i2i_qwen3vl_distillation_20261015/`；没有已经训练好、达到 20 秒的学生模型 |

本地与 sg 项目源码、实验目录及 `/data` 的相关目录均已搜索。当前可核验的 PE GGUF 实跑只有 **Q8_0＋MTP**；Q4_K_M 没有本机推理对照。第三方 XVerse README 提到 GGUF，不构成本项目 PE 实验。不能把社区两图的 67.8 / 85.2 tok/s 当作本机三图速度。

## 2. 量化并没有接入当前生产 PE worker

sg 上当前 `workflow/config.json` 的 `pe_model` 仍为原 HF PE 目录。`workflow/pe_worker.py:43` 使用 `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16).to('cuda')`。

GGUF 在仓外独立脚本中调用；`--prompt-file` 只在 prompt 阶段之后选择下游生成用文本，**不会自动把 PE worker 换成 GGUF，也不会自动跳过没有缓存的 BF16 PE 阶段**。所以从普通 workflow 的 prompt/run 时间，不能直接判定“量化 PE 的耗时”。

本次审计保存了生产代码 SHA 与相关行。当前运行中的两个 `pe_rewrite_tiktok.py --split val` shard 继续保留，没有停止、改参数或更新其提示词银行。

## 3. 统一计算所有尝试，才解释得通总耗时

以下 GGUF 时间在 server ready 之后开始，**已经排除了约 21.5 秒模型服务加载**。HF 时间包括 stage 内加载等开销；此差异偏向 GGUF，但二者仍缺少严格同窗、同采样、同负载的受控比较。

| case | HF 总秒数 | HF 所有尝试 token | GGUF 总秒数 | GGUF 所有尝试 token | 尝试次数 HF / GGUF |
|---|---:|---:|---:|---:|---|
| hard_02 | 316.82 | 4,479 | 571.7 | 4,821＋6,443＝11,264 | 1 / 2 |
| hard_05 | 637.17 | 3,927＋5,231＝9,158 | 415.5 | 4,421＋3,695＝8,116 | 2 / 2 |
| 合计 | 953.99 | 13,637 | 987.2 | 19,380 | 3 / 4 |

按完整工作量计算，观察到的吞吐分别约为 HF **14.29 token/stage 秒**、GGUF **19.63 token/request 秒**。GGUF 输出工作量多 **42.1%**，总延迟反而多 **3.48%**。这是这两对实际经历的解释，不是“量化使模型更爱重试”的因果结论。

旧报告的“每 token 约快 2×”有口径问题：例如 hard_05 的 HF `5,231 / 637.2 ≈ 8.2`，分子只取成功那次，分母却含两次尝试；GGUF 的 `3,695 / 188.6 ≈ 19.6` 又只算成功单次。统一分子分母后，不能保留这个 2× 结论。

两次 GGUF 首答都因“引用生成器看不到的 B RGB”被拒，第二次把整个请求重新生成，而非只修几个字符。这个成本必须进入 20 秒验收。不能只放宽正则、允许非法 B 引用，或只报告成功那次来假装加速。

## 4. 主要成本是生成几千 token

服务原始 [server.log](source/output/gate1/server.log) 中四次请求：

| 请求 | prompt 处理 | decode | 输出 token | 服务合计 |
|---|---:|---:|---:|---:|
| hard_02 attempt1 | 9.000 s | 237.128 s | 4,821 | 246.128 s |
| hard_02 attempt2 | 6.846 s | 318.006 s | 6,443 | 324.852 s |
| hard_05 attempt1 | 7.567 s | 218.486 s | 4,421 | 226.053 s |
| hard_05 attempt2 | 6.926 s | 181.088 s | 3,695 | 188.013 s |

合计 decode **954.707 秒**、prompt 处理 **30.339 秒**；decode 占服务计时 **96.92%**。与 HTTP 记录总时间只有约 2.15 秒差异，不能把主要耗时归给模型加载或网络传输。

MTP 已经启用：acceptance 0.623～0.649、mean len 2.25～2.30；服务记录大量 graph reuse，不能把“打开 MTP”列为尚未实施的新收益。后几次服务实际处理 2,730～2,850 prompt token，而 API 的完整 prompt 是 7,412～7,532，已有前缀复用迹象，也不能把常驻/前缀缓存当成数十倍的剩余空间。

Q8 压缩权重大小，既不把 9B 网络变成小模型，也不减少其输出的 token 数。每 token 变快后，如果仍生成数千 token、甚至再来一次，端到端仍可能很慢。社区作者也按 token/s 而不是固定秒数报告收益，且其输入、设备、负载与我们不同。[作者说明](https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF/blob/main/README.md)

### 最终文件很短，不代表模型只生成了这些内容

GGUF 脚本仅保存 `message.content`，没有保留完整 API response 和 `message.reasoning_content`。当前四份所谓 raw 文件都不含 `</think>`。用现有 PE tokenizer 在 **CPU** 重新统计：

| 请求 | API completion token | 保存的 content token |
|---|---:|---:|
| hard_02 attempt1 | 4,821 | 440 |
| hard_02 attempt2 | 6,443 | 365 |
| hard_05 attempt1 | 4,421 | 460 |
| hard_05 attempt2 | 3,695 | 358 |

保存的 content 合计 1,623，而 completion 合计 19,380；约 **91.6%** 的 completion 数量没有体现在这些短文件里。这里是同家族 tokenizer 的重数估计，包含特殊 token/后端记账差异，不能冒充 API 精确 reasoning_tokens；但足以否定“只生成了几百 token 却用了几分钟”的理解。

把 `reasoning_format` 改成不显示思考，只改变返回组织形式，不自动减少已经计算的推理 token。下一轮必须保存完整 response、reasoning 字段和 usage。

## 5. 有两个配置问题，但需按收益大小处理

**视觉在 CPU。** 两份脚本都显式带 `--no-mmproj-offload`；已安装二进制帮助及 [对应版本文档](https://github.com/ggml-org/llama.cpp/blob/14310190c/tools/server/README.md)说明它关闭视觉 projector 的 GPU offload。10 月 14 日“全 GPU 驻留”不适用于视觉部分，10 月 15 日审计已指出该问题。mmproj 本身仍是 BF16/F32，也没有因语言权重量化而自动压缩。应单独比较启用 GPU 的耗时与显存，但以本批约 3.1% 的非 decode 时间，单独改它不能实现 20 秒。

**公平对照尚未对齐。** 旧 GGUF 的 content 顺序是 instruction→三图，HF 是三图→instruction；GGUF 沿用 `min_p=0.05`，官方 I2I/HF 不启用该项。当前也确有两个 PE shard 同卡运行。这些都是混杂因素；不能跨设备套社区速度，也不能断言“慢全部由争用造成”。

## 6. 优先验证现有 runtime 的思考预算，再决定是否蒸馏

本机已安装的 b11617 实测 `--help` 支持：

```text
--reasoning-budget N
  -1: 不限制；0: 立即结束思考；N>0: 限制思考 token 预算
```

这不同于给 HF chat template 传 `enable_thinking=False`，也不同于把 `max_new_tokens` 改成 512。后者会截断包含答案在内的总输出，可能根本等不到 JSON。llama.cpp 的 [reasoning-budget 实现](https://github.com/ggml-org/llama.cpp/blob/14310190c/common/reasoning-budget.cpp)会在预算耗尽后强制结束思考序列，再让答案继续生成。

旧文档“模板没有 budget 字段”仅说明当时 HF 模板没有该旋钮，**不能扩展成已安装 llama.cpp 没有运行时预算**。目前只证明功能存在，尚未对本 checkpoint 验证预算是否按预期生效、MTP 下是否配合正确、答案格式和动作质量能否保持；不能直接宣布它已达到 20 秒。

推荐顺序：

1. 固定 Q8、A/B/骨架和完整答案要求，先对齐图片顺序、`min_p=0`、top_k=20、top_p=0.95、temperature=1 与 seed；记录 MTP、视觉设备及模型/模板 SHA。对齐后的基线另命名，不覆盖旧 Gate 1。
2. 在同一资源窗口，保持其余条件不变，比较 reasoning budget **128 / 256 / 512** 与无限制基线。总 max token 留足以写完整 JSON；要确认实际结束原因不是 length 截断。0 预算只作边界对照，不预设其质量。
3. 服务常驻、先完成 warmup；分别测新三图、共享 A 的新 B，不能用完全相同请求的缓存命中替代真实重写延迟。GPU 视觉与 CPU 视觉另作一个单变量对照。
4. 若完整 PE 答案仍过长，再试“只输出动作、头向、表情、左右手、可见性、持物关系槽位；固定程序补 A 身份/衣服/场景保留条款”。这是独立提示词协议，不能静默替换正在跑的训练/测试银行。不能为缩短答案而丢掉持物与手势要求。
5. 最后才评估 Q4_K_M 或短答案学生。Q4 是可测的额外吞吐优化；答案级蒸馏是另一项需要数据与质量验证的工作。不能拿纯文本 Pocket-T2I 直接代替需要看三图的 I2I。

### 20 秒意味着怎样的 token 预算

近似热请求成本：

`T = 排队＋图片/文本输入处理＋(思考 token＋答案 token)/实际 decode 速度＋校验/重试`

以本批约 20 token/s 和 7～9 秒输入处理计，20 秒内只能容纳 **约 220～260 个总输出 token**。现有完整 JSON 本身已有约 358～365 token，所以在旧速度下，即便零思考也不能稳定满足 20 秒。

若同机优化后实测达到 50 token/s，思考 128＋答案 365、输入处理 8 秒，数学情景为 **约 17.9 秒**；这是预算测算，不是实测承诺。若仍只有约 20 token/s，则还需提升 runtime 的实际速度或缩短输出协议，不能靠修改上限数字达标。

## 7. 验收、复核与没有做的事

先取至少 **8 个已用过的 case × 2 seed** 做受控探针，包含抱臂/持物、抬手、头转、近景手部等困难样本；完成筛选后再用未参与调参的独立样本确认。每个候选输出保存完整 API body、模型/输入/指令 SHA、采样、思考预算、prompt/decode timing、实际 token 数、finish_reason、全部失败与重试。

速度报告应同时列 median/p95/max、首答合法率、重试率和总失败率。硬要求为每次 ≤20 秒时，不能用均值 ≤20 秒代替；样本太少时 p95 也只是描述性读数。超时不能当成成功，回退教师的耗时也必须计入。

文本检查沿用真实角色契约；另外人工核验左右手、表情/头向、持物、A 外观与背景边界。合法候选再通过现有 `--prompt-file` 通道做同几何同 seed 出图，结合 PCK、两种 torso_ratio、identity、背景误差和标准六图面板判断。JSON 通过不等于动作正确。

本次没有新跑 GPU、没有安装包、没有下载 Q4、没有停止现有任务，也没有覆盖或删移图片/视频及历史记录。没有测得新的 20 秒结果。

如需复核，只读命令为（执行后在标准输出得到新快照）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python \
  /data/shangguanrz/pe_i2i_gguf_latency_audit_20261016_121207/collect_readonly.py
```

该命令不启动模型服务或训练；本地 `audit.json` 和 `source/` 可直接检查原始记录。后续 GPU 探针归入已有 GGUF 实验族的新 output 子目录，不能复用旧输出路径。

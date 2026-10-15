# PE-I2I 社区量化权重选型：哪个最适配、哪个不浪费时间（2026-10-14）

对象：`workflow/` 单人线的 `prompt` 阶段用的 `Qwen/Qwen-Image-2.1-PE-I2I`（`qwen3_5` = Qwen3.5-VL 9B 微调，17.53 GiB BF16，32 层里 24 层是 Gated DeltaNet 线性注意力）。

触发：用户给出 `https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF`，问"这些量化模型哪个最适配我们的项目，而且推理不会浪费太长时间"。

**标注约定**：`[实测]` = 本机今天量/读到的；`[他测]` = 社区作者公布、有方法有数字但**不是我们机器上跑的**；`[未证实]` = 需我们自己验。

---

## 0. 结论

1. **位宽不是主因，runtime 才是。** `[实测+他测]`
   我们现在 HF transformers BF16 在 RTX 6000D 上只有 **18.4 tok/s**（4479 token / 243.7 s），而 llama.cpp 在同一模型上——**不量化、不投机，只 Q8_0**——作者实测 i2i 有 **46.6 tok/s（2.40×）**。
   原因与我们自己的诊断同源：batch=1 自回归解码的瓶颈是每步跨 32 层的 Python/launch 开销（45–55 ms/步），**不是搬字节**（带宽占空比只有 ≈0.16–0.18）。
   再叠 **MTP 投机解码**（`--spec-type draft-mtp`）→ `[他测]` i2i **67.8 tok/s（3.49×）**；换 Q4_K_M → **85.2 tok/s（4.39×）**。
   **落到我们的流水线：PE 单对中位 242.1 s 有望降到 60–110 s。**
2. **你给的这批 GGUF 里，最不该选的是 plain Q4_K_M。** `[实测，读文件头]`
   它的 `ssm_alpha`/`ssm_beta`/`ssm_out`/`attn_gate` **全在 Q4_K**——正好是另一个作者实测出 **mean KLD 0.146（他自己 0.05 阈值的 3 倍，判死）** 的那个配置；他后来把这些张量提到 Q8_0/Q6_K 才降到 0.0150。
   这批文件**没有 imatrix 校准、没有 MTP 头**（427 张量 vs 442）。
3. **最适配的不是这个仓库，而是 `mozophe/Qwen-Image-2.1-PE-MTP-GGUF` 的 i2i MTP 版**，优先级：
   **Q8_0（KLD 0.0019，"与 bf16 实际一致"，67.8 tok/s，9.79 GB）> Q4_K_M（KLD 0.0150，85.2 tok/s，5.99 GB）**；
   `heretic` 变体是改过语言模型的版本，我们这种**格式契约敏感**的任务不要用。
4. **两条"必须我们自测"的红线**：`[未证实]`
   ① 我们的 PE 喂 **3 张图**（A + B + 骨架），社区只公布 2 张图的数字；
   ② 官方 CUDA 预编译包是否含 **sm_120**（Blackwell 消费/工作站级）的 kernel。
5. **交付口径建议自建、不下第三方权重**：mozophe 公开了完整可复现配方（`convert → llama-imatrix → llama-quantize --tensor-type ssm_alpha=q8_0 --tensor-type ssm_beta=q8_0 --tensor-type attn_gate=q6_K --tensor-type ssm_out=q6_K → llama-perplexity --kl-divergence` 卡阈值），用官方预编译包就能做，**不需要 nvcc**。第三方的 MTP 头是从 `Qwen/Qwen3.5-9B` 嫁接的（他钉了 commit `c202236`），属派生权重，而 **盘上 LICENSE 是 Qwen RESEARCH LICENSE（商用需另行授权）**。
6. **闸 0 已于当晚实跑（§8）**：运行时与权重都已落 `/data`，**sm_120 上 CUDA 后端能用、3 图能吃下、MTP acceptance 0.644**；但那次有 2 个 HF PE worker 同卡抢跑，**19.4 tok/s 不是速度结论**，且暴露了两个必须先解释的差异（prompt_tokens +555、单发一次被契约拒绝）。


---

## 1. 我核对到的文件实况 `[实测，HTTP range 读 GGUF 头，不下载整档]`

工具：`experiments/pe_i2i_gguf_selection_20261014/scripts/gguf_probe.py`（range 读 48 MiB → 解析 KV + 全部张量元信息 + 按模块列 dtype）。

| 仓库 | 文件 | 体积 | 张量 | 架构 | chat_template | ssm_alpha/beta | ssm_out | attn_gate | MTP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prithivMLmods | `…Q4_K_M` | 5.63 GB | 427 | `qwen35` | **7756 字符，含 think** ✓ | **Q4_K** | **Q4_K** | **Q4_K** | 无 |
| mozophe | `…i2i.mtp.Q4_K_M` | 5.99 GB | **442** | `qwen35` | 7756 字符，含 think ✓ | **Q8_0** | **Q6_K** | **Q6_K** | **+15 张量** |
| 两者 | `mmproj-bf16` | 0.92/0.92 GB | 334 | `clip`（`projector_type qwen3vl_merger`，`image_size 768`，`patch 16`，`merge 2`，**deepstack 全 false**） | — | — | — | — | — |

- 两个仓库的 `general.license.name` 都是 **qwen-research**。
- 视觉塔都是 **F32 + BF16 混合、未量化**（`type30`=BF16 110 个、`type0`=F32 222 个）→ 读骨架线色这条通路没被动过，这是好消息。
- HF 侧对照：`Qwen-Image-2.1-PE-I2I/chat_template.jinja` **正好 7756 字节**，与两个 GGUF 内嵌模板同长 → 模板没有被改写，`enable_thinking` 那条分支还在。
- prithivMLmods 档位（同仓库文件表）：BF16 17.92 · Q3_K_M 4.62 · Q3_K_L 4.93 · Q4_K_S 5.35 · **Q4_K_M 5.63** · Q5_K_S 6.31 · **Q5_K_M 6.47**；mozophe：i2i/t2i × {Q8_0 9.79, Q4_K_M 5.99} × {正常, heretic}。
- **HF 下载不是瓶颈**：`[实测]` sg 直连 `resolve/main` 实测 **≈10 MB/s**（40 MiB / 4.2 s），下 6 GB ≈ 10 分钟。

---

## 2. 质量证据：作者的 KLD 表 `[他测，方法公开但样本小]`

方法：`llama-perplexity --kl-divergence`，ctx 4096，flash attention；评测文本 = 官方 PE system prompt + PE 自己写的 12 条改写（含 thinking 与 JSON，≈6k 计分 token）；**imatrix 校准文本 16 条、≈35k token，与评测文本不重叠**。

| 模型 | Q8_0 | Q4_K_M（保护版） |
| --- | --- | --- |
| t2i | 0.0054 | 0.0398 |
| **i2i（我们要的）** | **0.0019** | **0.0150** |
| t2i-heretic | 0.0060 | 0.0414 |
| i2i-heretic | 0.0017 | 0.0156 |

细节（i2i）：Q8_0 mean 0.0019 / median 0.00049 / 99% 0.0052 / 同 top-token **98.5%** / RMS Δp 0.77%；Q4_K_M mean 0.0150 / median 0.0080 / 99% 0.084 / 同 top-token **94.5%** / RMS Δp 2.98%。
配方消融（t2i，同一评测文本）：**plain Q4_K_M 0.146 → +imatrix 0.0717 → +imatrix&ssm q8_0 0.0740 → +attn_gate/ssm_out q6_K 0.0398**（体积只多 0.2 GB）。

三条要点：
- **KLD 是纯文本分布指标**，测不到"看图读骨架读错"。我们的任务 3/4 的成本在视觉条件上，所以**它只能当第一道筛，不能当验收**。
- **同作者的对照恰好说明 prithivMLmods 的 plain 路线为什么危险**：敏感层压到 4-bit 会把 mean KLD 推到 0.146 量级。
  注意诚实边界：**plain Q4_K_M 的 0.146 是在 t2i 上测的，i2i 的 plain 版没有公布数字** `[未证实]`。
- **8-bit KV cache 基本免费**：`-ctk q8_0 -ctv q8_0` 下 i2i Q4_K_M 的 KLD 不变（0.0150→0.0150），Q8_0 只从 0.0019 到 0.0023；因为只有 8/32 层留 KV，16k 上下文省 ~240 MiB。

---

## 3. 速度证据：为什么"换 runtime"比"降位宽"值钱 `[他测 + 我们的对照]`

作者在同一台 **RTX 4090 Laptop（16 GB）** 上逐级加一样东西，i2i（两张输入图）、官方采样、thinking on、max 24000：

| 步骤 | tok/s | 相对 |
| --- | --- | --- |
| ComfyUI int8 convrot，MTP off（基线） | 19.4 | 1.00× |
| + MTP | 32.6 | 1.68× |
| + 换成 llama.cpp（Q8_0） | **67.8** | **3.49×** |
| + 再用 Q4_K_M | 85.2 | 4.39× |

**和我们基线的对齐方式**（关键，别乱比）：
- 我们的 HF BF16 = **18.4 tok/s** `[实测]`；作者基线（ComfyUI int8）= 19.4 tok/s。**两者几乎一样慢**，而且都远慢于同尺寸卡应有的带宽上限水平 → 说明这是**软件栈的每步固定开销**，不是位宽、也不是显存。
- llama.cpp 不带 MTP 也不量化到 4-bit，纯 Q8_0 就到 46.6 tok/s（2.4×）→ **收益主体来自 runtime**。
- **不要跨设备直接套数**：4090 Laptop 的显存带宽和功耗墙都比 RTX 6000D 低，我们在同软件栈上大概率**不低于**这组数；但必须自己量（见 §5 闸 2）。
- `[他测]` llama.cpp 的速度**不依赖 max_length**（8192 / 16256 / 24000 同速）；而 ComfyUI 后端会因 max_length 变慢。这条对我们很重要：可以放心用官方的 24000。

**顺带把两个社区常见说法钉掉**（都是我们本机实测过的）：
- **不要关 thinking**：`[实测]` `enable_thinking=False` 不减 token（4674 vs 4615）、还会破坏 JSON 契约；作者也从模型行为角度给出同一结论——"Keep thinking on. Both models were trained to reason before answering and give worse prompts without it."
- **别指望 thinking 预算**：`[实测]` 该 checkpoint 的模板里没有 `budget` 相关字段（出现 0 次）。

---

## 4. 适配性核对表：我们的 7 条硬要求 vs 候选

| 要求 | 来源 | prithivMLmods GGUF | mozophe MTP GGUF | 备注 |
| --- | --- | --- | --- | --- |
| 线性注意力架构可跑 | `qwen3_5` | ✓（`general.architecture=qwen35`，llama.cpp 有 `src/models/qwen35.cpp`）`[实测]` | ✓ 同 | |
| **≥3 张图** | workflow 送 A/B/骨架 | `[未证实]`（官方节点写"Up to 10 input images, any size"，但**作者的 i2i 速度只在 2 图上测**） | 同左 | 我们的第一道自测项 |
| 内嵌模板支持 thinking | `pe_worker.py:67` | ✓ 7756 字符含 think `[实测]` | ✓ 同 | 与 HF 模板同长 |
| 三字段 JSON + `ratio_follow="<image1>"` | `src/pose_edit/prompt_contract.py:18-35` | `[未证实]` 需实测通过率 | 同 | 现成校验器直接可用 |
| 禁止 `<image2>` 出现在最终提示词 | `pose_transfer/prompts.py:15-29` 角色后缀 | **必须自己塞 system 后缀**（18,344 B 官方 system + 我们的后缀） | 同 | 节点支持 `system_prompt` 覆盖输入 → 可行 |
| 采样 `temp1.0/top_p0.95/top_k20`、无 presence_penalty | `PE_SAMPLING` | ✓ llama-server 全可设 | ✓ | 别动这些值，它们进缓存指纹 |
| 敏感层不被压到 4-bit | §1 | ✗ **ssm/attn_gate 全 Q4_K** | ✓ Q8_0/Q6_K | 这是两者的实质差别 |

**MTP 的复现性代价** `[他测]`：作者明确写"quality is unchanged, but the same seed gives different text than with MTP off"。
我们的 `prompt/stage.json` 把 `pe_seed` 与 `PE_SAMPLING` 钉进身份，**开 MTP 后同 seed 不再能逐字复现旧答案**，所以 backend/spec-type 必须作为新的身份字段记录，否则"可复现"这条会被悄悄破坏。

---

## 5. 建议的落地与验收（按成本排序，全都不动生产代码）

**闸 0｜装 runtime（≈0.7 GB，不需要 nvcc）** `[未证实→先证伪]`
下载 llama.cpp 官方预编译包 `bin-ubuntu-cuda-13`（备选 `bin-ubuntu-cuda-12`、`bin-ubuntu-vulkan-x64`；本机已确认有 `libvulkan.so.1` + `nvidia_icd.json`，`api_version 1.4.312`）。
用**最小文件**（mmproj 0.92 GB + Q4_K_M）跑一次：`llama-cli -m … -mmproj … --n-gpu-layers 99 -f 一条真实 PE prompt`，看两件事：**CUDA/SM_120 是否初始化成功**、`tok/s` 落在哪个量级。失败就换 vulkan 包，再不行退回"只做 vLLM/蒸馏"。
（`nvcc` 本机不存在，`/usr/local/cuda*` 也没有；gcc 13.3 / cmake 3.28 在，但 CUDA 后端要 nvcc → **只有预编译包或 pip 的 nvcc wheel 这两条路**。）

**闸 1｜功能契约对齐（CPU + 少量 GPU，24 对 × 1 seed）**
把 `workflow/runs/*/prompt/pe_result.json` 里的 `input_roles` 三张图 + `effective_system_prompt.txt` 原样喂给 llama-server，
用现成 `parse_pe_answer` + `validate_generator_prompt` 判：三字段合法率、`ratio_follow=="<image1>"` 率、**违禁 `<image2>` 必须 0**、重试率 ≤ BF16 基线（实测 2/24）。

**闸 2｜速度（同窗对照，绝不跨时段比）**
同一批对、同一时段内跑：HF BF16（qwen21，现基线）vs llama.cpp Q8_0 ±MTP vs Q4_K_M ±MTP，
记 `tok/s`、单对总秒数、`prefill` 秒数、峰值显存、`free_gib` 前后（这台卡 `nvidia-smi` 已坏，用 `torch.cuda.mem_get_info()`；他人负载会让绝对秒数摆 2.5 倍）。
判定线：**单对 PE < 120 s 才算值得接**（否则仍占一次 run 的 50%+，收益不抵引入新运行时的复杂度）。

**闸 3｜端到端（唯一算结论的一级）**
用现成 `--prompt-file` 通道把 GGUF 版提示词送进生成阶段（**PE 产物 `pe_compiled_prompt.txt` 只读、不被覆盖**，来源与哈希进 `prompt/generation_prompt.json`），
同几何同 seed 成对比 `body PCK@0.2 / torsoG / idcos / bgMAE`（`evaluate2.py` + `scorecard.py --baseline/--candidate`）。
阈值：Δbody-PCK@0.2 的 95% 下界 > −0.02；违禁引用 0；编译后词数留在 272–366。
诚实标注：n=24 只分辨到 ΔPCK≈0.05。

**闸 4｜如果要采纳，就自建权重**
用同一配方（`llama-imatrix` + `--tensor-type` 保护 ssm/attn_gate + `llama-perplexity --kl-divergence` 卡 0.01/0.05）在我们自己的 checkpoint 上产 Q8_0 与 Q4_K_M，
落 `/data/shangguanrz/pic-edit-models/`，登记进 `configs/asset_registry.json` + `docs/数据盘说明_20261012.md`，
`pe_worker.py` 改走 HTTP（**它在 prompt 阶段的 CODE 列表里，改动会让全部既有 run 的 prompt 缓存失效**，要排在 85 格矩阵与全量 bank 的空窗做）。

---

## 6. 明确不建议

1. **不要选 plain Q4_K_M / Q3 系列**（4-bit 压在 `ssm_alpha/ssm_beta/ssm_out/attn_gate` 上；t2i 对照实测 0.146）。
2. **不要为 prithivMLmods 的 Q5_K_M 单独下 6.5 GB**：它同样无 imatrix、无 MTP，比 mozophe 的 Q8_0 **更差还更慢**（MTP 那一档 1.68× 是白送的）。
3. **不要用 `heretic` 变体**做我们的契约任务（语言模型被改过，KLD 略高、行为未知）。
4. **不要把 Q4_K_M 直接换进生产**：同 top-token 只有 94.5%，意味着**5.5% 的决策位置改选**，在 4.5k token 的长答案上足以换掉整段动作描述——必须过闸 3 才谈替换；保守起点是 **Q8_0**。
5. **不要在共享卡上并发多个 llama-server 又同时跑生成段**：Q8_0+mmproj≈10.7 GB、Q4≈7.6 GB（含 3 图要重新量），与生成段峰值 33.97 GiB 相加会挤爆他人窗口；用 `gpu_gate.sh` 那套闸门。
6. **不要开 MTP 却不记录它**：它会改变同 seed 的文本（见 §4 复现性代价）。

---

## 7. 仍打开的问题

**已由 §8/§9 回答的**：sm_120 上预编译 CUDA 包能否初始化 → **能**（`CUDA0: NVIDIA RTX 6000D`，但要把 cudart 目录放进 `LD_LIBRARY_PATH`）；3 张图能否吃下 → **能吃下**（MTP acceptance 0.644）；
**prompt_tokens 差异** → 是 §8 探针自己的错（instruction 被我截断成 130 字符），真实同输入下 HF 7358 vs GGUF 7412、HF 7478 vs GGUF 7532，**稳定差 +54**（BOS/模板标记级），不是视觉记账差异；
**契约拒绝是否 GGUF 独有** → 不是，`hard_05` 的 HF 臂同样 attempt1 被拒、attempt2 才过。

**仍然打开的**：
- [ ] **端到端秒数**：§9 两对合计 GGUF 987 s vs HF 954 s＝**没赚**（tok/s 翻倍被重试率与多吐 token 吃掉）。需要 ≥8 对 × 2 seed 才能定这条。
- [ ] **重试率差**（GGUF 2/2 vs HF 1/2）在 n=24 上是否成立。
- [ ] 20 条 `find_slot: non-consecutive token position` 是否无害。
- [ ] 骨架图（黑底彩线，OOD）在 llama.cpp 视觉路径上是否与 HF 视觉塔近似一致。
- [ ] llama-server 连续批处理对 PE 的实际倍率（HF 侧 K=3 只有 1.29–1.36×，算力墙可能仍在）。
- [ ] Q4_K_M 那一档（+1.26×）值不值 KLD 0.0019→0.0150 的质量代价——本次只测了 Q8_0。
- [ ] **P0 定标改动必须与换 PE 实现一起测**：§9 显示换提示词会改变模型对"骨架尺寸"与"A 的体型"的服从权重，单独换会得出误导性的 PCK 结论。
- [ ] 蒸馏路线（4479 token → ~400 token ≈ **11×**）与这条 runtime 路线**不冲突**：llama.cpp 同样能跑蒸馏出来的小模型，两条要一起排。
- [ ] 许可证：Qwen RESEARCH LICENSE 下的派生/量化权重交付华为方的口径（本次下载的是第三方派生权重）。


---

## 8. 闸 0 实测（同日夜间，权重与运行时已落盘）

**落盘位置**：运行时 `/data/shangguanrz/tools/llama-cpp/b11617-cuda12.8/`（llama.cpp **b11617**，commit `14310190c`，预编译 CUDA 12.8 包 + 配套 cudart 包，**不需要 nvcc**）；
权重 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF/`：`…pe_i2i.mtp.Q8_0.gguf` **9,786,060,352 B**、`…mmproj.bf16.gguf` **921,704,640 B**，两者**字节数与 HF API 公布值逐位相符**，SHA256 记在同目录 `SHA256SUMS`。
脚本在 `/data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/`（仓外，避免与在跑会话抢 git）。

跑通的东西：

| 检查 | 结果 |
| --- | --- |
| CUDA 后端能否在这张卡上起来 | **能**。`llama-cli --list-devices` → `CUDA0: NVIDIA RTX 6000D (85040 MiB, 40883 MiB free)`。**前提**：`LD_LIBRARY_PATH` 必须同时含 `llama-b11617` 与**同级的 cudart 目录**（cudart 是解到兄弟目录的，只给主目录时 `--list-devices` 直接返回 `(none)`——我第一次就踩了这个） |
| 模型 + mmproj + 3 张图 | 全部接受。真实 case 用 `workflow/runs/wp1_accept_hard02` 的 A/B/骨架 + 该 run 的 `effective_system_prompt.txt`（18,572 字符）+ 官方 instruction（2,192 字符） |
| MTP 投机解码 | **生效**：`creating MTP draft context`；收尾统计 **draft acceptance 0.644**（4214 生成中 2714 被接受）、**mean len 2.29** |
| 全 GPU 驻留 | 日志无 OOM、无 CPU fallback、无 offload 告警；跑完回收后 `torch.cuda.mem_get_info()` 回到 39.9 GiB free |

**这次拿到的第一个以前拿不到的数**：prefill 与 decode 终于分开了——**7354 token 的 prompt 处理 9.10 s（808 tok/s）**，decode 4821 token / 248.4 s。**但这次 tok/s 不能当基线**：当时另一会话的 `pe_rewrite_tiktok` **两个 shard 正在跑**（3 方抢同一张卡），19.4 tok/s 与我们 HF 的 18.4 几乎一样，**这个"没变快"是争用造成的，不是结论**。

**两个必须追的异常**（都还没解释）：
1. **契约拒绝 1/1**：单发一次没有走 `pe_worker.py` 的两次重试逻辑，而答案里出现了被禁的 B 引用（校验器原话 `PE prompt refers to B RGB, which the generator cannot see`）。HF 侧同期重试率约 8%（2/24），**n=1 不构成退化证据，但也说明必须按同一套重试口径重测**。
2. **prompt_tokens 7358 vs HF 6803（差 +555）**：当时以为是视觉/模板记账不一致。
   **§9 追加更正**：这是我自己的探针错——那次用了截断成 130 字符的 instruction，真实是 2,192 字符。
   同输入重测为 **HF 7358 vs GGUF 7412、HF 7478 vs GGUF 7532 ⇒ 稳定 +54**（BOS/模板级），**视觉记账两边一致，这条不再是风险项**。
3. 附：日志里 20 条 `find_slot: non-consecutive token position`（与多图位置/MTP 草稿有关），需要确认无害。

**下一刀（要窗口，不要现在抢）**：等 `pe_rewrite_tiktok` 空窗后，同一时段内跑 **HF BF16 vs llama.cpp Q8_0（MTP 开/关）** 的成对吞吐，并把 `pe_worker.py` 的两次重试口径搬进 harness；判定线仍是 docs §5 的 **单对 PE < 120 s 才值得接**。

---

## 9. 效果实测：HF BF16 vs GGUF Q8_0+MTP，同几何同 seed 成对（2026-10-15 凌晨）

做法：两对（`hard_02`、`hard_05`）各建两个 run，**几何与 seed 完全相同**，只有提示词来源不同——
HF 臂用 `pe_compiled`，GGUF 臂用 `--prompt-file`（`provenance.source=manual`，PE 产物不被覆盖）。
GGUF 臂的两次重试口径与 `pe_worker.py` 一致（seed+1 + 纠正后缀）。`evaluate.py` + `evaluate2.py` 各跑一遍。

| pair | 臂 | body PCK@0.2 | torso_ratio **guide** | torso_ratio **source** | identity_cos | bg MAE | person MAE | PE tok/s | PE 秒 | attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_02 | HF BF16 | **0.944** | 0.815 | 0.496 | 0.656 | 18.31 | 55.47 | 14.1 | 317 | 1 |
| hard_02 | GGUF Q8_0+MTP | **0.000** | **1.623** | **0.987** | **0.969** | **4.39** | **11.41** | 19.8 | 571.7 | 2 |
| hard_05 (source_torso) | HF BF16 | 0.833 | 0.886 | 0.886 | 0.558 | 6.73 | 36.45 | 8.2 | 637 | 2 |
| hard_05 (source_torso) | GGUF Q8_0+MTP | **1.000** | 0.856 | 0.856 | 0.413 | 6.63 | 40.20 | 19.6 | 415.5 | 2 |

**四条读数，按重要性排**：

1. **每 token 快了约 2×，但端到端一秒没省。** GGUF 稳定 19.6–19.8 tok/s，HF 在同一争用时段里 8.2–14.1；
   可两对合计 GGUF **987.2 s** vs HF **954.0 s**——**0.97×，等于没赚**。原因是 GGUF 两对**都触发了第二次重试**（HF 是一对一次过、一对重试），
   而且平均多吐了 214 token（5069 vs 4855）。**在这条流水线上，重试率比 tok/s 更值钱**，这句话应该写进成本模型。
2. **`hard_02` 的 PCK 0.000 不是"量化把提示词写坏了"**，看图与看尺度就明白：两臂生成的都是"站立抱臂"，
   但 GGUF 臂把人物画成 **guide 骨架躯干的 1.623 倍**（`torso_ratio_source` 却是 **0.987**，即几乎等于 A 本人真实大小），
   HF 臂则把人缩到 A 的 **0.496** 去迁就那张**本来就偏小的骨架图**，于是 PCK 拿了 0.944。
   ⇒ **PCK 单独用会误判**：它奖励"服从（定标错误的）骨架"，不奖励"保持 A 的体型"。必须和 `torso_ratio_guide/source` 同屏看。
3. **同一只"坏"臂在保持性上反而更好**：identity_cos **0.969 vs 0.656**、背景 MAE **4.39 vs 18.31**、人物 MAE 11.4 vs 55.5。
   `hard_05` 那对则是 GGUF 臂姿态更好（1.000 vs 0.833）但身份略差（0.413 vs 0.558）。**两对方向相反，n=1/格，不构成任何结论**。
4. **契约拒绝不是 GGUF 独有**：`hard_05` 的 **HF 臂自己也是 attempt1 rejected → attempt2 accepted**（`PE_RETRY … refers to B RGB`），
   所以 §8 那次单发被拒确实只是 n=1 噪声；但"GGUF 2/2 重试 vs HF 1/2"这个苗头值得在 n=24 上钉死。

**尺度这条为什么重要**：workflow README 记的 `hard_05` 定标问题（`source_person` 把躯干压到 95.9 px、生成 225.6 px，错配 2.35 倍）
与这里 `hard_02` 的 1.623/0.496 是同一个机制的两面。**提示词换实现会改变模型对"骨架尺寸"与"A 的体型"的服从权重**，
所以换 PE 实现这件事，必须和 P0 定标改动一起测，不能单独换。

**下一步该测什么（要 GPU 窗口）**：≥8 对 × 2 seed，主判据不是 PCK 单项，而是
**（PCK, torso_ratio_guide, torso_ratio_source, identity_cos, bg MAE）五元组 + 重试率 + 端到端秒数**；
并且把 tok/s 与"含重试的每对总秒数"分开报，否则会再次得出"快 2× 但没省钱"的结论。

---

## 10. 参考

**本机实测**
- GGUF 头/张量 dtype：`experiments/pe_i2i_gguf_selection_20261014/scripts/gguf_probe.py`（HTTP range 解析）
- 基线：`workflow/runs/wp1_accept_hard02/prompt/pe_result.json`（4479 tok / 243.7 s / 19.65 GiB）、`runs/**/*/stage.json` 中位数（PE 242.1 s、几何 31.5 s、出图 47.6 s）
- thinking/并发/512 各臂：`experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014/ablation*/…/stats.json` 与 README §4.3
- 工具链：`nvcc` 与 `/usr/local/cuda*` 不存在；`cmake 3.28.3`、`gcc 13.3`、`libvulkan.so.1.3.275`、`nvidia_icd.json`（api 1.4.312）；驱动 `580.173.02` vs 用户态 `libcuda 580.178.04`（NVML 因此不匹配，`nvidia-smi` 不可用）；`torch.cuda` 正常（RTX 6000D、cc **12.0**、83.05 GiB）
- 下载带宽：HF `resolve/main` 40 MiB / 4.21 s ≈ 10 MB/s
- 契约校验器：`src/pose_edit/prompt_contract.py:18-58`；角色后缀 `workflow/pose_transfer/prompts.py:15-29`；`--prompt-file` 与 provenance `workflow/run.py:177-202`

**外部（候选权重与作者自测）**
- [prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF)（7 档 + mmproj，`license: other`）
- [mozophe/Qwen-Image-2.1-PE-MTP-GGUF](https://huggingface.co/mozophe/Qwen-Image-2.1-PE-MTP-GGUF)（i2i/t2i × Q8_0/Q4_K_M × 含/不含 heretic，KLD 与速度表、imatrix 配方、MTP 嫁接来源 commit `c202236`）
- [mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP](https://github.com/mozophe/ComfyUI-Qwen-Image-2.1-PromptEnhancer-MTP)（**"Multi-image editing. Up to 10 input images, any size."**、"Keep thinking on"、8 GB VRAM 可跑 Q4_K_M、llama.cpp 后端 ≈2×(t2i)–3.5×(edit)）
- [llama.cpp](https://github.com/ggml-org/llama.cpp)：`src/models/qwen35.cpp`（Gated DeltaNet 张量 + "MTP layers are dense"）、预编译资产 `bin-ubuntu-cuda-12/-cuda-13/-vulkan-x64`、`ggml.h` 的 `ggml_type` 枚举（BF16=30）
- [Qwen-Image-2.1 官方 `prompt_rewrite/`](https://github.com/QwenLM/Qwen-Image-2.1/tree/main/prompt_rewrite)（官方 system/采样/max_length 16256 t2i、24000 i2i；也有 vLLM server 入口）
- [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I)（`chat_template.jinja` 7756 B、`system_prompt.txt` 18,344 B、Qwen RESEARCH LICENSE）
- 内部前作：`docs/workflow_PE重写成本_量化与蒸馏调研_20261014.md`（成本解剖、量化工具链天花板 ≈1.5×、`linear_attn` 被官方 recipe 排除）、`docs/Qwen-Image-2.1-官方提示词增强调研_20261008.md`

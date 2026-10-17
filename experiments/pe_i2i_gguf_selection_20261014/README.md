# PE-I2I 量化选型：GGUF 探测（2026-10-14）

## 实验目的

不下载整档就判断社区公开的 PE-I2I 量化权重**哪个适配我们的单人 workflow、哪个不浪费时间**。
判据是三条：架构与模板能不能用、敏感层（24 层线性注意力的 gate/output/alpha/beta）被压到几 bit、有没有 MTP 投机头。

## 里面有什么

- `scripts/gguf_probe.py` —— **HTTP Range 只读 GGUF 头**（默认前 48 MiB），解析 KV 元数据 + 全部张量元信息，
  按模块打印 dtype 分布。用法：

  ```bash
  /usr/bin/python3 scripts/gguf_probe.py <repo_id> <file1.gguf> [file2.gguf ...]
  ```

  它顺带回答两个问题：`tokenizer.chat_template` 在不在、长度是否等于官方 `chat_template.jinja` 的 7756 字节；
  `blk.N.{ssm_alpha,ssm_beta,ssm_out,attn_gate,mtp.*}` 各自是什么位宽。
  为什么走 range 而不是下整档：单个 Q4_K_M 就有 5.6 GB，而我们要的信息全在头部；HF `resolve/main` 支持 Range，
  sg 实测 ≈10 MB/s，下全档 10 分钟、读头 15 秒。
- `logs/` —— 探测原始输出。

## 关键结论（数字与完整对照见 `docs/workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md`）

- 两个仓库的 `general.architecture` 都是 **`qwen35`**，内嵌 **chat_template 7756 字符、含 think**，与官方模板同长；
  mmproj 视觉塔 **F32+BF16 未量化**。
- **prithivMLmods 的 plain Q4_K_M：`ssm_alpha`/`ssm_beta`/`ssm_out`/`attn_gate` 全是 Q4_K，且无 imatrix、无 MTP（427 张量）**。
  同一配置在另一作者的消融里 mean KLD = **0.146**（他自己 0.05 阈值的 3 倍）。
- **mozophe 的 i2i MTP Q4_K_M（442 张量 = 427 + 15 个 MTP）：`ssm_alpha/beta` Q8_0、`ssm_out`/`attn_gate` Q6_K**，
  i2i mean KLD **0.0150**、同 top-token 94.5%；其 Q8_0 mean KLD **0.0019**、同 top-token 98.5%。
- 速度（作者自测，RTX 4090 Laptop，i2i 两图）：ComfyUI int8 基线 19.4 tok/s → llama.cpp Q8_0 **67.8**（3.49×）→ Q4_K_M **85.2**（4.39×）；
  llama.cpp 不叠 MTP 也有 46.6。**我们的 HF BF16 实测 18.4 tok/s**，与作者基线同量级 → 瓶颈是每步固定开销，不是位宽。
- 本机前置条件：`nvcc` 与 `/usr/local/cuda*` **不存在**（CUDA 后端只能靠 llama.cpp 官方预编译包 `bin-ubuntu-cuda-12/-13`，或 `bin-ubuntu-vulkan-x64`；`libvulkan.so.1` + `nvidia_icd.json` 已在位）。

## 闸 0 实跑结果（2026-10-14 夜）

落盘：运行时 `/data/shangguanrz/tools/llama-cpp/b11617-cuda12.8/`（llama.cpp b11617、commit `14310190c`、预编译 CUDA 12.8 + 配套 cudart）；
权重 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF/`（`mtp.Q8_0.gguf` 9,786,060,352 B、`mmproj.bf16.gguf` 921,704,640 B，与 HF API 公布字节数一致，SHA256 见同目录 `SHA256SUMS`）。
执行位置在仓外 `/data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/`（避免与在跑会话抢 git 工作树）。

- **CUDA 能用**：`llama-cli --list-devices` → `CUDA0: NVIDIA RTX 6000D (85040 MiB, 40883 MiB free)`。
  **坑**：cudart 解在**兄弟目录**，`LD_LIBRARY_PATH` 只给主目录时 `--list-devices` 返回 `(none)`——我第一趟就把它误判成"不支持 sm_120"。
- **3 图 + MTP 跑通**：真实 case（`runs/wp1_accept_hard02` 的 A/B/骨架 + 该 run 自己的 18,572 字符 system + 2,192 字符 instruction）
  → `creating MTP draft context` 生效，**draft acceptance 0.644、mean len 2.29**；日志无 OOM、无 CPU fallback，跑完显存回到 39.9 GiB free。
- **第一次把 prefill 与 decode 分开量到**：prompt 7354 token 用 **9.10 s（808 tok/s）**；decode 4821 token / 248.4 s。
- **两个当时不能定论、后来查清的点**：
  ① 当时另一会话 `pe_rewrite_tiktok` **两个 shard 正在跑**（3 方抢一张卡），19.4 tok/s 属争用下读数；
  ② 单发一次被契约拒绝——**后来证实不是 GGUF 独有**：`hard_05` 的 HF 臂同样 attempt1 被拒、attempt2 才过（见下一节）。
  ③ **`prompt_tokens +555` 是我自己探针的错**：那次我用了截断成 130 字符的 instruction，而真实是 2,192 字符。
     同输入重测：HF 7358 vs GGUF 7412、HF 7478 vs GGUF 7532 ⇒ **稳定差 +54**（BOS/模板级），视觉记账两边一致。

## 闸 1 效果实测（2026-10-15 凌晨）

`scripts/pe_llama_batch.py` 对 `q8s_hard_02` / `q8s_hard_05` 两对真实 case 跑 GGUF Q8_0+MTP（两次重试口径照抄 `pe_worker.py`），
再用 `run.sh --prompt-file` 生成第二臂，`evaluate.py` + `evaluate2.py` 打分。四臂 run 名：`q8s_*`（HF 提示词）/ `q8l_*`（GGUF 提示词）。

**数字的唯一来源是 `docs/workflow_PE_I2I量化选型_GGUF社区权重评估_20261014.md` §9 的表**，本处只记三条判读：

- **每 token 快约 2×，端到端一秒没省**（两对合计 GGUF 987 s vs HF 954 s）——GGUF 两对都触发第二次重试且平均多吐 214 token。**重试率比 tok/s 更值钱。**
- **`hard_02` 的 PCK 0.000 是尺度问题不是提示词问题**：GGUF 臂 `torso_ratio_source` 0.987（几乎等于 A 本人尺寸）但 `torso_ratio_guide` 1.623；
  HF 臂把人缩到 A 的 0.496 去迁就偏小的骨架图，反而拿了 0.944。**PCK 不能单看，必须与两个 torso_ratio 同屏。**
- **同一只"PCK 0.000"的臂在保持性上更好**（identity_cos 0.969 vs 0.656、背景 MAE 4.39 vs 18.31）；`hard_05` 那对方向相反（GGUF 姿态 1.000 vs 0.833、身份 0.413 vs 0.558）。**两对相反、n=1/格，不构成结论。**

## 与其他实验的关系

- 上游：`experiments/3.singel man/tiktok_pe_prompt_rewrite_20261014`（吞吐对照，已否定 thinking-off / 512 / K=3 / fla / mamba-dtype 五条加速线；本目录换角度查 runtime 与第三方量化权重）。
- 结论并入 `docs/workflow_PE重写成本_量化与蒸馏调研_20261014.md` 的量化章节：那份讲"自己压权重能到多少"（≈1.05–1.25×），本目录讲"社区已有权重里哪条路更快"（换 llama.cpp + MTP）。
- 与 `docs/workflow_大方向优化调研_20261014.md` 的 P0（定标）耦合：换 PE 实现会改变模型对"骨架尺寸"与"A 的体型"的服从权重，**两件事不能分开测**。

## 还没做

- 没下 Q4_K_M（位宽那一档没对照）；没跑 `llama-perplexity --kl-divergence` 自证 KLD。
- 样本量只有 2 对 × 1 seed，**不足以支撑任何"效果变好/变坏"的结论**；下一步是 ≥8 对 × 2 seed 的五元组（PCK + 两个 torso_ratio + idcos + bg MAE）加端到端秒数。
- 没有改 `workflow/pe_worker.py`、`config.json`、`configs/asset_registry.json`（前两者在 prompt 阶段的缓存指纹里，最后一个别的会话正在动）。

## 2026-10-16：thinking 开关对照

新增 `scripts/thinking_switch_benchmark_v2.py`，在同一 Q8_0＋MTP 模型、三图、system/instruction、seed42、采样和 CPU 视觉路径下，分别测默认 thinking、模板 `--reasoning off`、运行时 `--reasoning on --reasoning-budget 0`，以及 128-token 短思考。每组使用独立服务，加载时间与请求时间分开；保存完整 API response（包括 reasoning）、原始文本和角色契约判定。每组只请求一次，不把格式失败伪装成可用延迟。

原版 `thinking_switch_benchmark.py` 保留：零预算组已完成，但切换下一组时遇到 TCP TIME_WAIT；v2 每组用独立端口，支持 `--arms` 选择未跑的组，不覆盖已经完成的零预算产物。新增 128-budget 是因为零预算文本违反源物品保持，需要检查折中方案。

`--dry-run` 只核对真实输入 SHA 与条件，不启动服务或创建 output。正式执行需要新 `--out`，已有目录会拒绝覆盖。输入从 workflow run 的 `prompt/pe_result.json` 读取；旧绝对路径因整理失效时，只接受新 run 的 inputs/geometry 中 SHA 完全相同的图片，并记录映射。

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python scripts/thinking_switch_benchmark_v2.py \
  --run /home/shangguanrz/project/pic-edit/workflow/runs/test/q8s_hard_02 \
  --out /data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014/output/thinking_switch_20261016_<时间戳> \
  --dry-run
```

去掉 `--dry-run` 后实测；先检查 GPU 余量，低于 24 GiB 会停止，不关闭其他会话进程。默认总输出上限 8192 在三组完全相同；如果触发 length 截断，则记为失败。这个探针不出图、不训练、不改生产 worker/提示词银行。共享 GPU、单个 case/seed 只能回答此配置下的行为，不构成稳定 20 秒或无质量损失的结论。

旧速度统计的口径修正与只读证据在 `research/pe_i2i_gguf_latency_audit_20261016/report.md`；原始实验记录保留。

实测已完成：[thinking 开关报告](report/thinking_switch_20261016_122659.md)。同一困难样本/seed，默认 168.86 秒/3319 token，模板关闭 31.35 秒/407 token，零预算 30.26 秒/367 token，128 预算 43.07 秒/622 token。均为排除加载、无重试的单次请求。普通关闭在 GGUF 上生效，但本例关思考的三个候选都要求删除源麦克风，默认组也有图号错误，尚无可直接替换生产且达到 20 秒的配置。初次与后续两份产物目录及原始回复均保留。

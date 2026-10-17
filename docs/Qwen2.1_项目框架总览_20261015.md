# Qwen-Image-2.1 人物姿态编辑：项目框架总览

**编写时间**：2026-10-15，Asia/Shanghai
**核对范围**：本地 `D:/codeplus/pictureedit`（Git HEAD `2b44828`）＋ 服务器 `sg`:`/home/shangguanrz/project/pic-edit`（Git HEAD `17991d3`，另有未提交修改）
**读者**：AI 大模型 / 新加入的工程师。本文是**给机器读的地图**，不是一篇宣传稿。
**语言约定**：中文叙述，术语保留英文原文（Adapter / PCK / latent / token / hook）。

---

## 0. 如何读这篇文档（LLM 读者须知）

本文每一条结论都带证据标签，请**按标签决定可信度权重**：

| 标签 | 含义 | 你该怎么用 |
|---|---|---|
| **[实测]** | 本次在磁盘上直接读到、或由代码逐行核对过（含文件路径/行号/JSON 键） | 可以直接引用 |
| **[文档]** | 只出现在 README / 设计文档里，磁盘上没有找到对应产物，或无法独立复算 | 引用时必须带"未见物证"限定 |
| **[已撤回]** | 曾写进文档、后来被控制组或索引错误证伪 | **不要**当结论用，只可作为"如何避免同类错误"的教训 |
| **[计划]** | 接口/文件尚未实现，只存在于 spec 或 plan 里 | 不要当成现有能力 |

三条最重要的前提：

1. **本地与服务器不是同一棵树**。目录布局、文件存在性、`experiments/` 的组织方式都不一样（见 §7.4）。任何"某文件不存在"的判断必须先声明在哪一侧。
2. **GPU 只在服务器上**。本地 Windows 是 CPU-only（python 3.13.13 / torch 2.12.0+cpu），只能读代码、跑 CPU 契约测试，不能训练或出图。
3. **仓库里的旧 README 有大量滞后**（根 `README.md`、`experiments/README.md`、部分 `docs/`）。本文 §8 专门列出**已被证伪的文档声明清单**。如果你只能读一份文件，读本文的 §8 再决定信谁。

---

## 1. 任务定义

### 1.1 一句话任务

给定 **源图 A**（一个人或两个人）与 **目标姿态来源 B**（一张图 / 一段视频帧 / 外部骨架），生成一张图：

- 人物的**动作姿态与面部表情**重定向到 B；
- **身份（Face ID）、服装与材质、背景**严格保持为 A；
- 允许身高、体型（betas）与视角保持 A 的原样 —— 即**体型与动作解耦**。

方向上这与"换脸/Head Swap"类工作**恰好相反**：BFS 那类开源 LoRA 是"保姿态、换身份"，本项目是"保身份、换姿态"。这个方向差异是后续多个实验负面结果的根因（§5.2）。

### 1.2 输入/输出契约（训练侧与推理侧不同，别混）

| 视角 | 看到什么 | 说明 |
|---|---|---|
| 训练 | A 的条件图（法线/骨架/部件/mask）+ A 的 latent + 目标 B 的 latent + 目标 RGB | 目标 RGB 只作监督，绝不进条件通道（有专门的泄漏门禁测试） |
| Adapter 推理 | A 的 RGB/latent + B 的几何条件 + 固定 prompt | 残差注入，基座冻结 |
| workflow 推理 | A 的参考图 + B 的 2D 骨架引导图（`B_3d_dwpose.png`）+ PE 重写后的 prompt | **不用任何自训 Adapter**，纯 prompt + 多图条件（§6.5） |

### 1.3 验收门槛（OKR）与达标情况

冻结在 `reports/OKR_人物图像编辑_20260929.md`，同时写进根 `README.md` §1.1：

| 指标 | 门槛 | 当前最好的实测值 | 判定 |
|---|---|---|---|
| PCK@0.02 | ≥ 0.98 | **0.9800**（iPER test 50，Qwen+Adapter，`qwen21_iper_body/four_arm_eval_50_s40/summary.json`）；另处文档写 0.979 | 达标（边界） |
| NME | ≤ 0.005 | **0.0047** | 达标 |
| ArcFace 余弦（身份） | ≥ 0.65 | 512 下最好 0.480；1024 严格子集 n=21 达 **0.660** | **未达标**（512 全线不达标） |
| 双人接触合理率 | ≥ 90 % | 双人线未得到有效度量：自动检测器在拥抱图只检出 1 人 | **未测量** |
| 主观优于开源基线 | ≥ +5 % | 官方 Qwen-Image-Edit-2511 基线**从未跑过**（P0 待办） | 未测量 |

**这就是整个项目的核心矛盾**：姿态几何已经打满，身份保持还差一大截，而两者互相拉扯——把姿态练得更狠，脸就掉。

### 1.4 三个被点名的技术瓶颈

1. **肢体接触交互**（握手/搭肩/拥抱）：接触归属、遮挡归属、人物不互换。
2. **衣物几何保真**：中晚期去噪被局部特征注入破坏全局服装/背景流。
3. **面部高频微纹理重定向**：512 全身图下 VAE 16× 下采样 ⇒ 整张脸只有约 32×32 token 空间，人脸约占 8×8 token，手指级信息在 latent 里就是亚像素（`docs/BFS_Body_Global_架构审查_20261008.md`）。**这是分辨率问题，不是模型选择问题**——换 20B 基座也一样。

---

## 2. 三条技术线

```
线一 DeepGen-1.0 + V6/V6.5/V6.6 适配器      ← 已完成，现在是"对照基线"，不再投入
线二 Qwen-Image-2.1 + 自训 Adapter（主线）    ← §3–§5
线三 workflow / workflow-multiple 流水线       ← 不训练，纯 prompt+几何引导，§6
```

三条线**共享**同一套几何条件构建代码、同一套评测器、同一份数据集。线二的实验结论直接催生线三：适配器身份掉得太狠 ⇒ 先做一条"不动权重"的流水线把交付兜住。线二和线三之间的接口**至今没打通**（workflow 明确 `adapter:false, lora:false`，见 §6.5）。

**为什么要有线一**：DeepGen-1.0（2.47 B MMDiT）上的 V6 适配器是本项目最早的成功控制方案，四臂对照里它就是"Qwen 那一行的对照组"。它的 20k 步权重还是 Qwen 适配器的**初始化来源**——Qwen Body Adapter 里 76.1 M / 92.19 M 参数是从 V6 迁移过来的（§4.5）。这条线没死，它在扮演"控制基线 + 权重供体"。

---

## 3. 模型选择

### 3.1 为什么最终选 Qwen-Image-2.1 作基座

理由**按证据强度排序**（来源：`docs/开源基线调研_Qwen同基座_20261004.md`、`reports/Qwen-Image-2.1_架构对比与迁移路线_20260924.md`、`docs/架构说明_body_face_hand_适配器.md`）：

1. **[实测] 四臂对照的量化优势**（iPER held-out 50 对，同评测协议）：见 §5.1 表。同适配器思路下 Qwen 比 DeepGen 多 **+1.3312 dB PSNR（CI [0.9808, 1.6681]，显著）**、PCK@0.02 高 2.5 点（CI [0.0109, 0.0406]，显著）。**限定**：这个优势只在"受控（adapter ON）"条件下成立；**裸基座对照不显著**（+0.2970 dB，CI 跨零），且两臂采样设置不对称（DeepGen 用 CFG 4.5 + 负向提示词，Qwen 用 CFG 1.0 无负向），详见 §5.1。
2. **[实测] 单流（single-stream）架构允许一个关键技巧**：把控制残差**只加在 target-token 后缀**上，前缀（文本 + 参考图）一个字都不动。双流 MMDiT 里 joint attention 序列做不到这么干净。这个技巧在代码里就是 `output[:, prefix:] + strength*delta`（`src/pose_control/qwen21/body_adapter.py:141-163`）。
3. **[实测] 原生多图条件 + Qwen3-VL 文本编码器 + 原生 RGBA**：这三项与项目自建的身份/纹理分支功能重叠，省掉一堆外挂工程。
4. **[实测] 显存放得下**：512 分辨率、单卡峰值 32.76 GiB，一张 83 GiB 的卡够。
5. **[实测] 32 层单流有已训练的同基座控制先验可借**（Qwen-Image-2.1-Fun-Controlnet-Union，见 §6.11）。

**明确不是理由的一条**：latent 分辨率。1024² 下两个基座都给 64×64=4096 token × 64 通道，手脸该糊还是糊。

### 3.2 评估过并被否决的选项

| 候选 | 结论 | 否决理由 [实测] |
|---|---|---|
| DeepGen-1.0（2.47 B MMDiT） | 保留为对照基线，不作主线 | `diffusers 0.35.2 / transformers 4.56.1` 环境**根本装不进** Qwen21 需要的 `QwenImage21Pipeline`（≥0.37.dev）与 `Qwen3VLForConditionalGeneration`（≥4.57.1）；就地升级会破坏 V6 已完成的 20k 步结果 |
| Qwen-Image-Edit-20B / **实际是 2511** | 不作基座；列为 **P0 外部基线，至今未跑** | `models/Qwen-Image-Edit-20B/`（du 54 G）的 `model_index.json` 是 `QwenImageEditPlusPipeline`，自带 model card 是上游 `Qwen/Qwen-Image-Edit-2511`。目录名"20B"与注册表名 `legacy_qwen_edit_20b` **都是错的**，只有 `docs/开源基线调研_Qwen同基座_20261004.md` 写对了。`src/` 与 `workflow/` 对它**零引用** |
| Qwen-Image **1.0** ControlNet（InstantX 2B）、DiffSynth In-Context-Control-Union | **结构不可移植** | 1.0 是 60 层双流 / 3072 维 / AdaLN / GELU；2.1 是 32 层单流 / 4096 / 共享 modulation / SwiGLU。没有可对齐的挂载点 |
| 2.1 生态里的现成姿态控制 | 只有一个，且不够 | `alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union`（7.55 GB ≈ 3.8 B，16/32 层，DWPose 输入，**无身份通道**）。2.1 上不存在 SMPL-X 控制器，也不存在保身份控制器 |
| 开源 BFS LoRA（BFS = Alissonerdx/BFS-Best-Face-Swap, MIT） | 结构可加载，**协议上负收益** | 见 §5.2，身份 −0.1075 [−0.1639, −0.0523]（8 演员 cluster bootstrap） |
| PE-I2I 的各种量化/加速路线 | 全线否决为"主手段" | 见 §6.11，llm-compressor 官方配方把 `linear_attn`（24/32 层）与 visual 锁 BF16 ⇒ 量化天花板只约 1.5× |

### 3.3 基座事实对照表 [实测]

| 项 | Qwen-Image-2.1（主线） | DeepGen-1.0（对照） |
|---|---|---|
| 类 | `QwenImage21Pipeline` / `QwenImage21Transformer2DModel` | `DeepGenPipeline` / `SD3Transformer2DModel` |
| 架构 | 单流 DiT，32 层 | 双流 MMDiT，24 层，`dual_attention_layers=[0..12]` |
| hidden | 32 heads × head_dim 128 = **4096** | 24 heads × head_dim 64 = 1536 |
| MLP | SwiGLU（`img_mlp.proj/gate_layer/out`），`mlp_ratio=3` | GELU |
| latent | **64 通道**，`patch_size=1` | 16 通道，`patch_size=2` |
| VAE | `AutoencoderKLQwenImage21`，空间下采样 **16×**，temporal 8，`z_dim=64`，base_dim 96 / decoder_base_dim 144 / dim_mult [1,2,4,8,8]，带 64 维 `latents_mean/std` | 8× 下采样，16 通道 |
| 位置编码 | RoPE 三轴 `axes_dims_rope=[16,56,56]` | `pos_embed_max_size=384` |
| 时间调制 | 共享 `modulation`；`causal_condition=true` 时**文本与参考图用 t=0，目标用当前 t** | AdaLN |
| 注意力 | block-causal（同图块内全注意力，跨块因果） | joint attention，`context_pre_only=True` ⇒ block 23 不出残差 |
| 文本/图像编码器 | **Qwen3-VL**（`text_encoder` 17.5 GB，8.767 B 参数，有 `lm_head`）；`context_in_dim=4096` | Qwen2.5-VL-3B（hidden 2048，36 层） |
| 权重体量 | 31 GB（transformer 14 GB = 9,968,332,504 + 4,261,951,904 B；text_encoder 4 shards ≈ 17.5 GB；vae 1.3 GB） | 14 GB |
| revision | **`790c92633540aa0cb11d9abf19eb46d861714758`**（代码里三处 pin：`qwen21/runtime.py:16`、`qwen21_full/qwen_runtime.py:18`、`prompt_study/protocol.py:27`），`_diffusers_version 0.37.0.dev0`，实际装 0.41.0 | Git commit `fbaaaf4ae1c79450d6a34b30beb4f59840568c69` |
| 许可 | **Qwen Research License**（仅研究/评估；商用需单独授权；派生权重须标 "Built with Qwen"） | 上游 DeepGen 条款 |

> ⚠️ `reports/Qwen-Image-2.1_架构对比与迁移路线_20260924.md` 把 DeepGen 的文本编码器写成"T5 + CLIP 拼接"——那是 SD3 的通用配方，**不是这个 checkpoint**。磁盘上 `vlm/config.json` 是 `Qwen2_5_VLForConditionalGeneration`。

### 3.4 辅助模型清单 [实测]

| 模型 | 路径 / 体量 | 用途 | 关键细节 |
|---|---|---|---|
| DWPose | `models/DWPose/dw-ll_ucoco_384.onnx` 134,399,116 B + `yolox_l.onnx` 216,746,733 B | 2D 关键点、评测 reader | ONNX 输出是 **COCO-WholeBody 133 点**；本仓渲染与 Hi4D 文件用 **134 点**（在 slot 1 插入 neck=肩中点）。**这个 133/134 差异出过事故**，转换函数集中在 `src/pose_edit/geometry.py::cocowholebody133_to_dwpose134` 与 `src/pose_edit/dwpose.py` |
| SMPLest-X (`smplest_x_h`) | `add/tools/SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar` ≈ 8.25 GB；模板 `human_model_files/smplx` | 从单帧回归 SMPL-X 参数 | **137 点 = body25 + hands40 + face72**；DWPose 包装是 body18+feet6+face68+hands42 = 134。**两套点集不可 reshape 互换**（`docs/experiments/2026-10-11-single-person-native-workflow-assessment.md`）。`pyrender` 需 EGL |
| SMPL-X / SMPL | `models/smpl/SMPL_NEUTRAL.pkl` | 网格/法线/深度/部件渲染 | 注意：这个目录里是 **SMPL（6,890 顶点）而非 SMPL-X（10,475 顶点）**，README 里有警告 |
| InsightFace antelopev2 | `scrfd_10g_bnkps.onnx` 16.9 MB（检测）、`glintr100.onnx` 260.7 MB（ArcFace R100）、`1k3d68.onnx`、`2d106det.onnx`、`genderage.onnx` | 身份余弦、5 点对齐、训练侧人脸监督 | L2 归一化后取 cosine |
| DINOv2 | `models/antelopev2/dinov2-giant/model.safetensors` 4,546,005,432 B（hidden 1536, 40 层, patch 14） | 外观/纹理特征（dual_v8 外观库的 1536→512 支路） | **只能当外观代理，不能当身份真值** |
| face_eval TorchScript | `identity_encoder.ts` 261 MB / `landmark_encoder.ts` 5.2 MB / `perceptual_encoder.ts` 30.6 MB / `lpips.ts` 59 MB | 训练期 identity / landmark / VGG-perceptual / LPIPS 损失 | `torch.jit.load(..., map_location='cuda')`，冻结；**这就是本地跑不了测试的原因** |
| SAM2 / Multi-HMR / nvdiffrast / Sapiens | `add/tools/*` + `add/envs/*` | 分割、3D 人体回归、离屏渲染法线/深度/部件、人体分割 | iPER 的 `iper_assets_512_nvdiffrast/` 条件包由 nvdiffrast 离线渲染 |
| CHAMP | `third_party/champ_guidance` + `datasets/champ_motions_example` | DeepGen 时代 3D 引导过拟合实验 | **已被 Qwen 线取代**，产物仅作历史证据 |
| PE-I2I | 见 §3.5 | 提示词重写 | 不是生成模型，是 VLM 重写器 |

### 3.5 提示词重写模型 PE-I2I 与其 runtime 备选 [实测]

- **模型**：`Qwen/Qwen-Image-2.1-PE-I2I`，与基座同日（2026-09-20）发布，**从 Qwen3.5-VL 9B 微调**（`general.architecture = qwen35`，`Qwen3_5ForConditionalGeneration`），**9.410 B 参数**，32 层中 **24 层是 Gated DeltaNet 线性注意力**，BF16。
  落盘 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I`（18 GB = 4 shards: 5,074,583,064 + 5,089,028,664 + 5,089,028,760 + 3,567,080,680 B）。
  `system_prompt.txt` **18,344 B**，`chat_template.jinja` 7,756 B（含 think 段），vocab 248,320。
- **官方调用契约**（`docs/Qwen-Image-2.1-官方提示词增强调研_20261008.md`）：一条编辑指令 + 最多 10 张参考图，引用写 `<image1>…`；采样 `max_new_tokens=24000, do_sample, temperature=1.0, top_p=0.95, top_k=20`；**I2I 绝不能设 `presence_penalty`**（T2I 才需要 1.5）；`enable_thinking=True` 是强制的。输出 JSON `{rewritten_prompt, wh_ratio, ratio_follow}`，后两者互斥。
- **runtime 备选（已下载但收益为零）**：`llama.cpp b11617` CUDA12.8（`/data/shangguanrz/tools/llama-cpp/`）+ `mozophe` 的 **i2i MTP Q8_0** GGUF `9,786,060,352 B` + `mmproj.bf16.gguf` `921,704,640 B`（合计 10.7 GB，比 BF16 省 43.1 %）。选型三条硬标准：架构/模板兼容、**24 个线性注意力的 `ssm_alpha/ssm_beta/ssm_out/attn_gate` 敏感层的位宽**、有无 MTP 草稿头。被否决的是 prithivMLmods 的 plain Q4_K_M（这些层全在 Q4_K，无 imatrix，作者自测 mean KLD **0.146** = 其本人 0.05 淘汰线的 3 倍）；保留敏感层到 Q8_0/Q6_K 后 KLD 降到 0.0150，Q8_0 是 0.0019。
- **BFS LoRA**：`bfs_body_swap_v1.0_qwen_2.1.safetensors` 209,753,576 B（sha `7dc0a53a…`，136 modules / 272 tensors / 104,857,600 参数，rank 64，**注入 24 层 = 0–15 + 24–31，跳过 16–23**）；`bfs_head_v1.1_qwen_2.1.safetensors` 260,096,144 B（sha `d1d748d5…`，352 raw → 384 tensors / 192 modules）；`bfs_head_v1.1_alternative` 448 tensors / 224 modules。
  **移植陷阱**：ComfyUI 的融合 `img_mlp.gate_up`（24576）必须**先 gate 后 proj** 拆成两个 12288，否则 diffusers 静默丢 MLP 权重。转换函数 `convert_bfs_state_dict` 在根目录 `bfs_explore.py`，`attached_targets=304/384`。`~/models/bfs/` 下 krea2 / ltx2 / flux-klein / 2509 / 2511 各族**基座不匹配，禁止使用**。

---

## 4. 模型架构（Qwen 侧）

### 4.1 基座的四个不可破坏契约 [实测]

任何控制/条件方案都必须服从（细节与源码快照见 `research/qwen21_controlnet_transfer_20261015/REPORT.zh-CN.md`）：

1. **Latent 与空间坐标**：64 通道、16× 下采样；512×512 ⇒ 32×32 = **1024 latent token**；Transformer `patch_size=1`。**VLM 的一个图像 slot 展开成 2×2 = 4 个 latent token**，不能把 slot 数当 token 数。原生 pipeline 要求宽高对齐 **32**（不是模型卡说的 16）。
2. **文本与图像联合序列**：源图先由 Qwen3-VL 提供图文条件，VAE latent 填入对应图像位置，**目标噪声图在序列末段**。序列里文本与图像 slot 交错，不能假设前缀都是文本。
3. **block-causal attention**：目标能读前面的文本与参考图，前缀**不能**回读目标。改注意力规则等于改"身份条件"的语义。
4. **原生时间调制与 RoPE**：`modulation` 全 block 共享；`causal_condition=True` 下文本/参考图用 t=0、目标用当前 t；RoPE 三轴 `[16,56,56]`。控制侧必须复用同一套坐标、有效 token mask 与时间语义。

### 4.2 Qwen 侧 I/O 装配 [实测]（`src/pose_control/qwen21/runtime.py`）

```
encode_prompt(prompt, image=[vlm_image]) -> (prompt_embeds, prompt_mask, image_mask)   # 必须在 GPU 上
RGBA -> image_processor.preprocess(...).unsqueeze(2) -> _encode_vae_image(Generator(seed=0))
     -> flatten(3).squeeze(2).transpose(1,2) -> [B, tokens, 64]
断言：tokens == (resolution//16)**2 且 tokens % 4 == 0
transformer_inputs:
    hidden_states = cat(source_latent, noisy_target, dim=1)      # 两条各 1024 token（512 下）
    img_shapes    = [[(1,32,32)] * 2] * B
    encoder_hidden_states = prompt_embeds
    img_mask      = 各条件 image_mask 拼接 + ones(1, tokens//4)   # 注意 /4 打包系数
    return_dict=False
CFG：true_cfg_scale=1.0（所有评测脚本硬编码）；Qwen 路径无负向 prompt（DeepGen 对照臂有，见 §5.1）
采样步数：eval body 默认 40；eval face / eval full 默认 30；controlled_pipeline 默认 28
冻结：transformer / text_encoder / vae 全部 requires_grad_(False)，transformer.eval()
```

`use_kv_cache` 是真实存在的 pipeline kwarg。仓库里**不构造 scheduler**：flow-matching 由 pipeline 继承，训练直接对 `sigma ~ U[0,1)` 采样。

### 4.3 条件契约（raster + global + token） [实测]

公开维度**被冻结成契约**，偏离即 raise（`src/pose_control/v6/condition_injector.py:41-51,97-102`）：
`spatial_dim=256, token_dim=256, contact_channels=128, max_contact_relations=8`。

| 张量 | 形状 / 范围 | 来源 |
|---|---|---|
| `normal` | `[B,3,H,W]` ∈ [−1,1] | nvdiffrast 离线渲染 |
| `pose_heatmap` | `[B,25,H,W]` ∈ [0,1] | 25 点高斯热图，score 门限 >0.05（`src/data/iper_dataset.py:121-137`）；Hi4D 侧门限 ≥0.25 |
| `part_onehot` | `[B,14,H,W]` one-hot | 14 个 `PART_NAMES`（顺序与 `Body_Adapter_SMPLX_输入审查` 一致） |
| `human_mask` | `[B,1,H,W]` | — |
| `depth` | `[B,1,H,W]` 可选 | qwen21 lite 里 `use_depth=False` |
| `smplx_global` | `[B,26]` | **betas[:10] ‖ rot6d[6] ‖ transl[3] ‖ cam[7]**；逐字核对自 `scripts/build_iper_v6_conditions.py:318` 与 `build_tiktok_v6_conditions.py:349`。**没有手参数，也没有脸参数** |
| `relative_geometry` | `[B,11]`（仅双人） | — |
| `contact_raster` | `[B,2,H,W]`（仅双人） | — |
| `person_valid` / `person_b_valid` | `[B,2]` / `[B]` | 单人训练时第二人必须全 0，否则 raise |
| `task_id` | SINGLE 0 / DUAL 1 / HUG 2 / HANDSHAKE 3 / SHOULDER_HUG 4 / HOLD_HAND 5 / UNKNOWN 6 | — |

编码器（`geometry_encoder.py:114-155`、`token_encoders.py:117-176`）：

- `SpatialConditionEncoder`：normal stem 3→(32,64,128)；pose stem 25→(64,96,128) `project_to=32, sparse_init`；part stem 14→(64,96,128) 同样 32；`fusion_input = 128*3 = 384` → Conv1×1→256 + GroupNorm/SiLU + 2 个 `ConditionResBlock` + `RoleFiLM(256)`。
- `GlobalConditionTokenEncoder`：`LN(26)→Linear(26,512)→SiLU→Linear(512,1024)→SiLU→Linear(1024,4*256)` ⇒ **每人 4 个 token**。
- `RelativeGeometryTokenEncoder`：`LN(11)→512→2*256` ⇒ 每人 2 token。`ContactRasterEncoder`：Conv 2→32→64→128，stride 2 ×3。
- `PersonTokenBinder`：`Embedding(2, token_dim) ⊕ Embedding(16, token_dim)`，要求每个有效 pair 的 `source_indices` **唯一**。
- `AdapterReasoningConfig`：`condition_dim=256, hidden_dim=512, downsample=2, heads=8, layers global=1/cross_person=2/contact=1, ffn_ratio=4, contact_gate_init=-4.0`。

### 4.4 注入机制：本项目最核心的一个设计 [实测]

**不是 cross-attention，不是额外 token 流，不是 `img_cfg_in`（这个名字全仓不存在）。** 是**只加在 target-token 后缀上的、逐 block 的加性残差**，靠 `register_forward_hook` 实现：

```python
# src/pose_control/qwen21/body_adapter.py:141-163（简化版）
prefix = output.shape[1] - self.target_tokens
target = output[:, prefix:]
controlled = target + strength * delta
return torch.cat((output[:, :prefix], controlled), dim=1)
```

- 一个 `register_forward_pre_hook(with_kwargs=True)` 挂在 transformer 上，从入参里刮 `timestep`→sigma 与 `img_shapes[0][-1]`→`target_tokens = h*w`；
- 每层一个 hook 挂在 `transformer_blocks[i] for i in (3,8,13,18,23,28)`；断言恰好 32 个 block；用 `_pic_edit_qwen21_control_active` 单标志位防止上下文并发。
- **零初始化 head ⇒ 控制 ON 与 OFF 在第 0 步逐位相同**。训练前有一道不变式 `verify_zero_control`：max|native − controlled| ≤ **1e-3**，否则不开训（resume / `--init-checkpoint` 时跳过）。
- `body_face_scale`：身体残差**只在 layer index ≥ 18 时**于人脸 ROI 内被衰减，`delta = delta*(1-(1-body_face_scale)*face_state.mask)`。

**两种注入面必须分清**（这是文档最常搞错的地方）：

| 路径 | 注入面 | 代码位置 |
|---|---|---|
| `qwen21/`（lite / dual lite） | 6 个 block：`(3,8,13,18,23,28)` | `body_adapter.py:16` |
| `qwen21_full/`（主线） | **全部 32 个 block**，分 6 组 `0–5│6–10│11–15│16–21│22–26│27–31`（尺寸 `[6,5,5,6,5,5]`），同组复用同一残差 `residuals[6*layer//32]` | `qwen21_full/qwen_adapter.py:32,585`、`qwen_runtime` 上下文 `:649-650` |

Full 版一个去噪 step 的残差在 **pre-hook 里一次性算完**（读 `hidden[:, -target_tokens:]`、缓存的 prompt embeds、`img_shapes`），并用 sigma 相等去重，使 CFG 的同一 timestep 双次调用只算一次。

### 4.5 适配器谱系：三代实现 [实测]

| 代 | 类 / 文件 | 参数量（本机实例化） | 状态 |
|---|---|---|---|
| lite Body | `Qwen21BodyAdapter(width=512, hidden_size=4096)` `src/pose_control/qwen21/body_adapter.py:19-93` | **92,186,060**（= 7.155 M injector + 64.093 M reasoner + 4.850 M bridge = 76.1 M 从 V6 迁移 + **16.089 M** Qwen 侧新增） | **live**，iPER 里程碑用它 |
| lite Face | `Qwen21FaceAdapter` 同目录 `:23-101` | **17.987 M** | live |
| lite Dual | `Qwen21DualAdapter` 同目录 `:29` | **121.465 M**（+29.279 M over body） | **dead**（唯一 importer 是 `dual_inference.py`，而它 0 引用者；服务器侧 `scripts/{train,eval}/` 有入口） |
| **Full** | `Qwen21FullAdapter` `qwen21_full/qwen_adapter.py:124-453`，内部包 `UnifiedSMPLXAdapterV6`（`deepgen_adapter.py:38`） | 控制核维持 **1536 维**（DeepGen/SD3 计算图） | **live 主线** |
| dual_v8 | `research/qwen21_dual_upgrade_20261011/dual_code/`（16 文件 2195 行，`architecture_version="dual_v8"`） | 设计预算"数千万" | **[计划→已部分落地为 pilot]**：不在 `src/`，设计文档自己写"模块和 CLI 是计划接口，尚不可运行"；但 `experiments/qwen21_dual_v8/` 确有 300 步 pilot 权重 |

Full 版是"**旧控制核 + 新边界桥**"：`latent_bridge=Conv2d(64,64,1)`（identity-init）、`text_bridge=Linear(4096→context_dim)`（等维则 eye-init）、`pooled_bridge=Linear(4096,2048)`、`output_bridges = 5 支 Linear(1536→4096, bias=False)`；`BRANCH_NAMES=("geometry","interaction","detail","face","hand")`。`reset_output_bridges` 强制**恰好一个零边界**（继承 head 非零才零初始化，否则 xavier），注释原话："two consecutive zero maps cannot learn"。

> **这就是 §6.11 里 Union ControlNet 报告的靶子**：Full 版跑的是 DeepGen 的 SD3 控制计算图（`SD3ControlNetModel(num_layers=1)` 循环 6 个 stage），只在出入口做维度投影。中间 attention、位置编码、时间调制**都不是 Qwen 原生的**。文档与实际还有一处名字骗人：`QwenFullControlOutput.block_controlnet_hidden_states` 只是属性名，本机 Qwen21 的 `forward` 没有这个入参，实际注入靠 forward hook。

**lite Dual（= `legacy_dual_lite`，命名来自 `research/.../dual_code/api.py:19`）的特征表示**：每人 `[B,2,...]`；`_pool_geometry` 要求 token 数是**完全平方数**，adaptive-pool 到 (2,2) ⇒ 每人 4 token；`_source_tokens` 按 bbox 裁共享 source latent 后 pool 到 (2,4) ⇒ 每人 8 token；深度以 `depth_in=Conv2d(2→512,1)` 加进几何（尽管注入器仍 `use_depth=False`）；三层残差各自被 `target_mask` 门控。**新组件并非全零初始化**：只有 `depth_in`、`interaction_heads`、`appearance_heads` 和继承来的 body `zero_heads` 是零，`dual_global_encoder`/`source_projection`/`person_binder`/`interaction_in`/`interaction_fusion`/`appearance_fusion` 用 PyTorch 默认初始化（文档 §4.4 说"全部零初始化"是**错的**）。

**dual_v8 相对 lite dual 的改动**（`2026-10-11-qwen21-dual-v8-design.md`，423 行）：typed `_Transfer` schema 逐字段 assert 形状；几何 `normal [B,2,3,H,W]` 等 20 个张量；外观库 `[B,2,16,512]` = 8 个 VAE token + 8 个 DINO token，经 2 层 masked cross-attention resampler；几何节点 **14 个解剖节点/人 + 28 节点接触图层**（邻接 `eye(28) ⊕ ANATOMICAL_EDGES`）；**槽位对称的深度路由**（`max_pool2d(kernel=17,pad=8)` 做 8 px 膨胀，深度完全相等时按 0.5 平分，**故意不用槽位序号做打破**）；`DualGeometryEncoder` 对 A/B 两个槽位复用同一条 `RoleFiLM` 路径（"B identity is never learned"）；每层 **3 个 head**（`Linear(512,64)+SiLU+Linear(64,4096)`），只有**最后一个投影**零初始化，理由写在 `controller.py:68-80`——"so gradients reach it on step one"。

**LoRA 现状**：全仓**没有任何 PEFT 训练**。`rank=64` 只作为复制来的 V6 内部低秩模块存在（`FaceControlAdapter(face_dim=512, deepgen_dim=1536, num_stages=6, rank=64, heads=8)`、`HandControlAdapter(dim=512, deepgen_dim=1536, rank=64, heads=8)`、`LowRankStageAdapter`）。外部 LoRA 只有 **只读加载**：`frozen_bfs_lora.py` 解析 `diffusion_model.<name>.lora_[AB].weight`，rank 从 `lora_A.shape[0]` 读，并把 transformer 强制 `requires_grad_(False)`。

**Hand 分支是"看起来已发布，其实关着"** [实测]：`qwen21_full/__init__.py:34-38` 导出了 `HandFineCondition` 等，`deepgen_adapter.py:106-108` 会构造 `HandControlAdapter`，但**所有 Qwen 调用点硬传 `hand_strength=0, hand_mode='off'`**（`execution.py:635,641`；`train_iper_qwen21_full.py:244,247`；`eval_iper_qwen21_full.py:143`）；`hand/vton_layers.py`、`hand/refiner_export.py` 零引用者。唯一的例外是 joint pilot 的 **BFH（Body+Face+Hand）** 分支——它确实跑了，见 §5.3。

### 4.6 训练 [实测]

**Body flow loss**（`train_iper_qwen21_body.py:144-161`，等价实现 `src/data/letterbox.py:194-202`）：

```
sigma   = rand(B)
noisy   = (1-sigma)*target + sigma*noise
velocity= noise - target
mask    = bilinear_resize(human_mask, (grid,grid))
weight  = (1 + 2*mask).clamp(1,3)          # 人体 3×，背景 1×
loss    = ((pred-velocity)^2 * weight).sum() / (weight.sum() * C)
```

`valid_body_velocity_loss` 额外乘 `valid_tokens`，分母为 0 时 raise。

**Face 侧总损失**（`qwen21_full/execution.py:716`）：

```
total = flow + 0.10*identity + 0.05*perceptual + 0.02*outside
identity    = mean(1 - cosine(ArcFace(gen), ArcFace(tgt)))
perceptual  = MSE(VGG_feats)   # 从 x0 = noisy - sigma*prediction 解码得到
flow        = 'global'(全图均值, 历史默认) | 'roi'((sq*mask).sum()/(mask.sum()*C))
outside     = 与一次 face_strength=0 参照前向的 MSE，权重 (1-mask)
sigma 受限：rand * (0.35 若 auxiliary 分支, 否则 0.65)，且重抽直到 cosine schedule 非零
```

**优化器与分组学习率**：`AdamW(weight_decay=0.01)`，grad clip **1.0**，出现非有限值直接 `FloatingPointError`。

- `qwen21`(lite)：两组 —— 继承权重 `--transferred-lr 5e-5`，新参数 `--new-lr 1e-4`。
- `qwen21_full`：三组 —— `heads_and_interface 1e-4`、`encoders 5e-5`、`control_core 1e-5`，统一乘 `--lr-scale`。
- **Face 分支的 LR 后来被下调到 1e-6 / 5e-7 / 1e-7**（`docs/tiktok_qwen21_full_training.md`）：原先用 1e-4/5e-5/1e-5 时，固定噪声下 face loss **2 次更新内从 0.437 冲到 2.792**。

**批与精度**：lite `--accumulation 4 × --micro-batch-size 1` ⇒ 有效 batch 4；Full 无累积、B=1、`--max-steps 100`、`--warmup-steps 20`。transformer 在 `torch.autocast('cuda', bfloat16)`，**适配器本体保持 FP32**。分辨率默认 512，`choices=(512,1024)`。**全仓无 EMA**（穷举 grep）。**全量微调外置适配器，不是 LoRA**，且基座冻结是断言级：`if any(p.grad is not None for p in pipe.transformer.parameters()): raise AssertionError`。

**Checkpoint**（原子写：`.tmp` + `replace`）：

- lite：`{adapter_state_dict, optimizer_state_dict, step, micro_step, args, best_validation, torch_rng_state, cuda_rng_state}`
- full 额外：`{architecture, model_revision, branch, resolution, initialization(loading_report), initialization_provenance, sample_order, train_pairs_sha256, val_pairs_sha256, python/numpy rng}`

**Resume 契约**：拒绝分辨率不一致；恢复 adapter/optimizer/step/micro-step/全部 RNG；`best_validation` 仅在 `val_count` 与 `val_pairs` 都匹配时保留；`load_adapter_checkpoint` 拒绝 `architecture != "qwen21_full"`；`validate_resume_configuration` + `reconcile_resume_metrics` 账本处理截断；face 分支的损失配置键不匹配就 raise。`--init-checkpoint` = 只取权重、优化器全新。

### 4.7 评测实现 [实测]

**两个世界，不要混：**

1. **有 GT 的 root 级评测**（`eval_iper_qwen21_{body,face,full}.py`）：只算 `mae / mse / psnr / ssim(skimage) / human_mae / background_mae`，且 **full 把所有指标先 LANCZOS 统一到 512 网格再算**（`metric_resolution: 512`），即使输出是 1024。`eval_iper_qwen21_body.py:130-131` 断言每个控制层在采样期间**至少触发过一次**。**这些脚本自己不实现任何姿态指标。**
2. **关键点/身份指标**（`src/evaluation/`）：**四臂评测表里的 `pose_pck_002/005`、`pose_nme_diagonal`、`identity_cosine` 全部由这里产出**，口径是 **COCO-17 身体关节 + 图像对角线归一 + 生成图漏检关节计为失败 + 身份只在双侧脸都检出的配对上算（n 会小于样本数）**。
   - `joint_pilot_metrics.py`：DWPose ONNX（CPUExecutionProvider，4 线程，只取面积最大的人）；`COCO_TO_BODY25 = (0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11)`；可见门限 target score > 0.3，匹配门限 det score > 0.3；**距离按图像对角线归一化**；`pose_pck_002 = (dist < 0.02)/denominator`，另有 `_005`、`pose_nme_diagonal`、`pose_joint_recall`。身份 = insightface antelopev2（scrfd 检测 + glintr100 嵌入），L2 归一化 cosine。
   - `no_gt_metrics.py`：134 点布局，`GROUPS = body[0:18], feet[18:24], face[24:92], left_hand[92:113], right_hand[113:134]`；`PCK_THRESHOLDS = (0.2, 0.5)`；归一化优先用躯干像素，否则 `hypot(w,h)/4`；**`_strict` 变体把分母换成"请求的关节数"，这样把人裁出画外就无法得分**。
   - `v6/detail/metrics.py`：手 **PCK@0.1**（`threshold = 0.10*hbox_scale`），脸 NME 用 24:92 点；检测门限 mean score < 0.15、有效门限 score ≥ 0.30。
   - **LPIPS 不在 Qwen 路径里**（只在 `train_iper_v65_face.py:492-516`）。

### 4.8 一次训练/评测的资源账 [实测]

| 项 | 值 |
|---|---|
| 512 分辨率 Qwen 受控推理峰值 | **32.76 GiB**（DeepGen 同题 14.53 GiB） |
| 1024 单图成本 | 18.9 s/图、39.19 GiB |
| 512、30 步、单样本（workflow） | 47.6 s 中位、33.97 GiB 峰值 |
| 85 GB 卡上的并发 | 单卡串行是默认姿势；GPU 与人类共用（见 §9） |

---

## 5. 实验方案与进展

### 5.0 一览表（Qwen 相关）

| 实验目录 | 侧 | 状态 | 一句话结论 | 可引用性 |
|---|---|---|---|---|
| `qwen21_iper_body` | 服务器 `3.singel man/` | **完成** | Qwen+Adapter：PSNR 25.86、PCK@0.02 0.980，但身份从 0.8187 掉到 0.3510 | **[实测]** summary.json 逐行核对 |
| `bfs_compare_30_20261004` | 本地 + 服务器 | 完成 | BFS head LoRA 在身份上**输给**基座二次采样 | **[实测]** 102/102 槽位 + 演员 cluster CI |
| `bfs_body_face_joint_pilot_20261008` | 两侧数字**不同** | 本地=被取代的 512 归档；服务器=native 1024 步 + BFH | 联合 Body+Face(+Hand) 适配器胜 BFH 身份；BFS 两支身份都输 | **[实测]** 但本地 handoff 文档数字**全部对不上** |
| `qwen-2.1-test` | 两侧结构不同 | 早期原型，完成 | 纯 prompt 路线的能力上限探针；单样本过拟合 12.35→29.85 dB | **[实测]** results.json |
| `hi4d_qwen21_dual_20260930` / `qwen21_hi4d_dual` | 服务器 `4.two man ` | **停在 step 410，未存 checkpoint** | 10k 双人全量训练从未发生；"检测率 100%/swap 0.0" 属 **[已撤回]** | **[实测]** stopped_status.json |
| `qwen21_dual_v8` | 服务器 `4.two man/`，本地只有 5 张 PNG | pilot 跑完（300 步、12/12 评测格） | **两个头号门禁全失败**：双人检出 0.0，外观/身份指标 undefined | **[实测]** eval.log |
| `tiktok_pe_prompt_rewrite_20261014` | 服务器 | **正在运行** | 10,850 对逐对重写，截至 2026-10-15 10:42 完成 **503/10,850** | **[实测]** 进程与 records/ 均验证 |
| `pe_i2i_gguf_selection_20261014` | 服务器 | 完成（范围自我限制） | 换 runtime 才是加速落点；但端到端实测**净收益为零** | **[实测]** gate0/gate1 JSON |
| `tiktok_overfit_sanity_full` | 本地 | 完成，**verdict FAIL** | 六门禁 C3/C6 失败，直接终结"单层脸特征强注入"路线 | **[实测]** train_metrics.jsonl 1500 行完整 |
| `tiktok_00001_condition_repair` | 本地 | 残缺 | README 声称 44 图 + 权重 + 日志；磁盘上 **8 个文件、6 张 PNG** | **[文档]，不可引用** |

### 5.1 里程碑：`qwen21_iper_body` 四臂对照 [实测]

配置：基座 revision `790c926…`；DeepGen 控制臂 = `iper_v6_native_20k_v1/checkpoint_last.pt`（20k 步），Qwen 臂 = `full_20k/checkpoint_last.pt`（771,939,025 B）；**50 对 held-out iPER**（`iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl`），69 训练 / 19 测试 appearance、**0 外观重叠**；512²、40 步、seed 42、CFG 4.5（DeepGen）/ 1.0（Qwen）、KV cache 开。

来源 `four_arm_eval_50_s40/summary.json`：

| 指标 | deepgen_off | deepgen_on | qwen21_off | **qwen21_on** |
|---|---|---|---|---|
| PSNR | 17.2377 | 24.5321 | 17.5347 | **25.8633** |
| SSIM | 0.8017 | 0.8756 | 0.8363 | **0.8969** |
| PCK@0.02 | 0.1744 | 0.9550 | 0.2031 | **0.9800** |
| PCK@0.05 | 0.4848 | 0.9812 | 0.4705 | **0.9918** |
| human_mae | 60.008 | 20.031 | 61.165 | **17.575** |
| **identity_cosine** | 0.2612 | 0.2917 | **0.8187** | **0.3510** ← 灾难 |
| 秒/峰值 GiB | 5.08/14.51 | 5.94/14.53 | 5.84/32.76 | 5.87/32.76 |

**必须同时说清的四件事**（最后两条由本次直接读取 `paired_deltas_second_minus_first` 得到，**任何 README 都没写**）：

1. **5000 次配对 bootstrap（seed 20260927）的 CI 是"固定 50 对的描述性不确定性"**，原话 `descriptive uncertainty for the fixed 50 pairs` —— 不是对总体的推断。
2. **适配器的因果增益在两个基座上几乎相同**：`deepgen_control_gain.pose_pck_002` = +0.7805 [0.7191, 0.8362]，`qwen21_control_gain` = +0.7769 [0.7184, 0.8302]，CI 高度重叠。**"控制有没有用"这件事上 Qwen 不比 DeepGen 更强。**
3. **基座裸模型之间的差异：姿态与像素误差都不显著**。`backbone_native_change`（qwen_off − deepgen_off）里 **PSNR +0.2970 dB CI [−0.2422, +0.8397]、human_mae +1.156 CI [−3.716, +6.609]、PCK@0.02 +0.0287 CI [−0.0212, +0.0786]、PCK@0.05 −0.0143 CI [−0.0896, +0.0612]、NME +0.00759 CI [−0.00009, +0.01611] 全部跨零**；跨零之外**显著**的是身份与结构/背景项：identity_cosine +0.5575 CI [0.5082, 0.6049]、identity_target_cosine +0.2863 CI [0.228, 0.341]、SSIM +0.0346 CI [0.0134, 0.0594]、human_ssim +0.0162 CI [0.0074, 0.0269]、background_mae_source −6.9416 CI [−7.7764, −6.1269]、background_ssim_source +0.0832 CI [0.063, 0.1064]。
   ⇒ 裸 Qwen 相比裸 DeepGen 的可靠优势是**身份保持与背景/结构稳定**，不是姿态精度。
4. **但受控条件下的 Qwen 优势是显著的**：`controlled_model_change`（Qwen_on − DeepGen_on）PSNR **+1.3312 dB，CI [0.9808, 1.6681]**、PCK@0.02 **+0.0250，CI [0.0109, 0.0406]**、PCK@0.05 +0.0106 CI [0.0012, 0.0200]、human_mae **−2.456，CI [−3.589, −1.279]**、identity_cosine +0.0593 CI [0.0229, 0.0990]。
   ⇒ 正确表述是：**Qwen 的优势体现在"与适配器协同"上，而不是基座本身**。因此根 `README.md` 的"全面超越 DeepGen"在受控对比上成立（PSNR 与姿态都显著），但**不能**用来描述"裸基座谁更强"。§3.1 选基座的第一条理由也据此限定。

**两个指标定义的陷阱**（来自同一文件的 `metric_definitions`）：

- `pose_pck_*` 与 `pose_nme_diagonal` 用的是 **COCO-17 关节**（不是 25 点、不是 134 点），按 512×512 **图像对角线**归一，**生成图里漏检的关节直接计为失败**。
- `identity_cosine` **只在"源脸与生成脸都检出"的配对上计算 ⇒ n = 47，不是 50**；`identity_target_cosine` n = 48/49；`source_target_identity_cosine`（身份上限，即"同一个人在目标姿态下的脸"与源脸的余弦）= **0.6132，n = 46**。
  ⇒ 这解释了 §8 第 15 条的三个"上限值"（0.613 / 0.6272 / 0.683）其实是**不同样本集上的同一个量**；同时也意味着：**报身份必须连检出率一起报**（此处 `identity_detected` = 0.94）。

**一个未被记录的实验不对称**：`generation_settings.deepgen` 带 `negative_prompt`（"blurry, low quality, … missing fingers, mutated hands…"）与 `guidance_scale: 4.5`；`generation_settings.qwen21` **既没有 negative prompt 也没有 guidance 字段**（走 `true_cfg_scale=1.0`）。⇒ 四臂对照里两个基座的**采样设置不对称**，PSNR/SSIM 差异里有多少来自 CFG 与负向提示词，**无法从现有数据分离**。这是后续重跑必须固定的一项。


### 5.2 面部保持与两阶段级联 [实测]

**`bfs_compare_30_20261004`**：Stage 1 = Qwen2.1 + BodyAdapter(`full_20k/checkpoint_last.pt`, sha `23be5631…`) @512；Stage 2 @768 三臂（无 LoRA 基座 / BFS head v1.1 sha `d1d748d5…` / alternative sha `c5332bbc…`），strength 1.0，固定 `head_swap:` prompt，seed 42、30 步、CFG 1，两阶段间 body/face 适配器关闭。样本从 50 对里按时间序每组取首/中/尾选出 30 对（10 个服装组、**8 位演员**），无结果筛选。**槽位 30 + 24×3 = 102/102 全跑完。**

| 臂 | identity_A | PCK@0.02 | 头外 MAE/255 |
|---|---|---|---|
| Stage1 C（只身体） | 0.3591 | **0.9899** | 0.0000 |
| 基座二次采样（无 LoRA） | **0.5210** | 0.9498 | 3.1268 |
| BFS V1.1 | 0.4422 | 0.9629 | 4.3518 |
| BFS alternative | 0.4527 | 0.9678 | 4.2717 |

8 演员等权 cluster bootstrap：`bfs − nolora` 身份 **−0.1075 [−0.1639, −0.0523]，胜/负/平 7/17/0**；`alternative − nolora` −0.0828 [−0.1213, −0.0399]。反过来对"真人 B"的余弦是 0.3340/0.4082/0.4048/0.4054，视线角误差 3.74°/6.72°/5.08°/4.66°。
**结论**：BFS 在身份上输、在头部朝向上略赢，且每一支 stage-2 都让 PCK 变差、都在头框外漏像素。决策：先用 Full Body+Face 造 C，再做**局部**头部修复 —— 这条决策就是 §5.3 pilot 的立项依据。

**架构文档里的 face 四臂（n=30）**：0.395 → **0.480**。但这个 +0.085 **主要来自"把 A 的脸裁图喂进去"**，Face 分支自身的净增量只有 **+0.008，CI 跨 0**。
**1024 杠杆（严格 n=21）**：0.508 → **0.660**，同时 PCK@0.02 仍 0.990，代价 18.9 s/图、39.19 GiB。**这是目前唯一把身份推过 0.65 门槛的方案**，代价是分辨率与显存。

### 5.3 joint pilot：Body+Face(+Hand) vs 冻结 BFS [实测，但两套数字并存]

问题：**等优化器预算**下，冻结的开源 BFS LoRA 是不是比"纯 Qwen 适配器继续微调"更好的保脸支路？

- 共同起点：`tiktok_face_v1_20261004/training/checkpoint_step_003000.pt`（sha `4f7eb5a1…`，branch=face，**不恢复优化器状态**）；基座 297 个张量 / 7,115,124,736 参数**全冻结**；`body_lr_scale 0.25`、`face_lr_scale 0.5`、`face_weight 0.5`、`aux_every 4`、损失 = body_flow 1.0 + face_roi 0.5 + identity 0.1 + perceptual 0.05；`controls body=1 face=1 hand=0 interaction=0`。
- 切分（seed 1708）：train 512 对 / 512 唯一目标（从 1000 合格里选），20 个视频组、203 序列；common256 / branch256 各 256；val 24 TikTok + 12 iPER = **36 固定例**；`train_val_original_groups_disjoint: true`，`final_sealed_test_used: false`。

| 分支 | 本地 512 归档（组宏均值 ID / PCK2 / PCK5） | 服务器 native 1024 步（同一指标） |
|---|---|---|
| init | 0.4125 / 0.6552 / 0.9233 | 0.4231 / 0.7468 / 0.9530 |
| **J（共同 Body+Face 基座）** | 0.4369 / 0.6513 / 0.9310 | **0.4455** / 0.7316 / 0.9564 |
| Qwen_final | 0.3974 / 0.6254 / 0.9205 | 0.3849 / 0.7076 / 0.9571 |
| BFSBody | **0/36 未完成**（`--`） | 0.3282 / 0.6818 / 0.9578 |
| BFSHead | **0/36 未完成** | 0.3072 / 0.7082 / **0.9621** |

服务器 native 轮另有**第五支 BFH（Body+Face+Hand）**，本地和 README 都没有：
`BFH − Qwen` **+0.0279 身份**（24/24 配对），−0.0100 PCK@0.02，−0.0027 PCK@0.05；`BFH − BFSBody` +0.0622；`BFH − BFSHead` **+0.0901**；`BFH − J` −0.0264。BFH 绝对值：身份 **0.4648**、PCK@0.05 0.8647、NME 0.0335。
分离实验很干净：`JFaceOnly_common_tiktok` 身份 **0.9043** 但 PCK@0.02 只有 0.2307；`JBodyOnly` 身份 0.3875 / PCK@0.02 0.4468。

**判定**：**BFS LoRA 两支在身份上落败；联合 Body+Face(+Hand) 适配器胜出**。文档明确拒绝下"优胜结论"（只有 20 个组、不报 CI）。
`BFSBody_hotload_iper` 被 **SIGINT 手工中断**（`interruption_note.json`：用户纠正预处理为原始像素尺寸时中断，随后要求完整交接、不再追加实验；`not_a_model_quality_failure: true`），因此无 summary.json/scores.json、仅 10 张图。`data/handoff_integration.json` 记 `"native_pixel_dimensions_implemented": false, "experiments_stopped": true`。

> **警告**：`docs/experiments/2026-10-08-bfs-body-face-joint-handoff.md` 里的五个核心数字（BFSBody 0.3583/0.5248/0.9111、BFSHead 0.3333/0.5481/0.9206）**与本地 512 归档和服务器 native 报告都对不上**，它声称的"121/121 pytest 全通过""72 张拼图 + 30 张脸表"在磁盘上不可验证。只有**方向性**结论（Qwen < BFSBody < BFSHead on PCK@0.05）幸存。**不要引用这份 handoff 的数字。**

### 5.4 双人线（三代尝试）[实测]

**(a) `qwen-2.1-test`（prompt-only 原型）**：基座 `sha 790c926…`，源图来自 Wikimedia Commons，冻结长边 1280；A_heavy 19 / B_slim 21 / 7 对（`hard_01..05`、`medium_01/02`）+ `interaction/testset/inter_01..12`。生成 768×1024、40 步、seed 42、CFG 1.0、`model_cpu_offload`。`gen_v1/results.json` 9 条记录（59.8–127.4 s，峰值 19.53–19.67 GiB），`interaction_v1/results.json` 18 条 = 6 样本 × 3 臂（detailed/minimal/w_ref）。
单样本过拟合 `overfit_inter01`（500 步，样本 `inter_01_pair00_taichi`：源 `taichi00#41` → 目标 `hug00#108`，cam4，task_id=2 HUG）：`on_psnr` 12.354(step0) → 16.919 → 19.683 → 27.401 → **30.142(step400)** → 29.846(500)；`on_ssim` 0.40093 → **0.92449**；`off_*` 六行全部冻结在 12.354/0.40093。`train_log.json` loss 0.1019→0.0232(260)，`control_calls: 6`/step。
> 本地该目录只有 41 个文件、**没有任何 .pt**；README 描述的 `data/`、`experiment/`、`adapter_overfit.pt`、"309 张评测生成图"只存在于服务器 `2.qwen-test/`。

**(b) `hi4d_qwen21_dual`（热启动 + 10k 设计）**：warm start 自 `qwen21_iper_body/full_20k/checkpoint_best.pt`；`dual_adapter.py`；控制缓存版本 `hi4d_qwen21_dual_v2`；512 letterbox；计划 `max_steps 10000`、`stage1_steps 2000`、micro 2 × accum 2、混 10 % iPER 回放、val 64。
**实际**：`full_run_stopped_status.json` 写 `state: stopped_after_user_scope_correction`、`last_recorded_step: 410`、**`checkpoint_saved: false`**；目录根**没有 checkpoint_best.pt**。存在的是两个**单样本 300 步过拟合**（`single_hug_overfit/`、`single_pair14_action_v2_overfit/`，各 1,445,408,821 B，`iper_replay_rate 0.0, val_count 1`），`validation_full.json` = `{count: 1, flow_loss: 0.5215}`。
`single_hug_eval/summary.json`（n=1）：`detected_person_count **1.0**`，`pose_pck_b / contact / identity_swap` 全 `null`，body_identity_a 0.8504 / b 0.8574，bg_mae 4.061。`prompt_audit_pair14_v2/seed42`（full_detailed）：pck_a 0.4667，pck_b null，body_identity_a **0.8689** / b 0.7409，bg_mae 3.1645，检出人数仍是 **1.0**。
> **[已撤回]**：`experiments/README.md` §6 的"双人检测率 100%（平均 2.0 人）、Identity Swap = 0.0"、"混入 iPER 单人样本做回放排练"。磁盘上：适配器每一臂都只检出 **1.0** 人、swap 是 `null`（分母 0）、两个过拟合 manifest 里 `iper_replay_rate = 0.0`。达到 2.0 人的只有 **native/off** 臂，swap=0.0 也来自那些对照组，不是训练出来的适配器。
> 这条线**真正的资产是泄漏审计**：交换 target RGB + GT mask 的最大绝对差 **0.0**；交换 target geometry 才让输出变化 4.670/255。另外 `test_control_audit.md` 记录了 606/606 测试控制图缓存，以及 index-573 的垂直翻转 bug 用"≥80 % 覆盖且 +25 pt 才允许翻转"的门禁修掉（A/B 覆盖 44.7 %/12.5 % → 96.6 %/97.3 %）。

**(c) `qwen21_dual_v8`（v8 设计的对拍 pilot）**：`run_manifest.json` 固定 `geometry_encoder_version dual_v8_geometry_v1` + `appearance_encoder_version dual_v8_vae_dino_v1`、Hi4D world-metres Y-up、6 个编码器版本 SHA256、seed **14062**、prompt "Make the two existing people hug…"。warm start 只 **178 个张量 / 20,281,376 个元素**（`geometry_encoder.spatial_encoder.` 141 + `reasoner.person_reasoner.` 36 + `reasoner.shared_role_offset` 1）；300 步 / 2775.6 s；`best_noise_validation 0.0704`；`pilot_pair14_v8_300_v1/checkpoint_{best,last}.pt` 各 **567,957,785 B**（确实存在）；门禁矩阵里**只有 `gates_pair14_v3` 带 `real_model_gates.json`**（passed，zero_init_exact，178/178 matched，backbone SHA 前后一致），v1/v2/v4 只有 manifest 和 native_cache。评测网格 2 臂 × 2 prompt × 3 seed = **12/12 生成并评分两次**。

| 组 | 双人检出 | body pck@.02 | body pck@.05 | wholebody pck@.02 | 外观均值/swap | 脸 ID/swap | bg MAE |
|---|---|---|---|---|---|---|---|
| native_short | 1.0 | 0.2197（132 点） | 0.7121 | 0.2314 | 0.7226 / **0.0** | 0.0853 / **0.3333** | 0.0178 |
| native_detailed | 1.0 | 0.2348（132 点） | 0.6515 | 0.2314 | 0.7339 / **0.0** | 0.1597 / **0.3333** | 0.0171 |
| dual_v8_short | **0.0** | 0.8986（**69 点**） | 0.9855 | 0.9697 | **null**（3 次失败） | **null** | 0.0140 |
| dual_v8_detailed | **0.0** | 0.9130（**69 点**） | 0.9855 | 0.9835 | **null** | **null** | 0.0138 |

网格重建佐证：`dual_v8_*_seed43/44 → reconstructed_people: 1`，native 全部为 2；两支 dual_v8 各有 `mesh_failed_outputs 2`。
**判定：dual_v8 的两个头号门禁（双人检出、外观/身份可度量）都失败。**
> **[已撤回]** ×2：① `qwen21_dual_v8/README.md` 把 `face_identity swap_rate 0.333` 记在 dual_v8 名下 —— 那是 **native** 臂的值，dual_v8 的 swap 是 null（分母 0）。② "PCK 从 0.22 跳到 0.90"是**分母假象**：`scored_joint_denominator` 从 132 缩到 69，`failures: ["person_not_detected"] × 3` —— 人被画丢了，点少了，分数就"高"了。README 只说对了 native 那一侧。

### 5.5 提示词工程线（当前唯一在跑的 GPU 任务）[实测]

**`tiktok_pe_prompt_rewrite_20261014`** —— 目标：把硬编码 prompt
`Change the person pose to match the target pose. Preserve identity, clothing, and background.`（`src/pose_control/qwen21/runtime.py:17` 与 `qwen21_full/qwen_runtime.py:19`）换成 **10,850 对逐对 PE-I2I 重写**。

配置：输入是 A + B + **B 的真实 DWPose 渲染**（`TikTok_v3/dwpose_v1`，295 序列 / 7,481 帧），**故意不用** workflow 的 `B_3d_dwpose.png`；清单 `pairs_strict_v1/{train 8574, val 1120, test 1156}`；每对编译两种形态（`compiled_prompt` 把图像索引剥成角色词；`compiled_prompt_inference` 把 `<image3>`→`<image2>`）；K=2 个 worker；顺序 test→val→train；watchdog 10 分钟、≤6 次重启。

**2026-10-15 10:42 实测进度**：`records/test/` 504 个文件（**503 complete、1 failed**），val/train 均 0；`logs/full_run.log` 末行 `split=test alive=2 complete=502/1156`；`pgrep -cf` = 2 与 2。**即 test 502/1156 ≈ 43 %，总体 503/10,850 = 4.6 %。**
⚠ `prompts/test.jsonl` 只有 **72 行** —— 提示词库比记录滞后约 7×，读之前必须先跑 `assemble`。

吞吐消融（同窗口可比）：

| 臂 | 结果 |
|---|---|
| A thinking + native @ K=1 | **162.9 s/对、22.1 对/小时**，全量估算 492 GPU·h |
| B2 thinking + 512 | 150.6 s，只有 **1.08×** |
| C thinking @ K=3 | 延迟反而涨到 328.9 s，净加速 **1.29×** |
| notk（关思考） | 295.2 s / 6,447 tokens，比开着思考（6,008）**更慢**，且 1/6 的样本违反 JSON 契约 → **否决** |
| D flash-linear-attention | 无可测收益 |
| E fla + `mamba_ssm_dtype=bfloat16` | 16.0/16.1 tok/s vs 对照 21.6/21.7，**且改变输出**（278→315、364→341 词）→ **否决** |

其他有分量的测量：DWPose 数据审计 5,435 个唯一目标帧，`body_mean_conf` p10/p50/p90 = 0.626/0.731/0.872；低置信点 ≥4 个的帧占 **2,065/5,435 = 38.0 %**；但下肢外推 0/1/2/3 个 = 25.6/32.4/41.5/0.6 %、**≥4 个 = 0 帧** ⇒ 腿从未被"编造"。切分卫生：序列 / 视频组 / 精确对 三级交集全为 0，**但 test 只有 29 个序列 / 3 个视频组** ⇒ 任何 CI 都必须按组做 cluster bootstrap。逐样本性：305 词/条、**560 句里 556 句互不相同**、单句最多复用 2/64、词表 Jaccard 中位数 **0.419 [0.310, 0.527]**。成本：prompt 从 14 词 = 18 token 涨到 278 词 = **352 token**，`prompt_embeds` 18×4096 → 352×4096 = **0.15 → 2.88 MB/对（19.4×，全量约 +30–32 GB）**，必须落 `/data`。语义一致性审计：v1 规则产出 46 对里 30 个"假矛盾"（65 % 误报来自 `button-up shirt`/`chin raised`），收紧后剩 1 个候选、同样是假阳性 ⇒ **未检出真实矛盾**。
> **[已撤回]**：§4.1"左腿槽位坏了 / 只有 0.8 % 连贯 / 足以解释训练效果差"于 19:43 撤回，根因是**把 DWPose 的 18 点顺序当成了 COCO-17**（`(11,13,15)` 其实是 Lhip/Lankle/**Leye**）。用正确三元组 (8,9,10)/(11,12,13) 在同样 900 帧上重算：右腿 85.2 %（213/250）、左腿 92.9 %（224/241）⇒ **条件图没有系统缺陷，不需要重跑 DWPose**。另有两张表被标【已作废】。隔离区：`defective_instruction_v1/`（8 个 train + 2 个 GROUNDING 之前的记录，不入库）、`pilot_v0/`（5 条排除）。事故成本：`init_initial_iper_missing_cache` 类操作烧掉 14 对（≈65 分钟），后来同一个"先挪目录再杀进程"的错误又烧掉 2 对。

**下游缺口（阻塞项）**：`train_iper_qwen21_body.py:141,:183` 与 `train_iper_qwen21_full.py:235` 调 `cache.pair(pipe, batch, device)` 时**不传 `prompts=`**，所以提示词库现在喂不进训练；而 `precompute()`（`:121-127`）的两处存在性检查都**不含 `prompt=`** ⇒ 逐样本 embeds 永远不会被预计算，每一步都会 cache-miss。**两个文件、三个调用点，至今未改。**

### 5.6 负面结果：`tiktok_overfit_sanity_full` 六门禁 [实测]

架构 **v6.5（DeepGen MMDiT，不是 Qwen）**，TikTok 序列 00001、`0014`→`0074`，seed 42 / eval_seed 1234，Phase A 200 步固定 t=150、`max_steps 1500`、Phase B `uniform[0,999]`、全适配器 778 个优化器张量。产物完整性：`train_metrics.jsonl` **正好 1500 行**、16 个评测槽位（`gen_eval_history.jsonl` 16 行、`gen_evals/` 48 文件 = 16×3）、4 个 checkpoint、`elapsed_seconds 711.99`。

`summary.json → verdict: "FAIL"`：

| 门禁 | 判据 | 实测 | 结果 |
|---|---|---|---|
| C1 | 初始 ON≡OFF ≤ 1e-5 | 0.0 | PASS |
| C2 | Phase A flow tail20 ≤ 0.7×0.65047 = 0.4553 | 0.41958 | PASS |
| **C3** | Phase B flow tail20 ≤ 0.6×0.15210 = 0.0913 | **0.23826**（loss 反升 56 %） | **FAIL** |
| C4 | ΔPSNR ≥ 5.0 | +7.947 | PASS |
| C5 | final ON ≥ 20.0 | 20.039（标 `stretch: true`） | PASS |
| **C6** | ON − OFF | **−0.0440 dB** | **FAIL** |

`grad_flow_report` 定位到病灶：`face_preparer` **149 个参数里 0 个有非零梯度**；`condition_bridge` 3/5、`detail_preparer` 9/21、`control_interface` 146/266、`condition_injector` 150/179。诊断上脸的相似度 0.9749→0.9947、脸 LPIPS 0.0345→0.0140 在改善，而**全图 PSNR 在变差**。
独立反证在 `tiktok_legacy_evidence_202609/`：`train.log` 末行 `step 1500/1500 total=0.186339 flow=0.178280 id=0.000000 perc=0.000000` —— **identity 与 perceptual 两项在整个 1500 步里恒为 0**。同档案 1500 步量化：full_body 10.9792→10.7650、face_roi 11.3298→10.8123、右手 SSIM 0.1325→**0.0212**。
**判定**：这是全仓最有价值的否定结果。它终结了"MMDiT 中晚期单层特征强注入"路线，直接推动转向 Qwen-Image 与解耦两阶段。

---

## 6. workflow：不训练的那条线探索到哪了

### 6.1 定位与存在理由

`workflow/`（单人）与 `workflow-multiple/`（双人）是**独立的动作迁移流水线**，与 `experiments/` 的约定完全不同（各有自己的 `runs/` + `scripts/<工具名>/` 布局）。它们**不训练任何权重**：

```
一张源图 A + 一张参考图 B
   → 几何：SMPLest-X + DWPose 把 B 变成一张 2D 骨架引导图
   → 提示词：PE-I2I VLM 看图把"改成 B 的动作"这句话写成详细的编辑指令
   → 生成：Qwen-Image-2.1 原生 backbone（无 Adapter、无 LoRA），两张参考图 + 指令
```

它的价值有三层：① 适配器身份掉得太狠，先用流水线把可交付效果兜住；② 它是最便宜的**反事实对照组**——"如果什么都不训练，只用提示词 + 骨架图，能做到什么"；③ 它是 PE-I2I、无 GT 评测、人工盲评这三件基础设施的实际承载体。

> **两侧不同步**：远端 `workflow/` 在持续演进（mtime 2026-10-15 01:17），本地 Windows 是**过期镜像**——`run.py` 本地 SHA256 `2b847414…` vs 远端 `6f9c2f73…`，`smplx_worker.py` `e33af3a3…` vs `4e1144a7…`，只有 `pe_worker.py` 两侧一致 `40e0e858…`。**编排结论以远端为准**（`docs/workflow_PE蒸馏到Qwen3VL任务分支_20261015.md` 自己就这么声明）。下文行号除非注明，均指远端文件。

### 6.2 阶段图与编排 [实测]

`STAGES = ('geometry', 'prompt', 'generate')`（`run.py:25`），目录 `geometry/ prompt/ generation/`，worker `smplx_worker.py / pe_worker.py / generate_worker.py`（`:26-28`）。**"四阶段"里的第四段是 `inputs/` 准备**，由 `prepare()`（`run.py:152-179`）在进程内完成：EXIF 归一 A/B → `inputs/A_rgb.png`、`B_rgb.png`、letterbox 后的 `A_reference.png`、`source_preprocess.json`；此外 `evaluation` 作为外部产物阶段由 `batch.sh:71` 调 `evaluate.py`。

每阶段以子进程运行：`subprocess.run([python, '-u', ROOT/script, '--request', request.json])`（`run.py:67-75`）；**几何用 `config['geometry_python']`（`add/envs/smplestx_bw`），提示词与生成用 `config['qwen_python']`（`~/nihaochen/envs/qwen21`）**（`:314`）——三阶段跨两个 Python 环境。

CLI（`run.py:235-255`）：`--config --pair --a --b --name --until {geometry,prompt,generate} --prompt-file --resolution --steps --seed --subject {auto,largest,INDEX} --framing {source_person,source_torso} --check --verify --force`。`--force` 与 `--verify` **互斥**（`:257-258`）。

缓存/续跑语义：每阶段先算 `stage_reasons(record, request, run_dir)`，**理由列表为空才算 `CACHE_HIT` 不重算**（`:305-307`）；`--verify` 只把理由抛出来而不计算（`:309-310`）。循环结束后 `furthest_valid_stage()`（`common.py:179-196`）重新测量这个 run 实际最远到哪，避免 `--until geometry` 重跑把已完成的 generation 降级；然后**再复核一遍每个被触碰的阶段**，把 `verification.status ∈ {validated, mismatch}` **写成实测结果而非声明**（`run.py:341-350`）。全局 `flock` 锁在 `workflow/.workflow.lock` 串行化调用（`:288`）。`run.json`（schema `single_person_pose_workflow_v2`）状态机 running/complete/partial/failed，`report.py:96` 会把"running 但锁持有者不存在"重标为 `abandoned`。入口：`run.sh`（exec qwen21 python，`:27`）、`batch.sh`（遍历 `$PAIRS/*/`，7 对测试图，`RESOLUTION` 默认 **768**，`:50`；先按严格 `auto` 选主体，失败则回退 `--subject largest` 并记 `SUBJECT_FALLBACK`，`:63-69`，每个 run 后跑 `evaluate.py --run`，末尾 `report.py`）。

### 6.3 几何阶段 [实测]

`smplx_worker.py`：YOLOv8x 检 A 与 B（`person_detection_confidence=0.5`，`config.json:9`）→ `choose_subject`（`geometry.py:30-63`，`auto` 遇到多于 1 人直接拒绝）。SMPLest-X 经 `src/pose_edit/estimator.py:estimate_single` 调用（`:412-445`），每次调用建一个 **run-local estimator sandbox**（复制 `main/` + 入口脚本、软链权重，`common.py:71-88`），把 B 以 JPEG q100/subsampling 0 暂存成 `demo/input_frames/<job>/000001.jpg`，用 `SMPLESTX_DET_CONF` 环境变量运行 `scripts/run_smplestx_params_only.py`（这个 vendored hook 依赖由 `--check` 检查，`run.py:88-96`），校验 12 个必需参数键，再 `reconstruct()` 重跑 SMPL-X neutral 层并断言网格误差 ≤ **1e-4 m**，`project()` 做针孔投影并与估计器自己的 `joint_proj_model` 交叉核对到 **0.25 px** 以内。

产物：`B_smplx.pt`、`B_keypoints.npz`（`smplx137_cam, dwpose134_cam, keypoints_B_px, keypoints_generation_px, valid_B, valid_generation, source_person_bbox_canvas`）、`B_camera.json`，以及 PNG `B_poseguide_native.png`、**`B_3d_dwpose.png`（真正喂给 PE 和生成器的那张）**、`B_poseguide_overlay.png`、`A_poseguide_alignment.png`（`:111-120`）。

点集换算：`smplx137_to_dwpose134`（`geometry.py:107-119`），body-18 顺序 `(24,7,9,11,13,8,10,12,2,4,6,1,3,5,23,21,22,20)` = nose, neck, 右臂链, 左臂链, 髋/腿, `R_Eye,R_Ear,L_Eye,L_Ear`。DWPose 逻辑集中在 `src/pose_edit/dwpose.py`（94 行，CPU-only ONNX）；`cocowholebody133_to_dwpose134`（`geometry.py:132-148`）把 ONNX 的 133 点补成 134 点布局。`dwpose.py:295-296` 的 docstring 原话：保留两份这个约定的副本，"就是 133-vs-134 slot bug 曾经发生一次的原因"。

画布与定标：生成画布 = A 的宽高比 @ `resolution²` 像素预算、两边对齐 32（`canvas_size`，`common.py:221-239`），A letterbox 上去（灰 128 pad，`:242-251`），`final_image.png` 再按 `content_box` 裁回（`generate_worker.py:331-333`）。`config.json:11` 默认 resolution **1024**，而 `batch.sh` 用 **768**（这是很多数字对不上的来源之一）。两种 framing：

- `source_person`：把 B 的框塞进 A 的框里，`scale = min(w-ratio, h-ratio)`；
- `source_torso`：把 B 实测的"颈→髋中心"躯干长度缩放到 **A 的实测躯干**（对 A 跑 DWPose，`smplx_worker.py:89-100`，再乘 letterbox 的 `sy`），**测不到就硬失败而不是静默回退**（`:93-98`）。常数 `TORSO_TO_PERSON_BOX_HEIGHT = 0.31`（`geometry.py:27`）**已被降级为 fallback**，transform 会记录 `torso_reference ∈ {measured_A, constant_fallback, not_applicable}`。

`B_camera.json` 记 `focal_xy, principal_xy, coordinate_system: 'camera XYZ; pixels x-right y-down; projection uses Z+1e-4'`、`mesh_max_error_m`、`projection_max_error_px`、逐图检测框与分数，以及主体一致性检查（关节质心必须落在选定 B 框的 15 % 边距内，否则硬失败，`smplx_worker.py:68-80`）。

> **两个必须记住的坑**：
> ① `B_3d_dwpose.png` 是"估计 3D 关节的 2D RGB 投影"，遮挡点**是模型的猜测**，有效性判据只是"有限且在画内"，**不是遮挡置信度**；
> ② 它**不能**和训练侧的 `render_iper_v1_dwpose_wholebody.py` 条件互换 —— 同名、同点数、不同来源与配色，喂给训练好的 BodyAdapter 属于**分布外输入**（`docs/workflow_大方向优化调研_20261014.md` P5）。

### 6.4 提示词阶段（PE-I2I）[实测]

模型路径 `config.json:7` → `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I`。`pe_worker.py` 读官方 `system_prompt.txt`（**18,344 B**）+ `PE_ROLE_SYSTEM_SUFFIX`（`prompts.py:15-27`），按 `A_reference, B_rgb, B_3d_dwpose` 顺序喂三图（`pe_worker.py:192`），`AutoModelForImageTextToText` bf16，`enable_thinking=True`，采样 `PE_SAMPLING = {max_new_tokens:24000, do_sample:true, temperature:1.0, top_p:0.95, top_k:20}`（`prompt_contract.py:107`）。

**重试**：最多 **2 次**，seed = `pe_seed+attempt`，第 2 次会把上一次被拒的错误原因作为纠正句追加（`pe_worker.py:199-236`）。
**契约**：官方三字段 JSON `{rewritten_prompt, wh_ratio:"", ratio_follow:"<image1>"}`（`prompt_contract.py:112-129`）；角色后缀**禁止**最终答案引用 `<image2>`（即 B 的照片），并把骨架图 `<image3>` 映射为生成器的 `<image2>`（`compile_prompt`，`prompts.py:65-94`，遇到 `<image2>` 残留或未解析图像标签**硬失败**，`validate_generator_prompt` 只允许 {1,2}）。
**产物**：`pe_raw.txt`、`pe_compiled_prompt.txt`（只读）、`rewritten_prompt_raw.txt`、`pe_result.json`（角色、SHA、采样、尝试次数、峰值显存）、`pe_attempts.json`、`effective_system_prompt.txt`。
**提示词溯源**：`select_generation_prompt()` 把 PE 输出或 `--prompt-file` **冻结**成 `prompt/generation_prompt.txt` + `generation_prompt.json`（`{source: pe_compiled|manual, source_path, source_sha256, sha256}`，`run.py:182-207`）——手工实验**永不覆盖** PE 输出。提示词阶段的缓存身份 = 设置 `pe_model, pe_seed, prop_policy` + 三张输入图内容哈希 + `model_signature(pe_model)` + 代码哈希（`pe_worker.py`、`pose_transfer/prompts.py`）+ `PE_SAMPLING, enable_thinking`（`run.py:125-140`）。

**成本解剖**（`docs/workflow_PE重写成本_量化与蒸馏调研_20261014.md` §1）：一次调用 prefill **6,803 token**（2,240 视觉 + 4,563 文本），decode **4,479 token / 243.7 s = 18.4 tok/s**，最终只有 **310 词**（约 400 个学生 token）进入生成器 ⇒ **约 93 % 是脚手架**。中位耗时 PE **242.1 s** vs 生成 47.6 s vs 几何 31.5 s ⇒ **PE 占三阶段 GPU 时间的 75.4 %**（含评测则 68.9 %）。关思考**不省 token**（4,674 vs 4,615）且 6 次里违反 JSON 契约 1 次。根因是 kernel 回退：`transformers/models/qwen3_5/modeling_qwen3_5.py:479` 的**逐 token Python 循环**，因为没装 `fla`/`causal_conv1d`，而 FLA 拒绝 float32、config 又写 `mamba_ssm_dtype: float32`。
**仍在磁盘上的缺陷指令**：`pe_instruction()` 硬编码了 "The target basketball must NOT appear. Describe the raised/turned hand…"（`prompts.py:44-45`），`compile_prompt` 的固定契约也写 "Do not add the basketball or any other target-only object"（`:88`）—— **通用契约被 hard_05 的篮球样例污染了**。
**同 seed 复现的真相**：两次 GGUF 门禁运行产出了**逐字节相同**的首次答案（SHA `e287ea85…`）⇒ 那是同 seed 重放，不是独立样本；而带 MTP 的 llama.cpp **破坏**字面级 seed 可复现性，所以 backend / spec-type 必须成为身份字段。

### 6.5 生成阶段 [实测]

`generate_worker.py`：`QwenImage21Pipeline.from_pretrained('/data/shangguanrz/pic-edit-models/Qwen-Image-2.1', dtype=bf16)`。
**没有 Adapter，也没有 LoRA**：config 里 `adapter:false, lora:false`，`run.py:273-277` 拒绝其它取值，worker 检测到任何 PEFT config 直接 raise（`generate_worker.py:292-293,315-317`）。**这条线是纯 prompt + 图像条件。** 训练好的 BodyAdapter（iper_v6 / BFH）**从未接进来**，"隐式注入"那一臂至今 0 % 覆盖（§6.11）。

只喂两张图（`:297`）：`<image1>` = `A_reference`（身份/脸/发/服装/场景/A 的道具），`<image2>` = `B_3d_dwpose` 骨架引导；**`B_rgb` 故意不给**（`'B_rgb_given_to_generator': False`，`:335`）。两张必须同画布，worker 断言 pipeline 不会重采样（`:302-309`）。默认参数 `config.json:11-16`：resolution 1024（batch 用 768）、steps 30、seed 42、`true_cfg_scale 1.0`、`use_kv_cache false`。身份/服装/背景保持**完全靠提示词契约**（`source_only` 道具策略：保留 A 的物体、排除 B 独有物体）。输出 `final_canvas.png`（RGBA 合成）、`final_image.png`（裁回 content_box）、`generation.json`（带 SHA256 的参考图清单、prompt 溯源与哈希、峰值显存、torch 版本）。

### 6.6 溯源与缓存契约 [实测]

`src/pose_edit/common.py`（272 行）是两条流水线**共享**的层（2026-10-13 WP1 抽出）。请求 schema `single_person_pose_request_v2` 把 `identity`（每阶段的设置子集、输入内容 SHA256、模型目录的 `model_signature` = 逐文件 size+mtime_ns（小 json/txt/py 才取 sha256）、代码 sha256 清单）与 `runtime`（绝对路径，**永不参与指纹**）分开——这条就是 W3 修复：**改名或挪动 run 目录不会让缓存失效**。
`stage.json`（schema `single_person_pose_stage_v2`）字段：`schema, status, identity, request_sha256（identity 的指纹）, outputs{name→{file, sha256}}, metadata, completed_at_unix`（`common.py:91-102`）。`stage_reasons` 枚举**每一种**过期原因（schema 不符、status、identity 差异并点名变化的设置/代码文件 + 8 字符哈希前缀、产物缺失、产物写完后被改）（`:113-161`）= W5 修复。`EXTRA_RUNTIME_ROLES = ('generation_prompt_provenance',)` 记录但不入指纹（`run.py:32-34`），因为里面含绝对路径。
**没有 `caches/` 目录参与 workflow**：唯一的缓存层就是 per-run、per-stage 的 `stage.json` 内容哈希校验；跨 run 的内容缓存被列为后续工作。代码哈希进指纹 ⇒ **任何 workflow 代码改动会让全部历史 run 的 `--verify` 失效**，这是**声明过的设计取舍，不是 bug**。

### 6.7 评测栈：无 GT 六轴 [实测]

| 文件 | 角色 | 计算什么 |
|---|---|---|
| `evaluate.py`（8.9 KB，几何环境） | **第一 reader**，单轴姿态 | 用**同一个 SMPLest-X** 重估 `final_image.png`，与 134 点引导按组比较（body 0:18 / feet 18:24 / face 24:92 / left_hand 92:113 / right_hand 113:134，`:37-39`）；指标 `mpjpe_px, mpjpe_norm, p95_norm, pck@0.2, pck@0.5, worst_joint_offset_px`，按引导骨架的"颈→髋中心"躯干归一（`:56-63`）；出画引导点剔除；写 `evaluation/{eval.json, final_keypoints.npz, final_smplx.pt, eval_overlay.png, stage.json}`，自带请求指纹缓存。它的 `limits` 字段明说：**只测姿态保真** |
| `evaluate2.py`（23 KB，qwen 环境，**写在旁边、不改 evaluate.py**） | **第二 reader / 无 GT 六轴** | DWPose 姿态（CPU，含 **strict PCK**：丢点要罚，`body_pck@0.2_strict`）、`pose_generated_vs_source`（身体到底动了没）、`estimator_agreement`（SMPLest-X vs DWPose 的 `body_agree@0.2` + verdict）、shape（肢长比漂移 + `view_max_rel_change`）、identity（ArcFace antelopev2 cosine，带 yaw 警告）、preservation（肢体外凸包 mask、背景/人物 MAE/PSNR/SSIM + 可选 LPIPS alex）、`scale_fidelity`（`torso_ratio_generated_over_guide` 与 `_over_source`——引导比例就是为 hard_05 的 2.35× 缩放灾难而设）、可选 `--fit-source` 拟合 A 以算 β 距离、多人画面的 `nonsubject_*`；**相同 SHA256 的跨 run 重复图直接记 `duplicate`**；写 `eval2.json / eval2_overlay.png / stage2.json`。CLI `--run(可重复) --all --force --fit-source --no-lpips --fit-only` |
| `scorecard.py` | 跨 run 对照表 | 列 `pck3D, pckDW, pckDW*, agree, torsoG, moved, idcos, bgMAE, bgLPIPS, perMAE, shape, people`；`--baseline X --candidate Y` 出带方向的配对差；**图像在盘上变了就把记录标 stale**；显式警告"每个 run n=1" |
| `report.py` | 只读索引 | 递归发现所有 `runs/**/run.json`（2026-10-14 修了三个崩溃 bug：缺 `import os`、缺 `import json`、只做一层遍历漏掉批量/嵌套 run）；打每阶段秒数、prompt 来源、评测摘要、配对覆盖、`legacy`/`abandoned` 标签 |
| `human_eval/` | 人工盲评 | `protocol.md` + `prepare.py`（只出配对臂的问题、左右随机化、真值藏 `truth.json`、符号链接）+ 离线 `annotate.html`（每图 4 个轴 + 强制二选一，localStorage→jsonl）+ `tally.py`（win_rate 相对参照臂带符号、**逐问题** bootstrap CI95、合理率、逐轴失败率、Fleiss κ、哈希不符拒收）。已做冒烟：来自双人 run 的 12 题；合成"永远选左"选票 → win_rate +0.167 CI [−0.333, +0.667] `NOT_MET`，**证明盲评程序确实防住了位置偏好**。参照臂（`--condition none`，纯 Qwen-Edit 基线）**还没实现**，需要改 run.py 的 INPUTS_CODE 指纹 |

**为什么必须有两个 reader**：用生成时同一个估计器去评自己的结果有自偏。`无GT调研` 设计了六轴（姿态第二 reader + strict PCK、shape、identity、preservation、NR-IQA 延后、统计 ≥3 seed + 配对 bootstrap，三个头牌指标 `pck_torso@0.2` / Δidentity / `background_mae`）；PyiQA/CLIP 分支未安装，VIEScore 式的本地 Qwen3VL 打分器仍是开放项。
**两条协议陷阱（PoC 实测，15 个 run）**：同一张图跨 run 重复（cosine 完全相同即同一输出，必须按 SHA256 去重）；ArcFace 绝对值跨对不可比（要用相对 pseudo-GT 上限的 Δ）。身份均值 0.5592、中位 0.5406、最低 `hard_03` 0.2259、最高 `testset_hard05_promptv2` 0.8400；**hard_05 家族 PE prompt 0.8183/0.8400 vs 手写 prompt 0.3430/0.3149/0.3892，而姿态轴反向**（手写 body PCK 0.667 vs PE 0.389）——**两个优化方向指向相反的结果，任何只报单轴的结论都不可信。**

### 6.8 runs 现状与基线数字 [实测]

远端 `workflow/runs/` 顶层：`q8l_hard_02 / q8l_hard_05 / q8s_hard_02 / q8s_hard_05`、`scale_anchor_20261013/`（6 个子 run + summary.json）、`test/`（14 个 legacy + formal v1/v2：demo、hard_01–05、hard_05_manual、hard_05_torso、medium_02、testset_*）、`tiktok/`（8 个舞蹈测试 + results.md）、`torso_check_person_20261014`、`torso_measured_hard02_20261014`、`wp1_accept_hard02`，另有 `runs-face-lora/`。`report.py` 在 2026-10-14 测得 **19 个 run、7/7 对覆盖**；tiktok 再加 8 个（768/seed42：**7 个 body PCK 1.000，tt_00304 0.938**，竖屏 604×1080，取 `pairs_strict_v1` 的 test 划分）。

单人基线表（`evaluate.py` body PCK@0.2，768/seed42/30 步，PE prompt，workflow README "实况"表）：

| run | PCK@0.2 | 备注 |
|---|---|---|
| hard_01 | 1.000 | `--subject largest`（画面里 3 个人） |
| hard_02 | 0.944 | |
| hard_03 | 0.944 | |
| hard_04 | 1.000 | |
| **hard_05** | **0.056** | **缩放锚点灾难**：引导躯干 95.9 px vs 生成 225.6 px（2.35×） |
| hard_05_torso | 0.389 | scale 0.977、ratio 1.02（尺寸对了，姿态没跟上） |
| hard_05_manual | 0.667 | 同 framing + 手写 prompt，源哈希 `17b9c608…` |
| medium_01 / medium_02 | 1.000 / 1.000 | |

> ⚠ 这些数字来自**三个不同的代码状态**（`6603a819…`、`6a57ff88…`、`5ac98986…`），**全套从未在同一状态下重跑过**。`q8s_hard_05` 的 `eval.json` 现在是 0.778，而 GGUF 文档 §9 表格写 0.833 —— 两处**未对齐**，以 `eval.json` 为当前事实。

`workflow/scripts/` 只有一个工具目录 `six-picture-create/`（`script/` + `output/` + `README.md`，即 AGENTS.md 指定的标准样板）：把每 run 的六个关键产物拼成一张图（A_reference、B_rgb、B_poseguide_overlay、**B_3d_dwpose = 唯一进生成器的那张**、A_poseguide_alignment、final_image），flags `--all/--check/-o/--cell 620/--gap 22/--scale-threshold 0.65/--lang`，标题自动从 `geometry/stage.json` 的 `metadata.camera_checks` 生成。`workflow/patches/` 里是 `SMPLest-X_local.diff`（相对上游 `fdebd887`）+ `SMPLESTX_DET_CONF` 入口脚本（R12）。整改前的 run 备份在 `/data/shangguanrz/pic-edit-archive/workflow_runs_backup_20261012/`（带 SHA256SUMS）。

### 6.9 workflow-multiple：双人流水线 [实测]

`workflow-multiple/` 在远端完整（约 40 个文件），本地也有一份（较旧）。它的核心是一个 **85 格矩阵审计**（`matrix_audit.py`），2026-10-14 12:52 实测：**planned 85 / covered 85，退出码 0**（68 个 dwpose 格 + 17 个 fit 孪生格；四个标记 BATCH_DONE / RETRY_DONE / FINALIZE_DONE / FIT_ALL_DONE 全部出现；83 个有姿态分、83 个有身份分、**43 个人数正确**）。

四个臂的**人数正确率**（这是全部矛盾的中心）：

| 臂 × framing | 人数正确率 | body PCK 均值 |
|---|---|---|
| skeleton_only × b_canvas | **70.6 %** | 0.786 |
| skeleton_only × source_fit | 47.1 % | 0.779 |
| with_b × b_canvas | 47.1 % | 0.828 |
| with_b × source_fit | **17.6 %** | **0.815** |

**关键教训（记忆条目已被磁盘确认）**：**"哪个臂更好"依赖档位（framing）**。给生成器看 B 的照片（`with_b`）会**提高 PCK 但大幅降低人数正确率**；只看骨架则相反。配对投票也随上下文翻转（"臂 @ b_canvas 17：PCK 4/6/5"；"臂 @ source_fit：PCK 6/6/5"），早期在部分格子未跑完时下的方向性结论在跑满后被改写（README change-log 213–217 行）。

- **swap 结论站得住**：`identity_swap_suspected` 9/55（后为 68 个 dwpose 格里 7 个，约 10 %），幅度 0.022–0.120，而未标记组最大值 0.018，阈值 0.02 由 `swap_calibration.py` 定标（README:79,122,197）。`dance_pair02_059` 在 5 次 run 里全被标记，**尽管它的 PCK 是 0.875–1.000**——"两人站到了对方位置上，关节却摆对了"（README:94）。这条指标是有效的。
- **import 结论已被控制组证伪并撤回** [已撤回]：`target_import`（逐人"与 B 的相似度 − 与 A 的相似度"）原本报告 3/17、4/16 被标记 ⇒ 泄漏；55-run 控制聚合证明它由 **crop/scale 与参照集差异**驱动，明确撤回（README:69-77）。连"偏置来自裁剪"这个逃逸假设也被 `import_crop_probe.py` 在全部 85 run / 190 个身上杀死（README:96-112，"这条路走不通，已被实验关掉"）。现在**只允许**报"减去控制基线后的数值"，不允许报布尔判定。
- **0.31 常数属于单人线** [实测]：`TORSO_TO_PERSON_BOX_HEIGHT = 0.31` 在共享的 `src/pose_edit/geometry.py` 里，只被单人 `source_torso` framing 使用（且已降为 fallback）。双人线自己 38 个躯干的实测中位数 0.285、IQR 0.273–0.300，双人 README 明确声明其结果"与 0.31 常数无关"。
- **向导对比 n=17 后 GT 仍是默认** [实测]：`--by source` 的 fit-vs-GT 孪生对（n=17）：PCK 7/2/7 平、人数 3/11/3 平、MPJPE 9:16 与 identity 10:16 **偏向 GT**（中位差 +0.025）；早期 n=3 的方向性读数被判不可靠。`keypoints_source` 默认保持 `dwpose`，fit 臂重新定位为"度量参照"。WP7 门禁：fit 向导与 DWPose 的髋差 0.120 > 0.10 **FAIL**，但与投影后的 Hi4D 几何比是 0.021 PASS —— backhug 向导自身错位约 **68 px**。

双人线的已解决问题还包括：四图输入 34.04 GiB/142 s vs 三图 86 s；骨架与 Hi4D mask 对齐 90.8 % / 关键点 100 % 落在框内；hard_02 端到端不变式 **PCK 0.944 / MPJPE 0.057 / hands 0.179 / face 0.021**；以及一条通用的门禁学原则——**门禁阈值要和"参照自身的噪声地板"比（真实几何 vs DWPose 的互差是 0.909/0.105），不能和生成质量比**。

### 6.10 workflow 已修 bug 清单（这是最有复用价值的部分）[实测]

1. **W7 / R6：DWPose body-18 尾部顺序错**。旧代码尾巴写成 `23,22,21,20`，把 L_Eye/R_Ear 放到 slot 15/16；改成 `R_Eye,R_Ear,L_Eye,L_Ear`，并加了一个**去解析已安装 SMPLest-X 的 `joints_name`/`flip_pairs`** 的测试，而不是再抄一遍常数。
2. **"0.31 常数在 5 对上都在 ±4 % 内"被实测证伪** [已撤回]：`workflow-multiple/torso_ratio.py` 量了单人线自己的 8 张源图，躯干/框高比跨度 **0.259–0.524** ⇒ 有效缩放误差 **0.59×–1.20×，仅 2/8 落在 ±10 %**。在 hard_02@1024：实测 A 躯干 585.7 px，常数臂只给到正确缩放的 59 %、`source_person` 给 61 %（两者仅差 2.7 %，"两个代理量都在猜"）。
3. **fit worker 的 `ImageOps.contain` bug**：它返回**新图**，第一版 worker 读的是旧尺寸 ⇒ scale=1.0、负 pad、所有拟合点整体偏移 **(22, 192) px**，而模型自己的 j2d↔j3d 自检**完全看不出异常**。
4. **`133 vs 134` slot bug**（见 §3.4）。
5. **`report.py` 三个崩溃 bug**（缺 import、只做一层遍历）。
6. **run 改名/挪动会毁掉缓存** → `identity`/`runtime` 拆分修复（W3）。
7. **产物写完后被人改动不被发现** → `stage_reasons` 逐因枚举（W5）。
8. **重复图把跨 run 均值悄悄拉偏** → 按 SHA256 记 `duplicate`。
9. **`docs/workflow审查与整改_20261012.md` 的 W1–W12 → R1–R12**：prompt-file 溯源、identity/runtime 分离、逐因过期、实测校验、`report.py`、body-18 修复、`--subject`、conf-hook 检查、pytest 接入、`--pair`/batch/evaluate、`--framing source_torso`、`patches/`。
   当时明确**未解决**：多人语义超出姿态的部分、0.31 只在 5 对上验证、用同一个估计器自评的偏置、run 身份 = 目录名、全套未在同一代码状态重跑。

### 6.11 workflow 的四个开放分支（2026-10-14 → 10-15 的决策状态）

| 分支 | 决策状态 | 关键证据 |
|---|---|---|
| **P0 官方外部基线** | **未跑**。`docs/开源基线调研_Qwen同基座_20261004.md` §4 把 Qwen-Image-Edit-2511 排第一（"审稿人第一问"），20 B bf16 ≈ 55–60 GB，84 GB 卡放得下。`--condition none` 那一臂同样缺实现 | 全仓对 Edit-2511 **零引用** |
| **控制信号消融 / Union ControlNet** | `research/qwen21_controlnet_transfer_20261015/REPORT.zh-CN.md`（**完成的是架构研究，没有新的出图/收敛/显存实测**）。结论：**保留我们的人物几何/接触/身份绑定/脸手特征，逐步用原生 Qwen 控制块替换 DeepGen 控制核；优先借 Union 的现成预训练权重 + 小模块/LoRA；先立"原样官方基线"，再处理参考图身份路径与 target-only 注入；不要从零训 37.75 亿参数的控制器**。Union 事实：**180 个 BF16 张量、3,775,479,808 参数**，16 个控制块、索引 `[0,2,…,30]`、`control_in_dim=129` = 64 control latent + 1 mask + 64 masked-image latent；官方训练配方 batch1@1024、LR 2e-5、warmup 100、max_grad_norm 0.05、wd 3e-2、**只训 `control.*`**、10 % control-dropout、两阶段 CFG 蒸馏（teacher 3.5 → 推理 cfg 1.0）。文件头审计已完成，**7 GiB 权重本体未下载**。审计脚本 `verify_audit.py` 输出 `PASS`（28 个源哈希、37 个可解析 py、144 个 shape 匹配）；固定版本 VideoX-Fun `13bb3c625e…`、Union HF revision `8a4702014d…` | 报告同时指出：不能说"所有版本都不读 Qwen hidden"（lite 的 `residual()` 明确读当前层 target hidden，这反而是值得保留的设计）；Full 的 32 层注入**不等于** 32 层原生控制（同组残差被连续写入 5–6 个 block，"重复写入可能使局部修正过强"是**待消融的机制假设，不是已证明的失败原因**）；`block_controlnet_hidden_states` 只是属性名 |
| **方案 B：Union 数据集适配 + 小规模训练** | `docs/superpowers/plans/2026-10-15-qwen21-union-dataset-adaptation-plan-b.md`（39.4 KB，**[计划]，未实现、未下载权重、未启训**）。四臂 `native / union_frozen / union_lora_joint / union_lora_target`（后两组同初始权重、同初始 LoRA、同采样轨迹，**只改残差注入范围**）+ 三个诊断 `union_zero / union_target_frozen / wrong_pose`（**置零控制图 ≠ 关闭控制，不能代替 scale=0**）。预算固定 **train 1536（iPER/TikTok/Hi4D 各 512）/ val 72 / test 128（+3DPW 32 纯域外）+ overfit6**；seed 20261015，排序键 `SHA256("union_b_v1:"+sample_id)`；每步各取三数据集一条、micro 1 × accum 3 = 512 次更新正好各过一遍。LoRA 精确白名单 **130 个模块**（`control_img_in`、`control_blocks.0.before_proj` + 16 块 × 8 后缀），rank 16/alpha 16/dropout 0，A=Kaiming、B=0，**新增 23,267,344 参数**；AdamW lr 1e-4、wd 0.01、32 步 warmup 后 constant、clip 1.0；checkpoint 0/128/256/512，最终**固定 step 512 不挑 best**；先 10 microstep 显存探针（BF16 模型本体约 20.29 GiB，**不得当作峰值显存预测**），再 overfit6 200 updates，**平均 loss 相对 step0 至少降 20 % 才允许开正式两组**（这是可学习性门禁，不是质量承诺）。新代码全部落在 `src/pose_control/qwen21_union/` 与 `src/data/qwen21_union_dataset.py` + `tests/test_qwen21_union_*.py` | 明确的"不做"清单：不加 DeepGen 控制核、不加 BFS LoRA、不加 Face/Hand Adapter、不训全量 Union、不自动搜 scale、不用 PE（提示词模板固定，双人模板要求 "Keep the two people distinct."） |
| **PE 加速：量化 / runtime / 蒸馏** | 量化**降为次要**（官方 llm-compressor 配方锁 `linear_attn` + visual 在 BF16 ⇒ 天花板约 1.5×）；换 runtime 曾被寄予 2.4–4.4×，但 §9 配对端到端实测（2 对）：GGUF 每 token 快约 2×、总时间却 **987 s vs 954 s，净节省为零**，因为 GGUF 在 2/2 对上触发二次重试、平均多吐 214 token。⇒ **"重试率比 tok/s 更值钱"**。蒸馏方向：在仓内 Qwen3-VL `text_encoder`（8.767 B、有 `lm_head`）上加 Prompt-LoRA 技术可行（rank-16 全语言投影 = 43,646,976 参数 ≈ 83.25 MiB），但**2026-10-15 00:52 重排优先级：先把公平的 Q8+MTP 评测做完**；Pocket-2B 判为不可用（`qwen3_5_text`，无 vision_config，吃不下三图输入）；提出的验收线：契约 ≥99 %、ΔPCK 非劣 −0.02、重写中位提速 ≥3×；**token 计费、骨架 OOD 的学生视觉通路、许可（Qwen Research License 非商用条款 vs 蒸馏并把派生权重交付华为）三项被标为需要人类决策** | GGUF 的另一条硬教训：**PCK 必须和两个 `torso_ratio` 一起报**。q8l_hard_02 的 PCK 0.000 不是画坏了——它按引导的 1.623× 画，而 A 的真实比例是 0.987×（HF 臂是 0.496→0.944），**PCK 在惩罚"画对了尺寸"**；同一臂反而在保持项领先（identity_cos 0.969 vs 0.656、bg MAE 4.39 vs 18.31）。`hard_05` 则完全反向（pose 1.000 vs 0.833、identity 0.413 vs 0.558）。**两对相反、每格 n=1，不构成结论**；下一步 ≥8 对 × 2 seed 并报 5 元组 |

---

## 7. 数据、环境、目录与复现

### 7.1 数据集与划分 [实测]

| 数据集 | 清单 | train / val / test | 分组与用途 | 必须知道的限制 |
|---|---|---|---|---|
| **iPER** | `datasets/iPER/iper_sampled_6src64tgt/splits/{dev_train,dev_val,test}_pairs.jsonl` | **26,496 / 3,840 / 7,680** | 69 / 10 / 20 个 appearance，三者无重叠；单人主训练与评测 | `train_pairs.jsonl` 有 30,336 行且**含 dev_val**，训练必须用 `dev_train`。appearance 前缀在 train/val、train/test、val/test 分别有 **7 / 9 / 5 个交集** ⇒ 只能说"appearance 留出"，**不能说"完全未见的身份"**。离线条件包 `iper_assets_512_nvdiffrast/`，共享缓存 `experiments/iper_adapter_finetune_v1/condition_cache`（584 个张量） |
| **TikTok** | `datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1/{train,val,test}_pairs.jsonl` | **8,574 / 1,120 / 1,156** | 20 / 3 / 3 个 `source_video_group`，组间无重叠；舞蹈、脸手、小人物 | 用 `pairs_strict_v1`，**不混** `pairs/` 的宽松版。`sequence_id ≠ 独立视频组`，防泄漏按 `source_video_group`。**test 只有 29 序列 / 3 个视频组** ⇒ CI 必须按组 cluster bootstrap。条件过滤后 8,548 / 1,112（剔 26 / 8）。已知缺陷：604×1080 → 512 的宽高比挤压 1.79×；**身份未做认证**。`stats.json`：340 序列 / 92,961 帧；`TikTok_v3` 761 MB = manifests 401 MB + dwpose_v1 275 MB + qwen21_face_feature_cache 58 MB + qc 29 MB。`tiktok_verify.log` 是 **0 字节** |
| **Hi4D** | `datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_{train,val,test}.jsonl` | **1,312 / 822 / 606** | 6 / 2 / 2 个人物 pair；双人姿态、遮挡、接触 | 训练 pair01/09/10/12/14/16，验证 pair22/28，测试 pair00/02；唯一目标 774/274/330 ⇒ **多条 source 对同一 target 不能当独立样本计数**。`native_interaction_v4_2` 数量相同但**生产入口读 `final_assets_v4_2`**；首样本 RGB/geometry 是 **940×1280**（目录名不是分辨率依据）。NPZ 关键点生成器是 `Hi4D_mask_bbox_DWPose_official` ⇒ 历史上条件准备**用过目标人物 mask**；"生成器不读目标 RGB/mask" ≠ "上游条件无 GT 辅助"。`target_contact=true` 不保证接触顶点 > 0，抽样与评估按实际顶点数、零接触单列。Hi4D-origin 359 GB、**32 个 tar.gz 未解**（真正需要的几何+标注约 85 GiB，不是 445 GiB） |
| **3DPW** | `datasets/3dpw_single_final_v2/{train,validation,test}_pairs.csv` | 14,508 / 600 / 600 | 14 / 7 / 9 个 sequence，无序列重叠 | 本轮**只用 test 做域外**，不参与训练与调参 |

评测集：`baseline_50_eval_pairs.jsonl`（iPER held-out 50，69 训练/19 测试 appearance、0 重叠）；joint pilot 固定 36 例（24 TikTok 608×1088 + 12 iPER 512）；`tiktok_pe_prompt_rewrite` 的监控子集 12（9 tiktok idx 0,4,7,8,13,14,16,19,22 + 3 iper 0,2,3）。

### 7.2 三套环境（不可混用）[实测]

| 环境 | 路径 | 关键版本 | 谁在用 |
|---|---|---|---|
| **qwen21**（主线） | `/home/shangguanrz/nihaochen/envs/qwen21/bin/python` | py 3.12.14、torch **2.8.0+cu128**、**diffusers 0.41.0.dev0**（git pin `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`）、**transformers 5.17.0**、hub 1.32.0、tokenizers 0.23.2、peft 0.20.0、safetensors 0.8.0、onnxruntime-gpu 1.30.0、insightface 2.0、scikit-image 0.25.2；清单只在远端 `requirements_qwen21.txt`（500 B） | 所有 Qwen 训练/评测、workflow 的 prompt+generate 阶段 |
| **deepgen**（对照） | `/home/shangguanrz/miniconda3/envs/deepgen/bin/python` | py 3.12.14、torch 2.8.0+cu128、**diffusers 0.35.2**、**transformers 4.56.1**、peft 0.17.0、numpy 2.1.2；`requirements_lock.txt` 242 包 | V6/V6.5 全部、DeepGen 评测臂 |
| **smplestx_bw** | `add/envs/smplestx_bw/bin/python` | py 3.12.3、torch 2.8.0+cu128、torch/smplx/nvdiffrast/cv2/onnxruntime/ultralytics 已验；`pyrender` 需 EGL | workflow 的 geometry 阶段 |

另有 `deepgen_pose`、`multihmr`（py3.9 + numpy 1.26.4）、`sapiens2_hf`、`smpl_clean` 四个外部 env。
**红线**：`diffusers 0.41.0.dev0` 是为 Qwen-Image-2.1 专门保留的版本；**不往 qwen21 塞可能破坏它的包**；deepgen env 就地升级会让它既能装又不坏地加载 Qwen21 是不可能的（版本要求直接冲突），而升级会破坏已完成的 V6 结果。

### 7.3 硬件与磁盘 [实测]

- 单卡 **NVIDIA RTX 6000D**，compute capability (12,0) = sm_120，可用 **83.05 GiB**（`total_bytes = 89,171,099,648`）。
  ⚠ 同一张卡在文档里被写成 **83.05 / "80GB" / "85GB" 三种值**（`system_environment_lock.md` 与 `readiness.json` / `docs/experiments/*` / `AGENTS.md` §6）。以 83.05 GiB 为准。
- **NVML 坏了**：内核驱动 `580.173.02` vs NVML 库 `580.178`，`nvidia-smi` 返回码 18 / `Driver/library version mismatch`。但 **torch 查 CUDA 可用、能分配能算**（`tensor_operation_passed: true`）。⇒ 任何脚本不要依赖 `nvidia-smi`；显存要靠 `torch.cuda.max_memory_allocated/reserved`。
- 主机 node01，Ubuntu 24.04.2，125 GiB RAM（113 free），git-lfs 3.6.0。
- **`/` 已用 95 %（约 76–91 GB free）；`/data` = `/dev/sdb1` 22 TB 只用了 6 %。⇒ 一切新产物必须落 `/data/shangguanrz/`。**
  `/data` 六区：`pic-edit-datasets/` 463 GB、`pic-edit-archive/` 95 GB、`pic-edit-models/` 49 GB、`pic-edit-caches/` 9.6 GB（**1,390 个 `control_<20hex>.pt`**）、`dataset_samples/` 7.0 GB、`pic-edit-experiments/` 1.4 GB。
- **sg 有外网**（实测 HF ~10 MB/s）——早期"sg 无外网、必须在 Windows `pip download` 再 `scp`"的记录是**错的**（`docs/workflow_PE重写成本_量化与蒸馏调研_20261014.md` §1.5 已明确纠正，但仍有文档在重复这个前提）。

### 7.4 本地 ↔ 服务器目录差异（动手前必看）[实测]

| 项 | 本地 Windows `D:/codeplus/pictureedit` | 服务器 `sg:/home/shangguanrz/project/pic-edit` |
|---|---|---|
| Git HEAD | `2b44828`（main） | `17991d3` + 未提交修改；9 条分支已于 2026-10-15 全推 GitHub |
| Python | 3.13.13 / torch 2.12.0+**cpu** | 见 §7.2 |
| **训练/评测入口** | **在仓库根**：`train_iper_qwen21_{body,face,full}.py`、`eval_iper_qwen21_{body,face,full}.py`、`train_tiktok_overfit_sanity.py`、`train_iper_v65_face.py`、`bfs_explore.py` | 入口在 **`scripts/train/`**（`train_iper_qwen21_full.py`、`train_iper_qwen21_body.py`、`train_iper_qwen21_face.py`、`train_hi4d_qwen21_dual.py`、`train_qwen21_joint_pilot.py`）、`scripts/eval/`（**36 个**，含 `eval_iper_four_arm.py`、`eval_hi4d_qwen21_dual.py`、`eval_qwen21_bfs_rgb.py`）、`scripts/run/`；根目录只留 `bfs_explore.py` |
| `scripts/` 规模 | 顶层散着大量历史脚本 + `train/eval/run/data/audit/reports` | 150+ 个 .py，其中**只有 16 个进 git** |
| `experiments/` | **10 个平铺目录** + `README.md` + `实验中文索引.md`；**除 README 外全部未进版本控制**（`.gitignore:16` = `/experiments/*`，仅 `!/experiments/README.md`） | **5 个编号族目录**（`1.deepgen experiment` / `2.qwen-test` / `3.singel man` / `4.two man␣` / `5.workflow`，名字含空格与拼写错误）；2026-10-11 时 32 个目录 / 3,200 文件 / 66 GB |
| `docs/` | 33 个 md + `experiments/` + `superpowers/` + `qwen21_architecture/` | 10 个顶层 md，**其中 5 个本地没有**（`架构审查_20261012.md`、`adapter_v6_architecture.md`、`qwen21_hi4d_dual.md`、`iper_v6_training_plan.md`、`champ_deepgen_single_sample_runbook.md`）；本地另有 ~18 个远端没有（全部 `BFS_*`、`workflow_PE*`、`架构说明_*`、`远端_*` 等）。`docs/README.md` 索引的是**远端那套** |
| 有 / 无 | 本地**无** `models/` 内容、`datasets/`、`add/`、`vram_lab/`、`deepgen/`、`results/`、`inputs/`、`caches/`、`patches/`；本地**多** `workflow/`（旧）、`workflow-multiple/`（旧）、一堆 `.` 开头的暂存目录与 14 个远端根目录没有的文件 | 权重合计 19 GB（`/home`）+ 31 GB Qwen + 54 GB Edit + 49 GB（`/data`） |
| 根目录 `*.png` / `_probe_*.sh` | 有（`bfs768_004.png`、`panel_004.png`、13 个 `_probe_*.sh`）——**AGENTS.md 要求用完即删，尚未清** | — |

> 后果：`config.json` / `run_manifest.json` / `request.json` 里的**绝对路径**在另一侧一律解析不了。拷贝 run 目录做实验时如果手改过路径，副本会去读写源目录 —— 检测标准是**副本产物与对照的 SHA256 完全相同 = 副本没生效**。

### 7.5 复现命令矩阵（按机器分列，均已核对文件存在）[实测]

```bash
# ---------- 服务器（有 GPU；权威侧）----------
source /home/shangguanrz/miniconda3/etc/profile.d/conda.sh
conda activate /home/shangguanrz/nihaochen/envs/qwen21
cd /home/shangguanrz/project/pic-edit

python3 scripts/train/train_iper_qwen21_full.py --branch body --resolution 512
python3 scripts/train/train_iper_qwen21_full.py --branch face --resolution 512 --face-aux-alignment five_point
python3 scripts/eval/eval_iper_qwen21_full.py   --modes body body_face --resolution 512
python3 scripts/train/train_hi4d_qwen21_dual.py --steps 500
python3 scripts/train/train_qwen21_joint_pilot.py          # BFH 联合线
python3 scripts/eval/eval_qwen21_joint_pilot.py
python -m pytest tests/test_qwen21_*.py -v

# workflow（单人流水线）
bash workflow/run.sh  --pair <dir> --name <run>            # 或 --a/--b/--resolution/--steps/--seed/--subject/--framing
bash workflow/batch.sh                                      # 遍历 $PAIRS/*/，RESOLUTION 默认 768
python3 workflow/run.py --name <run> --verify               # 只校验，不计算
python workflow/evaluate.py  --run <run>                    # 第一 reader（smplestx_bw 环境）
python3 workflow/evaluate2.py --run <run> --force           # 无 GT 六轴（qwen21 环境）
python3 workflow/scorecard.py --baseline <A> --candidate <B>
python workflow/report.py ; python workflow/run.py --check  # 自检 vendored hook

# 两阶段 BFS 高清换脸精修（根目录）
python3 bfs_explore.py --pairs <pairs.jsonl> --resolution 768 --steps 30

# ---------- 本地 Windows（CPU，只读代码 / 契约测试）----------
python -m pytest --collect-only -q      # 808 tests collected / exit 0 / 0 collection error
                                        #   tests/ 704 + workflow/tests 58 + workflow-multiple/tests 46
```

`pytest.ini` 实际内容：`testpaths = tests workflow/tests workflow-multiple/tests`、`pythonpath = .`、`norecursedirs = add datasets models experiments results demooutput model-qwen-edit data vram_lab .git`。
**测试通过状态无法在本地断言**：`torch 2.12.0+cpu`、`cuda.is_available() = False`，所有 `map_location='cuda'` 的 TorchScript 评测器与 GPU 冒烟测试都会 skip 或 error。历史记载（**属 [文档]**）：远端 2026-10-12 审计 **573 passed / 30 failed**（剔除 7 个文件后）；joint pilot 六文件远端 **121/121**、本地 120 passed/1 skipped；`tiktok_qwen21_full_training.md` 远端 188 + 本地 20；`hi4d_smplx_pilot` **12/12**。

---

## 8. 不可引用清单：文档与磁盘的矛盾登记

这一节是给 LLM 读者的**防错表**。左边是文档里的话，右边是磁盘事实。

| # | 文档声明 | 磁盘事实 | 级别 |
|---|---|---|---|
| 1 | `README.md` §3.3 "双人检测率 100%（平均 2.0 人），Identity Swap = 0.0" | 适配器每臂 `detected_person_count = 1.0`，swap 为 `null`（分母 0）；2.0 与 0.0 来自 native/off 对照组 | **[已撤回]** |
| 2 | `README.md` / `docs/qwen21_hi4d_dual.md` "10k 双人训练、822 对验证" | `full_run_stopped_status.json`: step **410**、`checkpoint_saved: false`；`validation_full.json` `count: 1` | **从未发生** |
| 3 | `qwen21_dual_v8/README.md` "dual_v8 的 face swap_rate 达 0.333"；"PCK 0.22→0.90" | 0.3333 属于 **native** 臂，dual_v8 是 `null`；PCK 跳升是分母 132→69 的假象 | **误归因 + 分母假象** |
| 4 | `docs/experiments/2026-10-08-…-handoff.md`：1,024 步 + BFSBody 0.3583/0.5248/0.9111、BFSHead 0.3333/0.5481/0.9206、"121/121 全通过"、"72 拼图 + 30 脸表" | 本地 `report/summary.json` = `total_formal_steps 512`、两支 `formal_steps 0`、`valid_pair_n 0/36`；远端 native = 1024 但值是 0.3282/0.6818/0.9578 与 0.3072/0.7082/0.9621 | **五个数字两边都对不上，只有方向幸存** |
| 5 | `experiments/README.md` 头部"本目录 34 个实验 / 7 大族" | 本地 10 个目录；远端已重构成 5 个编号族；两份索引（`实验中文索引.md`）都没提 | **索引过期** |
| 6 | `tiktok_00001_condition_repair/README.md` "44 张对比图 + `train_pose_part_1500/` 权重与日志" | 目录共 **8 个文件、6 张 PNG**，无权重、无日志、无指标 | **无物证** |
| 7 | `qwen-2.1-test/README.md` `data/`、`experiment/`、`adapter_overfit.pt`、"309 张生成图" | 本地 41 文件、`find *.pt` 为空；只在 `sg:…/2.qwen-test/` 存在 | **路径失效** |
| 8 | `docs/架构说明_body_face_hand_适配器.md` §3.1 `src/pose_control/v6/checkpoint.py` 的 `ARCHITECTURE_VERSION = "v6.6"` | 该文件第 10 行是 **`"v6.5"`**；`v6.6` 在 `qwen21_full/checkpoint.py:10`。`src/pose_control/v6/README.md:3` **重复同一错误**，还引用了不存在的 `V65/V66_BRANCH_NAMES` | **版本标签错** |
| 9 | 同文档 §0 表格把 `src/pose_control/qwen21/` 列为"当前主力"、注入层写 3/8/13/18/23/28 | `train_iper_qwen21_full.py` 用 `qwen21_full/`，hook **全部 32 层**、6 组 `[6,5,5,6,5,5]`。**注入面被低估 5.3×** | **主线认错**（`docs/架构审查_20261012.md` P1-1 用 import 计数佐证：v6 84、qwen21_full 77、qwen21 40） |
| 10 | 同文档 §4.4 "新增组件**全部**零初始化以免破坏已训权重" | `dual_adapter.py` 里只有 `depth_in`、`interaction_heads`、`appearance_heads`、继承的 `zero_heads` 是零，另外 6 个模块是 PyTorch 默认初始化 | **错** |
| 11 | `configs/adapter_v6_architecture.json` 被 `configs/README.md` 与根 `README.md:269` 当权威（"注入层索引 [3,8,13,18,23,28]"） | 文件**已从 worktree 删除**（`git status` 显示 ` D`）；从 `.wt/check-main-clean/` 恢复出的真实内容是 `deepgen_injection = {transformer_layers: 24, block_ranges: [[0,3],…,[20,22]], last_block_injected: false}` —— `[3,8,13,18,23,28]` 是 Qwen 侧常量，**从未出现在该契约里** | **权威文件缺失 + 引用内容错** |
| 12 | `README.md` 命令矩阵 `python3 scripts/train/train_iper_qwen21_full.py` | 远端**对**（`scripts/train/` 有 5 个训练入口）；本地**没有**（`scripts/train/` 只有 `train_qwen21_joint_pilot.py`，本地根有 `train_iper_qwen21_full.py`） | **两侧不同，须声明在哪侧** |
| 13 | `docs/架构说明…` §9 "Qwen21 30 步" | `true_cfg_scale=1.0` ✔，但 `eval_iper_qwen21_body.py:39` 默认 **40** 步；30 只是 face/full | **差 10 步** |
| 14 | 同一个四臂结果两处不同小数 | PSNR 25.91 / PCK 0.979（`架构说明…`、OKR）vs **25.86 / 98.00 %**（`README.md:222`）；DeepGen+V6 24.54 vs **24.00**（后者其实是 `iper_v6_20k_eval_final` 的另一个数） | **一处结果两处写法** |
| 15 | 身份上限同一量三个值 | `架构说明…` §1.1 "source-target 基线 0.613" / `BFS_面部保持移用评估…` "`source_target_ceiling` 0.6272" / `OKR…` 表2 "约 0.683" | **未统一** |
| 16 | `docs/qwen21_architecture/README.md` | 目录名叫 architecture，实为生成层空话桩文件（6 条"该模块下实现/记录文件"），**零架构内容**；真正的载荷是 3 张 PNG + 2 个绘图脚本 | **不可当权威** |
| 17 | `tests/README.md` "实际测试文件共 **60 个**" + 约 20 个模块名；并称 `pytest.ini` 的 testpaths 只有 `tests` | 本地 **54** 个测试文件；列举的 `test_qwen21_control_contract.py`、`test_qwen21_dual_control.py`、`test_v66_hand_core.py` 等 22 个文件本地不存在；`pytest.ini` 实际含 `workflow/tests workflow-multiple/tests` | **文档写的是远端树** |
| 18 | `models/Qwen-Image-Edit-20B`（注册表 `legacy_qwen_edit_20b`） | 实际是 **Qwen-Image-Edit-2511**：`QwenImageEditPlusPipeline`、`_diffusers_version 0.36.0.dev0`、Qwen2.5-VL 编码器、卡是上游 2511；目录名与注册表名都误导 | **命名错误** |
| 19 | `reports/Qwen-Image-2.1_架构对比与迁移路线_20260924.md` §1 DeepGen 文本编码器 = "T5 + CLIP 拼接" | `models/DeepGen-1.0-diffusers/vlm/config.json` = `Qwen2_5_VLForConditionalGeneration`（hidden 2048、36 层） | **SD3 通用配方 ≠ 本 checkpoint** |
| 20 | 一张卡三种显存值（83.05 / 80 / 85 GB）；`docs/远端_目录总览_20261011.md` 称 `configs/` = 2 文件；`README_部署清单_20261012.md` §六自称"约 130 个人工 README"但同文 §八又删掉 25,417 个生成 README 后称剩 ~150 | 83.05 GiB 为准；`configs/` 现在确实 2 个文件但那正是缺失 JSON 后的状态；`research/README.md` 只索引 25 个子目录中的 10 个，且 **13 个 `research/*/README.md` 仍是生成桩**（AGENTS.md §4.3 明令禁止的那种句子） | **过期 / 未清理** |

**值得逐字引用的可靠核心**：`qwen21_iper_body/four_arm_eval_50_s40/summary.json`（README 与之一致，且它自己的 bootstrap CI 就是最诚实的限定语）；`bfs_compare_30_20261004/results.json` + `docs/BFS_扩大样本对照_20261004.md`（完全一致，演员 cluster CI）；`tiktok_overfit_sanity_full/summary.json` + `train_metrics.jsonl`（1500/1500 完整）；两个 2026-10-14 的 PE 目录（内部带撤回痕迹与实时日志，自洽度最高）。

---

## 9. 硬性纪律与踩过的坑

来自 `AGENTS.md`，加上本次实测补充，按"违反代价"排序：

1. **图片/视频是受保护资源**（`.png/.jpg/.jpeg/.mp4`）。任何整理、清理、迁移**不得**把它们纳入删移范围。本仓有大量评测生成图是唯一证据。
2. **不静默改实验变量**：拷贝 run 目录时 `request.json` 里的绝对路径不会自动改写；不改则副本读写源目录、实验白做。检测：副本产物与对照 SHA256 完全相同 = 没生效。
3. **不覆盖既有产物**：重跑加 `--force` 或换 `--name`。
4. **删除前先审计 + dry-run**（文件名+尺寸查唯一性，列清单，人确认再落地）。远端与本地会双向漂移：动远端前先 `git log` 核对，快照带时间戳。
5. **沙箱/权限被拒就停下回来问**，不绕。
6. **大文件不进 Git**（`.pt` / `.safetensors` / 条件缓存默认忽略）；**experiments/ 除 README 外整体未版本化** ⇒ 证据链本身没有备份，这是当前最大的可复现性风险。
7. **不往 `qwen21` env 塞不兼容的包**（`diffusers 0.41.0.dev0` 专用）。
8. **SSH 投递纪律**：含转义或重格式的脚本一律 `scp` 直传；本机 bash 的 heredoc 经 `ssh` 曾把 `\n` 污染成 `/n`。改过的远端文件必须 `py_compile` 或等价校验。
9. **`sgrzhh` 别名已失效**（`Connection closed`），只用 `sg`。
10. **GPU 与人类共用**：不要 `pkill`，不要依赖 `nvidia-smi`（NVML 坏了），不要用过时的 `miniconda3` 路径。
11. **workflow 代码改动会让全部历史 run 的 `--verify` 失效**（代码哈希进 identity）；批量任务进行中**不要动 `geometry.py`**（大方向 doc 的显式"不要做"清单第 1 条）。
12. **不要用 `--framing` 的常数回退值出结论**（0.31 已被降级）。
13. **不要只报 PCK**（§6.11 的 hard_02 反例）；不要把不同归一化方式混在一张表里比较。
14. **不要把 workflow 的 `B_3d_dwpose.png` 喂给训练好的 BodyAdapter**（分布外）。
15. **PE-I2I 的 backend / spec-type / MTP 必须写进身份字段**，否则同 seed 复现会骗人。

---

## 10. 实验血缘与当前状态

```
iper_v6_native_20k_v1 (DeepGen 20k, 远端) ─┐  权重供体(76.1M)
                                            ├─► qwen21_iper_body ──► full_20k/checkpoint_{best,last}.pt
tiktok_overfit_sanity_full (v6.5 六门禁 FAIL)┘        │                     │
   证据档案: tiktok_legacy_evidence_202609             │                     │
   助手: tiktok_00001_condition_repair (证据不足)       │                     │
                                                        │                     ▼
qwen-2.1-test (prompt-only 原型, 9+18 槽) ──────────────┤      bfs_compare_30_20261004 (BFS head 筛查, 102/102)
   └─ 16-test (stage1..6, 远端 2.qwen-test) ────────────┘                  │
                                                            两阶段级联决策 ▼
                                          bfs_body_face_joint_pilot_20261008
                                             ├─ 本地 = 512/letterbox 归档【已被取代】
                                             ├─ 远端 = native / 1024 步 / +BFH【当前权威】
                                             └─ docs/experiments/…handoff.md【数字不可用】
                                                        │
     hi4d_qwen21_dual_20260930（代码）→ single_hug / pair14 过拟合【10k 停在 410，无 checkpoint】
                                                        │
                              qwen21_dual_v8（v8 设计对拍 pilot，300 步，12/12 格）【两门禁失败】
                                                        │
     tiktok_pe_prompt_rewrite_20261014（503/10,850，运行中）──► pe_i2i_gguf_selection_20261014（端到端零收益）
                                                        │
     【2026-10-15 前沿，均未落地】
     ├─ qwen21_controlnet_transfer_20261015  —— Union 架构审计（文件头 only，7 GiB 未下）
     ├─ plans/2026-10-15-…-union-dataset-adaptation-plan-b —— 1536/72/128 四臂 + 130-LoRA 计划
     ├─ pe_i2i_qwen3vl_distillation_20261015 —— Prompt-LoRA 蒸馏（已降级：先做完公平 Q8+MTP 评测）
     └─ workflow P0 官方 Edit-2511 基线 + `--condition none` 参照臂 —— 都还没跑
```

**一句话状态**：姿态控制已经在 Qwen-Image-2.1 上做到 PCK@0.02 = 0.980（达标），但**身份保持未达标**（512 线 0.480 vs 门槛 0.65，只有 1024 严格子集 0.660 勉强过），**双人线全部失败于"只画出一个人"**，**提示词重写是唯一在跑的 GPU 长任务且还没接进训练器**，**"不训练"的 workflow 线反而是当前最稳的交付路径**。

---

## 11. 术语表

| 术语 | 含义 |
|---|---|
| **Adapter / 适配器** | 外挂的可训练模块，输出加性残差写进冻结基座；本项目的"训练线"核心 |
| **Full / lite / dual_v8** | 三代适配器实现：`qwen21_full`（32 层 6 组，DeepGen 控制核 + Qwen 边界桥）/ `qwen21`（6 层）/ `dual_v8`（typed schema + 原生注意力重采样，仍在设计→pilot 阶段） |
| **target-suffix-only residual** | 残差只加在序列末段目标图像 token 上，前缀（文本+参考图）不动 —— 本项目核心创新 |
| **zero-init head** | 每层注入头零初始化 ⇒ 控制 ON 与 OFF 在第 0 步逐位相同，保证不破坏基座 |
| **PCK@0.02 / NME** | 关键点对精度（阈值按图像对角线或躯干归一）/ 归一化平均关节误差 |
| **strict PCK** | 分母 = **请求的**关节数，把人裁出画外要罚；非 strict 变体允许靠丢难点刷分 |
| **BFH** | Body+Face+Hand 联合训练分支（joint pilot 的第五支，**未写进任何 README**） |
| **BFS** | Alissonerdx/BFS-Best-Face-Swap 开源 LoRA 家族（换脸方向，与本项目相反） |
| **PE-I2I** | Qwen-Image-2.1 官方图像编辑提示词重写 VLM（Qwen3.5-VL 9B 微调） |
| **workflow 的 4 阶段** | prepare(`inputs/`) → geometry(`geometry/`) → prompt(`prompt/`) → generation(`generation/`)，`evaluation/` 为外部产物阶段 |
| **stage.json / identity / runtime** | 阶段指纹：identity 参与缓存判定（含代码哈希），runtime（绝对路径）**永不参与** |
| **fitgeo / promptv2** | 两个 legacy run 名；它们的 `geometry/stage.json` 与 `prompt/stage.json` **逐字节相同**（md5 `fd932242…`、`3234112a…`），只差提示词文本 —— 不是两种几何方法 |
| **`B_3d_dwpose.png`** | 估计 3D 关节的 2D RGB 投影，唯一进生成器的引导图；与训练侧同名条件**不可互换** |
| **[实测]/[文档]/[已撤回]/[计划]** | 本文的证据分级，见 §0 |

---

## 12. 本文结论的来源文件索引

- **架构与代码**：`src/pose_control/qwen21/{body,face,dual}_adapter.py`、`src/pose_control/qwen21/runtime.py`、`src/pose_control/qwen21_full/{qwen_adapter,control_core,execution,qwen_runtime,conditions,condition_injector,geometry_encoder,frozen_bfs_lora,face_supervision,face_loss_alignment}.py`、`src/pose_control/qwen21_full/{face,hand,detail}/*`、`src/pose_control/v6/{condition_injector,reasoning/config,token_encoders}.py`、`src/data/{letterbox,iper_dataset,hi4d_qwen21_dataset}.py`、`src/evaluation/{joint_pilot_metrics,no_gt_metrics,hand_plausibility,dwpose_onnx/*}.py`、`src/pose_edit/{common,geometry,estimator,dwpose,prompt_contract}.py`；根 `train_iper_qwen21_{body,face,full}.py`、`eval_iper_qwen21_{body,face,full}.py`、`bfs_explore.py`；`configs/asset_registry.json`、`pytest.ini`、`requirements_lock.txt`、远端 `requirements_qwen21.txt`。
- **实验**：`experiments/*/README.md`、`experiments/README.md`、`experiments/实验中文索引.md`，以及各 run 的 `summary.json / results.json / run_manifest.json / per_sample.csv / *.log / eval.log`。
- **设计文档**：`docs/架构说明_body_face_hand_适配器.md`、`docs/BFS_LORA_实验说明.md`、`docs/BFS_Body_Global_架构审查_20261008.md`、`docs/Body_Adapter_SMPLX_输入审查_20261008.md`、`docs/BFS_面部保持移用评估_20261004.md`、`docs/BFS_扩大样本对照_20261004.md`、`docs/开源基线调研_Qwen同基座_20261004.md`、`docs/Qwen-Image-2.1-官方提示词增强调研_20261008.md`、`docs/tiktok_qwen21_full_training.md`、`docs/superpowers/specs/2026-10-11-qwen21-dual-v8-design.md`、`docs/superpowers/plans/2026-10-15-qwen21-union-dataset-adaptation-plan-b.md`、`docs/experiments/2026-10-0{8,9}-*.md`、`docs/experiments/2026-10-1{0,1}-*.md`。
- **workflow**：`workflow/{run,smplx_worker,pe_worker,generate_worker,evaluate,evaluate2,scorecard,report,batch,run}.py|.sh`、`workflow/config.json`、`workflow/README.md`、`workflow/pose_transfer/*`、`workflow/tests/*`、`workflow/scripts/six-picture-create/*`、`workflow-multiple/README.md`；远端 `workflow/runs/**`（19 + 8 个 run）；`docs/workflow{审查与整改_20261012, 无GT定量评测调研_20261013, _multiple_设计_20261013, _大方向优化调研_20261014, _PE重写成本_量化与蒸馏调研_20261014, _PE_I2I量化选型_GGUF社区权重评估_20261014, _PE蒸馏到Qwen3VL任务分支_20261015}.md`。
- **研究/证据**：`research/qwen21_dual_upgrade_20261011/{evidence.json,readiness.json,dual_code/*,repositories/}`、`research/qwen21_controlnet_transfer_20261015/{REPORT.zh-CN.md,remote_audit.json,source_snapshot/,verify_audit.py}`、`research/pe_i2i_qwen3vl_distillation_20261015/{audit.json,quantized_gate*.json}`、`research/hi4d_smplx_pilot_20261011/`、`research/diagnostic 系列`；`workflow/diagnostics/2026-10-13-scale{,-generation}/`。
- **交付与运维**：`reports/*.docx`（11 份，最大 `BFS_组会技术报告_20261004.docx` 25.5 MB）、`reports/*.md`（含 `OKR_人物图像编辑_20260929.md`、`Qwen21_双人Adapter论文与代码审查_20261011.md`）、`docs/system_environment_lock.md`、`docs/数据盘说明_20261012.md`、`docs/README_部署清单_20261012.md`、`AGENTS.md`。

---

## 13. 如果只能记三件事

1. **姿态已解决，身份没解决，且两者互斥**。四臂表里 0.8187 → 0.3510 那一行就是整个项目的发动机；所有支线（Face 分支、两阶段级联、BFH、1024 杠杆、无训练 workflow、PE 重写）都是在还这笔债。
2. **双人问题的真瓶颈是"只画出一个人"，不是姿态不准**。dual_v8 的 PCK 0.90 是 132→69 的分母假象；hi4d 的"swap = 0.0" 属于没训练过的对照组。先让人数正确，再谈双人姿态。
3. **这个仓库的文档层不可全信**。§8 的 20 条里，第 4/5/6/11/17 条会让你直接引用到不存在的文件或数字。**任何数字请先回到 §5 表格里那份 JSONL/JSON 源文件。**



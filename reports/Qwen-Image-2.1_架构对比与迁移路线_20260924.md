# Qwen-Image-2.1 架构对比与迁移路线评估

**日期**：2026-09-24
**目的**：评估把现有 SMPL-X Unified Adapter（V6.x）从 DeepGen-1.0（SD3 MMDiT）迁移到 Qwen-Image-2.1 的可行性
**数据来源**：全部来自两侧模型的 `config.json` 实测值，非文档转述

---

## 1. 两代骨干的硬指标对比

| 维度 | DeepGen-1.0-diffusers（现架构） | Qwen-Image-2.1 | 差异性质 |
|---|---|---|---|
| Transformer 类 | `SD3Transformer2DModel` | `QwenImage21Transformer2DModel` | 实现完全不同 |
| 注意力结构 | **双流 MMDiT**（`dual_attention_layers=[0..12]`，13 层双流） | **单流 single-stream** | ⚠️ 本质差异 |
| 层数 | **24** | **32** | 注入层需重排 |
| `attention_head_dim` × `num_attention_heads` | 64 × 24 → **hidden 1536** | 128 × 32 → **hidden 4096** | ⚠️ 适配器宽度需 ×2.67 |
| 文本条件维度 | `joint_attention_dim = 4096` | `context_in_dim = 4096` | ✅ 恰好一致 |
| pooled 条件 | `pooled_projection_dim = 2048`（CLIP） | 无 pooled 投影 | 需去除 |
| caption 投影 | `caption_projection_dim = 1536` | 融入 context 4096 | 需重做 |
| `patch_size` | 2 | **1** | — |
| transformer `in_channels` | 16 | **64** | — |
| VAE 下采样 | **8×** | **16×**（`scale_factor_spatial=16`） | latent 网格粗 2 倍 |
| VAE latent 通道 | 16 | **64**（`z_dim=64`） | 每通道信息量加倍补偿 |
| VAE 类 | `AutoencoderKL`（SD3） | `AutoencoderKLQwenImage21` | 不同实现 |
| VAE scaling | `1.5305` / shift `0.0609` | `latents_mean` / `latents_std`（向量） | 归一化方式不同 |
| 位置编码 | SD3 2D | RoPE 三轴 `[16,56,56]` | 不同 |
| 参数量 | ≈2.47 B | ≈7 B | 容量 ×2.8 |
| 文本编码器 | T5 + CLIP 拼接 | **Qwen3-VL**（36 层，hidden 4096，GQA 8 KV heads，bf16） | 语义空间不同 |
| 主干显存（bf16 权重） | ~5 GB | **~14 GB** | — |

---

## 2. 关于「潜空间分辨率瓶颈」的关键结论

### 2.1 计算（1024×1024 输入）

| | DeepGen-1.0（SD3） | Qwen-Image-2.1 |
|---|---|---|
| VAE 输出 latent | 128 × 128 × 16 | **64 × 64 × 64** |
| transformer patchify | 2 × 2 | 1（无压缩） |
| transformer 看到的 token 网格 | **64 × 64 = 4096** | **64 × 64 = 4096** |
| 每个 token 的数值维度 | 16 ch × 2 × 2 = **64** | **64** |

### 2.2 结论

**Qwen-Image-2.1 在「transformer 可寻址的空间分辨率」上与现有骨干完全相同，不会缓解手/脸的潜空间瓶颈。**

以一只手占约 60×60 像素为例：

- DeepGen：60 / 8 = 7.5 latent px → patchify 后约 **4×4 token**
- Qwen-2.1：60 / 16 = 3.75 latent px → 约 **4×4 token**

两者一致。Qwen 用「16× 下采样 + 64 通道」换了「8× 下采样 + 16 通道」，**空间粒度让给了通道容量**，token 层面的空间分辨率没有提升。

> **对 V6.4 / V6.5 / V6.6（detail / face / hand 三支）的直接含义**：这三支都是针对「VAE 8× 下采样导致手脸亚像素」设计的。换到 Qwen-Image-2.1 **不会自动解决该问题**，三支的设计前提依然成立。
>
> **但有一个值得实测的变量**：Qwen 的 VAE 用 64 个通道去编码每个 16×16 px 块，而 SD3 只用 16 通道编码每个 8×8 px 块。**VAE 本身的重建上限可能更高**——这是纯经验问题，必须实测，不能从架构推断。

---

## 3. Qwen-Image-2.1 的真正价值点

迁移的收益不在 latent 分辨率，而在这四处：

1. **文本条件质量跃升**：Qwen3-VL（36 层 / 4096 dim）替代 T5+CLIP。对指令跟随型编辑任务是量级改善。
2. **原生多参考图编辑**：支持最多 10 张参考图，直接在注意力内做 reference conditioning——**与你的身份保持（V6.5 ArcFace + DINOv2）目标高度重合**，可能替代一部分自研身份分支。
3. **原生 RGBA 透明通道**：生成/编辑透明图层。
4. **参数量 2.47B → 7B**：容量 ×2.8，且是更高效的单流架构（同 token 数下 FLOPs 更低）。

同时提供 `QwenImageEditPlusPipeline` 系列（已在 `model-qwen-edit/`，20B 版）作为编辑专用路径。

---

## 4. 环境问题（当前阻塞点）

| 组件 | deepgen 现有 | Qwen-Image-2.1 要求 | 结论 |
|---|---|---|---|
| diffusers | **0.35.2** | **0.37.0.dev0**（`QwenImage21Pipeline` 等类在 0.35.2 中不存在） | ❌ 差 2 个小版本 |
| transformers | **4.56.1** | **≥ 4.57.1**（`Qwen3VLForConditionalGeneration`；README 写 ≥5.17） | ❌ 不足 |
| torch | 2.8.0+cu128 | ≥ 2.4.0 | ✅ |

**症状**：`QwenImage21Pipeline.from_pretrained(...)` 会直接抛 `ImportError` / `ValueError: unknown class` —— 权重下好了也加载不了。

**风险**：`deepgen` 是现有 V6.x 全链路（SD3 + `src/pose_control/v6/`）的运行环境。就地升级 diffusers 0.35.2 → 0.37.dev 会改变 `SD3Transformer2DModel` / `AutoencoderKL` / scheduler 的行为，**存在破坏现有管线与已复现实验的风险**。

---

## 5. 推荐路线

### 阶段 0 — 权重落地（已完成）

- 实体权重：`/data/shangguanrz/pic-edit-models/Qwen-Image-2.1`（33.1 GB）
- 软链：`models/Qwen-Image-2.1 -> /data/...`（对代码路径完全等价）
- 理由：`/data` 与 `/` 同为机械 RAID 卷、吞吐一致，**权重加载无性能差异**（且 128 GiB 内存可缓存），但可保住这两天从 98% 降到 77% 的成果。

### 阶段 1 — 独立环境 + 冒烟（推荐立刻做）

```
conda create -n qwen21 python=3.12
# diffusers 从 git main 安装，transformers 升到 ≥4.57.1，torch 保持 2.8.0+cu128
# 不动 deepgen
```
跑 40 步 base 推理（README 推荐：`torch_dtype=bfloat16`、`num_inference_steps=40`、2048×2048），确认链路通。

### 阶段 1.5 — 决定性小实验（成本最低、信息量最高）

**用同一张手部占比大的图，分别过 DeepGen VAE 与 Qwen VAE，算手部 ROI 的重建 LPIPS。** 这一步直接回答「Qwen 的 16×/64ch VAE 是否真的对手部更友好」，也决定后面三支适配器要不要保留。**强烈建议在动骨干之前先做这个。**

### 阶段 2 — 适配器移植（真正的工程）

| 改动点 | 工作量 | 风险 |
|---|---|---|
| 宽度 1536 → 4096（所有 adapter 线性层） | 机械改动 | 低 |
| 层数 24 → 32，注入层重排（原 0-22，23 不注入） | 需重新实验扫描 | 中 |
| **双流 → 单流**：attention 内 text 与 image token 同序列，残差注入需按 token 位置 mask | **重设计** | **高** |
| 去除 pooled/CLIP 分支，改用 Qwen3-VL 条件 | 重设计 | 中 |
| RoPE 三轴 `[16,56,56]` 下的位置对齐 | 需验证 | 中 |
| VAE 归一化（scaling_factor → latents_mean/std 向量） | 小 | 低 |
| 保持「零初始化 ⇒ OFF≡ON（≤1e-6）」不变量 | 必须守住 | — |

**最本质的风险是「双流 → 单流」**：V6.3 的 geometry / interaction 分支建立在 MMDiT 双流结构上（image stream 与 text stream 各有独立权重）。单流架构下 attention 对拼接序列做全注意力，注入方式必须重新设计，**不是改宽度就能搞定**。

### 阶段 3 — 评测口径

沿用既定铁律：**只用 LPIPS(ROI)、FaceSim vs 源帧、真实多步生成指标**；禁用单步 x0 估计指标（输入泄漏，结论方向可能相反）。

---

## 6. 一句话建议

**值得走这条路，但主要理由不是「解决手脸瓶颈」，而是「Qwen3-VL 条件 + 原生多参考图编辑 + 7B 容量」。** 建议顺序：先做阶段 1（独立环境冒烟）+ 阶段 1.5（VAE ROI 对比实验）这两个低成本高信息量的动作，拿到 VAE 实测结论后再决定是否投入阶段 2 的重设计。

---

## 7. 附：本次落地的权重清单

| 文件 | 体积 |
|---|---|
| `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` | 9.968 GB |
| `transformer/diffusion_pytorch_model-00002-of-00002.safetensors` | 4.262 GB |
| `text_encoder/model-00001..00004-of-00004.safetensors` | 17.534 GB |
| `vae/diffusion_pytorch_model.safetensors` | 1.351 GB |
| 配置 / processor / scheduler / LICENSE / README 等 | ≈ 0.02 GB |
| **合计** | **33.13 GB** |

- 仓库 `sha = 790c92633540aa0cb11d9abf19eb46d861714758`，`lastModified = 2026-09-21`
- 非 gated，直连 `huggingface.co`（无需镜像）

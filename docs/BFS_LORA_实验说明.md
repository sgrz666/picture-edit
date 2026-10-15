# BFS Qwen-Image-2.1 LoRA 实验说明

更新：2026-10-08
来源：https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap （MIT）

## 文件位置

```
/home/shangguanrz/models/bfs/loras/bfs_body_swap_v1.0_qwen_2.1.safetensors   -> ../bfs_body_swap_v1.0_qwen_2.1.safetensors
/home/shangguanrz/models/bfs/loras/bfs_head_v1.1_qwen_2.1.safetensors       -> ../bfs_head_v1.1_qwen_2.1.safetensors
```

均已通过完整性校验（sha256 与 HF 权威值逐字节一致）：

| 文件 | 字节 | sha256 |
|---|---|---|
| bfs_body_swap_v1.0_qwen_2.1 | 209,753,576 | 7dc0a53aba4dbc70c204f498936a619d226559da11011f748c389189af46b664 |
| bfs_head_v1.1_qwen_2.1 | 260,096,144 | d1d748d5601077f3b6d05766f6823510e901970d916afa404a97e85dc92fa88e |

其余家族权重（krea2 / ltx2 / flux-klein / 2509 / 2511）也在上级 `~/models/bfs/`，本项目基座不匹配，不要用。

## 基座兼容性（已核对）

基座 `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1/transformer/config.json`：
32 层 / hidden 4096 / 单流 DiT / causal_condition=true / in-out channels 64。

**与 BFS 的 Qwen-2.1 权重完全对齐，rank-64 LoRA 无需重投影即可加载。**
环境必须用 `envs/qwen21`（diffusers 0.41.0.dev0）。**deepgen 环境加载不了**，别用。

## 必读：两个会导致「静默无效果」的坑

### 坑 1：键名必须转换，否则 img_mlp 部分被静默丢弃

ComfyUI 把 SwiGLU 融合成一个 Linear：`img_mlp.gate_up`（输出 24576）
diffusers 拆成两个：`img_mlp.gate_layer`（12288）+ `img_mlp.proj`（12288）

直接 `pipe.load_lora_weights(path)` 会报
`unexpected keys found ... img_mlp.gate_up.lora_A/lora_B.<adapter>.weight`
然后**把这些键全部丢掉**，attention 部分生效、MLP 部分静默失效 —— 结果看起来「跑了但没效果」。

**用这个函数转换**（已在 `bfs_explore.py:101-140` 验证，attached_targets=304/384）：

```python
from bfs_explore import convert_bfs_state_dict, load_lora
notes = load_lora(pipe, Path('/home/shangguanrz/models/bfs/loras/bfs_body_swap_v1.0_qwen_2.1.safetensors'), 'bfs_body')
# converted (gate_first=True, attached_targets=304)
```

`gate_first` 决定 24576 如何切分给 gate/proj，两种都试过。`load_lora` 内含断言：若 `img_mlp.proj` 或 `attn.to_q` 没挂上会直接抛错，可当加载自检。

### 坑 2：输入顺序不能反

- **body swap**：image1 = 场景（保它的姿态/光照/背景），image2 = 参考人（贡献脸/衣服/体型）
- **head swap（多个版本故意）**：image1 = 身体，image2 = 参考脸

## 官方 prompt（照抄，不要自己改写）

body_swap:
```
body_swap: start with <image1> as the base image, keeping its lighting, environment, and background. replace the body from <image1> with the body from <image2>, strictly preserving the clothing, body shape and proportions from <image2>. copy the pose, direction of the eye, head rotation, micro expressions from <image1>
```

推荐 LoRA strength = **1.0** 起步。

## 官方关于参考图的实测结论（可直接复用于我们的多参考图编辑）

1. **训练参考图格式是固定的**：站立全身、正面、纯浅背景、方形、占满画幅高度。
   非该格式的参考图 → 参考图自身背景会渗进输出、身份变弱。
2. **给参考图 token 预算，不是给分辨率**：
   把参考图缩到 **≈0.59 MP**（约 768×768 面积）并保持宽高比，背景存活明显更好。
   **把参考图放大到与场景等大是所有设置里最差的**，比什么都不做还糟。
   原理：参考图占 token 多就会压过场景。
3. **参考图不是全身时，先用 Qwen-2.1 无 LoRA 归一化**到上述格式再送入（1024×1024）。
   代价：会「补出」原图没有的腿和鞋，参考图本身的人脸相似度会掉。

## 与我们项目的关系（务必先读）

**任务方向相反，不能当解决方案替换。**
BFS = 保姿态/背景、换身份；我们 = 保身份、换姿态。目标函数相反。

**训练分布错位。** iPER 是运动中裁剪、多姿态、侧身逆光，落在 BFS 训练分布外。
作者自述的主要缺陷「人越小脸越差」与我们的核心痛点（脸小→身份掉）是同一个问题。

**注入机制不同。**
- BFS：改基座 attn+MLP 投影，body 版注入 24 层（**0-15 + 24-31，跳过 16-23**），104.86 M / rank 64
- 我们：旁路加残差，基座全程冻结，注入 6 层 (3,8,13,18,23,28)，92.19 M

叠加技术上无冲突，但 BFS 会改写基座权重，可能干扰已训好的姿态控制。
**建议实验时先单独跑 BFS，再跑 adapter+BFS 组合，并保留纯 adapter 对照臂。**

## 10-04 实测现状（证据不足，勿直接引用）

- `bfs_explore_run.log`（@512, 6 对）：body LoRA 加载时 gate_up 键被静默丢弃 → **该轮 body swap 结果无效**。
- `bfs_compare_30_20261004/results.json`（30 对 / 24 严格可见脸）：

| 臂 | 身份余弦 | PCK | 头外 MAE |
|---|---|---|---|
| body（我们的 adapter，单图） | 0.3591 | 0.9900 | — |
| **nolora（无 LoRA 修脸）** | **0.5210** | 0.9498 | **3.13** |
| bfs | 0.4422 | 0.9629 | 4.35 |
| alternative | 0.4527 | 0.9678 | 4.27 |

→ BFS head LoRA 在 768 上为**负增益 −0.079**，且头外 MAE 从 3.13 恶化到 4.35（背景/服装被污染）。
→ 注意 `bfs_explore.py` 的 `variants`（:285）不含 body_swap，body_swap 走独立分支（:303-310）
   只逐条落盘、**不进 summary**，且只跑了 2 条、无对照基线。

**建议**：本次正式实验用 iPER test 50 对、seed 42、30 步、cfg 1.0，
四臂 = {纯基座, 纯adapter, 纯BFS, adapter+BFS}，并把 body_swap 纳入统一统计。

## 相关文件

- 实测脚本：`/home/shangguanrz/project/pic-edit/bfs_explore.py`（含 `convert_bfs_state_dict` / `load_lora`）
- 结果：`/home/shangguanrz/nihaochen/qwen21_full/bfs_explore/`、`bfs_compare_30_20261004/`
- 官方文档：`/home/shangguanrz/models/bfs/docs/qwen-image-2.1-body.md`（body）、`qwen-image-2.1.md`（head）

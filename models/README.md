# models/ —— 模型权重目录（不纳入 Git）

本目录存放生成主干 **DeepGen-1.0-diffusers** 的权重文件，完整约 **14 GB**，体积过大不纳入仓库，需要自行下载。

## 模型来源

- Hugging Face 官方仓库：[`deepgenteam/DeepGen-1.0-diffusers`](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers)
- 国内镜像（推荐）：https://hf-mirror.com/deepgenteam/DeepGen-1.0-diffusers
- 模型主页（非 diffusers 版）：https://huggingface.co/deepgenteam/DeepGen-1.0
- License：Apache-2.0；基座：Qwen/Qwen2.5-VL-3B-Instruct

## 下载后的目录结构

```
models/DeepGen-1.0-diffusers/
├── model_index.json
├── prompt_template.json
├── deepgen_pipeline.py          # 自包含 pipeline（trust_remote_code=True 加载）
├── scheduler/scheduler_config.json
├── tokenizer/                   # tokenizer.json / vocab / merges / chat_template 等
├── connector/model.safetensors  # 约 1.7 GB
├── vlm/                         # Qwen2.5-VL 文本/视觉编码器，共约 7.5 GB
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00002.safetensors   # 约 4.9 GB
│   └── model-00002-of-00002.safetensors   # 约 2.6 GB
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  # DiT 主干，约 4.9 GB（存储张量元素约 2.47B）
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## 下载方式

### 方式一：huggingface-cli（推荐）

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com   # 国内加速，海外可省略
huggingface-cli download deepgenteam/DeepGen-1.0-diffusers \
  --local-dir models/DeepGen-1.0-diffusers --resume-download
```

### 方式二：Python API

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepgenteam/DeepGen-1.0-diffusers",
    local_dir="models/DeepGen-1.0-diffusers",
    resume_download=True,
)
```

## 加载验证

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "models/DeepGen-1.0-diffusers",   # 或直接写 repo_id 在线加载
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.to("cuda")
```

下载完成后可检查各分片大小是否与上文目录结构一致，并尝试用上面的加载代码实例化 pipeline，以确认权重完整、无损坏。

## 服务器上的完整权重清单

权威副本在 `sg`（node01）`/home/shangguanrz/project/pic-edit/models/`，共 **19 GB**
（2026-10-15 `du` 实测）。本地 Windows 这份仓库**没有 `models/`，也没有 `datasets/`**——
两者只存在于服务器，本地需要跑推理时按本节自行下载或建软链。

| 子目录 | 大小 | 是什么 |
|---|---|---|
| `DeepGen-1.0-diffusers/` | 14G | 生成主干（即上文的下载目标） |
| `antelopev2/` | 4.7G | InsightFace antelopev2，面部相似度评测用 |
| `face_eval/` | 340M | 面部评测其余模型 |
| `DWPose/` | 335M | 姿态估计 onnx，骨架条件图与 PCK 评测都依赖它 |
| `smpl/` | 75M | SMPL 体型模型，workflow 几何阶段用 |

> 引用这些路径前先 `ls models/` 核对目录名，别按记忆写。

## 训练产出的 checkpoint 不在本目录

训练自己写出的权重按实验目录约定留在各实验下，例如
`experiments/tiktok_overfit_sanity_full/checkpoints/` 有 4 个 1.87 GB 的 `.pt`（合计 7.5 GB）。
它们同样被 `.gitignore` 的 `*.pt` 排除，不进 Git，也不属于 `models/` 的职责——
`models/` 只放**外部下载的基座与工具模型**。


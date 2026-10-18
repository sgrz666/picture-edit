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

---

## 其他模型子目录

除 DeepGen-1.0-diffusers 外，`models/` 下还有六项辅助权重，各有独立说明（两个 Qwen 权重目录保留上游官方模型卡为 `README.md`，项目侧说明以 `README.pic-edit.md` 承载）：

| 子目录 | 内容 | 在项目中的角色 |
|---|---|---|
| [`antelopev2/`](antelopev2/README.md) | InsightFace 检测/关键点/ArcFace onnx + DINOv2-giant | 人脸检测与身份特征提取（Face ID、身份评测） |
| [`DWPose/`](DWPose/README.md) | DWPose ONNX（dw-ll_ucoco_384 + yolox_l） | 全身 134 关键点估计，骨架条件的来源 |
| [`Qwen-Image-2.1/`](Qwen-Image-2.1/README.pic-edit.md) | Qwen 2.1 7B 单流 DiT 基座，软链 → `/data/shangguanrz/pic-edit-models/Qwen-Image-2.1` | Qwen 迁移线的生成基座（body/face/dual/full 适配器均挂在它上面） |
| [`Qwen-Image-Edit-20B/`](Qwen-Image-Edit-20B/README.pic-edit.md) | Qwen-Image-Edit-20B，软链经 `model-qwen-edit` → `pic-edit-archive/cleanup_20261003/models/` | 历史/备用权重：`src/` 与 `workflow/` 均无代码引用，作为基线对照留存，注意其位于归档盘、存在被清理的风险 |
| [`smpl/`](smpl/README.md) | SMPL / SMPL-X 官方拓扑模型文件 | 三维人体参数模型（适配器条件编码的人体先验） |
| [`face_eval/`](face_eval/README.md) | 面部评测专用权重（身份/关键点/LPIPS/感知编码器） | 面部保持与画质量化评测 |

两项 Qwen 权重都不在 Git 内，实体位于数据盘；换机器时只需重建软链，不要移动上游模型卡。

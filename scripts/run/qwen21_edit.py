import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

MODEL = os.environ.get(
    "QWEN21_MODEL",
    "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"
)

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("当前环境无法使用 CUDA，停止推理。")

if not Path(MODEL, "model_index.json").is_file():
    raise FileNotFoundError(f"模型配置不存在：{MODEL}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

with Image.open(args.image) as im:
    source = im.convert("RGB")

print(f"[编辑] 输入：{args.image}", flush=True)
print(f"[编辑] 模型：{MODEL}", flush=True)
print(f"[编辑] 原图尺寸：{source.size}", flush=True)

pipe = QwenImage21Pipeline.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_model_cpu_offload()

generator = torch.Generator("cuda").manual_seed(args.seed)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()

result = pipe(
    prompt=args.prompt,
    image=source,
    num_inference_steps=args.steps,
    generator=generator,
).images[0]

result.save(args.out)

print(f"[完成] {args.out}", flush=True)
print(f"[耗时] {time.time() - t0:.1f} 秒", flush=True)
print(
    f"[显存峰值] {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
    flush=True,
)

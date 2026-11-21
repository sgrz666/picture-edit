import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

models_dir = Path("/home/shangguanrz/project/pic-edit/models")
models_dir.mkdir(parents=True, exist_ok=True)

# 1. Download antelopev2
antelope_dir = models_dir / "antelopev2"
print(f"Downloading antelopev2 to {antelope_dir}...")
snapshot_download(
    repo_id="DIAMONIK7777/antelopev2",
    local_dir=str(antelope_dir),
    resume_download=True,
)
print("antelopev2 downloaded successfully!")
print("Files in antelopev2:", [p.name for p in antelope_dir.glob("*.onnx")])

# 2. Download dinov2-giant
dinov2_dir = models_dir / "dinov2-giant"
print(f"Downloading facebook/dinov2-giant to {dinov2_dir}...")
snapshot_download(
    repo_id="facebook/dinov2-giant",
    local_dir=str(dinov2_dir),
    allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    resume_download=True,
)
print("dinov2-giant downloaded successfully!")
print("Files in dinov2-giant:", [p.name for p in dinov2_dir.iterdir() if not p.name.startswith(".")])

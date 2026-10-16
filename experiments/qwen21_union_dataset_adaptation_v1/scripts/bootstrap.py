"""建立方案 B 独立 venv，只复用方案 A 的只读源码和权重。"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-python", default="/home/shangguanrz/nihaochen/envs/qwen21/bin/python")
    args = parser.parse_args()
    work = args.workspace.resolve()
    if not work.is_relative_to(Path("/data/shangguanrz")):
        raise ValueError("experiment workspace must be below /data/shangguanrz")
    work.mkdir(parents=True, exist_ok=True)
    env = work / "env"
    python = env / "bin/python"
    if not python.exists():
        subprocess.run([args.base_python, "-m", "venv", "--system-site-packages", str(env)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "omegaconf==2.4.0", "ftfy==6.3.1"], check=True)
    probe = "import torch,diffusers,transformers,json;print(json.dumps(dict(torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__)))"
    versions = json.loads(subprocess.check_output([str(python), "-c", probe], text=True).strip())
    expected = dict(torch="2.8.0+cu128", diffusers="0.41.0.dev0", transformers="5.17.0")
    if versions != expected:
        raise RuntimeError(f"core versions changed: {versions}")
    versions.update(python=str(python), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source="/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/source/VideoX-Fun")
    output = work / f"environment_{time.strftime('%Y%m%dT%H%M%S')}.json"
    output.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()

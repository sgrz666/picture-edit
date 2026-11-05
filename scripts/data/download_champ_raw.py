#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHAMP (fudan-generative-ai/champ_trainning_sample) 原始训练视频全量下载脚本。

目标目录: /data/shangguanrz/data(to-train)/champ/raw
数据集结构: 832 个单人运动 MP4 视频，总计约 5.83 GB。
缓存落盘: /data/shangguanrz/.cache/huggingface (避免挤占 / 根分区)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 确保在导入 huggingface_hub 前重定向缓存目录到 /data
DEFAULT_CACHE_DIR = "/data/shangguanrz/.cache/huggingface"
os.environ.setdefault("HF_HOME", DEFAULT_CACHE_DIR)


def setup_rawdata_symlink(target_dir: Path):
    """
    HuggingFace 仓库中的视频存放在 rawdata/ 子目录。
    通过在 target_dir 中建立 rawdata -> . 的软链接，
    使得 snapshot_download 直接将 832 个视频平铺下载到 target_dir 根目录下，
    同时保持对 target_dir/rawdata/xxx.mp4 路径访问的兼容。
    """
    rawdata_link = target_dir / "rawdata"
    if not rawdata_link.exists():
        try:
            rawdata_link.symlink_to(".")
            print(f"[INFO] 已创建软链接: {rawdata_link} -> . (文件将直接平铺在 {target_dir})")
        except Exception as e:
            print(f"[WARN] 创建软链接失败，将以普通目录下载: {e}")


def main():
    parser = argparse.ArgumentParser(description="下载 CHAMP 官方全量原始视频数据集")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/shangguanrz/data(to-train)/champ/raw",
        help="视频保存目录 (默认: /data/shangguanrz/data(to-train)/champ/raw)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="fudan-generative-ai/champ_trainning_sample",
        help="HuggingFace 数据集仓库 ID",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="并发下载线程数 (默认: 12)",
    )
    parser.add_argument(
        "--use_mirror",
        action="store_true",
        default=False,
        help="是否使用 hf-mirror.com 镜像源 (国内直连加速)",
    )
    parser.add_argument(
        "--custom_endpoint",
        type=str,
        default="",
        help="自定义 HuggingFace Endpoint",
    )
    args = parser.parse_args()

    target_dir = Path(args.output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    setup_rawdata_symlink(target_dir)

    # 缓存目录重定向
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    print(f"[INFO] 目标下载目录: {target_dir}")
    print(f"[INFO] HF 缓存目录:  {os.environ.get('HF_HOME')}")
    print(f"[INFO] 并发线程数:   {args.max_workers}")

    endpoint = None
    if args.custom_endpoint:
        endpoint = args.custom_endpoint
    elif args.use_mirror:
        endpoint = "https://hf-mirror.com"

    if endpoint:
        print(f"[INFO] 使用 Endpoint: {endpoint}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 未找到 huggingface_hub，请使用包含 huggingface_hub 的 Python 环境，例如:")
        print("        /home/shangguanrz/nihaochen/envs/qwen21/bin/python download_champ_raw.py")
        sys.exit(1)

    t0 = time.time()
    print("[INFO] 开始解析并下载 CHAMP 原始视频数据集 (832 个 MP4 文件, ~5.83GB)...")

    download_kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": str(target_dir),
        "allow_patterns": ["rawdata/*.mp4"],
        "max_workers": args.max_workers,
    }
    if endpoint:
        download_kwargs["endpoint"] = endpoint

    snapshot_download(**download_kwargs)

    elapsed = time.time() - t0
    # 统计下载结果
    mp4_files = list(target_dir.glob("*.mp4"))
    total_size_mb = sum(f.stat().st_size for f in mp4_files) / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] CHAMP 全量原始数据下载完成!")
    print(f"耗时:       {elapsed:.1f} 秒 ({elapsed/60:.2f} 分钟)")
    print(f"有效 MP4 数量: {len(mp4_files)} / 832")
    print(f"总计数据大小: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"存放目录:   {target_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

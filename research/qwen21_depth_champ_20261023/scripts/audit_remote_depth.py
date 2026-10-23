"""Read-only depth asset/source audit; writes a new JSON and optional public weight cache only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_stats(path, mask_path=None):
    a = np.array(Image.open(path))
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = np.array(Image.open(path).convert("L"))
    result = {"path": str(path), "sha256": sha(path), "shape": list(a.shape),
              "dtype": str(a.dtype), "range": [int(a.min()), int(a.max())],
              "unique_gray": int(np.unique(gray).size),
              "rgb_channels_equal": bool(np.array_equal(rgb[..., 0], rgb[..., 1])
                                         and np.array_equal(rgb[..., 1], rgb[..., 2]))}
    if mask_path is not None and mask_path.is_file():
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        result["mask_path"] = str(mask_path)
        if gray.shape == mask.shape:
            result["mask_coverage"] = float(mask.mean())
            result["foreground_quantiles"] = np.quantile(gray[mask], [0, .5, 1]).tolist() if mask.any() else []
            result["background_quantiles"] = np.quantile(gray[~mask], [0, .5, 1]).tolist() if (~mask).any() else []
    return result


def iper_samples(root):
    result = []
    for app in sorted(root.iterdir()):
        paths = sorted((app / "depth_npy_512" / "target").glob("*.npy"))
        if not paths:
            continue
        for path in dict.fromkeys((paths[0], paths[-1])):
            raw = np.load(path, allow_pickle=False).squeeze()
            valid = np.isfinite(raw) & (raw > 0)
            entry = {"path": str(path), "sha256": sha(path), "shape": list(raw.shape),
                     "dtype": str(raw.dtype), "finite": bool(np.isfinite(raw).all()),
                     "positive_coverage": float(valid.mean()), "units": "not inferred from magnitude",
                     "positive_quantiles": np.quantile(raw[valid], [0, .25, .5, .75, 1]).tolist()}
            vis = app / "depth_vis_512" / "target" / (path.stem + ".png")
            mask_path = app / "mask_512" / "target" / (path.stem + ".png")
            if vis.is_file():
                entry["visualization"] = image_stats(vis, mask_path)
                gray = np.array(Image.open(vis).convert("L"), dtype=np.float64) / 255
                if raw.shape == gray.shape and np.std(raw[valid]) > 1e-8:
                    x, y = raw[valid].astype(np.float64), gray[valid]
                    slope, intercept = np.polyfit(x, y, 1)
                    entry["vis_linear_fit"] = {"slope": float(slope), "intercept": float(intercept),
                                               "correlation": float(np.corrcoef(x, y)[0, 1]),
                                               "rmse": float(np.sqrt(np.mean((y - slope*x - intercept)**2)))}
                    lo, hi = x.min(), x.max()
                    linear = (x-lo) / (hi-lo)
                    entry["vis_rmse_near0_far1"] = float(np.sqrt(np.mean((y-linear)**2)))
                    entry["vis_rmse_near1_far0"] = float(np.sqrt(np.mean((y-(1-linear))**2)))
            result.append(entry)
        if len(result) >= 6:
            break
    return result


def champ_samples(root):
    result = []
    for motion in sorted((root / "motions").iterdir()):
        paths = sorted((motion / "depth").glob("*.png"))
        if not paths:
            continue
        for path in dict.fromkeys((paths[0], paths[-1])):
            result.append(image_stats(path, motion / "mask" / path.name))
        if len(result) >= 6:
            break
    return result


def tiktok_samples(root):
    result = []
    for seq in sorted(root.iterdir()):
        paths = sorted((seq / "depth_vis").glob("*.png"))
        if paths:
            result.append(image_stats(paths[0], seq / "mask" / paths[0].name))
        if len(result) >= 3:
            break
    return result


def official_weight(cache):
    import torch
    repo = json.load(urlopen("https://huggingface.co/api/models/fudan-generative-ai/champ", timeout=30))
    commit = repo["sha"]
    url = f"https://huggingface.co/fudan-generative-ai/champ/resolve/{commit}/champ/guidance_encoder_depth.pth"
    expected = "4da437759e68cfe1c6361432305cc1ce02e1d04651db0d813b01eaaabd4665a1"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (expected + ".pth")
    if not path.exists():
        data = urlopen(url, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Official depth checkpoint hash changed")
        with path.open("xb") as f:
            f.write(data)
    if sha(path) != expected:
        raise ValueError("Cached official checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    keys = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "numel": v.numel()}
            for k, v in tensors.items()}
    def count(prefix):
        return sum(v.numel() for k, v in tensors.items() if k.startswith(prefix))
    active = sum(count(p) for p in ("conv_in.", "blocks.", "attentions.5.", "conv_out."))
    return {"url": url, "hf_commit": commit, "path": str(path), "sha256": sha(path),
            "bytes": path.stat().st_size, "tensor_count": len(tensors),
            "total_elements": sum(v.numel() for v in tensors.values()), "active_forward_elements": active,
            "inactive_elements": sum(v.numel() for v in tensors.values()) - active,
            "conv_out_weight_abs_max": float(tensors["conv_out.weight"].abs().max()),
            "conv_out_weight_std": float(tensors["conv_out.weight"].float().std()),
            "keys": keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weight-cache", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.project
    names = ("third_party/champ_guidance/guidance_encoder.py", "src/pose_control/qwen21_full/geometry_encoder.py",
             "src/pose_control/qwen21_full/native_geometry.py", "src/pose_control/qwen21_full/native_body.py",
             "src/data/iper_dataset.py", "src/data/tiktok_dataset.py", "src/data/canvas_contract.py",
             "src/data/geometry_pair_dataset_v2.py", "src/controls/generator.py")
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "project": str(root),
              "source_hashes": {n: sha(root/n) for n in names if (root/n).is_file()},
              "iper": iper_samples(root/"datasets/iPER/iper_assets_512_nvdiffrast"),
              "champ_examples": champ_samples(root/"datasets/champ_motions_example"),
              "tiktok": tiktok_samples(root/"datasets/TikTokDataset/TikTok_3d_assets_native")}
    if args.weight_cache:
        result["official_depth_weight"] = official_weight(args.weight_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "iper_samples": len(result["iper"]),
                      "champ_samples": len(result["champ_examples"]), "tiktok_samples": len(result["tiktok"]),
                      "official_weight_elements": result.get("official_depth_weight", {}).get("total_elements")}))


if __name__ == "__main__":
    main()

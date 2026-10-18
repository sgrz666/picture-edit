"""在已有 SMPL 环境 CPU 导出真实 LBS 顶点部位，供 Hi4D 接触端点绑定。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle


SMPL_TO_V6 = (1, 8, 11, 1, 9, 12, 1, 10, 13, 1, 10, 13,
              0, 1, 1, 0, 2, 5, 3, 6, 4, 7, 4, 7)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smpl-root", required=True)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--output", required=True, help="新建 .npz 绝对路径，不覆盖")
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if not destination.is_absolute() or destination.suffix != ".npz":
        parser.error("output 必须是绝对 .npz 路径")
    destination = destination.resolve()
    if any(part.lower() in {"models", "datasets"} for part in destination.parts):
        parser.error("产物不能写入 models 或 datasets")
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"拒绝覆盖现有导出: {destination}")
    root = Path(args.smpl_root)
    weights = root / "smpl" / f"SMPL_{args.gender.upper()}.pkl"
    if not weights.is_file():
        raise FileNotFoundError(weights)

    import numpy as np

    # Only the literal LBS matrix is used. The legacy male pickle needs an
    # environment with its existing chumpy dependency, but never needs torch.
    with weights.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")
    lbs = np.asarray(payload["weights"], dtype=np.float32)
    if lbs.shape != (6890, 24) or not np.isfinite(lbs).all():
        raise ValueError(f"SMPL LBS 拓扑或数值异常: {lbs.shape}")
    parts = np.asarray(SMPL_TO_V6, dtype=np.int64)[lbs.argmax(axis=1)]
    if parts.shape != (6890,) or parts.min() < 0 or parts.max() >= 14:
        raise ValueError("导出的部位标签无效")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, parts=parts)
    metadata = {
        "schema": "qwen21_pro_smpl_vertex_parts_v1",
        "codec": "smpl24_lbs_argmax_to_v6_14_zero_based",
        "model_type": "smpl", "gender": args.gender, "vertices": 6890,
        "parts": 14, "background_label": -1, "smpl_to_v6": SMPL_TO_V6,
        "lbs_dtype": "float32_match_smplx_default",
        "model_path": str(weights.resolve()), "model_sha256": sha256(weights),
        "output_sha256": sha256(destination),
        "counts": np.bincount(parts, minlength=14).tolist(),
        "scope": "LBS topology labels only; body reconstruction requires a separate params/vertices audit",
    }
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

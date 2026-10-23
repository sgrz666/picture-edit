"""Fetch small, pinned upstream text files for the CHAMP depth review; never execute them."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


COMMIT = "4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140"
FILES = (
    "LICENSE", "README.md", "models/guidance_encoder.py", "models/transformer_3d.py",
    "models/attention.py", "models/resnet.py", "models/champ_model.py",
    "models/unet_3d.py", "configs/train/stage1.yaml", "configs/inference/inference.yaml",
    "datasets/image_dataset.py", "datasets/data_utils.py", "train_s1.py",
    "pipelines/pipeline_guidance2image.py", "pipelines/pipeline_guidance2video.py",
    "scripts/data_processors/smpl/render_condition_maps.py",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = []
    for name in FILES:
        url = f"https://raw.githubusercontent.com/fudan-generative-vision/champ/{COMMIT}/{name}"
        data = urlopen(url, timeout=30).read()
        if len(data) > 500_000:
            raise ValueError(f"Unexpectedly large source: {name}")
        target = root / "sources" / "champ" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite changed source: {target}")
        if not target.exists():
            target.write_bytes(data)
        records.append({"path": target.relative_to(root).as_posix(), "upstream_path": name,
                        "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = root / "upstream_sources.json"
    if manifest.exists():
        print(f"Source files verified; existing manifest retained: {manifest}")
        return
    manifest.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
                                   "commit": COMMIT, "files": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": COMMIT, "files": len(records),
                      "total_bytes": sum(x["bytes"] for x in records)}))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "qwen21_pei2i_body_face_study_v1"
PROTOCOLS = ("p0", "pg", "pb", "pbg")
P0_PROMPT = "Change the person pose to match the target pose. Preserve identity, clothing, and background."
ARMS = tuple(f"{base}_{p}" for base in ("qwen", "head") for p in PROTOCOLS)
PROMPT_ROLES = {
    "p0": (),
    "pg": ("source_A", "target_depth", "target_normal", "target_dwpose"),
    "pb": ("source_A", "target_B"),
    "pbg": ("source_A", "target_B", "target_depth", "target_normal", "target_dwpose"),
}
EVAL_STEPS = tuple(range(0, 1025, 128))
FULL_VAL_STEPS = (0, 256, 512, 768, 1024)
TRAIN_MONITOR = (0, 64, 128, 192, 256, 320, 384, 448)
VAL_MONITOR = (0, 4, 7, 8, 13, 14, 16, 19, 22)
INIT_SHA256 = "9e0f048c585937ab74be792eb6d9730b0621eccda16e1744a6837b0eb9d24651"
TRAIN_SHA256 = "854133776d2067aa392442553f3003ef9e6d45107ccc9cffe5a4af8600fb929f"
VAL_SHA256 = "cece883b96dbbc9270fd77d4bb1e973470f52f941af85c2a0746947f5d91e10a"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(path)


def validate_arm(arm: str) -> tuple[str, str]:
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")
    base, protocol = arm.split("_", 1)
    return base, protocol


def validate_study_config(config: dict[str, Any]) -> None:
    """Validate study-wide fixed configuration according to the experiment protocol."""
    if config.get("schema") != SCHEMA:
        raise ValueError(f"Invalid schema: {config.get('schema')}")
    if config.get("max_updates") != 1024:
        raise ValueError(f"max_updates must be 1024, got {config.get('max_updates')}")
    if config.get("eval_every") != 128:
        raise ValueError(f"eval_every must be 128, got {config.get('eval_every')}")
    if config.get("resolution") != "native" or config.get("align_mode") != "native":
        raise ValueError("Resolution and align_mode must be 'native'")
    if config.get("with_hand") is True or config.get("hand_active") is True:
        raise ValueError("Hand adapter must remain disabled in this study")
    if config.get("use_depth") is True:
        raise ValueError("Adapter Depth channel must remain disabled in this study")
    arms = config.get("arms")
    if arms is not None and tuple(arms) != ARMS:
        raise ValueError(f"Configured arms must match exact 8 ARMS, got {arms}")

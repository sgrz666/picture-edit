"""Source-only identity and target-only geometry for the complete Qwen adapter.

The feature cache follows the existing V6.5 ArcFace/DINO format. Target frames
may contain geometry-only records; their appearance vectors are never read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate

from src.data.iper_dataset import IPERPoseDataset
from src.data.tiktok_dataset import TikTokV3PoseDataset, is_tiktok_v3_manifest
from .conditions import AdapterIdentityCondition, ConditionBundle
from .face.conditions import FaceFineCondition, FaceReferenceFeatures

ROOT = Path(__file__).resolve().parents[3]
SAMPLED = ROOT / "datasets/iPER/iper_sampled_6src64tgt"
ASSETS = ROOT / "datasets/iPER/iper_assets_512_nvdiffrast"
TIKTOK_RAW = ROOT / "datasets/TikTokDataset/TikTok_dataset/TikTok_dataset"
TIKTOK_ASSETS = ROOT / "datasets/TikTokDataset/TikTok_3d_assets_native"


def read_pairs(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pair_metadata(row: Mapping[str, Any], sampled_root: str | Path = SAMPLED) -> dict[str, Any]:
    if "sequence_id" in row:
        appearance = str(row["sequence_id"])
        source_stem = str(row["source_stem"] if "source_stem" in row else row["source"]["stem"])
        target_stem = str(row["target_stem"] if "target_stem" in row else row["target"]["stem"])
        return {
            "pair_id": str(row.get("pair_id", f"{appearance}_{source_stem}__{target_stem}")),
            "source_appearance": appearance, "target_appearance": appearance,
            "source_stem": source_stem, "target_stem": target_stem,
            "source_rgb": str(row["source_rgb"]), "target_rgb": str(row["target_rgb"]),
            "cross_person": False, "dataset_type": "tiktok_v3",
        }
    source_app = str(row.get("source_appearance", row.get("appearance")))
    target_app = str(row.get("target_appearance", row.get("appearance")))
    root = Path(sampled_root)
    return {
        "pair_id": str(row.get("pair_id", f"{source_app}_{row['source']['stem']}__{target_app}_{row['target']['stem']}")),
        "source_appearance": source_app,
        "target_appearance": target_app,
        "source_stem": row["source"]["stem"],
        "target_stem": row["target"]["stem"],
        "source_rgb": str(root / source_app / row["source"]["rgb_1024"]),
        "target_rgb": str(root / target_app / row["target"]["rgb_1024"]),
        "cross_person": source_app.split("_")[0] != target_app.split("_")[0],
    }


class QwenFullDataset(torch.utils.data.Dataset):
    """Existing iPER loading with explicit A/B provenance, including cross-person.

    Cross-person B RGB is available only as offline inspection data and is not
    a paired reconstruction target. Training callers must reject such rows.
    """
    def __init__(self, pairs_jsonl: str | Path, resolution: int = 512,
                 sampled_root: str | Path = SAMPLED, assets_root: str | Path = ASSETS,
                 dataset_type: str = "auto", tiktok_raw_root: str | Path = TIKTOK_RAW,
                 tiktok_assets_root: str | Path = TIKTOK_ASSETS, align_mode: str = "resize"):
        self.rows = read_pairs(pairs_jsonl)
        if not self.rows:
            raise ValueError(f"Empty pair manifest: {pairs_jsonl}")
        self.dataset_type = ("tiktok_v3" if is_tiktok_v3_manifest(pairs_jsonl) else "iper") if dataset_type == "auto" else dataset_type
        self.sampled_root = Path(sampled_root)
        if self.dataset_type == "tiktok_v3":
            if align_mode not in ("resize", "letterbox"):
                raise ValueError("Full TikTok conditioning requires resize or letterbox with shared RGB/geometry coordinates")
            self.base = TikTokV3PoseDataset(pairs_jsonl, raw_root=tiktok_raw_root,
                                           assets_root=tiktok_assets_root, resolution=resolution,
                                           align_mode=align_mode, strict_conditions=True)
            self.rows = self.base.samples
            return
        if self.dataset_type != "iper":
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        self.base = IPERPoseDataset(str(pairs_jsonl), str(sampled_root), str(assets_root), resolution, augment=False)
        # The existing reader binds both RGB files to one appearance. Load B's
        # geometry through a B/B proxy, then replace the source with the actual A.
        self.base.samples = [
            {**row, "appearance": pair_metadata(row, sampled_root)["target_appearance"],
             "source": row["target"]}
            for row in self.rows
        ]

    def __getattr__(self, name: str):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def samples(self):
        return self.rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self.base[index]
        meta = pair_metadata(self.rows[index], self.sampled_root)
        if self.dataset_type == "iper":
            batch["src_image"] = self.base.rgb_transform(self.base._load_image(meta["source_rgb"]))
        batch["source_stem"] = meta["source_stem"]
        batch["appearance"] = meta["source_appearance"]
        batch["pair_metadata"] = meta
        return batch


def qwen_full_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = default_collate([{k: v for k, v in row.items() if k != "pair_metadata"} for row in rows])
    result["pair_metadata"] = [dict(row["pair_metadata"]) for row in rows]
    return result


def build_condition_bundle(adapter, batch: Mapping[str, Any], *, device, dtype) -> ConditionBundle:
    """Build the complete geometry bundle exclusively from B's conditions."""
    def floating(name):
        return batch[name].to(device=device, dtype=dtype)
    kwargs = dict(normal_a=floating("normal"), pose_heatmap_a=floating("pose_heatmap"),
                  part_onehot_a=floating("part_onehot"), smplx_global_a=floating("smplx_global"),
                  human_mask_a=floating("human_mask"), task_id=batch["task_id"].to(device=device, dtype=torch.long))
    if bool(getattr(adapter.condition_injector, "use_depth", False)):
        kwargs["depth_a"] = floating("depth")
    return adapter.condition_injector(**kwargs)


def build_identity_condition(source_person_latents: torch.Tensor,
                             source_indices: torch.Tensor | None = None) -> AdapterIdentityCondition:
    """Accept old-core latent format after the Qwen interface's input mapping."""
    if source_indices is None:
        source_indices = torch.arange(2, device=source_person_latents.device).expand(source_person_latents.shape[0], -1)
    return AdapterIdentityCondition(source_person_latents=source_person_latents, source_indices=source_indices)


class FaceConditionStore:
    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self._caches: dict[str, dict[str, Any]] = {}

    def _load(self, appearance: str):
        if appearance not in self._caches:
            path = self.cache_root / f"{appearance}.pt"
            from scripts.precompute_iper_face_features_v65 import load_v65_face_cache
            cache = load_v65_face_cache(path)
            if 'geometry_valid' not in cache or cache['geometry_valid'].shape != cache['valid'].shape:
                raise ValueError('Full Face cache requires per-frame geometry_valid')
            if cache['geometry_valid'].tolist() != cache['metadata'].get('geometry_valid'):
                raise ValueError('Geometry validity differs from the fingerprinted cache metadata')
            self._caches[appearance] = cache
        return self._caches[appearance]

    def for_pairs(self, pairs: Sequence[Mapping[str, Any]], *, device, dtype, alignments=None):
        count = len(pairs)
        condition = FaceFineCondition(
            landmarks=torch.zeros(count, 2, 72, 3), jaw_pose=torch.zeros(count, 2, 3),
            expression=torch.zeros(count, 2, 10), source_boxes=torch.zeros(count, 2, 4),
            target_boxes=torch.zeros(count, 2, 4), face_valid=torch.zeros(count, 2, dtype=torch.bool),
            source_indices=torch.arange(2).expand(count, -1).clone())
        references = FaceReferenceFeatures(
            arcface=torch.zeros(count, 2, 1, 512), dino_patches=torch.zeros(count, 2, 1, 256, 1536),
            reference_valid=torch.zeros(count, 2, 1, dtype=torch.bool))
        for i, row in enumerate(pairs):
            if "source_appearance" not in row or "target_appearance" not in row:
                row = pair_metadata(row)
            source = self._load(str(row["source_appearance"]))
            target = self._load(str(row["target_appearance"]))
            si = source["stem_to_idx"][row["source_stem"]]
            ti = target["stem_to_idx"][row["target_stem"]]
            valid = bool(source["valid"][si]) and bool(target["geometry_valid"][ti])
            if not valid:
                continue
            # Appearance is always from A, never from target cache vectors.
            references.arcface[i, 0, 0] = source["arcface"][si].float()
            references.dino_patches[i, 0, 0] = source["dino_patches"][si].float()
            references.reference_valid[i, 0, 0] = True
            condition.landmarks[i, 0] = target["face_landmarks"][ti]
            condition.jaw_pose[i, 0] = target["jaw_pose"][ti]
            condition.expression[i, 0] = target["expression"][ti]
            condition.source_boxes[i, 0] = source["face_boxes"][si]
            condition.target_boxes[i, 0] = target["face_boxes"][ti]
            condition.face_valid[i, 0] = True
        from .face_coordinates import transform_face_condition
        condition = transform_face_condition(condition, alignments)
        return (condition.to(device=device, dtype=dtype).validate(),
                references.to(device=device, dtype=dtype).validate())

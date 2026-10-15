"""Source-only identity and target-only geometry for the complete Qwen adapter.

The feature cache follows the existing V6.5 ArcFace/DINO format. Target frames
may contain geometry-only records; their appearance vectors are never read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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


def study_prompt_signature(bank) -> dict[str, Any]:
    from .prompt_study.protocol import P0_PROMPT, text_sha256
    identities = {(row['instruction_version'], row['instruction_sha256']) for row in bank.records.values()}
    if len(identities) != 1:
        raise ValueError('Prompt manifest mixes instruction templates')
    version, sha = next(iter(identities))
    return {'instruction_version': version,
            'prompt_instruction_sha256': text_sha256(P0_PROMPT) if bank.protocol == 'p0' else sha}


def validate_study_supervision(store, rows) -> None:
    if store.data.get('align_mode') != 'native':
        raise ValueError('Study supervision requires native coordinates')
    for row in rows:
        record = store.record(row)
        native = record.get('by_resolution', {}).get('native')
        if not isinstance(native, dict) or not isinstance(native.get('valid'), bool):
            raise ValueError('Study supervision missing explicit native gate')


def study_asset_signature(features: str | Path, data_dir: str | Path) -> dict[str, Any]:
    from .prompt_study.protocol import sha256_file
    features, data_dir = Path(features), Path(data_dir)
    if features.resolve() != (data_dir / 'face_cache').resolve():
        raise ValueError('Study features must use the run data/face_cache')
    path = data_dir / 'face_assets.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'qwen21_study_face_cache_v2' or manifest.get('source_identity_only') is not True:
        raise ValueError('Study Face asset manifest must contain verified source-only v2 caches')
    records = manifest.get('sequences', [])
    if not records or len({record['sequence_id'] for record in records}) != len(records):
        raise ValueError('Study Face assets missing or duplicated')
    for record in records:
        cache = features / f"{record['sequence_id']}.pt"
        if Path(record['path']).resolve() != cache.resolve() or sha256_file(cache) != record['sha256']:
            raise ValueError(f"Study Face cache changed: {record['sequence_id']}")
    from .prompt_study.protocol import load_study_lock
    from .prompt_study.measurement import loss_signature
    lock = load_study_lock(data_dir / 'lock.json', verify_assets=False)
    return {'face_assets_sha256': sha256_file(path),
            'loss_models_sha256': loss_signature(lock, ROOT / 'models')}


def study_code_signature() -> dict[str, Any]:
    """Bind the real controller, native transforms and installed Qwen runtime."""
    import importlib.metadata
    import importlib.util
    from .prompt_study.protocol import sha256_file
    paths = list((ROOT / 'src/pose_control/qwen21_full').rglob('*.py'))
    for relative in ('third_party/v65_face', 'third_party/champ_guidance'):
        paths.extend(path for path in (ROOT / relative).rglob('*.py') if '__pycache__' not in path.parts)
    paths.extend(ROOT / relative for relative in (
        'src/adapter/pose_adapter.py', 'src/data/letterbox.py', 'src/data/tiktok_dataset.py',
        'scripts/precompute_iper_face_features_v65.py'))
    hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha256_file(path) for path in sorted(set(paths))}
    spec = importlib.util.find_spec('diffusers')
    if spec is None:
        raise ValueError('Study requires the installed Qwen Diffusers runtime')
    package = Path(spec.origin).parent
    runtime_files = list((package / 'pipelines/qwenimage').glob('*.py'))
    runtime_files.extend((package / 'models/transformers').glob('*qwenimage*.py'))
    if not runtime_files:
        raise ValueError('Installed Diffusers lacks Qwen runtime files')
    return {'runtime_code_sha256': hashes, 'native_preprocessing': 'native_v1',
            'diffusers_version': importlib.metadata.version('diffusers'),
            'diffusers_qwen_code_sha256': {str(path.relative_to(package)).replace('\\', '/'): sha256_file(path)
                                        for path in sorted(runtime_files)}}


def validate_study_cache(cache, dataset, bank, *, indices, require_target: bool) -> None:
    """Require prepared CPU cache entries before allocating a generation model."""
    from .qwen_runtime import dataset_image_to_pil
    for index in indices:
        sample = dataset[index]
        pair_id = sample['pair_metadata']['pair_id']
        source = dataset_image_to_pil(sample['src_image'])
        source_path = cache._path(source, 'source', prompt=bank.prompt_for(pair_id))
        if not source_path.is_file():
            raise FileNotFoundError(f'Missing source cache for {pair_id}: {source_path}')
        entry = torch.load(source_path, map_location='cpu', weights_only=True)
        hw = (source.height // 16, source.width // 16)
        keys = {'reference_latent', 'prompt_embeds', 'prompt_mask', 'image_mask', 'token_hw'}
        if (not isinstance(entry, dict) or not keys <= entry.keys() or tuple(entry['token_hw']) != hw
                or entry['reference_latent'].shape != (1, hw[0] * hw[1], 64)
                or entry['prompt_embeds'].shape[:2] != entry['prompt_mask'].shape
                or any(not torch.isfinite(entry[key]).all() for key in keys - {'token_hw'})):
            raise ValueError(f'Invalid source cache for {pair_id}')
        if require_target:
            target = dataset_image_to_pil(sample['tgt_image'])
            target_path = cache._path(target, 'target')
            if not target_path.is_file():
                raise FileNotFoundError(f'Missing target cache for {pair_id}: {target_path}')
            latent = torch.load(target_path, map_location='cpu', weights_only=True)
            expected = (1, (target.height // 16) * (target.width // 16), 64)
            if not torch.is_tensor(latent) or latent.shape != expected or not torch.isfinite(latent).all():
                raise ValueError(f'Invalid target cache for {pair_id}')


def pair_metadata(row: Mapping[str, Any], sampled_root: str | Path = SAMPLED) -> dict[str, Any]:
    if "sequence_id" in row:
        appearance = str(row["sequence_id"])
        source_app = str(row.get("source_appearance", appearance))
        target_app = str(row.get("target_appearance", appearance))
        if target_app != appearance:
            raise ValueError("TikTok sequence_id must select target geometry")
        source_stem = str(row["source_stem"] if "source_stem" in row else row["source"]["stem"])
        target_stem = str(row["target_stem"] if "target_stem" in row else row["target"]["stem"])
        meta = {
            "pair_id": str(row.get("pair_id", f"{source_app}_{source_stem}__{target_app}_{target_stem}")),
            "source_appearance": source_app, "target_appearance": target_app,
            "source_stem": source_stem, "target_stem": target_stem,
            "source_rgb": str(row["source_rgb"]), "target_rgb": str(row["target_rgb"]),
            "cross_person": source_app != target_app, "dataset_type": "tiktok_v3",
        }
        for key in ("sequence_id", "source_video_group", "target_video_group", "video_group",
                    "source_dwpose_json", "target_dwpose_json", "target_dwpose_png",
                    "target_depth_vis", "target_normal", "filtered_dwpose", "split"):
            if key in row:
                meta[key] = row[key]
        return meta
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
    def __init__(self, pairs_jsonl: str | Path, resolution: int | tuple[int, int] | str = 512,
                 sampled_root: str | Path = SAMPLED, assets_root: str | Path = ASSETS,
                 dataset_type: str = "auto", tiktok_raw_root: str | Path = TIKTOK_RAW,
                 tiktok_assets_root: str | Path = TIKTOK_ASSETS, align_mode: str = "resize"):
        self.rows = read_pairs(pairs_jsonl)
        if not self.rows:
            raise ValueError(f"Empty pair manifest: {pairs_jsonl}")
        self.dataset_type = ("tiktok_v3" if is_tiktok_v3_manifest(pairs_jsonl) else "iper") if dataset_type == "auto" else dataset_type
        self.sampled_root = Path(sampled_root)
        if self.dataset_type == "tiktok_v3":
            if align_mode not in ("resize", "letterbox", "native"):
                raise ValueError("Full TikTok conditioning requires resize, letterbox, or native with shared RGB/geometry coordinates")
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


class HandConditionStore:
    def __init__(self, raw_root=None, assets_root=None, align_mode="native"):
        self.raw_root = Path(raw_root or "/home/shangguanrz/project/pic-edit")
        self.assets_root = Path(assets_root or "/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_3d_assets_native")
        self.align_mode = align_mode
        self._smpl_cache: dict[str, tuple[Any, dict[str, int]]] = {}
        self._dw_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _get_dwpose(self, seq_id: str, stem: str, default_path: str | None = None) -> dict[str, Any] | None:
        key = (seq_id, stem)
        if key not in self._dw_cache:
            p = None
            if default_path and Path(default_path).exists():
                p = Path(default_path)
            else:
                for candidate in (
                    self.assets_root.parent / f"TikTok_v3/dwpose_v1/{seq_id}/dwpose_json/{stem}.json",
                    self.raw_root.parents[2] / f"TikTok_v3/dwpose_v1/{seq_id}/dwpose_json/{stem}.json",
                    Path(f"/home/shangguanrz/project/pic-edit/datasets/TikTokDataset/TikTok_v3/dwpose_v1/{seq_id}/dwpose_json/{stem}.json"),
                    Path(f"/data/shangguanrz/pic-edit-datasets/TikTokDataset/TikTok_v3/dwpose_v1/{seq_id}/dwpose_json/{stem}.json"),
                ):
                    if candidate.exists():
                        p = candidate
                        break
            if p is None or not p.exists():
                return None
            self._dw_cache[key] = json.loads(p.read_text(encoding="utf-8"))
        return self._dw_cache[key]

    def _get_smpl(self, seq_id: str):
        if seq_id not in self._smpl_cache:
            smpl_dir = self.assets_root / seq_id / "smplestx"
            params_p = smpl_dir / "params.pt"
            map_p = smpl_dir / "frame_map.json"
            if not params_p.exists() or not map_p.exists():
                return None, None
            params = torch.load(params_p, map_location="cpu", weights_only=False)
            frame_map = json.loads(map_p.read_text(encoding="utf-8"))
            stem_to_idx = {item['stem']: item['index'] for item in frame_map.get('frames', [])}
            self._smpl_cache[seq_id] = (params, stem_to_idx)
        return self._smpl_cache[seq_id]

    def for_pairs(self, pairs: Sequence[Mapping[str, Any]], *, device="cpu", dtype=torch.float32):
        count = len(pairs)
        hand_keypoints = torch.zeros(count, 2, 2, 21, 3, dtype=torch.float32)
        hand_pose = torch.zeros(count, 2, 2, 45, dtype=torch.float32)
        source_boxes = torch.zeros(count, 2, 2, 4, dtype=torch.float32)
        target_boxes = torch.zeros(count, 2, 2, 4, dtype=torch.float32)
        region_valid = torch.zeros(count, 2, 2, dtype=torch.bool)
        source_indices = torch.arange(2, dtype=torch.long)[None].expand(count, -1).clone()

        dummy_box = torch.tensor([0.0, 0.0, 0.1, 0.1], dtype=torch.float32)
        source_boxes[...] = dummy_box
        target_boxes[...] = dummy_box

        from src.data.letterbox import NativePad
        from .hand.geometry import split_iper_134_hand_points
        pad = NativePad.from_size((604, 1080))

        for i, row in enumerate(pairs):
            seq_id = str(row.get("sequence_id") or row.get("source_appearance") or row.get("appearance") or "")
            src_stem = str(row.get("source_stem", ""))
            tgt_stem = str(row.get("target_stem", ""))
            dw_src = self._get_dwpose(seq_id, src_stem, row.get("source_dwpose_json"))
            dw_tgt = self._get_dwpose(seq_id, tgt_stem, row.get("target_dwpose_json"))
            if dw_src is None or dw_tgt is None:
                continue

            src_kps = np.array(dw_src.get('keypoints_norm', []), dtype=np.float32)
            src_sc = np.array(dw_src.get('scores', []), dtype=np.float32)
            tgt_kps = np.array(dw_tgt.get('keypoints_norm', []), dtype=np.float32)
            tgt_sc = np.array(dw_tgt.get('scores', []), dtype=np.float32)

            if len(src_kps) != 134 or len(tgt_kps) != 134:
                continue

            src_pts134 = torch.from_numpy(np.concatenate([src_kps, src_sc[:, None]], axis=-1))
            tgt_pts134 = torch.from_numpy(np.concatenate([tgt_kps, tgt_sc[:, None]], axis=-1))

            if self.align_mode == "native":
                src_pts134 = pad.points(src_pts134)
                tgt_pts134 = pad.points(tgt_pts134)

            src_hands = split_iper_134_hand_points(src_pts134).clamp(0.0, 1.0)
            tgt_hands = split_iper_134_hand_points(tgt_pts134).clamp(0.0, 1.0)

            hand_keypoints[i, 0] = tgt_hands

            def get_boxes(hands: torch.Tensor):
                boxes = torch.zeros(2, 4, dtype=torch.float32)
                valid = torch.zeros(2, dtype=torch.bool)
                for side in range(2):
                    pts = hands[side]
                    vis = pts[:, 2] > 0.15
                    if int(vis.sum()) >= 3:
                        xy = pts[vis, :2]
                        lo = xy.amin(dim=0)
                        hi = xy.amax(dim=0)
                        span = (hi - lo).clamp_min(1e-3)
                        margin = span * 0.2 + 0.01
                        b = torch.cat([(lo - margin).clamp(0, 1), (hi + margin).clamp(0, 1)])
                        min_dim = 0.06
                        w = b[2] - b[0]
                        h = b[3] - b[1]
                        if w < min_dim:
                            cx = (b[0] + b[2]) / 2
                            b[0] = max(0.0, float(cx - min_dim / 2))
                            b[2] = min(1.0, float(cx + min_dim / 2))
                            if b[0] == 0.0:
                                b[2] = min(1.0, min_dim)
                            if b[2] == 1.0:
                                b[0] = max(0.0, 1.0 - min_dim)
                        if h < min_dim:
                            cy = (b[1] + b[3]) / 2
                            b[1] = max(0.0, float(cy - min_dim / 2))
                            b[3] = min(1.0, float(cy + min_dim / 2))
                            if b[1] == 0.0:
                                b[3] = min(1.0, min_dim)
                            if b[3] == 1.0:
                                b[1] = max(0.0, 1.0 - min_dim)
                        if bool((b[2:] > b[:2]).all()):
                            boxes[side] = b
                            valid[side] = True
                    if not valid[side]:
                        boxes[side] = dummy_box
                return boxes, valid

            s_boxes, s_valid = get_boxes(src_hands)
            t_boxes, t_valid = get_boxes(tgt_hands)
            s_boxes = torch.where(s_valid[:, None], s_boxes, t_boxes)

            source_boxes[i, 0] = s_boxes
            target_boxes[i, 0] = t_boxes
            region_valid[i, 0] = t_valid

            smpl_params, stem_to_idx = self._get_smpl(seq_id)
            if smpl_params is not None and tgt_stem in stem_to_idx:
                ti = stem_to_idx[tgt_stem]
                hand_pose[i, 0, 0] = smpl_params['left_hand_pose'][ti].float().reshape(-1)
                hand_pose[i, 0, 1] = smpl_params['right_hand_pose'][ti].float().reshape(-1)

        from .hand.conditions import HandFineCondition, HandReferenceFeatures
        condition = HandFineCondition(
            hand_keypoints=hand_keypoints.to(device=device, dtype=dtype),
            hand_pose=hand_pose.to(device=device, dtype=dtype),
            source_boxes=source_boxes.to(device=device, dtype=dtype),
            target_boxes=target_boxes.to(device=device, dtype=dtype),
            region_valid=region_valid.to(device=device),
            source_indices=source_indices.to(device=device),
            mesh_bps=torch.zeros(count, 2, 2, 1024, device=device, dtype=dtype),
            mesh_valid=torch.zeros(count, 2, 2, device=device, dtype=torch.bool),
            depth=torch.zeros(count, 2, 2, 1, 16, 16, device=device, dtype=dtype),
            visibility=region_valid.to(device=device, dtype=dtype),
        ).validate()

        references = HandReferenceFeatures(
            dino_patches=torch.zeros(count, 2, 2, 1, 16, 1536, device=device, dtype=dtype),
            reference_valid=torch.zeros(count, 2, 2, 1, device=device, dtype=torch.bool),
            source_indices=source_indices.to(device=device),
        ).validate()

        return condition, references


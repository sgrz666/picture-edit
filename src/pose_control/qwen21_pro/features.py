"""源人物专属的冻结 VAE64 + DINOv2-G/14 1536 raw patch bank。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .cache import asset_hashes, file_sha256, ordinary_cpu, source_feature_fingerprint
from .preparation import (_exists, _read_array, _record, _transform, map_points,
                          resolve_condition_assets, source_asset_roles, validate_cached_binding, validate_source_assets)

EXTRACTION_CODEC = {"profile": "vae64_dinov2_giant1536", "crop_size": 224, "patch_grid": 16,
                    "padding": "masked_person_square_white", "margin": .06,
                    "processor_resize": False, "processor_center_crop": False,
                    "sampling": "same_source_coordinates;bilinear_vae;nearest_support",
                    "roi_blocks": [[24, 92], [92, 113], [113, 134]], "min_roi_pixels": 8,
                    "face_min_pixels": 64, "face_min_score": .6, "face_min_support": .8}


@dataclass(frozen=True)
class RawSourceFeatures:
    person_ids: tuple[int, ...]
    tokens: torch.Tensor
    valid: torch.Tensor
    face_identity: torch.Tensor
    face_identity_valid: torch.Tensor
    fingerprint: str
    provenance: dict

    def validate(self):
        p = len(self.person_ids)
        if p not in (1, 2) or len(set(self.person_ids)) != p:
            raise ValueError("raw source requires unique bound people")
        if self.tokens.ndim != 4 or self.tokens.shape[:2] != (1, p) or self.tokens.shape[-1] != 1600:
            raise ValueError("raw source features must be [1,P,M,1600] (VAE64+DINO1536)")
        if self.valid.shape != self.tokens.shape[:3] or self.valid.dtype != torch.bool:
            raise ValueError("raw source valid must be bool[1,P,M]")
        if self.face_identity.shape != (1, p, 512) or self.face_identity_valid.shape != (1, p):
            raise ValueError("FaceID must be independently bound [1,P,512]")
        if self.face_identity_valid.dtype != torch.bool:
            raise ValueError("FaceID valid must be boolean")
        for value in (self.tokens, self.valid, self.face_identity, self.face_identity_valid):
            if value.device.type != "cpu" or value.requires_grad or value.is_inference():
                raise ValueError("raw cache requires ordinary frozen CPU tensors")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("raw source contains nonfinite values")
        if not self.tokens.is_floating_point() or not self.face_identity.is_floating_point():
            raise ValueError("raw feature/identity dtype must be floating")
        if not self.fingerprint:
            raise ValueError("missing source feature fingerprint")
        return self

    def validate_binding(self, record, source_latent, assets, producer_signature):
        if self.provenance["model_signature"].get("source_vae_producer") != producer_signature:
            raise ValueError("raw source feature producer binding mismatch")
        expected = source_feature_fingerprint(source_assets=source_asset_roles(assets, record),
                    model_signature=self.provenance["model_signature"], person_order=record["person_order"],
                    source_transform=record["source_transform"], source_latent=source_latent,
                    extraction=self.provenance["extraction"])
        if expected != self.fingerprint:
            raise ValueError("raw source features fingerprint is stale or belongs to another source")

    def to_dict(self):
        return {"person_ids": self.person_ids, "tokens": self.tokens, "valid": self.valid,
                "face_identity": self.face_identity, "face_identity_valid": self.face_identity_valid,
                "fingerprint": self.fingerprint, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "person_ids": tuple(value["person_ids"])}).validate()


def square_person_crop(rgb, mask, *, size=224, margin=.06, box=None):
    """返回方形源 crop、support 与 16² patch center 的源图像坐标。

    超出画布的 padding 保留在坐标里，support 为假。与 DINO 配对的 VAE
    samples 必须使用这些同源坐标，不能从整图随意截前256个 tokens。
    """
    rgb, mask = np.asarray(rgb), np.asarray(mask, dtype=bool)
    if rgb.shape[:2] != mask.shape or rgb.shape[-1] != 3:
        raise ValueError("RGB and source person mask dimensions differ")
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("source person has no visible support")
    if box is None:
        box = (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1)
    x0, y0, x1, y1 = map(float, box)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    if not np.isfinite(side) or side <= 0:
        raise ValueError("invalid source crop box")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bounds = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    masked = np.where(mask[..., None], rgb, 255).astype(np.uint8)
    affine = (side / size, 0, bounds[0], 0, side / size, bounds[1])
    image = Image.fromarray(masked).transform((size, size), Image.Transform.AFFINE, affine,
                    resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    support = np.asarray(Image.fromarray(mask.astype(np.uint8)).transform((size, size), Image.Transform.AFFINE,
                       affine, resample=Image.Resampling.NEAREST, fillcolor=0)) > 0
    centers = (np.arange(16) + .5) / 16 * side
    py, px = np.meshgrid(centers + bounds[1], centers + bounds[0], indexing="ij")
    return image, support, np.stack([px, py], -1).reshape(256, 2)


class _LocalArcFace:
    """直接加载已安装 ONNX；只检测给定人物 support，绝不整图 largest-face。"""
    def __init__(self, root, device):
        from insightface.model_zoo import get_model
        root = Path(root)
        detector, recognizer = root / "scrfd_10g_bnkps.onnx", root / "glintr100.onnx"
        if not detector.is_file() or not recognizer.is_file():
            raise FileNotFoundError("local SCRFD/ArcFace ONNX files required")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
        self.detector = get_model(str(detector), providers=providers)
        self.recognizer = get_model(str(recognizer), providers=providers)
        ctx = 0 if str(device).startswith("cuda") else -1
        self.detector.prepare(ctx_id=ctx, input_size=(640, 640), det_thresh=.6)
        self.recognizer.prepare(ctx_id=ctx)

    def __call__(self, rgb, support):
        from insightface.utils.face_align import norm_crop
        bgr = np.ascontiguousarray(np.where(support[..., None], rgb, 255)[..., ::-1])
        boxes, points = self.detector.detect(bgr, max_num=0)
        qualified = []
        for box, keypoints in zip(boxes, points if points is not None else []):
            x0, y0, x1, y1, score = box
            lo = np.floor([x0, y0]).astype(int).clip(0, [support.shape[1], support.shape[0]])
            hi = np.ceil([x1, y1]).astype(int).clip(0, [support.shape[1], support.shape[0]])
            region = support[lo[1]:hi[1], lo[0]:hi[0]]
            if score >= .6 and min(x1 - x0, y1 - y0) >= 64 and region.size and region.mean() >= .8:
                qualified.append(keypoints)
        if len(qualified) != 1:
            return torch.zeros(512), False
        aligned = norm_crop(bgr, landmark=qualified[0], image_size=112)
        value = torch.from_numpy(self.recognizer.get_feat(aligned).reshape(-1)).float()
        if value.numel() != 512 or not torch.isfinite(value).all() or value.norm() < 1e-6:
            return torch.zeros(512), False
        return F.normalize(value, dim=0), True


class FrozenSourceFeatureExtractor:
    def __init__(self, dino, processor, *, model_signature, device="cpu", face_extractor=None,
                 include_roi=True):
        self.dino = dino.requires_grad_(False).eval().to(device)
        self.processor, self.device = processor, torch.device(device)
        self.model_signature = model_signature
        self.face_extractor = face_extractor
        self.extraction = {**EXTRACTION_CODEC, "include_roi": bool(include_roi)}
        self.include_roi = bool(include_roi)

    @classmethod
    def from_local(cls, dino_root, *, vae_root, device="cpu", face_root=None, include_roi=True):
        """仅本地 pretrained；不下载、不装包、不改 qwen21 环境。"""
        from transformers import AutoImageProcessor, AutoModel
        dino_root = Path(dino_root)
        required = [dino_root / "config.json", dino_root / "preprocessor_config.json", dino_root / "model.safetensors"]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("complete local DINO giant model/config/processor weights required")
        import json
        configuration = json.loads(required[0].read_text(encoding="utf-8"))
        if configuration.get("hidden_size") != 1536 or configuration.get("patch_size") != 14:
            raise ValueError("effect profile requires frozen DINOv2-G/14 width1536")
        signature = {"dino": {path.name: file_sha256(path) for path in required}}
        vae_root = Path(vae_root)
        vae_weights = sorted(vae_root.glob("*.safetensors"))
        vae_files = [vae_root / "config.json", *vae_weights]
        if not vae_weights or not all(path.is_file() for path in vae_files):
            raise FileNotFoundError("locked source VAE config and weight shards required for provenance")
        vae_index = vae_root / "diffusion_pytorch_model.safetensors.index.json"
        if vae_index.is_file():
            index = json.loads(vae_index.read_text(encoding="utf-8"))
            names = set(index["weight_map"].values())
            if any(not (vae_root / name).is_file() for name in names):
                raise FileNotFoundError("incomplete source VAE weight shards")
            vae_files.append(vae_index)
        signature["source_vae"] = {path.name: file_sha256(path) for path in vae_files}
        face = None
        if face_root is not None:
            signature["face"] = {name: file_sha256(Path(face_root) / name)
                                 for name in ("scrfd_10g_bnkps.onnx", "glintr100.onnx")}
            face = _LocalArcFace(face_root, device)
        return cls(AutoModel.from_pretrained(str(dino_root), local_files_only=True),
                   AutoImageProcessor.from_pretrained(str(dino_root), local_files_only=True),
                   model_signature=signature, device=device, face_extractor=face, include_roi=include_roi)

    @torch.no_grad()
    def extract(self, condition, cached_condition, *, project_root, assets=None):
        from .encoded import EncodedCondition
        if isinstance(cached_condition,EncodedCondition):
            record,order,hw,source_latent=cached_condition.validate_source_binding(_record(condition))
        else:
            record, order, hw, source_latent = validate_cached_binding(condition, cached_condition)
        assets = resolve_condition_assets(record, project_root) if assets is None else assets
        validate_source_assets(record, assets, project_root)
        source_geometry = assets.get("source_geometry")
        if source_geometry is not None:
            from .geometry_preparation import validate_source_geometry
            validate_source_geometry(source_geometry, record, assets)
        elif (record["dataset"] == "hi4d" and self.include_roi and _exists(assets.get("source_raw_parameters"))
              and not all(_exists(p) for p in assets.get("source_keypoints", []))):
            raise ValueError("Hi4D source geometry ROIs required: run geometry_preparation source_geometry_only in existing SMPL environment")
        with Image.open(assets["source_rgb"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        transform = _transform(record["source_transform"])
        if image.size != transform.original_wh:
            raise ValueError("raw source RGB differs from locked source_transform original_wh")
        all_tokens, all_valid, identities, identity_valid, coverage = [], [], [], [], []
        source_map = source_latent.float().transpose(1, 2).reshape(1, 64, *hw)
        paths = assets.get("source_masks", [])
        if len(paths) != len(order):
            raise ValueError("source masks must already follow stable slot order")
        for slot, path in enumerate(paths):
            raw_mask = _read_array(path)
            if raw_mask is None:
                raw_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.max(-1)
            mask = np.asarray(Image.fromarray((raw_mask > 0).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)) > 0
            crops = [("body", None)] if mask.any() else []
            if mask.any() and self.include_roi:
                kp_paths = assets.get("source_keypoints", [None] * len(order))
                kp_path = kp_paths[slot]
                if _exists(kp_path):
                    from src.data.qwen21_union_dataset import _keypoints
                    payload = _keypoints(kp_path, record["dataset"], transform)
                    xy = np.asarray(payload["xy_native"], dtype=float)
                    enabled = np.asarray(payload["valid"], dtype=bool)
                    for name, (lo, hi) in zip(("face", "left_hand", "right_hand"), EXTRACTION_CODEC["roi_blocks"]):
                        selected = xy[lo:hi][enabled[lo:hi]]
                        if len(selected) >= 3:
                            minimum, maximum = selected.min(0), selected.max(0)
                            if min(maximum - minimum) >= self.extraction["min_roi_pixels"]:
                                crops.append((name, (*minimum, *maximum)))
                elif source_geometry is not None:
                    for name in ("face", "left_hand", "right_hand"):
                        box = source_geometry["rois"][slot].get(name)
                        if box is not None:
                            x0, y0 = np.floor(box[:2]).astype(int).clip(0, image.size)
                            x1, y1 = np.ceil(box[2:]).astype(int).clip(0, image.size)
                            if x1 > x0 and y1 > y0 and mask[y0:y1, x0:x1].sum() >= 8:
                                crops.append((name, (x0, y0, x1, y1)))
            patches, valids, crop_coverage = [], [], []
            for name, box in crops:
                crop, crop_support, native_xy = square_person_crop(rgb, mask, margin=self.extraction["margin"], box=box)
                inputs = self.processor(images=crop, return_tensors="pt", do_resize=False, do_center_crop=False)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                dino = self.dino(**inputs).last_hidden_state
                if tuple(dino.shape) != (1, 257, 1536):
                    raise ValueError("DINO giant must return [1,257,1536] at224; no width fallback")
                canvas_xy = map_points(native_xy, transform)
                grid = torch.tensor(canvas_xy / np.asarray(transform.canvas_wh) * 2 - 1, dtype=torch.float32).reshape(1, 1, 256, 2)
                vae = F.grid_sample(source_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, :, 0].T
                support_grid = torch.tensor(native_xy / np.array([rgb.shape[1], rgb.shape[0]]) * 2 - 1,
                                            dtype=torch.float32).reshape(1, 1, 256, 2)
                usable = F.grid_sample(torch.from_numpy(mask.copy()).float()[None, None], support_grid,
                                      mode="nearest", padding_mode="zeros", align_corners=False)[0, 0, 0] > .5
                patches.append(torch.cat([vae, dino[0, 1:].float().cpu()], -1))
                valids.append(usable)
                crop_coverage.append({"kind": name, "valid_tokens": int(usable.sum()), "support_fraction": float(crop_support.mean()),
                                      "box": list(map(float, box)) if box is not None else None})
            all_tokens.append(torch.cat(patches) if patches else torch.zeros(256, 1600))
            all_valid.append(torch.cat(valids) if valids else torch.zeros(256, dtype=torch.bool))
            identity, available = (self.face_extractor(rgb, mask) if mask.any() and self.face_extractor is not None
                                   else (torch.zeros(512), False))
            identities.append(torch.as_tensor(identity).float())
            identity_valid.append(bool(available))
            coverage.append({"person_id": order[slot], "crops": crop_coverage,
                             "faceid_valid": bool(available), "source_support_pixels": int(mask.sum())})
        maximum = max(len(tokens) for tokens in all_tokens)
        tokens = torch.stack([F.pad(t, (0, 0, 0, maximum - len(t))) for t in all_tokens])[None]
        valid = torch.stack([F.pad(v, (0, maximum - len(v))) for v in all_valid])[None]
        roles = source_asset_roles(assets, record)
        model_signature = {"extractors": self.model_signature, "source_vae_producer": cached_condition.producer_signature}
        fingerprint = source_feature_fingerprint(source_assets=roles, model_signature=model_signature,
                       person_order=order, source_transform=record["source_transform"], source_latent=source_latent,
                       extraction=self.extraction)
        provenance = {"profile": "vae64_dinov2_giant1536", "model_signature": model_signature,
                      "extraction": self.extraction, "asset_hashes": asset_hashes(roles), "coverage": coverage}
        payload = {"person_ids": order, "tokens": tokens, "valid": valid,
                   "face_identity": torch.stack(identities)[None],
                   "face_identity_valid": torch.tensor([identity_valid]), "fingerprint": fingerprint,
                   "provenance": provenance}
        return RawSourceFeatures.from_dict(ordinary_cpu(payload))

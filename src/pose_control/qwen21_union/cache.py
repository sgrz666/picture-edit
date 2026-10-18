"""内容寻址的多参考图条件缓存与仅训练可写的监督缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from .conditions import expand_image_slots, valid_target_tokens

PREPROCESS_VERSION = "qwen21-union-prepared512-target-lanczos-white-v2"
ENCODING_POLICY = "posterior-mode;(latent-mean)/std;unpatched64;rgba;no-denoising-rng"


class _FingerprintMismatch(ValueError):
    pass


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _references(record):
    return [record["source_rgb"], *record.get("reference_pose_images", [])]


def _transforms(record):
    return record.get("transforms", {key: record[key] for key in ("source_transform", "target_transform") if key in record})


def condition_fingerprint(record, model_signature, dtype="torch.bfloat16"):
    """Hash condition bytes/config only; never inspect target RGB or masks."""
    return _hash({"model": model_signature, "preprocess": PREPROCESS_VERSION, "encoding": ENCODING_POLICY,
                  "dtype": str(dtype), "canvas": [512, 512], "transforms": _transforms(record),
                  "references": [_sha_file(p) for p in _references(record)],
                  "control": _sha_file(record["control_image"]), "prompt": record["prompt"],
                  "person_order": record.get("person_order", [])})


def _ordinary(value):
    # Cached tensors must be normal CPU tensors: inference tensors cannot be saved
    # for backward in the frozen base's computation graph.
    if torch.is_tensor(value):
        with torch.inference_mode(False):
            return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _ordinary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ordinary(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_ordinary(item) for item in value)
    return value


class _Cache:
    namespace = ""

    def __init__(self, root):
        self.root = Path(root) / self.namespace

    def _index(self, sample_id):
        return self.root / "index" / f"{_hash(str(sample_id))}.json"

    def get(self, sample_id, expected_hash=None):
        meta = json.loads(self._index(sample_id).read_text(encoding="utf-8"))
        if expected_hash is not None and meta["hash"] != expected_hash:
            raise _FingerprintMismatch(f"cache fingerprint mismatch for {sample_id}")
        path = self.root / f"{meta['hash']}.pt"
        if _sha_file(path) != meta["payload_sha256"]:
            raise ValueError(f"cache payload fingerprint mismatch for {sample_id}")
        return _ordinary(torch.load(path, map_location="cpu", weights_only=True))

    def metadata(self, sample_id):
        return json.loads(self._index(sample_id).read_text(encoding="utf-8"))

    def put(self, sample_id, payload, fingerprint):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{fingerprint}.pt"
        if not path.exists():
            with path.open("xb") as stream:
                torch.save(_ordinary(payload), stream)
        meta = {"sample_id": sample_id, "hash": fingerprint, "path": str(path),
                "payload_sha256": _sha_file(path)}
        index = self._index(sample_id)
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return meta

    def contains(self, sample_id, fingerprint):
        try:
            self.get(sample_id, fingerprint)
        except (FileNotFoundError, _FingerprintMismatch):
            return False
        return True


class UnionConditionCache(_Cache):
    namespace = "conditions"


class UnionSupervisionCache(_Cache):
    namespace = "supervision"


def load_condition(cache_root, sample_id, *, expected_hash=None):
    return UnionConditionCache(cache_root).get(sample_id, expected_hash)


def load_supervision(cache_root, sample_id, *, expected_hash=None):
    return UnionSupervisionCache(cache_root).get(sample_id, expected_hash)


def _open_prepared(path):
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError(f"expected prepared 512x512 image, got {image.size}: {path}")
        return image.convert("RGBA")


def _open_target(path, transform):
    """Apply the provider's exact affine to original supervision, in memory."""
    with Image.open(path) as image:
        if not transform:
            if image.size != (512, 512):
                raise ValueError("non-512 target requires its recorded target_transform")
            return image.convert("RGBA")
        if tuple(transform.get("canvas_wh", (512, 512))) != (512, 512):
            raise ValueError("target_transform canvas must be 512x512")
        if "original_wh" in transform and image.size != tuple(transform["original_wh"]):
            raise ValueError("target RGB dimensions differ from target_transform original_wh")
        resized_wh = tuple(transform.get("resized_wh", (512, 512)))
        offset_xy = tuple(transform.get("offset_xy", (0, 0)))
        # Validate before PIL paste can silently clip an invalid transform.
        valid_target_tokens((512, 512), resized_wh, offset_xy)
        canvas = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        canvas.paste(image.convert("RGBA").resize(resized_wh, Image.Resampling.LANCZOS), offset_xy)
        return canvas


def _vae_dtype(pipe):
    return next(pipe.vae.parameters()).dtype


def _encode(pipe, image):
    """Reference, control and supervision use the identical deterministic encoder."""
    pixels = pipe.image_processor.preprocess(image, height=512, width=512)
    if pixels.shape[1] == 3:
        pixels = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=1)
    pixels = pixels.unsqueeze(2).to(device=pipe._execution_device, dtype=_vae_dtype(pipe))
    latent = pipe.vae.encode(pixels).latent_dist.mode()
    channels = pipe.vae.config.z_dim
    mean = torch.as_tensor(pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, channels, 1, 1, 1)
    std = torch.as_tensor(pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, channels, 1, 1, 1)
    if channels != 64 or bool((std == 0).any()) or tuple(latent.shape[2:]) != (1, 32, 32):
        raise ValueError("VAE must encode prepared512 into [1,64,1,32,32] with nonzero std")
    return ((latent - mean) / std).flatten(2).transpose(1, 2)


def cache_model_signature(model_signature, vae_config):
    """Rebuild encoder normalization provenance from config without loading VAE."""
    def value(name):
        return vae_config[name] if isinstance(vae_config, dict) else getattr(vae_config, name)
    return {"model": model_signature, "vae_mean": list(value("latents_mean")),
            "vae_std": list(value("latents_std")), "vae_z_dim": value("z_dim")}


def _signature(pipe, signature):
    return cache_model_signature(signature, pipe.vae.config)


def _target_transform(record):
    transform = record.get("target_transform") or record.get("transforms", {}).get("target", {})
    return transform.to_dict() if hasattr(transform, "to_dict") else transform


def supervision_fingerprint(record, model_signature, dtype="torch.bfloat16"):
    """Analogous to condition_fingerprint; signature includes VAE config fields."""
    return _hash({"model": model_signature, "target": _sha_file(record["target_rgb"]),
                  "transform": _target_transform(record), "dtype": str(dtype),
                  "encoding": ENCODING_POLICY, "preprocess": PREPROCESS_VERSION})


def expected_condition_fingerprint(record, model_signature, vae_config, dtype="torch.bfloat16"):
    return condition_fingerprint(record, cache_model_signature(model_signature, vae_config), dtype)


def expected_supervision_fingerprint(record, model_signature, vae_config, dtype="torch.bfloat16"):
    return supervision_fingerprint(record, cache_model_signature(model_signature, vae_config), dtype)


def _freeze_encoders(pipe):
    for name in ("vae", "text_encoder"):
        model = getattr(pipe, name, None)
        if model is not None:
            model.requires_grad_(False).eval()


@torch.no_grad()
def precompute_conditions(pipe, records, cache_root, model_signature):
    """Encode ordered [source,pose...] once. Target RGB/mask have no read path."""
    cache = UnionConditionCache(cache_root)
    _freeze_encoders(pipe)
    result = {}
    signature = _signature(pipe, model_signature)
    for record in records:
        sample_id = record["sample_id"]
        fingerprint = condition_fingerprint(record, signature, _vae_dtype(pipe))
        if cache.contains(sample_id, fingerprint):
            result[sample_id] = cache.metadata(sample_id)
            continue
        images = [_open_prepared(path) for path in _references(record)]
        reference = torch.cat([_encode(pipe, image) for image in images], dim=1)
        control = _encode(pipe, _open_prepared(record["control_image"]))
        prompt, prompt_mask, slots = pipe.encode_prompt(
            image=images, prompt=record["prompt"], device=pipe._execution_device, num_images_per_prompt=1)
        if not torch.is_tensor(prompt) or prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[-1] != 4096:
            raise ValueError("Qwen3-VL embeddings must be [1,L,4096]")
        # Official single-sample encode_prompt omits the attention mask because
        # every sequence position is valid. Persist its explicit equivalent.
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt.shape[:2], dtype=torch.bool, device=prompt.device)
        if not torch.is_tensor(slots):
            raise ValueError("VLM image slot mask is required for reference-image conditions")
        if prompt_mask.shape != prompt.shape[:2] or slots.shape != prompt.shape[:2]:
            raise ValueError("VLM text/slot masks must match the multimodal sequence")
        img_mask = torch.cat([slots.bool(), slots.new_ones((1, 256), dtype=torch.bool)], dim=1)
        shapes = [[(1, 32, 32)] * (len(images) + 1)]
        expand_image_slots(img_mask, shapes[0])
        payload = {"reference_latents": reference, "control_latents": control, "prompt_embeds": prompt,
                   "prompt_mask": prompt_mask, "img_mask": img_mask, "img_shapes": shapes,
                   "target_hw": (32, 32), "canvas_hw": (512, 512), "transforms": _transforms(record),
                   "person_order": record.get("person_order", [])}
        result[sample_id] = cache.put(sample_id, payload, fingerprint)
    return result


@torch.no_grad()
def precompute_supervision(pipe, records, cache_root, model_signature=None):
    """Only train targets may be encoded; caller joins split into supervision records."""
    records = list(records)
    if any(record.get("split") != "train" for record in records):
        raise ValueError("supervision cache is train-only")
    cache = UnionSupervisionCache(cache_root)
    _freeze_encoders(pipe)
    result = {}
    for record in records:
        transform = _target_transform(record)
        fingerprint = supervision_fingerprint(record, _signature(pipe, model_signature), _vae_dtype(pipe))
        sample_id = record["sample_id"]
        if cache.contains(sample_id, fingerprint):
            result[sample_id] = cache.metadata(sample_id)
            continue
        target = _encode(pipe, _open_target(record["target_rgb"], transform))
        valid = valid_target_tokens((512, 512), transform.get("resized_wh", (512, 512)),
                                    transform.get("offset_xy", (0, 0)))
        result[sample_id] = cache.put(sample_id, {"target_latent": target, "valid_target_tokens": valid}, fingerprint)
    return result

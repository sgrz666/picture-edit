"""当前几何/源参考的变分辨率编码与独立、可复核的 Pro 缓存。"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .cache import _payload_digest, file_sha256, ordinary_cpu, stable_hash
from .contracts import TaskCondition

SCHEMA = "pro_native_encoded_v1"


def encoding_code_signature():
    root = Path(__file__).parent
    return {
        name: file_sha256(root / name)
        for name in (
            "encoded.py",
            "inference.py",
            "geometry_preparation.py",
            "refinement.py",
        )
        if (root / name).is_file()
    }


def _files(paths):
    return [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in paths]


def geometry_input_files(signature):
    """递归读取当前几何及 ROI parent 的原始参数/相机/模型文件绑定。"""
    result = []
    if isinstance(signature, Mapping):
        if "input_asset_files" in signature:
            expected = signature.get("input_asset_hashes", {})
            for role, path in signature["input_asset_files"].items():
                digest = file_sha256(path)
                if expected.get(role) != digest:
                    raise ValueError("Current geometry raw input SHA256 differs")
                result.append({"path": str(Path(path).resolve()), "sha256": digest})
        for value in signature.values():
            if isinstance(value, (Mapping, list, tuple)):
                result.extend(geometry_input_files(value))
    elif isinstance(signature, (list, tuple)):
        for value in signature:
            result.extend(geometry_input_files(value))
    return result


def _verify_files(binding):
    for role in ("source_files", "geometry_files", "inpaint_files"):
        for entry in binding.get(role, []):
            if file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Encoded {role} file SHA256 changed")


@torch.no_grad()
def encode_image(pipe, image, *, canvas_hw=None):
    """确定性 posterior mode，真实网格与 VAE stats；不消耗采样 RNG。"""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGBA")
    h, w = canvas_hw or (image.height, image.width)
    scale = getattr(pipe, "vae_scale_factor", 16)
    if h % (2 * scale) or w % (2 * scale):
        raise ValueError("Qwen21 encoded images must form four-token image slots")
    pixels = pipe.image_processor.preprocess(image, height=h, width=w)
    if pixels.shape[1] == 3:
        pixels = torch.cat([pixels, torch.ones_like(pixels[:, :1])], 1)
    dtype = next(pipe.vae.parameters()).dtype
    latent = pipe.vae.encode(
        pixels.unsqueeze(2).to(device=pipe._execution_device, dtype=dtype)
    ).latent_dist.mode()
    config = pipe.vae.config
    if config.z_dim != 64 or tuple(latent.shape) != (1, 64, 1, h // scale, w // scale):
        raise ValueError("VAE output must be actual unpatched64 image grid")
    mean = torch.as_tensor(
        config.latents_mean, device=latent.device, dtype=latent.dtype
    ).reshape(1, 64, 1, 1, 1)
    std = torch.as_tensor(
        config.latents_std, device=latent.device, dtype=latent.dtype
    ).reshape(1, 64, 1, 1, 1)
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid VAE std")
    return ordinary_cpu(((latent - mean) / std).flatten(2).transpose(1, 2))


class EncodedCondition(Mapping):
    def __init__(
        self, payload, binding, *, file_manifest=None, pack_digest=None, config=None
    ):
        self.payload, self.binding = ordinary_cpu(payload), copy.deepcopy(binding)
        self._digest = _payload_digest(self.payload)
        self._binding_digest = stable_hash(self.binding)
        self.file_manifest, self._pack_digest, self._config = (
            file_manifest,
            pack_digest,
            config,
        )

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        return iter(self.payload)

    def __len__(self):
        return len(self.payload)

    @property
    def producer_signature(self):
        return json.dumps(
            {
                "producer": self.binding["producer"],
                "vae_config": self.binding["vae_config"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def validate_source_binding(self, record):
        """给新记录 source extractor 的已验证 whole-frame VAE 入口。"""
        self.verify()
        if (
            tuple(record["person_order"]) != tuple(self["person_order"])
            or len(self.binding["source_files"]) != 1
            or Path(record["source_rgb"]).resolve()
            != Path(self.binding["source_files"][0]["path"]).resolve()
        ):
            raise ValueError(
                "New source extractor requires the same ordered people and one prepared whole-source frame"
            )
        shape = self["img_shapes"][0][0]
        source_hw = tuple(shape[-2:])
        count = source_hw[0] * source_hw[1]
        transform = record["source_transform"]
        if tuple(transform["canvas_wh"])[::-1] != tuple(v * 16 for v in source_hw):
            raise ValueError(
                "Encoded source VAE grid differs from recorded source canvas"
            )
        return (
            record,
            tuple(record["person_order"]),
            source_hw,
            ordinary_cpu(self["reference_latents"][:, :count]),
        )

    def with_source_binding(self, pack, *, config):
        """新 source1600 提取后固定最终 source fingerprint；已落盘缓存不可重绑。"""
        self.verify()
        if self.file_manifest is not None:
            raise ValueError(
                "A loaded encoded cache cannot be rebound; write a new cache"
            )
        if tuple(self["person_order"]) != pack.person_ids:
            raise ValueError("New source bank person order differs")
        binding = copy.deepcopy(self.binding)
        binding["source_fingerprint"] = pack.source_fingerprint
        return EncodedCondition(self.payload, binding).with_task_condition(
            pack, config=config
        )

    def with_task_condition(self, pack, *, config):
        """首次使用前封口：当前 native 请求必须与 typed geometry 完全一致。"""
        self.verify()
        if self.file_manifest is not None:
            raise ValueError("A loaded encoded cache cannot be rebound")
        pack.validate(config)
        if (
            not pack.provenance.get("geometry")
            or stable_hash(pack.provenance["geometry"])
            != stable_hash(self.binding["geometry"])
            or tuple(self["target_hw"]) != pack.target_hw
            or tuple(self["person_order"]) != pack.person_ids
            or self.binding["source_fingerprint"] != pack.source_fingerprint
        ):
            raise ValueError(
                "Encoded current geometry/source/grid/person binding differs from typed pack"
            )
        digest = _payload_digest(pack.to_dict())
        if (
            (self._pack_digest is not None and self._pack_digest != digest)
            or self.binding.get("task_pack_sha256", digest) != digest
            or self.binding.get("config", config.to_dict()) != config.to_dict()
        ):
            raise ValueError(
                "Encoded typed pack changed after its creation-time binding"
            )
        binding = copy.deepcopy(self.binding)
        binding.update(task_pack_sha256=digest, config=config.to_dict())
        return EncodedCondition(
            self.payload, binding, pack_digest=digest, config=config
        )

    def verify(self, pack=None):
        if (
            _payload_digest(self.payload) != self._digest
            or stable_hash(self.binding) != self._binding_digest
        ):
            raise ValueError("Encoded payload or binding changed after loading")
        if self.binding["code"] != encoding_code_signature():
            raise ValueError("Encoded producer code changed")
        _verify_files(self.binding)
        if self.file_manifest is not None:
            manifest_path, expected, payload_path, payload_sha = self.file_manifest
            if (
                file_sha256(manifest_path) != expected
                or file_sha256(payload_path) != payload_sha
            ):
                raise ValueError("Encoded cache manifest/payload file changed")
        if pack is not None:
            if self._pack_digest is None or self._config is None:
                raise ValueError(
                    "Encoded condition has no creation-time typed pack binding"
                )
            if self._config is not None:
                pack.validate(self._config)
            if (
                _payload_digest(pack.to_dict()) != self._pack_digest
                or tuple(self["target_hw"]) != pack.target_hw
                or tuple(self["person_order"]) != pack.person_ids
                or self.binding["source_fingerprint"] != pack.source_fingerprint
                or stable_hash(pack.provenance.get("geometry"))
                != stable_hash(self.binding["geometry"])
                or self.binding.get("task_pack_sha256") != self._pack_digest
                or self.binding.get("config") != self._config.to_dict()
            ):
                raise ValueError("Encoded typed pack/source binding differs")
        return self


@torch.no_grad()
def encode_condition(
    pipe,
    *,
    sample_id,
    source_images,
    control_image,
    canvas_hw,
    person_ids,
    prompt,
    producer_signature,
    geometry_signature,
    source_fingerprint,
    pose_images=(),
    negative_prompt=None,
    inpaint_image=None,
    edit_mask=None,
    provenance=None,
    source_person_ids=None,
    task_condition=None,
    config=None,
):
    """源参考按调用者稳定顺序；pose/control 是当前请求几何，GT 无入口。

    原始 refiner 调用仍应 cached=False；此入口用于离线训练条件与 coarse cache。
    source_images/pose_images 须预先裁剪/缩放到所需原生参考分辨率。
    """
    if (
        not source_images
        or not producer_signature
        or not geometry_signature
        or not source_fingerprint
    ):
        raise ValueError(
            "Explicit source, producer and current geometry bindings required"
        )
    if len(set(person_ids)) != len(person_ids) or len(person_ids) not in (1, 2):
        raise ValueError("One or two ordered stable person IDs required")
    if source_person_ids is None:
        source_person_ids = (
            tuple(person_ids)
            if len(source_images) == len(person_ids)
            else ("shared_frame",) if len(source_images) == 1 else None
        )
    if (
        source_person_ids is None
        or len(source_person_ids) != len(source_images)
        or any(
            pid != "shared_frame" and pid not in person_ids for pid in source_person_ids
        )
    ):
        raise ValueError(
            "Native source reference person IDs must explicitly match ordered source images"
        )
    if (inpaint_image is None) != (edit_mask is None):
        raise ValueError("Inpaint coarse image and white edit mask must occur together")
    references, shapes = [], []
    for path in [*source_images, *pose_images]:
        with Image.open(path) as im:
            image = im.convert("RGBA")
        references.append(image)
        shapes.append((1, image.height // 16, image.width // 16))
    hw = tuple(v // 16 for v in canvas_hw)
    if any(v % 32 for v in canvas_hw):
        raise ValueError("Canvas must be divisible by32")
    payload = {
        "reference_latents": torch.cat(
            [encode_image(pipe, image) for image in references], 1
        ),
        "control_latents": encode_image(pipe, control_image, canvas_hw=canvas_hw),
        "target_hw": hw,
        "canvas_hw": tuple(canvas_hw),
        "person_order": tuple(person_ids),
        "img_shapes": [[*shapes, (1, *hw)]],
    }

    def encode_prompt(text):
        embeddings, mask, slots = pipe.encode_prompt(
            image=references,
            prompt=text,
            device=pipe._execution_device,
            num_images_per_prompt=1,
        )
        mask = (
            torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
            if mask is None
            else mask.bool()
        )
        if (
            slots is None
            or slots.shape != embeddings.shape[:2]
            or mask.shape != slots.shape
        ):
            raise ValueError("VLM must provide aligned prompt and image slot masks")
        if int(slots.sum()) * 4 != payload["reference_latents"].shape[1]:
            raise ValueError(
                "VLM source image slots differ from actual VAE reference grids"
            )
        return {
            "prompt_embeds": embeddings,
            "prompt_mask": mask,
            "img_mask": torch.cat(
                [
                    slots.bool(),
                    slots.new_ones((1, hw[0] * hw[1] // 4), dtype=torch.bool),
                ],
                1,
            ),
        }

    payload.update(encode_prompt(prompt))
    if negative_prompt is not None:
        payload["negative"] = encode_prompt(negative_prompt)
    if inpaint_image is not None:
        with Image.open(inpaint_image) as im:
            base = im.convert("RGBA").resize(canvas_hw[::-1], Image.Resampling.LANCZOS)
        with Image.open(edit_mask) as im:
            edit = (
                np.asarray(
                    im.convert("L").resize(canvas_hw[::-1], Image.Resampling.NEAREST)
                )
                >= 128
            )
        pixels = np.array(base)
        pixels[edit, :3] = 128
        # preprocess maps128 to approximately zero; mask explicitly in normalized
        # pixel space below to preserve exact native black-zero inpaint semantics.
        image = Image.fromarray(pixels)
        payload["inpaint_latents"] = _encode_masked_image(pipe, image, edit, canvas_hw)
        keep = torch.from_numpy((~edit).copy()).float()[None, None]
        payload["keep_mask"] = (
            torch.nn.functional.interpolate(keep, size=hw, mode="nearest")
            .flatten(2)
            .transpose(1, 2)
        )
    binding = {
        "schema": SCHEMA,
        "sample_id": str(sample_id),
        "source_files": _files(source_images),
        "geometry_files": [
            *_files([*pose_images, control_image]),
            *geometry_input_files(geometry_signature),
        ],
        "inpaint_files": _files([inpaint_image, edit_mask]) if inpaint_image else [],
        "person_ids": list(person_ids),
        "canvas_hw": list(canvas_hw),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "source_reference_person_ids": list(source_person_ids),
        "producer": copy.deepcopy(producer_signature),
        "geometry": copy.deepcopy(geometry_signature),
        "source_fingerprint": source_fingerprint,
        "code": encoding_code_signature(),
        "provenance": provenance or {},
        "vae_config": {
            k: getattr(pipe.vae.config, k)
            for k in ("z_dim", "latents_mean", "latents_std")
        },
    }
    value = EncodedCondition(payload, binding)
    if task_condition is not None:
        if config is None:
            raise ValueError(
                "Typed condition encoding requires explicit adapter config"
            )
        value = value.with_task_condition(task_condition, config=config)
    return value


def encode_prepared_condition(
    pipe,
    source_pack,
    *,
    manifest_path,
    config,
    sample_id,
    source_images,
    prompt,
    producer_signature,
    negative_prompt=None,
    source_person_ids=None,
):
    """几何 worker → 当前 typed pack + 当前 native refs/control 的单一接线入口。"""
    from .geometry_preparation import load_geometry_task_condition

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    pack = load_geometry_task_condition(source_pack, manifest_path, config=config)
    canvas_hw = tuple(manifest["geometry_signature"]["canvas"]["output_hw"])
    condition = encode_condition(
        pipe,
        sample_id=sample_id,
        source_images=source_images,
        control_image=manifest["control_image"],
        pose_images=manifest["pose_images"],
        canvas_hw=canvas_hw,
        person_ids=pack.person_ids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        producer_signature=producer_signature,
        geometry_signature=manifest["geometry_signature"],
        source_fingerprint=pack.source_fingerprint,
        source_person_ids=source_person_ids,
        task_condition=pack,
        config=config,
    )
    return condition, pack


@torch.no_grad()
def _encode_masked_image(pipe, image, edit, canvas_hw):
    # Native pipeline zeros the normalized RGB before adding opaque alpha.
    pixels = pipe.image_processor.preprocess(
        image, height=canvas_hw[0], width=canvas_hw[1]
    )
    pixels[:, :3] *= torch.from_numpy((~edit).copy()).to(pixels)[None, None]
    if pixels.shape[1] == 3:
        pixels = torch.cat([pixels, torch.ones_like(pixels[:, :1])], 1)
    pixels[:, 3:] = 1
    dtype = next(pipe.vae.parameters()).dtype
    latent = pipe.vae.encode(
        pixels.unsqueeze(2).to(device=pipe._execution_device, dtype=dtype)
    ).latent_dist.mode()
    mean = (
        torch.as_tensor(pipe.vae.config.latents_mean).to(latent).reshape(1, 64, 1, 1, 1)
    )
    std = (
        torch.as_tensor(pipe.vae.config.latents_std).to(latent).reshape(1, 64, 1, 1, 1)
    )
    return ordinary_cpu(((latent - mean) / std).flatten(2).transpose(1, 2))


class EncodedConditionCache:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, condition, pack, *, config):
        condition.verify(pack)
        pack.validate(config)
        if condition.binding.get("config") != config.to_dict():
            raise ValueError("Encoded task config differs from creation-time binding")
        path = self.root / stable_hash(condition.binding["sample_id"])
        path.mkdir(parents=True, exist_ok=False)
        payload = {"condition": condition.payload, "pack": pack.to_dict()}
        with (path / "payload.pt").open("xb") as stream:
            torch.save(ordinary_cpu(payload), stream)
        manifest = {
            "schema": SCHEMA,
            "binding": condition.binding,
            "config": config.to_dict(),
            "payload_sha256": file_sha256(path / "payload.pt"),
            "pack_digest": _payload_digest(pack.to_dict()),
        }
        with (path / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return path

    def load(self, sample_id, *, expected_binding, config):
        path = self.root / stable_hash(str(sample_id))
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["schema"] != SCHEMA
            or manifest["binding"] != json.loads(json.dumps(expected_binding))
            or manifest["config"] != config.to_dict()
            or manifest["binding"]["sample_id"] != str(sample_id)
        ):
            raise ValueError(
                "Encoded expected producer/source/config/geometry binding differs"
            )
        payload_path = path / "payload.pt"
        if file_sha256(payload_path) != manifest["payload_sha256"]:
            raise ValueError("Encoded payload SHA256 differs")
        payload = torch.load(payload_path, weights_only=True, map_location="cpu")
        pack = TaskCondition.from_dict(payload["pack"]).validate(config)
        value = EncodedCondition(
            payload["condition"],
            manifest["binding"],
            file_manifest=(
                manifest_path,
                file_sha256(manifest_path),
                payload_path,
                manifest["payload_sha256"],
            ),
            pack_digest=manifest["pack_digest"],
            config=config,
        )
        return value.verify(pack), pack


def sample_validator(condition, pack, sample_id):
    from .training import SampleAttestation

    if not isinstance(condition, EncodedCondition) or condition.file_manifest is None:
        raise ValueError(
            "Training requires independently verified EncodedCondition cache"
        )
    condition.verify(pack)
    if str(sample_id) != condition.binding["sample_id"]:
        raise ValueError("Encoded sample ID differs")
    return SampleAttestation(
        str(sample_id),
        stable_hash(condition.binding),
        condition.file_manifest[3],
        pack.source_fingerprint,
        pack.target_hw,
        pack.person_ids,
    )

"""固定 Stage1 生成图的768 face/paired-hand 局部精修与离线训练工厂。"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .cache import _payload_digest, asset_hashes, file_sha256, ordinary_cpu, stable_hash
from .contracts import ContactRelations, TaskCondition
from .encoded import (
    EncodedConditionCache,
    encode_condition,
    encode_image,
    encoding_code_signature,
    geometry_input_files,
)
from .features import RawSourceFeatures
from .geometry import PixelCanvasTransform
from .geometry_preparation import P_ROOT, _exclusive_directory, _save_image, _save_json


@dataclass(frozen=True)
class CoarseArtifact:
    image_path: Path
    manifest_path: Path
    manifest: dict
    manifest_sha256: str

    @classmethod
    def load(cls, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("artifact_kind") != "generated_coarse"
            or manifest.get("sampling", {}).get("sampling")
            != "native_dynamic_shift_euler_unpatched64_v1"
        ):
            raise ValueError(
                "Refinement base must be a verified generated coarse artifact; GT/noisy GT is forbidden"
            )
        value = cls(
            Path(manifest["image_path"]),
            manifest_path,
            manifest,
            file_sha256(manifest_path),
        )
        return value.verify()

    def verify(self):
        if file_sha256(self.manifest_path) != self.manifest_sha256:
            raise ValueError("Generated coarse manifest changed")
        if json.loads(self.manifest_path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError("Generated coarse in-memory manifest changed")
        if file_sha256(self.image_path) != self.manifest["image_sha256"]:
            raise ValueError("Generated coarse image changed")
        if (
            asset_hashes(self.manifest["stage1_files"])
            != self.manifest["stage1_hashes"]
        ):
            raise ValueError("Locked Stage1 checkpoint changed since coarse generation")
        if (
            not self.manifest["stage1_signatures"]
            or not self.manifest["source_fingerprint"]
        ):
            raise ValueError("Generated coarse requires Stage1 and source bindings")
        geometry_input_files(self.manifest["geometry_signature"])
        return self


def write_coarse_artifact(
    result,
    *,
    output_dir,
    stage1_checkpoint,
    stage1_signatures,
    source_fingerprint,
    geometry_signature,
    selection_role,
    output_root=P_ROOT,
):
    """只接受采样结果，保存生成图+模型/协议签名，不能把 GT 路径当作底图。"""
    from .inference import SamplingResult

    if (
        not isinstance(result, SamplingResult)
        or result.images is None
        or result.audit.get("sampling") != "native_dynamic_shift_euler_unpatched64_v1"
    ):
        raise ValueError("A generated coarse SamplingResult is required")
    if (
        selection_role not in {"selected_stage1", "engineering_probe"}
        or not stage1_signatures
        or not geometry_signature
    ):
        raise ValueError(
            "Coarse generation requires explicit locked Stage1 selection/signature"
        )
    checkpoint = Path(stage1_checkpoint)
    files = {
        name: str((checkpoint / name).resolve())
        for name in ("checkpoint.json", "state.pt")
    }
    hashes = asset_hashes(files)
    if not all(hashes.values()):
        raise ValueError("Complete locked Stage1 checkpoint required")
    saved_metadata = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )["metadata"]
    if any(
        saved_metadata.get(key) != value for key, value in stage1_signatures.items()
    ):
        raise ValueError("Caller Stage1 signatures differ from checkpoint")
    actual_model = {"state_sha256": hashes["state.pt"], "metadata": saved_metadata}
    if result.audit.get("trained_model_signature") != actual_model:
        raise ValueError(
            "Stage1 checkpoint differs from the model actually used by coarse sampler"
        )
    if (
        result.audit.get("source_fingerprint") != source_fingerprint
        or result.audit.get("geometry_signature") != geometry_signature
    ):
        raise ValueError("Coarse sampler source/current geometry bindings differ")
    image = (
        result.images[0] if isinstance(result.images, (list, tuple)) else result.images
    )
    if not isinstance(image, Image.Image):
        raise ValueError("Decoded generated PIL image required")
    path = _exclusive_directory(output_dir, output_root)
    image_path = path / "coarse.png"
    _save_image(image_path, image.convert("RGB"))
    manifest = {
        "schema": "generated_coarse_v1",
        "artifact_kind": "generated_coarse",
        "image_path": str(image_path),
        "image_sha256": file_sha256(image_path),
        "image_hw": [image.height, image.width],
        "stage1_files": files,
        "stage1_hashes": hashes,
        "stage1_signatures": copy.deepcopy(stage1_signatures),
        "sampling": copy.deepcopy(result.audit),
        "selection_role": selection_role,
        "source_fingerprint": source_fingerprint,
        "geometry_signature": copy.deepcopy(geometry_signature),
    }
    _save_json(path / "coarse.json", manifest)
    return CoarseArtifact.load(path / "coarse.json")


@dataclass(frozen=True)
class PartEndpoint:
    person_id: int
    part_id: int


@dataclass(frozen=True)
class ROITransform:
    kind: str
    person_ids: tuple[int, ...]
    box_xyxy: tuple[int, int, int, int]
    canvas_hw: tuple[int, int]
    reflect_x: bool = False
    resolution: int = 768
    invalid_reason: str | None = None
    endpoints: tuple[PartEndpoint, ...] = ()

    @classmethod
    def from_box(
        cls,
        kind,
        person_ids,
        box,
        canvas_hw,
        *,
        reflect_x=False,
        margin_px=0.0,
        endpoints=None,
    ):
        if (
            kind not in {"face", "paired_hands", "hand"}
            or len(set(person_ids)) != len(person_ids)
            or not person_ids
        ):
            raise ValueError(
                "ROI requires face/hand kind and explicit stable person IDs"
            )
        endpoints = (
            tuple(PartEndpoint(pid, 0) for pid in person_ids)
            if endpoints is None and kind == "face"
            else tuple(endpoints or ())
        )
        if (
            len(endpoints) != (2 if kind == "paired_hands" else 1)
            or len(set(endpoints)) != len(endpoints)
            or set(e.person_id for e in endpoints) != set(person_ids)
            or any(
                e.part_id not in ((0,) if kind == "face" else (4, 7)) for e in endpoints
            )
        ):
            raise ValueError(
                "ROI must retain exact stable person/face-or-hand part endpoints"
            )
        h, w = canvas_hw
        if len(box) != 4 or not np.isfinite(box).all() or margin_px < 0:
            return cls(
                kind,
                tuple(person_ids),
                (0, 0, 0, 0),
                tuple(canvas_hw),
                reflect_x,
                invalid_reason="invalid_roi",
                endpoints=endpoints,
            )
        lo = np.floor(np.asarray(box[:2]) - margin_px).astype(int).clip(0, [w, h])
        hi = np.ceil(np.asarray(box[2:]) + margin_px).astype(int).clip(0, [w, h])
        x0, y0, x1, y1 = *lo, *hi
        reason = (
            "empty_roi"
            if x1 <= x0 or y1 <= y0
            else (
                "tiny_roi"
                if (x1 - x0) * (y1 - y0) < 8 or min(x1 - x0, y1 - y0) < 2
                else None
            )
        )
        return cls(
            kind,
            tuple(person_ids),
            tuple(map(int, (x0, y0, x1, y1))),
            tuple(canvas_hw),
            reflect_x,
            invalid_reason=reason,
            endpoints=endpoints,
        )

    @property
    def transform(self):
        x0, y0, x1, y1 = self.box_xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Empty ROI has no invertible transform")
        sx, sy = self.resolution / (x1 - x0), self.resolution / (y1 - y0)
        matrix = [
            [
                -sx if self.reflect_x else sx,
                0,
                self.resolution + sx * x0 if self.reflect_x else -sx * x0,
            ],
            [0, sy, -sy * y0],
            [0, 0, 1],
        ]
        return PixelCanvasTransform(
            torch.tensor(matrix), self.canvas_hw, (self.resolution, self.resolution)
        )

    def to_local(self, xy):
        return self.transform.apply(xy)

    def to_canvas(self, xy):
        matrix = torch.linalg.inv(self.transform.matrix).to(xy)
        return xy @ matrix[:2, :2].T + matrix[:2, 2]

    def crop(self, image, *, nearest=False):
        if image.size != self.canvas_hw[::-1]:
            raise ValueError("ROI input image canvas differs")
        if self.invalid_reason == "empty_roi":
            return Image.new(image.mode, (768, 768))
        inverse = torch.linalg.inv(self.transform.matrix).numpy()
        return image.transform(
            (768, 768),
            Image.Transform.AFFINE,
            tuple(inverse[:2].reshape(-1)),
            resample=Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC,
            fillcolor=0,
        )


def derive_refinement_rois(pack, boxes, *, canvas_hw, margin_px=12.0):
    """face 各归其主；显式手接触 ROI 取两端边界的并集。parts4/7 是左右手。"""

    def bounds(slot, name):
        item = boxes[slot]
        return item.get("mesh_part_bounds_xyxy", item).get(name)

    rois = []
    for slot, person in enumerate(pack.person_ids):
        face = bounds(slot, "face")
        if face is not None and bool(pack.person_valid[0, slot]):
            rois.append(
                ROITransform.from_box(
                    "face", (person,), face, canvas_hw, margin_px=margin_px
                )
            )
    seen = set()
    for edge in torch.nonzero(pack.relations.valid[0]).flatten().tolist():
        rel = pack.relations
        a, b = int(rel.src_person[0, edge]), int(rel.dst_person[0, edge])
        pa, pb = int(rel.src_part[0, edge]), int(rel.dst_part[0, edge])
        if pa not in (4, 7) or pb not in (4, 7):
            continue
        first, second = bounds(a, "left_hand" if pa == 4 else "right_hand"), bounds(
            b, "left_hand" if pb == 4 else "right_hand"
        )
        if first is None or second is None:
            continue
        key = tuple(sorted(((a, pa), (b, pb))))
        if key in seen:
            continue
        seen.add(key)
        box = [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        ]
        people = tuple(
            person for slot, person in enumerate(pack.person_ids) if slot in (a, b)
        )
        rois.append(
            ROITransform.from_box(
                "paired_hands",
                people,
                box,
                canvas_hw,
                margin_px=margin_px,
                endpoints=(
                    PartEndpoint(pack.person_ids[a], pa),
                    PartEndpoint(pack.person_ids[b], pb),
                ),
            )
        )
    return tuple(rois)


@dataclass(frozen=True)
class SourceCrop:
    person_id: int
    image_path: Path
    mask_path: Path
    files: dict
    hashes: dict
    original_box: tuple | None = None
    part_id: int = 0
    source_geometry_provenance: dict | None = None
    crop_margin: float = 0.06

    @classmethod
    def bind(
        cls,
        person_id,
        image_path,
        mask_path,
        *,
        original_assets,
        original_box=None,
        part_id=0,
        source_geometry_provenance=None,
        crop_margin=0.06,
    ):
        files = {
            "crop": str(Path(image_path).resolve()),
            "mask": str(Path(mask_path).resolve()),
            **{
                f"source_original_{i}": str(Path(p).resolve())
                for i, p in enumerate(original_assets)
            },
        }
        return cls(
            person_id,
            Path(image_path),
            Path(mask_path),
            files,
            asset_hashes(files),
            tuple(original_box) if original_box is not None else None,
            part_id,
            copy.deepcopy(source_geometry_provenance),
            crop_margin,
        ).verify()

    def verify(self):
        if self.part_id not in (0, 4, 7):
            raise ValueError("Source ROI part must be face0/left_hand4/right_hand7")
        if (
            Path(self.image_path).resolve() != Path(self.files["crop"]).resolve()
            or Path(self.mask_path).resolve() != Path(self.files["mask"]).resolve()
        ):
            raise ValueError(
                "Source crop path differs from its immutable file role binding"
            )
        if self.part_id in (4, 7) and self.source_geometry_provenance is None:
            raise ValueError(
                "Hand source crop requires actual source geometry ROI provenance"
            )
        if asset_hashes(self.files) != self.hashes or not all(self.hashes.values()):
            raise ValueError("Source crop/original asset SHA256 changed")
        with Image.open(self.image_path) as image, Image.open(self.mask_path) as mask:
            if image.size != (768, 768) or mask.size != image.size:
                raise ValueError(
                    "Native source crop/mask must be768 square before VLM/VAE encoding"
                )
        if self.source_geometry_provenance is not None:
            from .geometry_preparation import validate_source_geometry
            from .features import square_person_crop

            bundle = self.source_geometry_provenance
            original = self.files["source_original_0"]
            original_mask = self.files["source_original_1"]
            validate_source_geometry(
                bundle, {"person_order": bundle["person_ids"]}, {"source_rgb": original}
            )
            if self.person_id not in bundle["person_ids"]:
                raise ValueError("Source crop person absent from source geometry")
            slot = bundle["person_ids"].index(self.person_id)
            name = {0: "face", 4: "left_hand", 7: "right_hand"}[self.part_id]
            expected = bundle["rois"][slot].get(name)
            if (
                expected is None
                or self.original_box is None
                or tuple(expected) != tuple(self.original_box)
            ):
                raise ValueError(
                    "Source crop anatomical part/native ROI differs from source geometry endpoint"
                )
            mask_role = f"source_mask_{slot}"
            if (
                mask_role not in bundle["asset_files"]
                or file_sha256(original_mask) != bundle["asset_hashes"][mask_role]
            ):
                raise ValueError(
                    "Source crop visible mask belongs to another stable person"
                )
            with Image.open(original) as image:
                rgb = np.array(image.convert("RGB"))
            with Image.open(original_mask) as image:
                support = np.array(image.convert("L")) > 0
            reference, reference_support, _ = square_person_crop(
                rgb, support, box=self.original_box, size=768, margin=self.crop_margin
            )
            with Image.open(self.image_path) as image, Image.open(
                self.mask_path
            ) as mask:
                if not np.array_equal(
                    np.array(image.convert("RGB")), np.array(reference)
                ) or not np.array_equal(
                    np.array(mask.convert("L")) >= 128, reference_support
                ):
                    raise ValueError(
                        "Source crop pixels/support do not match its declared native geometry ROI"
                    )
        return self

    @property
    def kind(self):
        return {0: "face", 4: "left_hand", 7: "right_hand"}[self.part_id]


def _crop_signature(crop):
    return {
        "person_id": crop.person_id,
        "part_id": crop.part_id,
        "kind": crop.kind,
        "files": crop.hashes,
        "box": crop.original_box,
        "crop_margin": crop.crop_margin,
        "source_geometry_sha256": (
            None
            if crop.source_geometry_provenance is None
            else stable_hash(crop.source_geometry_provenance)
        ),
    }


def _verify_crop_endpoints(crops, person_ids, endpoints):
    required = {(e.person_id, e.part_id) for e in endpoints}
    actual = [(c.person_id, c.part_id) for c in crops]
    if (
        len(actual) != len(required)
        or set(actual) != required
        or set(pid for pid, _ in actual) != set(person_ids)
    ):
        raise ValueError(
            "Source crops must match exact ROI person/part endpoints; face or opposite hand substitution is forbidden"
        )
    for crop in crops:
        crop.verify()


def write_source_crop(
    *,
    person_id,
    source_image,
    source_mask,
    box,
    output_dir,
    output_root=P_ROOT,
    part_id=0,
    source_geometry_provenance=None,
    crop_margin=0.06,
):
    """源原图 native ROI + 本人物可见mask，正方形白底；无 target 输入。"""
    from .features import square_person_crop

    with Image.open(source_image) as image:
        rgb = np.array(image.convert("RGB"))
    with Image.open(source_mask) as image:
        if image.size != (rgb.shape[1], rgb.shape[0]):
            raise ValueError("Source RGB/mask dimensions differ")
        mask = np.array(image.convert("L")) > 0
    lo = np.floor(box[:2]).astype(int).clip(0, [rgb.shape[1], rgb.shape[0]])
    hi = np.ceil(box[2:]).astype(int).clip(0, [rgb.shape[1], rgb.shape[0]])
    if mask[lo[1] : hi[1], lo[0] : hi[0]].sum() < 8:
        raise ValueError("Source ROI has fewer than8 visible pixels")
    crop, support, _ = square_person_crop(
        rgb, mask, box=box, size=768, margin=crop_margin
    )
    path = _exclusive_directory(output_dir, output_root)
    _save_image(path / "source_crop.png", crop)
    _save_image(
        path / "source_support.png", Image.fromarray(support.astype(np.uint8) * 255)
    )
    return SourceCrop.bind(
        person_id,
        path / "source_crop.png",
        path / "source_support.png",
        original_assets=[source_image, source_mask],
        original_box=box,
        part_id=part_id,
        source_geometry_provenance=source_geometry_provenance,
        crop_margin=crop_margin,
    )


@torch.no_grad()
def encode_roi_source_bank(
    pipe,
    extractor,
    source_crops,
    *,
    person_ids,
    vae_producer_signature=None,
    endpoints=None,
):
    """每个真实 source crop 自己的 VAE48² 与 DINO16² 同坐标采样；不借 GT。"""
    endpoints = (
        tuple(PartEndpoint(pid, 0) for pid in person_ids)
        if endpoints is None
        else tuple(endpoints)
    )
    _verify_crop_endpoints(source_crops, person_ids, endpoints)
    vae_producer_signature = vae_producer_signature or extractor.model_signature.get(
        "source_vae"
    )
    if not vae_producer_signature:
        raise ValueError(
            "ROI source bank requires the active frozen VAE producer signature"
        )
    values, valids, signatures = [], [], []
    for person_id in person_ids:
        person_tokens, person_valid = [], []
        for crop in source_crops:
            crop.verify()
            if crop.person_id != person_id:
                continue
            with Image.open(crop.image_path) as image:
                image = image.convert("RGB")
            with Image.open(crop.mask_path) as support:
                support = np.array(support.convert("L")) > 0
            vae = (
                encode_image(pipe, image).float().transpose(1, 2).reshape(1, 64, 48, 48)
            )
            dino_image = image.resize((224, 224), Image.Resampling.BICUBIC)
            inputs = extractor.processor(
                images=dino_image,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
            inputs = {key: value.to(extractor.device) for key, value in inputs.items()}
            dino = extractor.dino(**inputs).last_hidden_state
            if dino.shape != (1, 257, 1536):
                raise ValueError("ROI source bank requires DINOv2-G/14 width1536")
            centers = (torch.arange(16).float() + 0.5) / 16 * 2 - 1
            yy, xx = torch.meshgrid(centers, centers, indexing="ij")
            grid = torch.stack([xx, yy], -1)[None]
            vae_patches = (
                F.grid_sample(vae, grid, align_corners=False, mode="bilinear")[0]
                .flatten(1)
                .T
            )
            valid = (
                F.grid_sample(
                    torch.from_numpy(support.copy()).float()[None, None],
                    grid,
                    mode="nearest",
                    align_corners=False,
                )[0, 0].flatten()
                > 0.5
            )
            person_tokens.append(
                torch.cat([vae_patches, dino[0, 1:].float().cpu()], -1)
            )
            person_valid.append(valid)
            signatures.append(_crop_signature(crop))
        values.append(
            torch.cat(person_tokens) if person_tokens else torch.zeros(256, 1600)
        )
        valids.append(
            torch.cat(person_valid)
            if person_valid
            else torch.zeros(256, dtype=torch.bool)
        )
    n = max(len(value) for value in values)
    fingerprint = stable_hash(
        {
            "schema": "roi_source_bank_v1",
            "sources": signatures,
            "person_ids": list(person_ids),
            "model": extractor.model_signature,
            "vae_producer": vae_producer_signature,
            "vae_config": {
                key: getattr(pipe.vae.config, key)
                for key in ("z_dim", "latents_mean", "latents_std")
            },
            "code": encoding_code_signature(),
        }
    )
    return RawSourceFeatures(
        tuple(person_ids),
        torch.stack([F.pad(v, (0, 0, 0, n - len(v))) for v in values])[None],
        torch.stack([F.pad(v, (0, n - len(v))) for v in valids])[None],
        torch.zeros(1, len(person_ids), 512),
        torch.zeros(1, len(person_ids), dtype=torch.bool),
        fingerprint,
        {"source_crops": signatures, "model_signature": extractor.model_signature},
    ).validate()


def localize_task_condition(pack, roi, *, source_features, config):
    """对每个 spatial/owner/ROI 做同一真实局部 affine，稳定 IDs 显式 remap。"""
    if source_features.person_ids != roi.person_ids:
        raise ValueError("ROI source crop bank stable person order differs")
    source_crops = source_features.provenance.get("source_crops")
    bank_endpoints = (
        []
        if not isinstance(source_crops, (list, tuple))
        else [(crop.get("person_id"), crop.get("part_id")) for crop in source_crops]
    )
    expected_endpoints = {
        (endpoint.person_id, endpoint.part_id) for endpoint in roi.endpoints
    }
    if (
        len(bank_endpoints) != len(expected_endpoints)
        or set(bank_endpoints) != expected_endpoints
    ):
        raise ValueError(
            "ROI source bank endpoints differ from requested face/hand endpoints"
        )
    slots = [pack.person_ids.index(pid) for pid in roi.person_ids]
    y, x = torch.meshgrid(
        torch.arange(48).float() + 0.5, torch.arange(48).float() + 0.5, indexing="ij"
    )
    original = roi.to_canvas(torch.stack([x, y], -1) * 16)
    grid = original / torch.tensor(roi.canvas_hw[::-1]) * 2 - 1

    def sample(value, *, labels=False, flattened=False):
        value = value[:, slots]
        prefix = value.shape[:-1] if flattened else value.shape[:-2]
        if flattened:
            value = value.reshape(*prefix, *pack.target_hw)
        shape = value.shape
        plane = value.reshape(-1, 1, *pack.target_hw).float()
        result = F.grid_sample(
            plane,
            grid[None].expand(len(plane), 48, 48, 2),
            mode="nearest" if labels else "bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        result = result.reshape(*shape[:-2], 48, 48)
        return result.flatten(-2) if flattened else result.long() if labels else result

    spatial = {
        name: sample(value, labels=name == "part")
        for name, value in pack.spatial.items()
    }
    # Bilinear FP32 weights can sum to 1+epsilon even on an all-one support map.
    for name in ("mask", "visibility", "contact"):
        spatial[name] = spatial[name].clamp(0, 1)
    spatial["part"][spatial["mask"] <= 0] = -1
    spatial["normal"] = (
        F.normalize(spatial["normal"], dim=2, eps=1e-8) * spatial["mask"]
    )
    masks = sample(pack.person_mask, flattened=True).clamp(0, 1)
    owner = torch.minimum(sample(pack.owner, flattened=True).clamp(0, 1), masks)
    owner = torch.minimum(owner, spatial["visibility"].flatten(2))
    pv = pack.person_valid[:, slots] & source_features.valid.any(-1)
    owner *= pv[..., None]
    relations = pack.relations
    enabled = relations.valid.clone()
    for slot in range(len(pack.person_ids)):
        if slot not in slots:
            enabled &= (relations.src_person != slot) & (relations.dst_person != slot)
    keep = enabled[0]
    mapping = {old: new for new, old in enumerate(slots)}
    rel = ContactRelations(
        torch.tensor(
            [[mapping[int(v)] for v in relations.src_person[0, keep]]], dtype=torch.long
        ),
        relations.src_part[:, keep],
        torch.tensor(
            [[mapping[int(v)] for v in relations.dst_person[0, keep]]], dtype=torch.long
        ),
        relations.dst_part[:, keep],
        relations.relative[:, keep],
        relations.confidence[:, keep],
        relations.valid[:, keep],
    )
    rel_valid = pv.gather(1, rel.src_person) & pv.gather(1, rel.dst_person)
    rel = replace(rel, valid=rel.valid & rel_valid)
    return replace(
        pack,
        target_hw=(48, 48),
        person_ids=roi.person_ids,
        spatial=spatial,
        spatial_valid={
            name: value[:, slots] & pv for name, value in pack.spatial_valid.items()
        },
        person_valid=pv,
        person_mask=masks,
        owner=owner,
        face_mask=sample(pack.face_mask, flattened=True).clamp(0, 1),
        hand_mask=sample(pack.hand_mask, flattened=True).clamp(0, 1),
        source_tokens=source_features.tokens,
        source_token_valid=source_features.valid & pv[..., None],
        face_identity=source_features.face_identity,
        face_identity_valid=source_features.face_identity_valid & pv,
        source_fingerprint=source_features.fingerprint,
        relations=rel,
        provenance={
            "source": source_features.provenance,
            "geometry": {
                "local_transform": roi.transform.to_dict(),
                "parent_geometry": pack.provenance.get("geometry", {}),
                "person_ids": list(roi.person_ids),
            },
        },
    ).validate(config)


@dataclass(frozen=True)
class RefinementRequest:
    coarse: CoarseArtifact
    roi: ROITransform
    source_crops: tuple[SourceCrop, ...]
    geometry_original_path: Path
    geometry_roi_path: Path
    edit_mask_path: Path
    coarse_roi_path: Path
    seed: int
    model_signature: dict
    hashes: dict
    source_bank_fingerprint: str
    task_condition: TaskCondition | None
    config: object
    _binding: str = field(init=False, repr=False)
    _task_digest: str | None = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_task_digest",
            (
                None
                if self.task_condition is None
                else _payload_digest(self.task_condition.to_dict())
            ),
        )
        object.__setattr__(self, "_binding", stable_hash(self.binding()))

    def binding(self):
        return {
            "coarse_manifest_sha256": self.coarse.manifest_sha256,
            "coarse_image_sha256": self.coarse.manifest["image_sha256"],
            "roi": asdict(self.roi),
            "source_crops": [_crop_signature(c) for c in self.source_crops],
            "hashes": self.hashes,
            "seed": self.seed,
            "model_signature": self.model_signature,
            "source_bank_fingerprint": self.source_bank_fingerprint,
            "local_task_pack_sha256": self._task_digest,
            "local_geometry": (
                None
                if self.task_condition is None
                else self.task_condition.provenance.get("geometry")
            ),
            "config": None if self.config is None else self.config.to_dict(),
        }

    def verify(self):
        self.coarse.verify()
        for crop in self.source_crops:
            crop.verify()
        if stable_hash(self.binding()) != self._binding:
            raise ValueError("Immutable refinement request binding changed")
        paths = {
            "geometry_original": self.geometry_original_path,
            "geometry_roi": self.geometry_roi_path,
            "edit_mask": self.edit_mask_path,
            "coarse_roi": self.coarse_roi_path,
        }
        if asset_hashes(paths) != self.hashes:
            raise ValueError("Refinement geometry/mask/coarse ROI SHA256 changed")
        if self.roi.invalid_reason is None:
            self.verify_task_condition(self.task_condition)
        return self

    def verify_task_condition(self, pack):
        if pack is None or self.config is None or self._task_digest is None:
            raise ValueError(
                "Refinement requires a creation-time local typed pack binding"
            )
        pack.validate(self.config)
        bank_crops = pack.provenance.get("source", {}).get("source_crops")
        if stable_hash(bank_crops) != stable_hash(
            [_crop_signature(crop) for crop in self.source_crops]
        ):
            raise ValueError(
                "Refiner source bank crop signatures differ from ordered native source crops"
            )
        expected = {
            "local_transform": self.roi.transform.to_dict(),
            "parent_geometry": self.coarse.manifest["geometry_signature"],
            "person_ids": list(self.roi.person_ids),
        }
        if pack.source_fingerprint != self.source_bank_fingerprint:
            raise ValueError("Refiner source bank differs from immutable crop binding")
        if (
            pack.target_hw != (48, 48)
            or pack.person_ids != self.roi.person_ids
            or stable_hash(pack.provenance.get("geometry")) != stable_hash(expected)
            or _payload_digest(pack.to_dict()) != self._task_digest
        ):
            raise ValueError(
                "Refiner local geometry/parent/affine/typed pack binding differs"
            )
        return self


def prepare_refinement_request(
    coarse,
    *,
    roi,
    source_crops,
    geometry_image,
    output_dir,
    seed,
    model_signature,
    source_bank_fingerprint,
    task_condition=None,
    config=None,
    edit_mask=None,
    output_root=P_ROOT,
):
    coarse.verify()
    if (
        not model_signature
        or not source_bank_fingerprint
        or set(c.person_id for c in source_crops) != set(roi.person_ids)
    ):
        raise ValueError(
            "Refiner requires trained model and actual source crops for each ROI person"
        )
    if tuple(coarse.manifest["image_hw"]) != roi.canvas_hw:
        raise ValueError("Coarse canvas/ROI differ")
    for crop in source_crops:
        crop.verify()
    _verify_crop_endpoints(source_crops, roi.person_ids, roi.endpoints)
    source_crops = tuple(
        crop for pid in roi.person_ids for crop in source_crops if crop.person_id == pid
    )
    path = _exclusive_directory(output_dir, output_root)
    if roi.invalid_reason in ("empty_roi", "invalid_roi"):
        local = Image.new("RGB", (768, 768))
        control = local.copy()
    else:
        with Image.open(coarse.image_path) as image:
            local = roi.crop(image.convert("RGB"))
        with Image.open(geometry_image) as image:
            control = roi.crop(image.convert("RGB"))
    if edit_mask is None:
        mask = Image.new("L", (768, 768), 255)
    else:
        with Image.open(edit_mask) as image:
            mask = (
                roi.crop(image.convert("L"), nearest=True)
                if image.size == roi.canvas_hw[::-1]
                else image.convert("L")
            )
        if mask.size != (768, 768):
            raise ValueError(
                "White edit mask must match local768 canvas or original canvas"
            )
    canvas_binding = coarse.manifest["geometry_signature"].get("canvas")
    if canvas_binding is not None and roi.invalid_reason not in (
        "empty_roi",
        "invalid_roi",
    ):
        physical = PixelCanvasTransform(
            torch.tensor(canvas_binding["matrix"]),
            tuple(canvas_binding["input_hw"]),
            tuple(canvas_binding["output_hw"]),
        )
        fov = Image.fromarray(physical.valid_fov_mask().byte().cpu().numpy() * 255)
        allowed = np.array(roi.crop(fov, nearest=True)) >= 128
        mask = Image.fromarray(
            ((np.array(mask) >= 128) & allowed).astype(np.uint8) * 255
        )
    _save_image(path / "coarse_roi.png", local)
    _save_image(path / "geometry_roi.png", control)
    _save_image(path / "edit_mask.png", mask)
    paths = {
        "geometry_original": Path(geometry_image),
        "geometry_roi": path / "geometry_roi.png",
        "edit_mask": path / "edit_mask.png",
        "coarse_roi": path / "coarse_roi.png",
    }
    request = RefinementRequest(
        coarse,
        roi,
        tuple(source_crops),
        Path(geometry_image),
        paths["geometry_roi"],
        paths["edit_mask"],
        paths["coarse_roi"],
        seed,
        copy.deepcopy(model_signature),
        asset_hashes(paths),
        source_bank_fingerprint,
        (
            None
            if task_condition is None
            else TaskCondition.from_dict(ordinary_cpu(task_condition.to_dict()))
        ),
        config,
    )
    _save_json(path / "request.json", request.binding())
    return request.verify()


@dataclass(frozen=True)
class RefinementResult:
    image_path: Path
    audit: dict


def _composite(coarse, refined, edit_mask, roi):
    refined = refined.convert("RGB")
    x0, y0, x1, y1 = roi.box_xyxy
    if roi.reflect_x:
        refined = refined.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        edit_mask = edit_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    patch = refined.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    mask = (
        np.array(edit_mask.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)) >= 128
    )
    result = np.array(coarse).copy()
    before = result.copy()
    region = result[y0:y1, x0:x1]
    region[mask] = np.array(patch)[mask]
    global_mask = np.zeros(before.shape[:2], bool)
    global_mask[y0:y1, x0:x1] = mask
    assert np.array_equal(result[~global_mask], before[~global_mask])
    return Image.fromarray(result), global_mask


@torch.no_grad()
def run_refinement(
    pipe,
    request,
    *,
    task_condition,
    output_dir,
    output_root=P_ROOT,
    prompt="Refine the person while preserving identity and the requested pose.",
    negative_prompt=None,
    num_inference_steps=40,
    true_cfg_scale=1.0,
):
    request.verify()
    path = _exclusive_directory(output_dir, output_root)
    with Image.open(request.coarse.image_path) as image:
        coarse = image.convert("RGB")
    with Image.open(request.edit_mask_path) as image:
        mask = image.convert("L")
    reason = request.roi.invalid_reason
    if reason is None:
        x0, y0, x1, y1 = request.roi.box_xyxy
        native_edit = (
            np.array(mask.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)) >= 128
        )
        if int(native_edit.sum()) < 8:
            reason = "empty_or_tiny_edit_mask"
    if reason is not None:
        _save_image(path / "refined.png", coarse)
        audit = {
            "fallback_reason": reason,
            "edited_pixels": 0,
            "outside_edit_mask_bitwise_equal": True,
            "request_binding": request._binding,
        }
    else:
        request.verify_task_condition(task_condition)
        if getattr(pipe, "text_encoder", None) is None:
            raise ValueError("Raw reference refiner requires cached=False and full VLM")
        if getattr(pipe, "trained_model_signature", None) != request.model_signature:
            raise ValueError(
                "Refiner loaded model signature differs from immutable request"
            )
        if (
            task_condition.target_hw != (48, 48)
            or task_condition.person_ids != request.roi.person_ids
        ):
            raise ValueError(
                "Refiner typed task must be localized to actual768/48² and ROI stable people"
            )
        if task_condition.source_fingerprint != request.source_bank_fingerprint:
            raise ValueError(
                "Refiner source bank differs from immutable source crop binding"
            )
        with Image.open(request.coarse_roi_path) as image:
            local = image.convert("RGB")
        with Image.open(request.geometry_roi_path) as image:
            control = image.convert("RGB")
        references = []
        for crop in request.source_crops:
            with Image.open(crop.image_path) as image:
                references.append(image.convert("RGB"))
        result = pipe(
            image=local,
            mask_image=mask,
            reference_images=references,
            control_image=control,
            height=768,
            width=768,
            output_resolution=768,
            use_kv_cache=False,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            cfg_norm_rescale=True,
            generator=torch.Generator(device=pipe._execution_device).manual_seed(
                request.seed
            ),
            task_condition=task_condition,
        )
        refined = result.images[0]
        _save_image(path / "generated_roi.png", refined)
        composited, edit = _composite(coarse, refined, mask, request.roi)
        _save_image(path / "refined.png", composited)
        audit = {
            "fallback_reason": None,
            "edited_pixels": int(edit.sum()),
            "outside_edit_mask_bitwise_equal": True,
            "request_binding": request._binding,
            "refiner_model_signature": request.model_signature,
            "coarse_manifest_sha256": request.coarse.manifest_sha256,
            "roi": asdict(request.roi),
            "sampling": {
                "steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
                "cfg_norm_rescale": True,
                "seed": request.seed,
                "use_kv_cache": False,
                "cached": False,
            },
        }
    _save_json(path / "audit.json", audit)
    return RefinementResult(path / "refined.png", audit)


def make_refiner_training_sample(
    pipe,
    request,
    *,
    task_condition,
    supervision_image,
    config,
    cache: EncodedConditionCache,
    sample_id,
    producer_signature,
    output_dir,
    prompt,
    negative_prompt=None,
    engineering_probe=False,
    output_root=P_ROOT,
    supervision_transform=None,
):
    """离线 coarse 固定底图；GT 只有此处独立监督 encoder 有读取入口。

    正式训练必须 locked selected_stage1；engineering_probe 只能验证链路，
    condition 绑定 coarse image/ROI/模型/协议 SHA，返回可给 training runner 的三元组。
    """
    request.verify()
    request.verify_task_condition(task_condition)
    if (
        request.coarse.manifest["selection_role"] != "selected_stage1"
        and not engineering_probe
    ):
        raise ValueError(
            "Formal refiner training requires locked selected Stage1 checkpoint"
        )
    if request.coarse.manifest["geometry_signature"].get("mode") != "driver_exact":
        raise ValueError("Original GT supervision requires exact driver pairing")
    if request.roi.invalid_reason:
        raise ValueError("Invalid refinement ROI cannot be a training sample")
    task_condition.validate(config)
    if (
        task_condition.target_hw != (48, 48)
        or task_condition.person_ids != request.roi.person_ids
    ):
        raise ValueError("Refiner training requires localized768 task/source bank")
    if task_condition.source_fingerprint != request.source_bank_fingerprint:
        raise ValueError(
            "Refiner training source bank differs from immutable crop binding"
        )
    path = _exclusive_directory(output_dir, output_root)
    condition = encode_condition(
        pipe,
        sample_id=sample_id,
        source_images=[c.image_path for c in request.source_crops],
        control_image=request.geometry_roi_path,
        canvas_hw=(768, 768),
        person_ids=request.roi.person_ids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        producer_signature=producer_signature,
        geometry_signature=task_condition.provenance["geometry"],
        source_fingerprint=task_condition.source_fingerprint,
        inpaint_image=request.coarse_roi_path,
        edit_mask=request.edit_mask_path,
        source_person_ids=[c.person_id for c in request.source_crops],
        task_condition=task_condition,
        config=config,
        provenance={
            "coarse_manifest": str(request.coarse.manifest_path),
            "coarse_manifest_sha256": request.coarse.manifest_sha256,
            "coarse_checkpoint_sha256": request.coarse.manifest["stage1_hashes"],
            "selection_role": (
                "engineering_probe" if engineering_probe else "selected_stage1"
            ),
        },
    )
    # Disk verifier must also attest the locked coarse artifact and original
    # source crops, beyond the derived input crops used by VAE/VLM.
    condition.binding["source_files"] += [
        {"path": p, "sha256": c.hashes[k]}
        for c in request.source_crops
        for k, p in c.files.items()
        if k.startswith("source_original_")
    ]
    condition.binding["inpaint_files"] += [
        {
            "path": str(request.coarse.manifest_path),
            "sha256": request.coarse.manifest_sha256,
        },
        {
            "path": str(request.coarse.image_path),
            "sha256": request.coarse.manifest["image_sha256"],
        },
    ]
    condition.binding["inpaint_files"] += [
        {"path": p, "sha256": request.coarse.manifest["stage1_hashes"][k]}
        for k, p in request.coarse.manifest["stage1_files"].items()
    ]
    condition._binding_digest = stable_hash(condition.binding)
    with Image.open(supervision_image) as image:
        image = image.convert("RGBA")
        if supervision_transform is not None:
            transform = supervision_transform
            if (
                image.size != transform.input_hw[::-1]
                or transform.output_hw != request.roi.canvas_hw
            ):
                raise ValueError("Supervision physical canvas transform differs")
            inverse = torch.linalg.inv(transform.matrix).cpu().numpy()
            image = image.transform(
                transform.output_hw[::-1],
                Image.Transform.AFFINE,
                tuple(inverse[:2].reshape(-1)),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(255, 255, 255, 255),
            )
        target = request.roi.crop(image)
    with Image.open(request.edit_mask_path) as image:
        edit = torch.from_numpy((np.array(image.convert("L")) >= 128).copy()).float()
    canvas_binding = request.coarse.manifest["geometry_signature"].get("canvas")
    if canvas_binding is not None:
        physical = PixelCanvasTransform(
            torch.tensor(canvas_binding["matrix"]),
            tuple(canvas_binding["input_hw"]),
            tuple(canvas_binding["output_hw"]),
        )
        fov = Image.fromarray(physical.valid_fov_mask().byte().cpu().numpy() * 255)
        local_fov = torch.from_numpy(
            (np.array(request.roi.crop(fov, nearest=True)) >= 128).copy()
        ).float()
    else:
        local_fov = torch.ones_like(edit)
    valid = F.interpolate(
        (edit * local_fov)[None, None], size=(48, 48), mode="area"
    ).flatten(1)
    if not valid.any():
        raise ValueError("Refiner training edit mask has no valid physical content")
    supervision = {
        "target_latent": encode_image(pipe, target),
        "valid_target_tokens": valid,
    }
    with (path / "supervision.pt").open("xb") as stream:
        torch.save(ordinary_cpu(supervision), stream)
    _save_json(
        path / "supervision.json",
        {
            "supervision_sha256": file_sha256(supervision_image),
            "payload_sha256": file_sha256(path / "supervision.pt"),
            "roi": asdict(request.roi),
            "condition_request_sha256": request._binding,
            "selection_role": (
                "engineering_probe" if engineering_probe else "selected_stage1"
            ),
        },
    )
    cache.put(condition, task_condition, config=config)
    loaded, pack = cache.load(
        sample_id, expected_binding=condition.binding, config=config
    )
    return loaded, ordinary_cpu(supervision), pack

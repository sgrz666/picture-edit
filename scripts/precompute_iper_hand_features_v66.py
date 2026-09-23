#!/usr/bin/env python3
"""Cache source-hand DINO patches from existing iPER V6 detail crops.

ViTPose/HaMeR outputs are accepted as independently-produced offline tensors;
target-frame RGB is never a runtime Hand Adapter input. This command never
downloads weights and does not overwrite an existing cache without --overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.v6.hand.quality import hand_quality_gate, merge_vitpose_keypoints
from src.pose_control.v6.hand.geometry import hand_heatmaps, local_hand_points


class OnDemandHandDINO:
    """Lazy local DINOv2-G/14 extractor. Call close() after caching new images."""

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.device = torch.device(device)
        self.processor = None
        self.model = None
        self._features: dict[str, torch.Tensor] = {}

    def _load(self):
        if self.model is None:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(str(self.path), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(self.path), local_files_only=True).eval().to(self.device)

    @torch.inference_mode()
    def extract(self, crop: torch.Tensor) -> torch.Tensor:
        from PIL import Image
        self._load()
        image = ((crop.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        output = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).last_hidden_state
        if output.shape[1:] != (257, 1536):
            raise RuntimeError(f"DINOv2-G must return [B,257,1536], got {tuple(output.shape)}")
        return output[0, 1:].half().cpu()

    def close(self):
        self.model = None
        self.processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def extract_cached(self, crop: torch.Tensor, *, source_key: str) -> torch.Tensor:
        """On first new-image request extract, retain CPU features, unload weights."""
        if source_key not in self._features:
            try:
                self._features[source_key] = self.extract(crop)
            finally:
                self.close()
        return self._features[source_key]


def build_cache(detail: dict, extractor: OnDemandHandDINO, *, vitpose: dict | None = None, hamer: dict | None = None) -> dict:
    mapping = detail["stem_to_idx"]
    stems = [stem for stem, _ in sorted(mapping.items(), key=lambda item: item[1])]
    n = len(stems)
    points = torch.as_tensor(detail["hand_keypoints"], dtype=torch.float32).clone()
    boxes = torch.as_tensor(detail["boxes"], dtype=torch.float32)[:, 1:].clone()
    valid = torch.as_tensor(detail["region_valid"], dtype=torch.bool)[:, 1:].clone()
    crops = torch.as_tensor(detail["reference_crops"], dtype=torch.float32)[:, 1:]
    if points.shape != (n, 2, 21, 3) or boxes.shape != (n, 2, 4) or valid.shape != (n, 2) or crops.shape != (n, 2, 3, 224, 224):
        raise ValueError("V6 detail cache hand shapes are incompatible")
    flagged = hand_quality_gate(points, boxes) & valid
    if vitpose is not None:
        refined = torch.as_tensor(vitpose["keypoints"], dtype=torch.float32)
        points = merge_vitpose_keypoints(points, refined, flagged)
    mesh_bps = torch.zeros(n, 2, 1024)
    mesh_valid = torch.zeros(n, 2, dtype=torch.bool)
    if hamer is not None:
        if torch.as_tensor(hamer["mesh_bps"]).shape != mesh_bps.shape:
            raise ValueError("HaMeR BPS must have shape [N,2,1024]")
        refined_valid = torch.as_tensor(hamer["valid"], dtype=torch.bool)
        if refined_valid.shape != valid.shape:
            raise ValueError("HaMeR valid must have shape [N,2]")
        accept = flagged & refined_valid
        mesh_bps[accept] = torch.as_tensor(hamer["mesh_bps"], dtype=torch.float32)[accept]
        mesh_valid = accept
    patches = torch.zeros(n, 2, 256, 1536, dtype=torch.float16)
    appearance_valid = torch.zeros(n, 2, dtype=torch.bool)
    failures = []
    for index, stem in enumerate(stems):
        for side in range(2):
            if not valid[index, side]:
                continue
            try:
                local = local_hand_points(points[index, side], boxes[index, side])
                if points[index, side, :, 2].mean() >= .35:
                    heatmaps = hand_heatmaps(local, size=56, sigma=2)
                    coarse = F.max_pool2d(heatmaps.amax(0)[None, None], 11, stride=1, padding=5)
                    source_mask = F.interpolate(coarse, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
                    source_crop = crops[index, side] * source_mask + (-1) * (1 - source_mask)
                else:
                    source_crop = crops[index, side]
                patches[index, side] = extractor.extract(source_crop)
                appearance_valid[index, side] = True
            except Exception as exc:
                failures.append({"stem": stem, "side": side, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "metadata": {"schema_version": 1, "dino_model": str(extractor.path), "quality_flagged": int(flagged.sum()), "hamer_accepted": int(mesh_valid.sum()), "extraction_failures": failures},
        "frame_stems": stems, "stem_to_idx": {stem: i for i, stem in enumerate(stems)},
        "dino_patches": patches, "reference_valid": appearance_valid,
        "hand_keypoints": points, "boxes": boxes, "region_valid": valid,
        "mesh_bps": mesh_bps, "mesh_valid": mesh_valid,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Precompute V6.6 iPER hand DINO features")
    parser.add_argument("--appearance", required=True)
    parser.add_argument("--assets-root", default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast")
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vitpose-refinements", help="optional precomputed mapping with keypoints [N,2,21,3]")
    parser.add_argument("--hamer-refinements", help="optional precomputed mapping with mesh_bps [N,2,1024] and valid [N,2]")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    folder = Path(args.assets_root) / args.appearance
    destination = folder / "v66_hand_features.pt"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(destination)
    source = folder / "v6_detail_conditions.pt"
    detail = torch.load(source, map_location="cpu", weights_only=False)
    extractor = OnDemandHandDINO(args.dino_model, args.device)
    try:
        vitpose = None if args.vitpose_refinements is None else torch.load(args.vitpose_refinements, map_location="cpu", weights_only=False)
        hamer = None if args.hamer_refinements is None else torch.load(args.hamer_refinements, map_location="cpu", weights_only=False)
        cache = build_cache(detail, extractor, vitpose=vitpose, hamer=hamer)
    finally:
        extractor.close()
    cache["metadata"]["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    cache["metadata"]["fingerprint"] = hashlib.sha256(cache["dino_patches"].numpy().tobytes()).hexdigest()
    torch.save(cache, destination)
    print(f"saved {destination}: {int(cache['reference_valid'].sum())} visible source-hand crops")


if __name__ == "__main__":
    main()

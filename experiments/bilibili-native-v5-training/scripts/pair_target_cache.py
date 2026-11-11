"""只复用最近一个 single-person driver 的原始条件；不缓存可训练特征。

Usage: cache = SingleTargetCache(assemble_full_cache)
       sample, local = cache.assemble(job, renderer=renderer)

Target tensors are shared read-only, exactly as the offline preparation expects.
Every returned metadata tree is independent. Unknown protocols use the original
assembler. This module never writes files or constructs a renderer/model.
"""
from copy import deepcopy
from dataclasses import replace
import json


_SOURCE_KEYS = frozenset(("pair_id", "source_asset", "source_canvas", "source_masks", "source_rgb",
                          "source_indices", "source_slot_ids", "references"))
_TARGET_KEYS = frozenset(("driver_asset", "target_canvas", "target_rgb", "units", "scale_status",
                          "dual_config", "bps_basis", "bps_seed", "expression_mode", "distances",
                          "surface_distance"))
_REQUIRED = _SOURCE_KEYS | frozenset(("driver_asset", "target_canvas", "target_rgb", "units", "scale_status"))


def _supported(job):
    return (isinstance(job, dict) and _REQUIRED <= job.keys()
            and not (job.keys() - _SOURCE_KEYS - _TARGET_KEYS)
            and job.get("source_indices") == [0, -1]
            and job.get("source_slot_ids") == [0, 1]
            and job.get("expression_mode") == "driver"
            and job.get("distances") == "absent_single_person")


def _copies(sample, local):
    """Copy mutable provenance, sharing only the expensive physical tensors."""
    result = dict(sample)
    for key in ("pair_metadata", "identity_metadata", "pose_provenance"):
        result[key] = deepcopy(sample[key])
    depth = sample["depth_condition_v2"]
    context = replace(depth.context, metadata=deepcopy(depth.context.metadata))
    result["depth_condition_v2"] = replace(depth, context=context, metadata=deepcopy(depth.metadata))
    return result, replace(local, metadata=deepcopy(local.metadata))


class SingleTargetCache:
    """One-entry cache for [source0,inactive] with the driver's own expression.

    hits/misses count supported single-person jobs; fallbacks count delegated
    protocols. Calls are sequential. Target tensors must not be mutated in place.
    """

    def __init__(self, original_assemble):
        self.original_assemble = original_assemble
        self.hits = self.misses = self.fallbacks = 0
        self._key = self._renderer = self._sample = self._local = None

    def assemble(self, job, *, renderer):
        if not _supported(job):
            self.fallbacks += 1
            return self.original_assemble(job, renderer=renderer)

        from src.pose_control.qwen21_full.native_full_assets import read_record
        source = read_record(job["source_asset"], schema="native_smplx_asset_v4")
        driver = read_record(job["driver_asset"], schema="native_smplx_asset_v4")
        if len(source["people"]) != 1 or len(driver["people"]) != 1:
            self.fallbacks += 1
            return self.original_assemble(job, renderer=renderer)

        try:
            key = json.dumps({name: value for name, value in job.items() if name not in _SOURCE_KEYS},
                             sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            self.fallbacks += 1
            return self.original_assemble(job, renderer=renderer)

        if key != self._key or renderer is not self._renderer:
            sample, local = self.original_assemble(job, renderer=renderer)
            cached_sample, cached_local = _copies(sample, local)
            self._key, self._renderer = key, renderer
            self._sample, self._local = cached_sample, cached_local
            self.misses += 1
            return sample, local

        result = self._rebind(job, source, driver, renderer)
        self.hits += 1
        return result

    def _rebind(self, job, source, driver, renderer):
        import numpy as np
        from PIL import Image
        import torch
        from src.data.canvas_contract import CanvasSpec
        from src.pose_control.qwen21_full.native_full_assets import read_record
        from src.pose_control.qwen21_full.native_full_data import _verified

        if job["units"] != driver["metadata"].get("units") or job["scale_status"] != driver["metadata"].get("scale_status"):
            raise ValueError("Depth scale must match imported geometry; metre needs an explicit bound calibration asset")
        source_spec = CanvasSpec.from_record(job["source_canvas"])
        target_spec = CanvasSpec.from_record(job["target_canvas"])
        if (source_spec.original_hw != tuple(source["metadata"]["original_hw"])
                or target_spec.original_hw != tuple(driver["metadata"]["original_hw"])):
            raise ValueError("Each fit must use its own original image and CanvasSpec")
        source_ids = [p["metadata"]["person_id"] for p in source["people"]]
        target_ids = [p["metadata"]["person_id"] for p in driver["people"]]
        if len(job["source_masks"]) != len(source_ids):
            raise ValueError("Declare one visible mask per real Source person")
        for asset, rgb in ((source, job["source_rgb"]), (driver, job["target_rgb"])):
            with Image.open(_verified(rgb)) as image:
                if (image.height, image.width) != tuple(asset["metadata"]["original_hw"]):
                    raise ValueError("Fit/RGB original dimensions differ")
            if asset["metadata"].get("rgb_sha256") != rgb["sha256"]:
                raise ValueError("Fit provenance belongs to another RGB frame")
        if source_ids[0] != target_ids[0]:
            raise ValueError("Stable source identity mapping differs")
        if job.get("bps_basis"):
            _verified(job["bps_basis"])

        sh, sw = source_spec.canvas_hw
        source_mask = torch.zeros(2, 1, sh, sw, dtype=torch.bool)
        source_head = torch.zeros_like(source_mask)
        for position, reference in enumerate(job["source_masks"]):
            with Image.open(_verified(reference)) as image:
                original = torch.from_numpy(np.asarray(image.convert("L")).copy() > 0)[None]
            mask = source_spec.stack(original.float(), nearest=True) > .5
            source_mask[position] = mask
            rendered = renderer.render(source["people"][position], source["K"], source_spec.original_hw)
            head = source_spec.stack(torch.from_numpy(rendered["part"] == 1)[None].float(), nearest=True) > .5
            source_head[position] = head & mask

        reference_meta = read_record(job["references"], schema="native_local_reference_v4").metadata[0]
        if reference_meta["reference_sources"][0]["geometry_sha256"] != job["source_asset"]["sha256"]:
            raise ValueError("Local Source ROI geometry differs from full Source asset")

        sample, local = _copies(self._sample, self._local)
        mapping = list(job["source_indices"])
        padded_source_ids = source_ids + ["__inactive__"]
        sample.update(source_canvas=source_spec.metadata(), source_token_hw=source_spec.token_hw,
                      source_person_mask=source_mask, source_head_mask=source_head,
                      source_indices=torch.tensor(mapping, dtype=torch.long))
        sample["pair_metadata"].update(pair_id=job["pair_id"], source_rgb_sha256=job["source_rgb"]["sha256"],
            source_asset_sha256=job["source_asset"]["sha256"])
        sample["identity_metadata"].update(source_canvas_id=source_spec.fingerprint(),
            source_rgb_sha256=job["source_rgb"]["sha256"], source_person_ids=list(padded_source_ids),
            source_indices=mapping, source_geometry_sha256=job["source_asset"]["sha256"],
            source_mask_sha256=[r["sha256"] for r in job["source_masks"]])

        record_id = job["pair_id"] + ":smplx:0"
        sample["pose_provenance"]["record_id"] = record_id
        depth = sample["depth_condition_v2"]
        for metadata in depth.metadata:
            metadata.update(source_id=record_id, pose_record_id=record_id)
        for metadata in depth.context.metadata:
            metadata["pose_record_id"] = record_id
        for metadata in local.metadata:
            metadata.update(source_canvas_id=source_spec.fingerprint(), source_rgb_sha256=job["source_rgb"]["sha256"],
                source_person_ids=list(padded_source_ids), source_asset_sha256=job["source_asset"]["sha256"],
                reference_sources=deepcopy(reference_meta["reference_sources"]))
        return sample, local.validate()

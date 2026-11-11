"""Align single-person local occlusion Z without changing published Body PNGs.

Compose as SingleTargetCache(LocalSceneDepth(original_assemble).assemble).
Only the current [source0,inactive], driver-expression protocol is handled;
other protocols and multi-person assets are delegated unchanged. No model is
loaded, no parameter is fitted, and no input/output file is written here.
"""
from copy import deepcopy
from dataclasses import replace


POLICY = "same_driver_full_mesh_float32_at_exact_target_canvas; body_png_values_unchanged"
_ALLOWED = frozenset(("pair_id", "source_asset", "driver_asset", "source_canvas", "target_canvas", "source_masks",
    "source_rgb", "target_rgb", "source_indices", "source_slot_ids", "units", "scale_status", "references",
    "expression_mode", "distances", "dual_config", "bps_basis", "bps_seed", "surface_distance"))


def _supported(job):
    return (isinstance(job, dict) and not (job.keys() - _ALLOWED)
            and {"source_asset", "driver_asset", "target_canvas"} <= job.keys()
            and type(job.get("source_indices")) is list and job["source_indices"] == [0, -1]
            and type(job.get("source_slot_ids")) is list and job["source_slot_ids"] == [0, 1]
            and job.get("expression_mode") == "driver" and job.get("distances") == "absent_single_person")


def _native_renderer(renderer):
    seen = set()
    while hasattr(renderer, "delegate"):
        if id(renderer) in seen:
            raise ValueError("Renderer delegate cycle cannot supply a trusted float scene")
        seen.add(id(renderer))
        renderer = renderer.delegate
    if not callable(getattr(renderer, "render", None)):
        raise ValueError("A native same-fit float renderer is required")
    return renderer


class LocalSceneDepth:
    """Use full-mesh float Z only for the local scene/submesh depth comparison."""

    def __init__(self, original_assemble):
        self.original_assemble = original_assemble
        self.aligned = self.fallbacks = 0

    def assemble(self, job, *, renderer):
        sample, local = self.original_assemble(job, renderer=renderer)
        if not _supported(job):
            self.fallbacks += 1
            return sample, local

        import numpy as np
        import torch
        from src.data.canvas_contract import CanvasSpec
        from src.pose_control.qwen21_full.native_full_assets import read_record

        source = read_record(job["source_asset"], schema="native_smplx_asset_v4")
        driver = read_record(job["driver_asset"], schema="native_smplx_asset_v4")
        if len(source["people"]) != 1 or len(driver["people"]) != 1:
            self.fallbacks += 1
            return sample, local
        if any(asset["metadata"].get("geometry_method") != "sapiens_v1" for asset in (source, driver)):
            self.fallbacks += 1
            return sample, local

        spec = CanvasSpec.from_record(job["target_canvas"])
        if spec.original_hw != tuple(driver["metadata"]["original_hw"]):
            raise ValueError("Local float scene must use the driver's exact original canvas")
        h, w = spec.canvas_hw
        if local.scene_z.shape != (1, 1, h, w) or local.scene_z.device.type != "cpu":
            raise ValueError("Offline local float scene requires one CPU target canvas")
        for metadata in local.metadata:
            if metadata.get("target_canvas_id") != spec.fingerprint():
                raise ValueError("Local float scene Canvas binding differs from the existing condition")

        engine = _native_renderer(renderer)
        camera = spec.camera(driver["K"]).astype(np.float32)
        surface = engine.render(driver["people"][0], camera, (h, w))
        z, visible = np.asarray(surface["z"]), np.asarray(surface["valid"])
        if z.shape != (h, w) or visible.shape != (h, w) or visible.dtype != np.bool_:
            raise ValueError("Native float scene depth/support shape or dtype differs")
        if not np.isfinite(z[visible]).all() or (z[visible] <= 0).any():
            raise ValueError("Native float scene requires positive finite Z on its real support")
        valid = torch.from_numpy(visible.copy())[None, None] & spec.valid_mask()[None].bool()
        scene = torch.from_numpy(z.astype(np.float32, copy=True))[None, None]
        scene = torch.where(valid, scene, 0.)
        metadata = deepcopy(local.metadata)
        provenance = {"policy": POLICY, "source_role": "driver",
                      "source_asset_sha256": job["driver_asset"]["sha256"],
                      "target_canvas_id": spec.fingerprint(), "renderer": type(engine).__name__}
        for item in metadata:
            item["local_scene_depth"] = dict(provenance)
        result = replace(local, scene_z=scene, scene_z_valid=valid, metadata=metadata).validate()
        self.aligned += 1
        return sample, result

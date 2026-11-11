"""Measure published v1 map versus same-fit Native rasterization, without training."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, "/home/shangguanrz/project/pic-edit")
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from src.pose_control.qwen21_full.native_full_assets import read_record, sha256
    from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer
    asset = read_record({"path": str(args.asset), "sha256": sha256(args.asset), "schema": "native_smplx_asset_v4"}, schema="native_smplx_asset_v4")
    original = asset["metadata"]["original_geometry"]
    depth = cv2.imread(original["depth"]["path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
    rendered = IndependentMeshRasterizer("cuda").render(asset["people"][0], asset["K"], depth.shape)
    published_mask, rendered_mask = depth > 0, rendered["valid"]
    shared = published_mask & rendered_mask
    outside = published_mask & ~rendered_mask
    distance = cv2.distanceTransform((~rendered_mask).astype(np.uint8), cv2.DIST_L2, 5)
    error = np.abs(depth[shared] - rendered["z"][shared])
    result = {"asset": str(args.asset), "geometry_method": "sapiens_v1", "hw": list(depth.shape),
        "published_pixels": int(published_mask.sum()), "rendered_pixels": int(rendered_mask.sum()),
        "outside_pixels": int(outside.sum()), "outside_max_distance_pixels": float(distance[outside].max()) if outside.any() else 0.,
        "silhouette_iou": float(shared.sum() / (published_mask | rendered_mask).sum()),
        "shared_depth_abs_error": {"median": float(np.median(error)), "p99": float(np.quantile(error, .99)), "max": float(error.max())},
        "smplx_model_sha256": asset["people"][0]["metadata"]["model_sha256"],
        "original_mesh_cam_max_difference": asset["metadata"]["original_mesh_cam_max_difference"]}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

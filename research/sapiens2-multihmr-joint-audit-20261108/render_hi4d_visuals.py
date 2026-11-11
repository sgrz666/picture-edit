"""Batch visualizer script for Sapiens 2 + MultiHMR Joint Optimization.
Renders standardized 6-strip / 2x3 grid comparison images for Hi4D pilot benchmark frames.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/data/shangguanrz/models/two-man")

from sapiens2_multihmr_joint.benchmark_runner import CANONICAL_PILOT_IDS
from sapiens2_multihmr_joint.config import JointOptimizationConfig
from sapiens2_multihmr_joint.pipeline import JointOptimizationPipeline
from sapiens2_multihmr_joint.vis_utils import render_6strip_comparison


def main():
    out_dir = "/data/shangguanrz/models/two-man/sapiens2_multihmr_joint/vis_results"
    os.makedirs(out_dir, exist_ok=True)

    config = JointOptimizationConfig()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = JointOptimizationPipeline(config=config, device=device)
    faces = pipeline.smplx_model.faces

    req_path = "/data/shangguanrz/pic-edit-experiments/mete-test/dual_pose_compare_20261029_v1/data/request.json"
    with open(req_path, "r") as f:
        req_data = json.load(f)

    records = [r for r in req_data["records"] if r["id"] in CANONICAL_PILOT_IDS]
    print(f"Rendering {len(records)} canonical pilot frames across categories...")

    mhmr672_pkl = "/data/shangguanrz/models/two-man/benchmark_results/M2_MultiHMR_672/predictions.pkl"
    with open(mhmr672_pkl, "rb") as f:
        m2_data = pickle.load(f)

    sapiens_dir = "/data/shangguanrz/models/two-man/benchmark_results/M3_Sapiens2_detr"

    generated_files = []

    for rec in tqdm(records, desc="Rendering Visual Strips"):
        rec_id = rec["id"]
        cat = rec.get("category", "separated")
        K_mat = np.array(rec["K"], dtype=np.float64)
        img_path = rec["image"]

        if not os.path.exists(img_path):
            continue

        # Load Sapiens 2D
        sap_file = os.path.join(sapiens_dir, f"{rec_id}.json")
        if os.path.exists(sap_file):
            with open(sap_file, "r") as sf:
                sap_instances = json.load(sf).get("instances", [])
        else:
            sap_instances = []

        # Load MultiHMR 3D
        m2_entry = m2_data.get(rec_id, {})
        m2_raw = m2_entry.get("predictions", [])

        mhmr_instances = []
        baseline_meshes = []
        for p in m2_raw:
            rotvec = np.asarray(p["rotvec"], dtype=np.float32)
            transl = np.asarray(p["transl_pelvis"], dtype=np.float32)
            shape = np.asarray(p["shape"], dtype=np.float32)

            with torch.no_grad():
                out0 = pipeline.smplx_model(betas=torch.from_numpy(shape.reshape(1, 10)).to(pipeline.device))
                pelvis_offset = out0.joints[0, 0].detach().cpu().numpy()
            transl_smplx = transl - pelvis_offset

            init_params = {
                "global_orient": torch.from_numpy(rotvec[0:1].reshape(1, 3)).to(pipeline.device),
                "body_pose": torch.from_numpy(rotvec[1:22].reshape(1, 63)).to(pipeline.device),
                "left_hand_pose": torch.from_numpy(rotvec[22:37].reshape(1, 45)).to(pipeline.device) if len(rotvec) > 36 else torch.zeros((1, 45), device=pipeline.device),
                "right_hand_pose": torch.from_numpy(rotvec[37:52].reshape(1, 45)).to(pipeline.device) if len(rotvec) > 51 else torch.zeros((1, 45), device=pipeline.device),
                "jaw_pose": torch.from_numpy(rotvec[52:53].reshape(1, 3)).to(pipeline.device) if len(rotvec) > 52 else torch.zeros((1, 3), device=pipeline.device),
                "betas": torch.from_numpy(shape.reshape(1, 10)).to(pipeline.device),
                "transl": torch.from_numpy(transl_smplx.reshape(1, 3)).to(pipeline.device),
            }

            with torch.no_grad():
                fwd_base = pipeline.optimizer.forward_smplx(init_params)
                v_base = fwd_base.vertices[0].detach().cpu().numpy()
                baseline_meshes.append({"vertices": v_base, "faces": faces})

            mhmr_instances.append({
                "init_params": init_params,
                "transl_pelvis": transl,
                "j3d": p["j3d"],
                "j2d": p["j2d_orig"],
                "bbox": p.get("bbox_orig", np.zeros(4, dtype=np.float32)),
            })

        # Run Joint Optimization
        res = pipeline.process_image(
            image=img_path,
            camera_K=K_mat,
            category=cat,
            sapiens_instances=sap_instances,
            mhmr_instances=mhmr_instances,
        )

        refined_meshes = []
        for p in res["predictions"]:
            refined_meshes.append({"vertices": p["vertices"], "faces": faces})

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Render 2x3 Grid View
        grid_vis = render_6strip_comparison(
            image=img_rgb,
            sapiens_instances=sap_instances,
            baseline_meshes=baseline_meshes,
            refined_meshes=refined_meshes,
            camera_K=K_mat,
            layout="grid",
            title=f"Hi4D [{cat.upper()}] Frame {rec_id}",
        )
        grid_file = os.path.join(out_dir, f"{cat}_{rec_id}_grid.jpg")
        cv2.imwrite(grid_file, cv2.cvtColor(grid_vis, cv2.COLOR_RGB2BGR))
        generated_files.append(grid_file)

    print(f"\nRendered {len(generated_files)} comparison images in {out_dir}:")
    for f in generated_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

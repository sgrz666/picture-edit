"""Verify translation and hand-mean conversion against the native SMPL-X layers."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
args = parser.parse_args()
directory = Path(args.dir).resolve()
base = Path("/home/shangguanrz/project/pic-edit")
repo = base / "add/tools/multi-hmr"
os.chdir(repo)
sys.path.insert(0, str(repo))
import numpy as np
import torch
import smplx
from blocks.smpl_layer import SMPL_Layer

multi = np.load(directory / "multihmr_hi4d.npz")
metadata = json.loads((directory / "multihmr_hi4d.json").read_text())
K = torch.tensor(multi["K_canvas"], dtype=torch.float32)
layer = SMPL_Layer(person_center="head").eval()
result = {"multi_flat_hand_mean": bool(layer.bm_x.flat_hand_mean), "multihmr_people": []}
with torch.no_grad():
    for i in range(metadata["person_count"]):
        def tensor(key):
            return torch.tensor(multi[f"person_{i}_{key}"], dtype=torch.float32).reshape(1, -1)

        pose = tensor("rotvec").reshape(1, 53, 3)
        betas, expr, head = tensor("shape"), tensor("expression"), tensor("transl")
        native = layer(pose, betas, None, None, head, K, expr)
        kwargs = dict(global_orient=pose[:, 0], body_pose=pose[:, 1:22].flatten(1),
                      left_hand_pose=pose[:, 22:37].flatten(1), right_hand_pose=pose[:, 37:52].flatten(1),
                      jaw_pose=pose[:, 52].flatten(1), betas=betas, expression=expr, transl=torch.zeros(1, 3))
        local = layer.bm_x(**kwargs)
        transl = native["transl_pelvis"][:, 0] - local.joints[:, 0]
        canonical = layer.bm_x(**dict(kwargs, transl=transl))
        wrong = layer.bm_x(**dict(kwargs, transl=native["transl_pelvis"][:, 0]))
        result["multihmr_people"].append({
            "index": i, "canonical_transl": transl.tolist(),
            "mesh_conversion_max_abs_m": float((canonical.vertices - native["v3d"]).abs().max()),
            "using_pelvis_as_transl_max_abs_m": float((wrong.vertices - native["v3d"]).abs().max()),
            "float32_vs_amp_native_mesh_max_abs_m": float((native["v3d"] - torch.tensor(multi[f"person_{i}_v3d"])).abs().max()),
        })
    single = np.load(directory / "smplestx_single.npz")
    bmdir = base / "add/tools/SMPLest-X/human_models/human_model_files"
    source = smplx.create(str(bmdir), "smplx", gender="neutral", use_pca=False, flat_hand_mean=False, use_face_contour=True)
    target = smplx.create(str(bmdir), "smplx", gender="neutral", use_pca=False, flat_hand_mean=True, use_face_contour=True)
    fields = {"global_orient": "smplx_root_pose", "body_pose": "smplx_body_pose",
              "left_hand_pose": "smplx_lhand_pose", "right_hand_pose": "smplx_rhand_pose",
              "jaw_pose": "smplx_jaw_pose", "betas": "smplx_shape", "expression": "smplx_expr", "transl": "cam_trans"}
    kwargs = {key: torch.tensor(single[value], dtype=torch.float32) for key, value in fields.items()}
    wrong = target(**kwargs)
    correct = target(**dict(kwargs, left_hand_pose=kwargs["left_hand_pose"] + source.left_hand_mean,
                           right_hand_pose=kwargs["right_hand_pose"] + source.right_hand_mean))
    expected = torch.tensor(single["smplx_mesh_cam"])
    result["smplestx_hand_mean_conversion"] = {
        "uncorrected_max_abs_m": float((wrong.vertices - expected).abs().max()),
        "corrected_max_abs_m": float((correct.vertices - expected).abs().max()),
    }
result["body_model_sha256"] = {}
for name, path in [("multihmr", repo / "models/smplx/SMPLX_NEUTRAL.npz"),
                   ("smplestx", bmdir / "smplx/SMPLX_NEUTRAL.npz")]:
    result["body_model_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
(directory / "convention_checks.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

"""S1: extract SHAPY artifacts, build the official HBW layout and config.

The only code patch is in the experiment's copy of the SHAPY repo:
  * BodyMeasurements is instantiated on CPU and called with vertices/faces,
    because the official CPU branch's forward() needs them explicitly.

PYTHONPATH must contain the shim at third_party/shapy_bm (only exposes the
CPU BodyMeasurements, avoiding the CUDA-only CWH extension import).
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import yaml

import config

WEIGHTS = config.RUN_ROOT / "data" / "shapy_weights"
UTILITY = config.UTILITY_DIR


def extract(z: zipfile.ZipFile, member: str, dest: Path):
    info = z.getinfo(member)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == info.file_size:
        return
    tmp = dest.with_name(dest.name + ".extracting")
    with z.open(info) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if tmp.stat().st_size != info.file_size:
        raise RuntimeError(f"size mismatch: {dest}")
    os.replace(tmp, dest)


def symlink(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main():
    with zipfile.ZipFile(config.SHAPY_ZIP) as z:
        names = set(z.namelist())
        wanted = {
            "trained_models/shapy/SHAPY_A/checkpoints/best_checkpoint":
                config.SHAPY_CKPT,
            "trained_models/b2a/polynomial/caesar-male_smplx-neutral-10betas/last.ckpt":
                WEIGHTS / "b2a" / "male_neutral.ckpt",
            "trained_models/b2a/polynomial/caesar-female_smplx-neutral-10betas/last.ckpt":
                WEIGHTS / "b2a" / "female_neutral.ckpt",
            "expose_release/data/all_means.pkl":
                WEIGHTS / "expose_release" / "data" / "all_means.pkl",
            "expose_release/data/SMPLX_to_J14.pkl":
                UTILITY / "SMPLX_to_J14.pkl",
            "expose_release/utility_files/flame/SMPL-X__FLAME_vertex_ids.npy":
                WEIGHTS / "expose_release" / "utility_files" / "flame" / "SMPL-X__FLAME_vertex_ids.npy",
            "utility_files/measurements/measurement_defitions.yaml":
                UTILITY / "measurement_defitions.yaml",
            "utility_files/measurements/smplx_measurements.yaml":
                UTILITY / "smplx_measurements.yaml",
            "utility_files/evaluation/eval_point_set/HD_SMPLX_from_SMPL.pkl":
                UTILITY / "HD_SMPLX_from_SMPL.pkl",
            "utility_files/smplx/smplx_correspondences.npz":
                UTILITY / "smplx" / "smplx_correspondences.npz",
            "utility_files/shape_priors/female_normal.npz":
                UTILITY / "shape_priors" / "female_normal.npz",
            "utility_files/shape_priors/male_normal.npz":
                UTILITY / "shape_priors" / "male_normal.npz",
        }
        for member, dest in wanted.items():
            if member not in names:
                raise RuntimeError(f"missing member in SHAPY zip: {member}")
            extract(z, member, dest)

    # Official HBW layout expected by human_bodies_in_the_wild.py
    symlink(config.SHAPY_HBW / "images" / "val", config.HBW_FULLRES / "images" / "val")
    symlink(config.SHAPY_HBW / "keypoints" / "val", config.KEYPOINTS)
    symlink(config.SHAPY_HBW / "smplx" / "val", config.DATA / "hbw_obj" / "val")
    symlink(config.SHAPY_HBW / "genders.yaml", config.GENDERS)
    body_models = config.SHAPY_HBW / "body_models" / "smplx"
    body_models.mkdir(parents=True, exist_ok=True)
    symlink(body_models / "SMPLX_NEUTRAL.npz", config.SMPLX_NEUTRAL)

    # Patch the experiment copy of the loader: CPU measurements + explicit
    # vertices/faces.  The shim package is selected through PYTHONPATH.
    loader = config.SHAPY_REPO / "regressor" / "human_shape" / "data" / "datasets" / "human_bodies_in_the_wild.py"
    text = loader.read_text(encoding="utf8")
    old_inst = "measurements = BodyMeasurements({"
    new_inst = "measurements = BodyMeasurements({"
    if "measurements = BodyMeasurements({" in text:
        text = text.replace(
            "measurements = BodyMeasurements({\n            'meas_definition_path': osp.join(meas_definition_path),\n            'meas_vertices_path': osp.join(meas_vertices_path),\n        }).to(device=self.device)",
            "measurements = BodyMeasurements({\n            'meas_definition_path': osp.join(meas_definition_path),\n            'meas_vertices_path': osp.join(meas_vertices_path),\n        }).to(device='cpu')",
        )
    old_call = "curr_subj_meas_val = measurements(curr_triangles)['measurements']"
    new_call = ("curr_subj_meas_val = measurements(\n"
                "                    curr_triangles.cpu(),\n"
                "                    vertices=curr_v_shaped.cpu(),\n"
                "                    faces=body_model_faces.unsqueeze(0).cpu(),\n"
                "                )['measurements']")
    if old_call in text:
        text = text.replace(old_call, new_call)
    if old_call in text or "measurements(curr_triangles)" in text:
        raise RuntimeError("loader patch did not apply; inspect the source")
    loader.write_text(text, encoding="utf8")

    # Config: start from the official eval config, override every path.
    src_cfg = config.SHAPY_REPO / "regressor" / "configs" / "b2a_expose_hrnet_eval_shape.yaml"
    cfg = yaml.safe_load(src_cfg.read_text(encoding="utf8"))
    cfg["part_key"] = "shape"
    cfg["output_folder"] = str(config.SHAPY_RUN)
    cfg["checkpoint_folder"] = "checkpoints"
    cfg["pretrained"] = ""
    cfg["datasets"]["batch_size"] = 8
    cfg["datasets"]["shape"]["splits"] = {"train": [], "val": ["hbw"], "test": []}
    cfg["datasets"]["shape"]["transforms"]["flip_prob"] = 0.0
    cfg["datasets"]["shape"]["vertex_flip_correspondences"] = str(UTILITY / "smplx" / "smplx_correspondences.npz")
    hbw = cfg["datasets"]["shape"]["hbw"]
    hbw.update({
        "data_folder": str(config.SHAPY_HBW),
        "img_folder": "images",
        "keyp_folder": "keypoints",
        "mesh_folder": "smplx",
        "gender_fname": "genders.yaml",
        "imgs_minimal": "",
        "keyps_minimal": "",
        "annot_fname": "",
        "body_model_folder": str(config.SHAPY_HBW / "body_models"),
        "meas_definition_path": str(UTILITY / "measurement_defitions.yaml"),
        "meas_vertices_path": str(UTILITY / "smplx_measurements.yaml"),
    })
    cfg["network"]["smplx"]["compute_measurements"] = False
    cfg["network"]["smplx"]["use_b2a"] = False
    cfg["network"]["smplx"]["meas_definition_path"] = str(UTILITY / "measurement_defitions.yaml")
    cfg["network"]["smplx"]["meas_vertices_path"] = str(UTILITY / "smplx_measurements.yaml")
    cfg["network"]["smplx"]["backbone"]["hrnet"]["pretrained_path"] = ""
    cfg["body_model"]["model_folder"] = str(config.SHAPY_HBW / "body_models")
    cfg["body_model"]["smplx"]["mean_pose_path"] = str(WEIGHTS / "expose_release" / "data" / "all_means.pkl")
    cfg["body_model"]["smplx"]["j14_regressor_path"] = str(UTILITY / "SMPLX_to_J14.pkl")
    cfg["body_model"]["smplx"]["head_verts_ids_path"] = str(
        WEIGHTS / "expose_release" / "utility_files" / "flame" / "SMPL-X__FLAME_vertex_ids.npy")
    cfg["j14_regressor_path"] = str(UTILITY / "SMPLX_to_J14.pkl")
    cfg["losses"]["body"]["shape"]["prior"]["gender_shape"]["female_stats_path"] = str(
        UTILITY / "shape_priors" / "female_normal.npz")
    cfg["losses"]["body"]["shape"]["prior"]["gender_shape"]["male_stats_path"] = str(
        UTILITY / "shape_priors" / "male_normal.npz")
    cfg["evaluation"]["body"]["p2p_t"]["input_point_regressor_path"] = str(UTILITY / "HD_SMPLX_from_SMPL.pkl")
    cfg["evaluation"]["body"]["p2p_t"]["target_point_regressor_path"] = str(UTILITY / "HD_SMPLX_from_SMPL.pkl")
    out_cfg = config.RUN_ROOT / "configs" / "shapy_eval.yaml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")

    config.SHAPY_RUN.mkdir(parents=True, exist_ok=True)
    print("[S1] checkpoint", config.SHAPY_CKPT, config.SHAPY_CKPT.stat().st_size)
    print("[S1] layout", config.SHAPY_HBW)
    print("[S1] config", out_cfg)
    print("[S1] loader patched")


if __name__ == "__main__":
    main()

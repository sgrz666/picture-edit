"""Build Native dual SMPL-X samples using the verified RGB and slot bindings."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qwen_cache import DATA, EVAL, EVAL_SHA, pinned
from select_pairs import read, ref, write

from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.conditions import PART_NAMES
from src.pose_control.qwen21_full.dual_hi4d_geometry import IndependentMeshRasterizer, SMPL_TO_PART
from src.pose_control.qwen21_full.native_full_assets import import_smplx, local_geometry, read_record, write_record
from src.pose_control.qwen21_full.native_full_data import BODY_SCHEMA, PAIR_SCHEMA, assemble_full_cache
from src.pose_control.qwen21_full.native_local_condition import NativeLocalReferencesV4, REFERENCE_SCHEMA, LOCAL_SCHEMA
from src.pose_control.qwen21_full.native_v5.contracts import DETAIL_SCHEMA, SUPERVISION_SCHEMA
from src.pose_control.qwen21_full.native_v5.data import source_detail_sidecar, training_sidecar, write_sidecar
from src.pose_control.qwen21_full.native_v5.losses import frozen_evaluators
from src.pose_control.qwen21_full.native_v7 import preservation_assets as preservation

SMPLX = DATA / "nihaochen/pose_1114/models/SMPLX_NEUTRAL_smplestx.npz"
OPTIONS = {"gender": "neutral", "use_pca": False, "num_pca_comps": 6, "flat_hand_mean": True,
           "num_expression_coeffs": 10, "use_face_contour": True}
PART_LAYOUT = ((0, 1), (1, 1), (0, 5), (0, 8), (1, 5), (1, 8))


def checked(bound):
    path = Path(bound["path"])
    if ref(path) != bound:
        raise ValueError(f"Selected Native input changed: {path}")
    return path


def save_mask(path, mask):
    with path.open("xb") as stream:
        Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return ref(path)


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return ref(path)


class AssetBuilder:
    def __init__(self, directory, face_models, appearance_module, *, enable_dino):
        import smplx
        from smplx.joint_names import JOINT_NAMES
        from importlib.util import module_from_spec, spec_from_file_location

        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.raster = IndependentMeshRasterizer("cuda")
        self.model = ref(SMPLX)
        model = smplx.SMPLX(str(SMPLX), batch_size=1, **OPTIONS).float().eval()
        lbs = model.lbs_weights.numpy()
        parts = np.array([*SMPL_TO_PART[:22], 0, 0, 0, *([4] * 15), *([7] * 15)])
        part_weights = np.stack([lbs[:, parts == i].sum(1) for i in range(14)], 1)
        segmentation = directory / "segmentation.npz"
        with segmentation.open("xb") as stream:
            np.savez_compressed(stream, face_part14=part_weights[model.faces].sum(1).argmax(1),
                                left_hand_vertices=np.flatnonzero(part_weights.argmax(1) == 4),
                                right_hand_vertices=np.flatnonzero(part_weights.argmax(1) == 7))
        hand_indices = []
        for side in ("left", "right"):
            names = [side + "_wrist"]
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                names.extend(side + "_" + finger + suffix for suffix in ("1", "2", "3", ""))
            hand_indices.append([JOINT_NAMES.index(name) for name in names])
        basis_sha = hashlib.sha256(model.expr_dirs.numpy().astype("<f4").tobytes()).hexdigest()
        self.import_options = {"model_options": OPTIONS, "model_file": str(SMPLX),
                               "model_sha256": self.model["sha256"], "segmentation": str(segmentation),
                               "expression_basis_sha256": basis_sha, "part_names": list(PART_NAMES),
                               "hand_joint_indices": hand_indices}
        self.evaluator = pinned(EVAL, EVAL_SHA)
        self.encoder = frozen_evaluators(str(EVAL), None, device="cuda", sha=[EVAL_SHA, None])[0]
        self.appearance = None
        self.dino_files = {}
        if enable_dino:
            spec = spec_from_file_location("bilibili_appearance", appearance_module)
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            dino = face_models / "dinov2-giant"
            self.dino_files = {str(path.relative_to(dino)): ref(path)["sha256"] for path in sorted(dino.iterdir())
                               if path.is_file() and path.suffix in (".bin", ".json", ".safetensors")}
            self.appearance = module.SourceFeatureExtractor({
                "arcface_model": str(face_models / "glintr100.onnx"),
                "arcface_sha256": ref(face_models / "glintr100.onnx")["sha256"],
                "dino_model": str(dino), "dino_files": self.dino_files, "device": "cuda"})
        self.arcface_sha = ref(face_models / "glintr100.onnx")["sha256"]
        self.frames = {}
        self.sources = {}

    def frame(self, frame, ids, note):
        key = (frame["rgb"]["sha256"], frame["id"], tuple(ids))
        if key in self.frames:
            return self.frames[key]
        suffix = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
        directory = self.directory / ("frame-" + suffix)
        directory.mkdir()
        fit = torch.load(checked(frame["files"]["smplx"]), map_location="cpu", weights_only=True)
        names = ("betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
                 "jaw_pose", "expression", "transl", "mesh_cam", "K")
        params = {name: fit[name].float().numpy() for name in names}
        params.update(leye_pose=np.zeros((2, 3), np.float32), reye_pose=np.zeros((2, 3), np.float32),
                      camera_R=np.eye(3, dtype=np.float32), camera_t=np.zeros(3, np.float32))
        param_path = directory / "original-fit.npz"
        with param_path.open("xb") as stream:
            np.savez_compressed(stream, **params)
        asset = import_smplx({"parameters": str(param_path), "original_hw": frame["hw"],
            "camera_id": f"xhs:{note}:camera", "scale_group_id": f"xhs:{note}:model",
            "provenance": {"rgb_sha256": frame["rgb"]["sha256"], "fitted_smplx": frame["files"]["smplx"],
                           "motion_backend": "Multi-HMR/PromptHMR/SDPose_opt_pen"},
            "people": [{**self.import_options, "person_id": person} for person in ids]})
        # These fits used zero expression and jaw. Face72 is limited to the observed
        # SDPose 68-point support via the precomputed paired 3D face projection.
        from smplx.joint_names import JOINT_NAMES
        if JOINT_NAMES[76] != "right_eye_brow1" or JOINT_NAMES[127] != "right_contour_1" or JOINT_NAMES[143] != "left_contour_1":
            raise ValueError("SMPL-X face joint convention differs")
        face_order = [*range(127, 144), *range(76, 127)]
        annotations = read(checked(frame["files"]["keypoints"]))["people"]
        for slot, person in enumerate(asset["people"]):
            person["face72_cam"][4:] = fit["joints_cam"][slot, face_order].numpy()
            person["face72_valid"][4:] = np.asarray(annotations[f"p{slot}"]["valid_mask134"][24:92], bool)
            person["metadata"]["face72_layout"] = "smplx137_65_136_first4_extras_then_dw68"
        bound = write_record(directory / "asset.npz", asset, schema="native_smplx_asset_v4")
        self.frames[key] = bound
        return bound

    @torch.no_grad()
    def reference(self, frame, asset, spec, ids):
        import cv2
        from insightface.utils import face_align

        key = (asset["sha256"], spec.fingerprint())
        if key in self.sources:
            return self.sources[key]
        directory = Path(asset["path"]).parent
        geometry = local_geometry(read_record(asset, schema="native_smplx_asset_v4"), spec, renderer=self.raster)
        with Image.open(checked(frame["rgb"])) as im:
            original = im.convert("RGB")
            image = spec.image(original, fill=(128, 128, 128))
        arrays = {}
        for name in ("owner", "part"):
            with Image.open(checked(frame["files"][name])) as mask:
                arrays[name] = np.asarray(mask)
            if arrays[name].shape != spec.original_hw:
                raise ValueError(f"Original owner/part grid differs: {frame['id']}")
        masks = [save_mask(directory / f"owner-{p}.png", arrays["owner"] == p + 1) for p in range(2)]
        details = [save_mask(directory / f"detail-{p}-{part}.png",
                             (arrays["owner"] == p + 1) & (arrays["part"] == part))
                   for p, part in PART_LAYOUT]
        bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
        crops = [face_align.norm_crop(bgr, landmark=np.asarray(points, np.float32), image_size=112)
                 for points in frame["points"]]
        pixels = torch.from_numpy(np.stack([crop[..., ::-1] for crop in crops]).copy()).permute(0, 3, 1, 2)
        features = self.encoder(pixels.to("cuda", dtype=torch.float32) / 127.5 - 1).cpu().reshape(2, 512)
        embeddings = torch.as_tensor(frame["features"], dtype=torch.float32)
        consistency = torch.nn.functional.cosine_similarity(features, embeddings)
        if not torch.isfinite(consistency).all() or consistency.min() < .995:
            raise ValueError(f"Frozen identity encoder / mining feature parity differs: {consistency.tolist()}")
        boxes = torch.from_numpy(geometry["boxes"]).float()
        patch_faces = torch.zeros((1, 2, 1, 256, 1536))
        patch_hands = torch.zeros((1, 2, 2, 1, 256, 1536))
        face_valid = torch.zeros((1, 2, 1), dtype=torch.bool)
        hand_valid = torch.zeros((1, 2, 2, 1), dtype=torch.bool)
        if self.appearance is not None:
            locations, images = [], []
            for p, part in PART_LAYOUT:
                slot = 0 if part == 1 else 1 if part == 5 else 2
                box = boxes[p, slot].tolist()
                if ((arrays["owner"] == p + 1) & (arrays["part"] == part)).any() and (
                        box[2] > max(0, box[0]) and box[3] > max(0, box[1]) and max(box[:2]) < 1):
                    locations.append((p, slot))
                    images.append(self.appearance._crop(image, box))
            if images:
                for (p, slot), tokens in zip(locations, self.appearance._dino_patches(images)):
                    if slot == 0:
                        patch_faces[0, p, 0], face_valid[0, p, 0] = tokens, True
                    else:
                        patch_hands[0, p, slot - 1, 0], hand_valid[0, p, slot - 1, 0] = tokens, True
        provenance = {"schema": REFERENCE_SCHEMA, "role": "source_only", "reference_count": 1,
            "source_rgb_sha256": frame["rgb"]["sha256"], "source_canvas_id": spec.fingerprint(),
            "source_person_ids": ids, "arcface_sha256": self.arcface_sha, "dino_files": self.dino_files,
            "arcface_preprocessing": "SCRFD original pixels; aligned RGB112; glintr100",
            "reference_sources": [{"rgb": frame["rgb"], "canvas": spec.metadata(), "canvas_id": spec.fingerprint(),
                                   "person_ids": ids, "geometry_sha256": asset["sha256"]}]}
        native = NativeLocalReferencesV4(embeddings[None, :, None], torch.ones(1, 2, 1, dtype=torch.bool),
            patch_faces, face_valid, patch_hands, hand_valid, boxes[None, :, 0], boxes[None, :, 1:], (provenance,)).validate()
        references = write_record(directory / "references.npz", native, schema=REFERENCE_SCHEMA)
        result = (references, masks, details, features)
        self.sources[key] = result
        return result


def assemble(root, selection, face_models, appearance_module, enable_dino):
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.60)
    inputs_file = root / "qwen-inputs.json"
    inputs = read(inputs_file)
    if inputs["selection"] != ref(selection):
        raise ValueError("Native and Qwen use different selection bytes")
    builder = AssetBuilder(root / "frames", face_models, appearance_module, enable_dino=enable_dino)
    rows, groups = defaultdict(list), defaultdict(dict)
    preservation_entries = {}
    selected = read(selection)["pairs"]
    for i, pair in enumerate(selected, 1):
        pid, split, note = (pair[k] for k in ("id", "split", "note"))
        source, target = pair["source"], pair["target"]
        mapping = pair["source_indices"]
        source_ids = [f"xhs:{note}:{p}" for p in range(2)]
        target_ids = [source_ids[p] for p in mapping]
        key = split + "\0" + pid
        condition = inputs["entries"][key]
        source_spec = CanvasSpec.from_record(condition["source_canvas"])
        target_spec = CanvasSpec.from_record(condition["target_canvas"])
        source_asset = builder.frame(source, source_ids, note)
        target_asset = builder.frame(target, target_ids, note)
        refs, masks, details, features = builder.reference(source, source_asset, source_spec, source_ids)
        directory = root / "native" / pid
        directory.mkdir(parents=True)
        sample, local = assemble_full_cache({"pair_id": pid, "source_asset": source_asset, "driver_asset": target_asset,
            "source_canvas": source_spec.metadata(), "target_canvas": target_spec.metadata(), "source_masks": masks,
            "source_rgb": source["rgb"], "target_rgb": target["rgb"], "source_indices": mapping,
            "source_slot_ids": [0, 1], "units": "model_unit", "scale_status": "model_scale",
            "references": refs, "expression_mode": "off", "surface_distance": "unavailable"},
            renderer=builder.raster)
        body = write_record(directory / "body.npz", sample, schema=BODY_SCHEMA)
        local_ref = write_record(directory / "local.npz", local, schema=LOCAL_SCHEMA)
        v4 = write_rows(directory / "v4.jsonl", [{"schema": PAIR_SCHEMA, "pair_id": pid, "scene_kind": "dual",
            "data_profile": "full_smplx", "body": body, "local": local_ref, "references": refs,
            "source_rgb": source["rgb"], "target_rgb": target["rgb"]}])
        sample.update(native_scene_kind="dual", local_condition_v4=local,
                      local_references_v4=read_record(refs, schema=REFERENCE_SCHEMA))
        detail = source_detail_sidecar(sample, {"availability": "available", "role": "primary_source_only",
            "support_kind": "visible_owned", "source_rgb_sha256": source["rgb"]["sha256"],
            "source_person_ids": source_ids, "masks": details})
        points = directory / "target-five-points.json"
        write(points, {"role": "training_only", "coordinate_space": "original_pixels",
                       "driver_rgb_sha256": target["rgb"]["sha256"], "target_person_ids": target_ids,
                       "landmarks5": target["points"], "valid": [True, True]})
        supervision = training_sidecar(sample, {"availability": "available", "record": ref(points)})
        detail_ref = write_sidecar(directory / "detail.npz", detail, DETAIL_SCHEMA)
        supervision_ref = write_sidecar(directory / "supervision.npz", supervision, SUPERVISION_SCHEMA)
        rows[split].append({"schema": "native_full_pair_v5", "pair_id": pid, "v4_manifest": v4,
                            "detail": detail_ref, "supervision": supervision_ref,
                            "qwen_condition": condition["condition"]})
        groups[split][pid] = pair["group"]
        metadata = preservation.sample_binding(sample, EVAL_SHA)
        metadata.update(background_margin_px=16, background_implementation=preservation.BACKGROUND_IMPLEMENTATION)
        support = (preservation.background_support(sample) if source_spec == target_spec else
                   torch.zeros(1, target_spec.token_hw[0] * target_spec.token_hw[1]))
        preservation_entries[key] = preservation.write_record(directory / "source-preservation.pt",
            features[mapping], torch.ones(2, dtype=torch.bool), support, metadata)
        if i == 1 or i % 20 == 0 or i == len(selected):
            print(f"Native dual assets {i}/{len(selected)} ({pid})", flush=True)
    for split, items in rows.items():
        write_rows(root / f"{split}.jsonl", items)
        write(root / f"{split}-groups.json", groups[split])
    write(root / "preservation.json", {"schema": preservation.INDEX_SCHEMA, "evaluator": builder.evaluator,
                                      "entries": preservation_entries})
    write(root / "native-ready.json", {"splits": {k: len(v) for k, v in rows.items()}, "smplx": builder.model,
          "selection": ref(selection), "dino_files": builder.dino_files, "identity_evaluator": builder.evaluator,
          "pose_backend": "sdpose", "contact_labels": "unavailable", "face_expression": "off"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--face-models", type=Path, required=True)
    parser.add_argument("--appearance-module", type=Path, required=True)
    parser.add_argument("--no-dino", action="store_true")
    args = parser.parse_args()
    assemble(args.root, args.selection, args.face_models, args.appearance_module, not args.no_dino)

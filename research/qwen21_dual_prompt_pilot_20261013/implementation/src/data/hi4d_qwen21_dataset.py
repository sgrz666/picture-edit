"""Hi4D N2C controls for the Qwen21 two-person adapter.

Only rendered target geometry is a control. Target RGB and GT person masks are
never read while constructing controls; GT masks are reserved for evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from src.pose_control.v6.conditions import ContactRelationBatch, TaskType


SIZE = 512
VERSION = "hi4d_qwen21_dual_v2"
# SMPL's 24 LBS joints -> the V6 fourteen-part convention (zero based).
SMPL_TO_V6 = np.array([1, 8, 11, 1, 9, 12, 1, 10, 13, 1, 10, 13,
                       0, 1, 1, 0, 2, 5, 3, 6, 4, 7, 4, 7], dtype=np.int64)


def letterbox_array(value: np.ndarray, *, nearest: bool = False, fill=0) -> np.ndarray:
    height, width = value.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    out_w, out_h = round(width * scale), round(height * scale)
    resized = cv2.resize(value, (out_w, out_h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    output = np.full((SIZE, SIZE) + value.shape[2:], fill, dtype=value.dtype)
    left, top = (SIZE - out_w) // 2, (SIZE - out_h) // 2
    output[top:top + out_h, left:left + out_w] = resized
    return output


def letterbox_points(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(SIZE / width, SIZE / height)
    out_w, out_h = round(width * scale), round(height * scale)
    result = xy.astype(np.float32).copy() * scale
    result[:, 0] += (SIZE - out_w) // 2
    result[:, 1] += (SIZE - out_h) // 2
    return result


def _rgb(path: str) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(letterbox_array(image).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1)


def _mask(path: str) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("L")) > 127
    return torch.from_numpy(letterbox_array(image.astype(np.uint8), nearest=True).copy()).float().unsqueeze(0)


def _bbox(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("valid Hi4D person has an empty rendered semantic mask")
    return np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=np.float32) / SIZE


def _orient_parts_to_semantic(parts: np.ndarray, semantic: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Detect a vertical raster orientation mismatch from supplied geometry.

    Some Hi4D camera renders have the SMPL mesh inverted vertically relative
    to the geometry NPZ's semantic raster. No RGB or GT person mask is used.
    """
    visible = parts > 0
    coverage = float((visible & semantic).sum()) / max(1, int(semantic.sum()))
    flipped = parts[::-1].copy()
    flipped_coverage = float(((flipped > 0) & semantic).sum()) / max(1, int(semantic.sum()))
    if flipped_coverage >= 0.8 and flipped_coverage > coverage + 0.25:
        return flipped, flipped_coverage, True
    return parts, coverage, False


def _heatmaps(path: str, native_width: int, native_height: int) -> np.ndarray:
    with np.load(path) as data:
        xy = data["xy_native"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
    # DWpose wholebody: 18 body joints, six feet landmarks and a derived mid-hip.
    points = letterbox_points(xy[:24], native_width, native_height)
    midhip = (points[8] + points[11]) / 2
    points = np.concatenate([points, midhip[None]], axis=0)
    scores = np.concatenate([scores[:24], [min(scores[8], scores[11])]])
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    heat = np.zeros((25, SIZE, SIZE), np.float32)
    for index, ((x, y), score) in enumerate(zip(points, scores)):
        if score >= 0.25 and 0 <= x < SIZE and 0 <= y < SIZE:
            heat[index] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5.0 ** 2))
    return heat


class Hi4DGeometryConverter:
    def __init__(self, smpl_model: str | Path, device: str = "cuda") -> None:
        import smplx
        import nvdiffrast.torch as dr
        model = smplx.create(str(smpl_model), model_type="smpl", gender="neutral")
        self.faces = torch.as_tensor(model.faces.astype(np.int32), device=device)
        self.parts = SMPL_TO_V6[np.asarray(model.lbs_weights.argmax(dim=1).cpu())]
        self.dr = dr
        self.context = dr.RasterizeCudaContext(device=torch.device(device))
        self.device = torch.device(device)

    def render_parts(self, verts: np.ndarray, extr: np.ndarray, K: np.ndarray,
                     width: int, height: int, semantic: np.ndarray) -> np.ndarray:
        camera = verts @ extr[:, :3].T + extr[:, 3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError("SMPL mesh has vertices behind the Hi4D camera")
        projected = camera @ K.T
        xy = letterbox_points(projected[:, :2] / projected[:, 2:3], width, height)
        ndc_x, ndc_y = xy[:, 0] * (2 / SIZE) - 1, 1 - xy[:, 1] * (2 / SIZE)
        depth = camera[:, 2]
        z_ndc = 2 * (depth - 0.1) / 20 - 1
        clip = np.stack([ndc_x * depth, ndc_y * depth, z_ndc * depth, depth], axis=1)
        positions = torch.from_numpy(clip).to(self.device, dtype=torch.float32)[None]
        raster, _ = self.dr.rasterize(self.context, positions, self.faces, resolution=(SIZE, SIZE))
        labels = F.one_hot(torch.as_tensor(self.parts, device=self.device), 14).float()[None]
        interpolated, _ = self.dr.interpolate(labels, raster, self.faces)
        result = interpolated[0].argmax(-1).cpu().numpy().astype(np.uint8) + 1
        result[raster[0, :, :, 3].cpu().numpy() <= 0] = 0
        result, coverage, flipped = _orient_parts_to_semantic(result, semantic)
        if flipped:
            print(f"Hi4D geometry-only SMPL part vertical flip, coverage={coverage:.1%}", flush=True)
        if coverage < 0.15:
            ry, rx = np.nonzero(result)
            sy, sx = np.nonzero(semantic)
            raise ValueError(f"SMPL part raster covers only {coverage:.1%}; raster bbox "
                             f"{(rx.min(), ry.min(), rx.max(), ry.max())}, semantic bbox "
                             f"{(sx.min(), sy.min(), sx.max(), sy.max())}; "
                             f"areas raster={len(rx)} semantic={len(sx)} overlap={int(((result > 0) & semantic).sum())}")
        # Fill minor raster/semantic edge gaps from the closest mesh part.
        missing = semantic & (result == 0)
        if missing.any():
            if not (result > 0).any():
                raise ValueError("SMPL part rendering is empty; check camera projection")
            nearest = distance_transform_edt(result == 0, return_distances=False, return_indices=True)
            result[missing] = result[tuple(axis[missing] for axis in nearest)]
        result[~semantic] = 0
        return result

    def convert(self, row: dict) -> dict[str, torch.Tensor]:
        with np.load(row["target_geometry_npz"]) as data:
            geo = {key: data[key] for key in data.files}
        with np.load(row["target_smpl"]) as data:
            smpl = {key: data[key] for key in data.files}
        with np.load(row["target_camera"]) as data:
            camera = {key: data[key] for key in data.files}
        camera_index = list(camera["ids"]).index(row["camera_id"])
        extr = camera["extrinsics"][camera_index].astype(np.float32)
        K = geo["K_runtime"].astype(np.float32)
        height, width = int(geo["height"]), int(geo["width"])
        semantic = [letterbox_array((geo[f"semantic_{side}"] > 0).astype(np.uint8), nearest=True).astype(bool)
                    for side in ("A", "B")]
        normal = letterbox_array(geo["normal_scene"].astype(np.float32))
        normal = np.clip(normal, -1, 1)
        if not np.isfinite(normal).all():
            raise ValueError("nonfinite Hi4D normal map")
        parts = [self.render_parts(smpl["verts"][person], extr, K, width, height, semantic[person])
                 for person in range(2)]
        source_boxes = [_bbox(mask) for mask in semantic]
        centers = [((box[:2] + box[2:]) / 2) for box in source_boxes]
        dimensions = [np.maximum(box[2:] - box[:2], 1 / SIZE) for box in source_boxes]
        roots = smpl["transl"].astype(np.float32) @ extr[:, :3].T + extr[:, 3]
        body_height = float(np.mean(np.ptp(smpl["verts"][:, :, 1], axis=1)))
        body_height = max(body_height, 0.5)
        delta = (roots[1] - roots[0]) / body_height
        yaw = float(smpl["global_orient"][1, 1] - smpl["global_orient"][0, 1])
        relative = np.concatenate([delta, centers[1] - centers[0],
                                   np.log(dimensions[1] / dimensions[0]),
                                   [np.linalg.norm(delta), np.sin(yaw), np.cos(yaw), np.sign(delta[2])]]).astype(np.float32)
        assert relative.shape == (11,)
        contact = np.stack([letterbox_array(geo[f"contact_{side}"].astype(np.float32)) for side in ("A", "B")])
        contact = np.nan_to_num(contact, nan=0, posinf=0, neginf=0).clip(0, 1)
        idx_a = np.flatnonzero(smpl["contact"][0] > 0)
        idx_b = np.flatnonzero(smpl["contact"][1] > 0)
        relation_rows = []
        if len(idx_a) and len(idx_b):
            tree = cKDTree(smpl["verts"][1, idx_b])
            distances, nearest_b = tree.query(smpl["verts"][0, idx_a], k=1)
            for ai, bi, distance in sorted(zip(idx_a, idx_b[nearest_b], distances), key=lambda x: x[2]):
                key = (int(self.parts[ai]), int(self.parts[bi]))
                if key not in {(a, b) for a, b, _ in relation_rows}:
                    relation_rows.append((key[0], key[1], float(distance / body_height)))
                if len(relation_rows) == 8:
                    break
        relations = {name: torch.zeros(8, dtype=torch.long) for name in
                     ("src_person", "src_part", "dst_person", "dst_part", "contact_type")}
        relations["dst_person"][:len(relation_rows)] = 1
        distance = torch.zeros(8, 1)
        valid = torch.zeros(8, dtype=torch.bool)
        for index, (part_a, part_b, dist) in enumerate(relation_rows):
            relations["src_part"][index] = part_a
            relations["dst_part"][index] = part_b
            distance[index, 0] = dist
            valid[index] = True
        result: dict[str, torch.Tensor] = {}
        for person, side in enumerate(("a", "b")):
            mask = semantic[person]
            part = parts[person]
            result[f"normal_{side}"] = torch.from_numpy((normal * mask[..., None]).copy()).permute(2, 0, 1)
            result[f"human_mask_{side}"] = torch.from_numpy(mask.copy()).float().unsqueeze(0)
            result[f"part_onehot_{side}"] = torch.from_numpy(np.stack([(part == index + 1) for index in range(14)]).astype(np.float32))
            result[f"pose_heatmap_{side}"] = torch.from_numpy(_heatmaps(row[f"target_dwpose_{side.upper()}"], width, height))
            depth = letterbox_array(geo[f"depth_{side.upper()}"].astype(np.float32))
            depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0) * mask
            result[f"depth_{side}"] = torch.from_numpy((depth / 10).copy()).unsqueeze(0)
            fx, fy, cx, cy = K[0, 0] / width, K[1, 1] / height, K[0, 2] / width, K[1, 2] / height
            rotvec, _ = cv2.Rodrigues(extr[:, :3])
            global_26 = np.concatenate([smpl["betas"][person], smpl["global_orient"][person],
                                        smpl["transl"][person], [fx, fy, cx, cy],
                                        rotvec.flatten(), extr[:, 3]]).astype(np.float32)
            result[f"smplx_global_{side}"] = torch.from_numpy(global_26)
        result["relative_geometry"] = torch.from_numpy(relative)
        result["contact_raster"] = torch.from_numpy(contact.copy())
        result["human_mask"] = torch.maximum(result["human_mask_a"], result["human_mask_b"])
        action = row["target_action"].lower()
        result["task_id"] = torch.tensor(int(TaskType.HUG if action.startswith("hug") else TaskType.DUAL), dtype=torch.long)
        result.update({f"contact_{name}": value for name, value in relations.items()})
        result["contact_distance"] = distance
        result["contact_valid_mask"] = valid
        result["person_b_valid"] = torch.tensor(True)
        return result


class Hi4DQwen21Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest: str | Path, cache_dir: str | Path, *, load_target: bool = True):
        self.manifest = Path(manifest)
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        self.cache_dir = Path(cache_dir)
        self.load_target = load_target
        if not self.rows:
            raise ValueError("empty Hi4D manifest")

    def __len__(self) -> int:
        return len(self.rows)

    def control_path(self, row: dict) -> Path:
        key = hashlib.sha256((VERSION + row["target_geometry_npz"] + row["target_dwpose_A"] + row["target_dwpose_B"]).encode()).hexdigest()[:24]
        return self.cache_dir / f"control_{key}.pt"

    def precompute(self, converter: Hi4DGeometryConverter) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(self.rows):
            path = self.control_path(row)
            if not path.exists():
                data = converter.convert(row)
                data = {key: value.half() if value.is_floating_point() else value
                        for key, value in data.items()}
                temporary = path.with_suffix(".tmp")
                torch.save(data, temporary)
                temporary.replace(path)
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(self.rows):
                print(f"Hi4D controls {index + 1}/{len(self.rows)}", flush=True)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self.control_path(row)
        if not path.exists():
            raise FileNotFoundError(f"missing control cache {path}; run precompute first")
        control = torch.load(path, map_location="cpu", weights_only=True)
        sample = {**control, "src_image": _rgb(row["source_rgb"]),
                  "source_mask_a": _mask(row["source_person_mask_A"]),
                  "source_mask_b": _mask(row["source_person_mask_B"]), "row": row}
        if self.load_target:
            sample["tgt_image"] = _rgb(row["target_rgb"])
        return sample


def collate_hi4d(samples: list[dict]) -> dict:
    batch = {key: torch.stack([sample[key] for sample in samples]) if torch.is_tensor(samples[0][key])
             else [sample[key] for sample in samples] for key in samples[0]}
    batch["contact_relations"] = ContactRelationBatch(
        src_person=batch.pop("contact_src_person"), src_part=batch.pop("contact_src_part"),
        dst_person=batch.pop("contact_dst_person"), dst_part=batch.pop("contact_dst_part"),
        contact_type=batch.pop("contact_contact_type"), distance=batch.pop("contact_distance"),
        valid_mask=batch.pop("contact_valid_mask"))
    return batch

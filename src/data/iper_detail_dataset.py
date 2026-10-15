import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.pose_control.v6.detail.conditions import (
    DetailReferenceBatch,
    FaceHandDetailCondition,
)


class IPERDetailDataset(Dataset):
    """PyTorch Dataset for iPER Detail Branch (V6.4) Single-Sample and Multi-Sample training.

    Preserves full compatibility with V6.3 Geometry branch conditions while attaching:
    - FaceHandDetailCondition: 2-person slot structure (slot 0 valid, slot 1 invalid zero)
    - DetailReferenceBatch: 2-person slot structure with R=1 crop references
    - Precomputed diagnostic targets for validation/evaluation
    """

    def __init__(
        self,
        sampled_root: str,
        assets_root: str,
        pairs_jsonl: Optional[str] = None,
        single_sample: Optional[Dict[str, str]] = None,
        resolution: int = 512,
        augment: bool = False,
        heatmap_sigma: float = 4.0,
    ):
        """Initializes the IPERDetailDataset.

        Args:
            sampled_root: Root directory of iper_sampled_6src64tgt.
            assets_root: Root directory of iper_assets_512_nvdiffrast.
            pairs_jsonl: Optional path to JSONL file containing pairs.
            single_sample: Optional dict specifying fixed single sample, e.g.:
                {
                    "appearance": "024_6",
                    "source_stem": "source_motion_f000375",
                    "source_role": "source_motion",
                    "target_stem": "target_f001492",
                    "target_role": "target",
                }
            resolution: Target image resolution (default 512).
            augment: Whether to apply data augmentation (False for single_overfit).
            heatmap_sigma: Gaussian sigma for pose heatmaps.
        """
        self.sampled_root = sampled_root
        self.assets_root = assets_root
        self.resolution = resolution
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma

        self.samples: List[Dict[str, Any]] = []
        if single_sample is not None:
            self.samples.append(single_sample)
        elif pairs_jsonl is not None:
            with open(pairs_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            # Default to the fixed single sample for 024_6
            self.samples.append({
                "appearance": "024_6",
                "source": {"stem": "source_motion_f000375", "role": "source_motion"},
                "target": {"stem": "target_f001492", "role": "target"},
            })

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.resolution, dtype=torch.float32),
            torch.arange(self.resolution, dtype=torch.float32),
            indexing="ij",
        )
        self.register_grid_x = x_grid.unsqueeze(0)
        self.register_grid_y = y_grid.unsqueeze(0)

        self._v6_condition_cache: Dict[str, Any] = {}
        self._v6_detail_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_v6_conditions(self, appearance: str) -> Optional[Dict[str, Any]]:
        if appearance not in self._v6_condition_cache:
            path = os.path.join(self.assets_root, appearance, "v6_conditions.pt")
            if os.path.exists(path):
                self._v6_condition_cache[appearance] = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            else:
                self._v6_condition_cache[appearance] = None
        return self._v6_condition_cache[appearance]

    def _get_v6_detail_conditions(self, appearance: str) -> Optional[Dict[str, Any]]:
        if appearance not in self._v6_detail_cache:
            path = os.path.join(self.assets_root, appearance, "v6_detail_conditions.pt")
            if os.path.exists(path):
                self._v6_detail_cache[appearance] = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            else:
                self._v6_detail_cache[appearance] = None
        return self._v6_detail_cache[appearance]

    def _render_pose_heatmap(self, kps_25: torch.Tensor) -> torch.Tensor:
        x_px = kps_25[:, 0:1, None] * self.resolution
        y_px = kps_25[:, 1:2, None] * self.resolution
        score = kps_25[:, 2:3, None]

        d2 = (self.register_grid_x - x_px) ** 2 + (self.register_grid_y - y_px) ** 2
        heatmap = score * torch.exp(-d2 / (2.0 * (self.heatmap_sigma ** 2)))

        valid = (score > 0.05).expand(-1, self.resolution, self.resolution)
        heatmap = torch.where(valid, heatmap, torch.zeros_like(heatmap))
        return heatmap.clamp(0.0, 1.0)

    def _load_image(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _load_grayscale(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")

    def get_detail_condition_and_reference(
        self,
        appearance: str,
        source_stem: str,
        target_stem: str,
    ) -> Tuple[FaceHandDetailCondition, DetailReferenceBatch]:
        """Builds FaceHandDetailCondition and DetailReferenceBatch for a given pair.

        Both use batch_size = 1.
        """
        detail_data = self._get_v6_detail_conditions(appearance)
        if detail_data is None:
            raise FileNotFoundError(f"Missing v6_detail_conditions.pt for appearance {appearance}")

        stem_to_idx = detail_data["stem_to_idx"]
        src_idx = stem_to_idx[source_stem]
        tgt_idx = stem_to_idx[target_stem]

        # Target Detail Geometry
        # Person slot 0
        tgt_face_kps = detail_data["face_keypoints"][tgt_idx]  # [68, 3]
        tgt_hand_kps = detail_data["hand_keypoints"][tgt_idx]  # [2, 21, 3]
        tgt_smplx_detail = detail_data["smplx_detail"][tgt_idx]# [103]
        src_boxes = detail_data["boxes"][src_idx]               # [3, 4]
        tgt_boxes = detail_data["boxes"][tgt_idx]               # [3, 4]
        tgt_region_valid = detail_data["region_valid"][tgt_idx] # [3] bool

        # Assemble into 2-person slot format: [1, 2, ...]
        face_kps_t = torch.zeros(1, 2, 68, 3, dtype=torch.float32)
        face_kps_t[0, 0] = tgt_face_kps

        hand_kps_t = torch.zeros(1, 2, 2, 21, 3, dtype=torch.float32)
        hand_kps_t[0, 0] = tgt_hand_kps

        smplx_detail_t = torch.zeros(1, 2, 103, dtype=torch.float32)
        smplx_detail_t[0, 0] = tgt_smplx_detail

        src_boxes_t = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
        src_boxes_t[0, 0] = src_boxes

        tgt_boxes_t = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
        tgt_boxes_t[0, 0] = tgt_boxes

        region_valid_t = torch.zeros(1, 2, 3, dtype=torch.bool)
        region_valid_t[0, 0] = tgt_region_valid

        source_indices_t = torch.tensor([[0, 1]], dtype=torch.long)

        condition = FaceHandDetailCondition(
            face_keypoints=face_kps_t,
            hand_keypoints=hand_kps_t,
            smplx_detail=smplx_detail_t,
            source_boxes=src_boxes_t,
            target_boxes=tgt_boxes_t,
            region_valid=region_valid_t,
            source_indices=source_indices_t,
        ).validate()

        # Source Appearance References: [1, 2, 3, 1, 3, 224, 224]
        src_crops = detail_data["reference_crops"][src_idx] # [3, 3, 224, 224]
        src_region_valid = detail_data["region_valid"][src_idx] # [3]

        ref_images_t = torch.zeros(1, 2, 3, 1, 3, 224, 224, dtype=torch.float32)
        ref_images_t[0, 0, :, 0] = src_crops

        ref_valid_t = torch.zeros(1, 2, 3, 1, dtype=torch.bool)
        ref_valid_t[0, 0, :, 0] = src_region_valid

        references = DetailReferenceBatch(
            images=ref_images_t,
            reference_valid=ref_valid_t,
        ).validate()

        return condition, references

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        appearance = sample.get("appearance", "024_6")

        if "source" in sample and isinstance(sample["source"], dict):
            source_stem = sample["source"]["stem"]
            source_role = sample["source"].get("role", "source_motion")
            source_rel = sample["source"].get("rgb_1024", f"rgb_1024/{source_role}/{source_stem}.png")
        else:
            source_stem = sample.get("source_stem", "source_motion_f000375")
            source_role = sample.get("source_role", "source_motion")
            source_rel = f"rgb_1024/{source_role}/{source_stem}.png"

        if "target" in sample and isinstance(sample["target"], dict):
            target_stem = sample["target"]["stem"]
            target_role = sample["target"].get("role", "target")
            target_rel = sample["target"].get("rgb_1024", f"rgb_1024/{target_role}/{target_stem}.png")
        else:
            target_stem = sample.get("target_stem", "target_f001492")
            target_role = sample.get("target_role", "target")
            target_rel = f"rgb_1024/{target_role}/{target_stem}.png"

        # Load RGB images
        src_path = os.path.join(self.sampled_root, appearance, source_rel)
        tgt_path = os.path.join(self.sampled_root, appearance, target_rel)

        src_img = self._load_image(src_path)
        tgt_img = self._load_image(tgt_path)

        # Load V6.3 assets
        normal_path = os.path.join(self.assets_root, appearance, "normal_512", target_role, f"{target_stem}.png")
        mask_path = os.path.join(self.assets_root, appearance, "mask_512", target_role, f"{target_stem}.png")
        part_path = os.path.join(self.assets_root, appearance, "part_512", target_role, f"{target_stem}.png")
        depth_vis_path = os.path.join(self.assets_root, appearance, "depth_vis_512", target_role, f"{target_stem}.png")

        normal_img = self._load_image(normal_path)
        mask_img = self._load_grayscale(mask_path)
        part_img = Image.open(part_path) if os.path.exists(part_path) else Image.new("L", (self.resolution, self.resolution), 0)

        if os.path.exists(depth_vis_path):
            depth_img = self._load_grayscale(depth_vis_path)
        else:
            depth_img = Image.new("L", (self.resolution, self.resolution), 0)

        # V6.3 condition bundle
        v6_data = self._get_v6_conditions(appearance)
        if v6_data is not None and target_stem in v6_data.get("stem_to_idx", {}):
            frame_idx = v6_data["stem_to_idx"][target_stem]
            smplx_global = v6_data["smplx_global"][frame_idx].clone()
            kps_25_coords = v6_data["kps_25_coords"][frame_idx].clone()
        else:
            smplx_global = torch.zeros(26, dtype=torch.float32)
            kps_25_coords = torch.zeros((25, 3), dtype=torch.float32)

        # RGB tensors: [-1, 1]
        src_tensor = self.rgb_transform(src_img)
        tgt_tensor = self.rgb_transform(tgt_img)

        # Resize for control images
        resize_nearest = transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST)
        to_tensor = transforms.ToTensor()

        mask_tensor = to_tensor(resize_nearest(mask_img))
        normal_raw = to_tensor(transforms.Resize((self.resolution, self.resolution))(normal_img))
        depth_tensor = to_tensor(transforms.Resize((self.resolution, self.resolution))(depth_img))

        # 1. Normal: scale from [0, 1] to [-1, 1], mask background to 0.0
        normal_v6 = (normal_raw * 2.0 - 1.0) * mask_tensor

        # 2. Part Map One-Hot: [14, H, W]
        part_arr = np.array(resize_nearest(part_img), dtype=np.int64)
        part_t = torch.from_numpy(part_arr)
        part_onehot = F.one_hot(part_t, num_classes=15)[:, :, 1:].permute(2, 0, 1).float()

        # 3. Pose Heatmap: [25, H, W] in [0, 1]
        pose_heatmap = self._render_pose_heatmap(kps_25_coords)

        # Detail conditions from v6_detail_conditions.pt
        detail_data = self._get_v6_detail_conditions(appearance)
        if detail_data is not None:
            stem_to_idx = detail_data["stem_to_idx"]
            s_idx = stem_to_idx[source_stem]
            t_idx = stem_to_idx[target_stem]

            f_kps = detail_data["face_keypoints"][t_idx]        # [68, 3]
            h_kps = detail_data["hand_keypoints"][t_idx]        # [2, 21, 3]
            smplx_det = detail_data["smplx_detail"][t_idx]      # [103]
            s_boxes = detail_data["boxes"][s_idx]               # [3, 4]
            t_boxes = detail_data["boxes"][t_idx]               # [3, 4]
            r_valid = detail_data["region_valid"][t_idx]        # [3] bool

            ref_crops = detail_data["reference_crops"][s_idx]   # [3, 3, 224, 224]
            ref_valid = detail_data["region_valid"][s_idx]      # [3] bool
        else:
            f_kps = torch.zeros(68, 3, dtype=torch.float32)
            h_kps = torch.zeros(2, 21, 3, dtype=torch.float32)
            smplx_det = torch.zeros(103, dtype=torch.float32)
            s_boxes = torch.zeros(3, 4, dtype=torch.float32)
            t_boxes = torch.zeros(3, 4, dtype=torch.float32)
            r_valid = torch.zeros(3, dtype=torch.bool)
            ref_crops = torch.zeros(3, 3, 224, 224, dtype=torch.float32)
            ref_valid = torch.zeros(3, dtype=torch.bool)

        # 2-person slot tensors (slot 0 populated, slot 1 zero)
        detail_face_kps = torch.zeros(2, 68, 3, dtype=torch.float32)
        detail_face_kps[0] = f_kps

        detail_hand_kps = torch.zeros(2, 2, 21, 3, dtype=torch.float32)
        detail_hand_kps[0] = h_kps

        detail_smplx = torch.zeros(2, 103, dtype=torch.float32)
        detail_smplx[0] = smplx_det

        detail_src_boxes = torch.zeros(2, 3, 4, dtype=torch.float32)
        detail_src_boxes[0] = s_boxes

        detail_tgt_boxes = torch.zeros(2, 3, 4, dtype=torch.float32)
        detail_tgt_boxes[0] = t_boxes

        detail_region_valid = torch.zeros(2, 3, dtype=torch.bool)
        detail_region_valid[0] = r_valid

        detail_source_indices = torch.tensor([0, 1], dtype=torch.long)

        # References: [2, 3, 1, 3, 224, 224]
        detail_ref_images = torch.zeros(2, 3, 1, 3, 224, 224, dtype=torch.float32)
        detail_ref_images[0, :, 0] = ref_crops

        detail_ref_valid = torch.zeros(2, 3, 1, dtype=torch.bool)
        detail_ref_valid[0, :, 0] = ref_valid

        return {
            "src_image": src_tensor,
            "tgt_image": tgt_tensor,
            "normal": normal_v6,
            "depth": depth_tensor,
            "part_onehot": part_onehot,
            "pose_heatmap": pose_heatmap,
            "smplx_global": smplx_global,
            "human_mask": mask_tensor,
            "task_id": torch.tensor(0, dtype=torch.long),
            "appearance": appearance,
            "source_stem": source_stem,
            "target_stem": target_stem,
            # Detail conditions
            "detail_face_kps": detail_face_kps,
            "detail_hand_kps": detail_hand_kps,
            "detail_smplx_detail": detail_smplx,
            "detail_src_boxes": detail_src_boxes,
            "detail_tgt_boxes": detail_tgt_boxes,
            "detail_region_valid": detail_region_valid,
            "detail_source_indices": detail_source_indices,
            "detail_ref_images": detail_ref_images,
            "detail_ref_valid": detail_ref_valid,
        }

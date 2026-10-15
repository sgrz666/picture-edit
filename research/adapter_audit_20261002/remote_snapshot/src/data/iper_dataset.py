import json
import os
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 14 body part swap pairs for horizontal flip
PART_FLIP_SWAP = {
    2: 5, 5: 2,   # left/right upper arm
    3: 6, 6: 3,   # left/right lower arm
    4: 7, 7: 4,   # left/right hand
    8: 11, 11: 8, # left/right upper leg
    9: 12, 12: 9, # left/right lower leg
    10: 13, 13: 10 # left/right foot
}

# 25 keypoint swap pairs for horizontal flip
KEYPOINT_FLIP_SWAP = {
    2: 5, 5: 2,   # RShoulder <-> LShoulder
    3: 6, 6: 3,   # RElbow <-> LElbow
    4: 7, 7: 4,   # RWrist <-> LWrist
    9: 12, 12: 9, # RHip <-> LHip
    10: 13, 13: 10, # RKnee <-> LKnee
    11: 14, 14: 11, # RAnkle <-> LAnkle
    15: 16, 16: 15, # REye <-> LEye
    17: 18, 18: 17, # REar <-> LEar
    19: 22, 22: 19, # LBigToe <-> RBigToe
    20: 23, 23: 20, # LSmallToe <-> RSmallToe
    21: 24, 24: 21, # LHeel <-> RHeel
}


class IPERPoseDataset(Dataset):
    """PyTorch Dataset for iPER pose editing training with DeepGen V6.3 Native condition support.
    
    Loads source and target image pairs along with:
    1. V6.3 Native conditions:
       - normal: [3, H, W] in [-1, 1]
       - depth: [1, H, W] in [0, 1]
       - part_onehot: [14, H, W] in {0, 1}
       - pose_heatmap: [25, H, W] in [0, 1]
       - smplx_global: [26] float32
       - human_mask: [1, H, W] in [0, 1]
       - task_id: scalar tensor (TaskType.SINGLE = 0)
    2. Legacy 8-channel control map: normal(3) + depth(1) + dwpose(3) + mask(1)
    """

    def __init__(
        self,
        pairs_jsonl: str,
        sampled_root: str,
        assets_root: str,
        resolution: int = 512,
        augment: bool = False,
        heatmap_sigma: float = 4.0,
    ):
        """Initializes the IPERPoseDataset.
        
        Args:
            pairs_jsonl (str): Path to the JSONL file containing image pairs.
            sampled_root (str): Root directory of iper_sampled_6src64tgt.
            assets_root (str): Root directory of iper_assets_512_nvdiffrast.
            resolution (int, optional): Target resolution for all images. Defaults to 512.
            augment (bool, optional): Whether to apply random horizontal flip. Defaults to False.
            heatmap_sigma (float, optional): Gaussian sigma for pose heatmaps. Defaults to 4.0.
        """
        self.pairs_jsonl = pairs_jsonl
        self.sampled_root = sampled_root
        self.assets_root = assets_root
        self.resolution = resolution
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma

        self.samples = []
        with open(self.pairs_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Precompute coordinate grids for fast vectorized Gaussian rendering
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.resolution, dtype=torch.float32),
            torch.arange(self.resolution, dtype=torch.float32),
            indexing='ij'
        )
        self.register_grid_x = x_grid.unsqueeze(0)  # [1, H, W]
        self.register_grid_y = y_grid.unsqueeze(0)  # [1, H, W]
        
        # In-memory cache for per-appearance v6_conditions.pt
        self._v6_condition_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_v6_conditions(self, appearance: str) -> Optional[Dict[str, Any]]:
        """Loads and caches v6_conditions.pt for a given appearance."""
        if appearance not in self._v6_condition_cache:
            path = os.path.join(self.assets_root, appearance, "v6_conditions.pt")
            if os.path.exists(path):
                self._v6_condition_cache[appearance] = torch.load(
                    path, map_location="cpu", weights_only=False
                )
            else:
                self._v6_condition_cache[appearance] = None
        return self._v6_condition_cache[appearance]

    def _render_pose_heatmap(self, kps_25: torch.Tensor) -> torch.Tensor:
        """Render 25-channel Gaussian heatmap from normalized coordinates.
        
        Args:
            kps_25: [25, 3] float tensor with (x_norm, y_norm, score).
            
        Returns:
            [25, H, W] float32 tensor in [0, 1].
        """
        x_px = kps_25[:, 0:1, None] * self.resolution  # [25, 1, 1]
        y_px = kps_25[:, 1:2, None] * self.resolution  # [25, 1, 1]
        score = kps_25[:, 2:3, None]                   # [25, 1, 1]
        
        d2 = (self.register_grid_x - x_px) ** 2 + (self.register_grid_y - y_px) ** 2
        heatmap = score * torch.exp(-d2 / (2.0 * (self.heatmap_sigma ** 2)))
        
        valid = (score > 0.05).expand(-1, self.resolution, self.resolution)
        heatmap = torch.where(valid, heatmap, torch.zeros_like(heatmap))
        return heatmap.clamp(0.0, 1.0)

    def _load_image(self, path: str) -> Image.Image:
        """Loads an image from path and converts to RGB."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _load_grayscale(self, path: str) -> Image.Image:
        """Loads an image from path and converts to Grayscale (L)."""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        appearance = sample['appearance']
        source_stem = sample['source']['stem']
        target_stem = sample['target']['stem']
        target_role = sample['target'].get('role', 'target')

        # Load RGB images
        src_path = os.path.join(self.sampled_root, appearance, sample['source']['rgb_1024'])
        tgt_path = os.path.join(self.sampled_root, appearance, sample['target']['rgb_1024'])
        
        src_img = self._load_image(src_path)
        tgt_img = self._load_image(tgt_path)

        # Load base assets
        normal_path = os.path.join(self.assets_root, appearance, 'normal_512', target_role, f'{target_stem}.png')
        dwpose_path = os.path.join(self.assets_root, appearance, 'dwpose_512', target_role, f'{target_stem}.png')
        mask_path = os.path.join(self.assets_root, appearance, 'mask_512', target_role, f'{target_stem}.png')
        part_path = os.path.join(self.assets_root, appearance, 'part_512', target_role, f'{target_stem}.png')

        normal_img = self._load_image(normal_path)
        dwpose_img = self._load_image(dwpose_path)
        mask_img = self._load_grayscale(mask_path)
        
        # Load Part Map (uint8, values 0..14)
        if os.path.exists(part_path):
            part_img = Image.open(part_path)
        else:
            part_img = Image.new('L', (self.resolution, self.resolution), 0)

        # Load Depth
        depth_vis_path = os.path.join(self.assets_root, appearance, 'depth_vis_512', target_role, f'{target_stem}.png')
        depth_img = None
        depth_np = None
        if os.path.exists(depth_vis_path):
            depth_img = self._load_grayscale(depth_vis_path)
        else:
            depth_path = os.path.join(self.assets_root, appearance, 'depth_npy_512', target_role, f'{target_stem}.npy')
            if os.path.exists(depth_path):
                depth_raw = np.load(depth_path).astype(np.float32)
                if len(depth_raw.shape) == 3 and depth_raw.shape[2] == 1:
                    depth_raw = depth_raw.squeeze(-1)
                depth_np = cv2.resize(depth_raw, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
                mask_bool = depth_np > 0
                if mask_bool.any():
                    d_min = depth_np[mask_bool].min()
                    d_max = depth_np[mask_bool].max()
                    if d_max > d_min:
                        depth_np = (depth_np - d_min) / (d_max - d_min)
                    depth_np[~mask_bool] = 0.0
            else:
                depth_np = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        # Load v6 conditions bundle
        v6_data = self._get_v6_conditions(appearance)
        if v6_data is not None and target_stem in v6_data.get("stem_to_idx", {}):
            frame_idx = v6_data["stem_to_idx"][target_stem]
            smplx_global = v6_data["smplx_global"][frame_idx].clone()  # [26]
            kps_25_coords = v6_data["kps_25_coords"][frame_idx].clone() # [25, 3]
        else:
            smplx_global = torch.zeros(26, dtype=torch.float32)
            kps_25_coords = torch.zeros((25, 3), dtype=torch.float32)

        # Horizontal flip augmentation
        flip = self.augment and torch.rand(1).item() < 0.5
        if flip:
            src_img = transforms.functional.hflip(src_img)
            tgt_img = transforms.functional.hflip(tgt_img)
            normal_img = transforms.functional.hflip(normal_img)
            dwpose_img = transforms.functional.hflip(dwpose_img)
            mask_img = transforms.functional.hflip(mask_img)
            part_img = transforms.functional.hflip(part_img)
            if depth_img is not None:
                depth_img = transforms.functional.hflip(depth_img)
            if depth_np is not None:
                depth_np = np.fliplr(depth_np)
                
            # Flip keypoint x coords and swap left/right joints
            kps_25_coords[:, 0] = 1.0 - kps_25_coords[:, 0]
            for src_j, dst_j in KEYPOINT_FLIP_SWAP.items():
                if src_j < dst_j:
                    tmp = kps_25_coords[src_j].clone()
                    kps_25_coords[src_j] = kps_25_coords[dst_j]
                    kps_25_coords[dst_j] = tmp

        # RGB tensors: [-1, 1]
        src_tensor = self.rgb_transform(src_img)
        tgt_tensor = self.rgb_transform(tgt_img)

        # Resize for control images
        resize = transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST)
        to_tensor = transforms.ToTensor()  # scales to [0, 1]

        mask_tensor = to_tensor(resize(mask_img))  # [1, H, W]
        normal_raw = to_tensor(transforms.Resize((self.resolution, self.resolution))(normal_img))  # [3, H, W] in [0, 1]
        dwpose_tensor = to_tensor(transforms.Resize((self.resolution, self.resolution))(dwpose_img)) # [3, H, W]

        # Depth tensor: [0, 1]
        if depth_img is not None:
            depth_tensor = to_tensor(transforms.Resize((self.resolution, self.resolution))(depth_img))
        else:
            depth_tensor = torch.from_numpy(depth_np.copy()).unsqueeze(0).clamp(0.0, 1.0)
            
        # Legacy control tensor (8 channels): normal(3) + depth(1) + dwpose(3) + mask(1)
        control_map = torch.cat([normal_raw, depth_tensor, dwpose_tensor, mask_tensor], dim=0)

        # --------------------------------------------------------------------
        # V6.3 Native Condition Tensor Construction
        # --------------------------------------------------------------------
        # 1. Normal: scale from [0, 1] to [-1, 1], mask background to 0.0
        normal_v6 = (normal_raw * 2.0 - 1.0) * mask_tensor
        if flip:
            normal_v6[0] = -normal_v6[0]  # Invert X normal when flipped

        # 2. Part Map One-Hot: [14, H, W]
        part_arr = np.array(resize(part_img), dtype=np.int64)
        if flip:
            # Swap left and right body parts
            flipped_part = part_arr.copy()
            for src_p, dst_p in PART_FLIP_SWAP.items():
                flipped_part[part_arr == src_p] = dst_p
            part_arr = flipped_part
        part_t = torch.from_numpy(part_arr)
        # 0 is background, classes 1..14 map to channels 0..13
        part_onehot = F.one_hot(part_t, num_classes=15)[:, :, 1:].permute(2, 0, 1).float()

        # 3. Pose Heatmap: [25, H, W] in [0, 1]
        pose_heatmap = self._render_pose_heatmap(kps_25_coords)

        return {
            'src_image': src_tensor,
            'tgt_image': tgt_tensor,
            'control_map': control_map,  # Legacy 8-channel compatibility
            # V6.3 Native condition bundle items:
            'normal': normal_v6,          # [3, H, W] in [-1, 1]
            'depth': depth_tensor,        # [1, H, W] in [0, 1]
            'part_onehot': part_onehot,   # [14, H, W] binary one-hot float
            'pose_heatmap': pose_heatmap, # [25, H, W] in [0, 1]
            'smplx_global': smplx_global, # [26] float32
            'human_mask': mask_tensor,    # [1, H, W] in [0, 1]
            'task_id': torch.tensor(0, dtype=torch.long),  # TaskType.SINGLE
            'appearance': appearance,
            'source_stem': source_stem,
            'target_stem': target_stem,
        }

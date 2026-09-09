import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard SMPL-X 24 body joint hierarchy connections for skeleton visualization
SMPLX_BODY_PAIRS = [
    (0, 1),
    (0, 2),
    (0, 3),  # Pelvis to hips & spine1
    (1, 4),
    (2, 5),  # Hips to knees
    (4, 7),
    (5, 8),  # Knees to ankles
    (7, 10),
    (8, 11),  # Ankles to feet
    (3, 6),
    (6, 9),  # Spine1 to spine2, spine2 to spine3
    (9, 12),  # Spine3 to neck
    (12, 15),  # Neck to head
    (9, 13),
    (9, 14),  # Spine3 to collars
    (13, 16),
    (14, 17),  # Collars to shoulders
    (16, 18),
    (17, 19),  # Shoulders to elbows
    (18, 20),
    (19, 21),  # Elbows to wrists
]

# 24 SMPL-X Body parts mapping
BODY_PART_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye",
]


class Camera:
    """OpenCV pinhole camera model (X right, Y down, Z forward)."""

    def __init__(
        self,
        fx: float = 1000.0,
        fy: float = 1000.0,
        cx: float = 256.0,
        cy: float = 256.0,
        image_size: Tuple[int, int] = (512, 512),
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_size = image_size  # (H, W)
        self.R = R if R is not None else np.eye(3, dtype=np.float32)
        self.T = (
            T if T is not None else np.array([0.0, 0.0, 2.5], dtype=np.float32)
        )

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D world points to 2D image plane.

        points_3d: (N, 3) in world coords Returns:
          points_2d: (N, 2) in pixel coords (u, v)
          depths: (N,) in camera coords (Z)
        """
        # Transform to camera coordinates: P_cam = R @ P_world + T
        points_cam = (self.R @ points_3d.T).T + self.T
        z = points_cam[:, 2]
        z_safe = np.clip(z, a_min=1e-4, a_max=None)

        u = self.fx * (points_cam[:, 0] / z_safe) + self.cx
        v = self.fy * (points_cam[:, 1] / z_safe) + self.cy

        points_2d = np.stack([u, v], axis=-1)
        return points_2d, z


class ControlSignalGenerator:
    """Generates multi-modal control maps from 3D Human (SMPL-X / meshes) and camera parameters.

    Outputs:
    1. 2D Joints / Skeleton (H, W, 3)
    2. Depth Map (H, W, 1 or 3)
    3. Surface Normal Map (H, W, 3)
    4. Person Instance & Body Part Mask (H, W, 3)
    5. Contact Heatmap (H, W, 1) and Structured Contact Table
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size  # (H, W)

    def render_skeleton(
        self, joints_2d_list: List[np.ndarray], colors: Optional[List] = None
    ) -> np.ndarray:
        """Renders 2D skeleton bones and joints for multiple persons.

        joints_2d_list: list of (J, 2) arrays for each person Returns:
        image (H, W, 3) uint8
        """
        H, W = self.target_size
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        palette = [
            [(0, 255, 255), (0, 165, 255), (0, 0, 255)],  # Person 0 (warm)
            [(255, 255, 0), (255, 128, 0), (0, 255, 0)],  # Person 1 (cool)
        ]

        for p_idx, joints_2d in enumerate(joints_2d_list):
            color_theme = palette[p_idx % len(palette)]
            # Draw bones
            for u_idx, v_idx in SMPLX_BODY_PAIRS:
                if u_idx < len(joints_2d) and v_idx < len(joints_2d):
                    p1 = (int(joints_2d[u_idx][0]), int(joints_2d[u_idx][1]))
                    p2 = (int(joints_2d[v_idx][0]), int(joints_2d[v_idx][1]))
                    # Check inside canvas bounds
                    if (
                        0 <= p1[0] < W
                        and 0 <= p1[1] < H
                        or 0 <= p2[0] < W
                        and 0 <= p2[1] < H
                    ):
                        bone_color = (
                            color_theme[0]
                            if u_idx % 2 == 0
                            else color_theme[1]
                        )
                        cv2.line(canvas, p1, p2, bone_color, thickness=3)

            # Draw joints
            for j_idx, pt in enumerate(joints_2d):
                p = (int(pt[0]), int(pt[1]))
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(canvas, p, 4, (255, 255, 255), -1)
                    cv2.circle(canvas, p, 2, color_theme[2], -1)

        return canvas

    def render_pointcloud_depth_normal(
        self,
        vertices_list: List[np.ndarray],
        normals_list: List[np.ndarray],
        camera: Camera,
        point_radius: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projects dense 3D vertices and normals into depth and surface normal maps with z-buffering.

        Returns:
          depth_map: (H, W, 1) float32 in [0, 1]
          normal_map: (H, W, 3) uint8 in [0, 255]
        """
        H, W = self.target_size
        z_buffer = np.full((H, W), np.inf, dtype=np.float32)
        normal_buffer = np.zeros((H, W, 3), dtype=np.float32)

        for vertices, normals in zip(vertices_list, normals_list):
            pts_2d, z = camera.project(vertices)

            # Map camera normals to RGB space [0, 1] -> [0, 255]
            # Normal in camera space: N_cam = R @ N_world
            normals_cam = (camera.R @ normals.T).T
            norms = np.linalg.norm(normals_cam, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            normals_cam = normals_cam / norms

            # Render to buffers
            for (u, v), depth, norm in zip(pts_2d, z, normals_cam):
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < W and 0 <= iv < H and depth > 0:
                    for du in range(-point_radius, point_radius + 1):
                        for dv in range(-point_radius, point_radius + 1):
                            nu, nv = iu + du, iv + dv
                            if 0 <= nu < W and 0 <= nv < H:
                                if depth < z_buffer[nv, nu]:
                                    z_buffer[nv, nu] = depth
                                    normal_buffer[nv, nu] = norm

        # Normalize depth: 0 is closest, 1 is furthest
        valid_mask = ~np.isinf(z_buffer)
        depth_map = np.zeros((H, W, 1), dtype=np.float32)
        if np.any(valid_mask):
            z_min = np.min(z_buffer[valid_mask])
            z_max = np.max(z_buffer[valid_mask])
            z_range = z_max - z_min if z_max > z_min else 1.0
            # Standard inverse depth or normalized linear depth (near = 1.0, far = 0.0)
            depth_map[valid_mask, 0] = 1.0 - (
                (z_buffer[valid_mask] - z_min) / z_range
            )

        # Normal map: [-1, 1] to [0, 255]
        normal_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        if np.any(valid_mask):
            norm_vis = (normal_buffer + 1.0) * 0.5 * 255.0
            normal_rgb[valid_mask] = np.clip(norm_vis[valid_mask], 0, 255).astype(
                np.uint8
            )

        return depth_map, normal_rgb

    def compute_two_person_contacts(
        self,
        verts_a: np.ndarray,
        verts_b: np.ndarray,
        camera: Camera,
        dist_threshold_m: float = 0.05,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Calculates contact heatmap and structured contact relationship table between two persons.

        Returns:
          contact_heatmap: (H, W, 1) float32 in [0, 1]
          contact_records: List of contact region descriptions
        """
        H, W = self.target_size
        contact_heatmap = np.zeros((H, W, 1), dtype=np.float32)

        # Downsample vertices for fast pairwise distance check if very dense
        stride = 2 if len(verts_a) > 2000 else 1
        va = verts_a[::stride]
        vb = verts_b[::stride]

        # Compute pairwise distance: (Na, Nb)
        # Using PyTorch for ultra-fast GPU / CPU matrix calculation
        t_va = torch.from_numpy(va)
        t_vb = torch.from_numpy(vb)
        dist_mat = torch.cdist(t_va.unsqueeze(0), t_vb.unsqueeze(0))[
            0
        ].numpy()  # (Na, Nb)

        # Contact vertices where distance < dist_threshold
        a_contacts, b_contacts = np.where(dist_mat < dist_threshold_m)

        contact_records = []
        if len(a_contacts) > 0:
            contact_points_3d = (va[a_contacts] + vb[b_contacts]) * 0.5
            pts_2d, depths = camera.project(contact_points_3d)

            for (u, v), depth in zip(pts_2d, depths):
                iu, iv = int(round(u)), int(round(v))
                if 0 <= iu < W and 0 <= iv < H and depth > 0:
                    cv2.circle(contact_heatmap, (iu, iv), 8, (1.0,), -1)

            # Blur heatmap for smooth spatial conditioning
            contact_heatmap = cv2.GaussianBlur(contact_heatmap, (15, 15), 0)[
                :, :, None
            ]
            contact_heatmap = np.clip(
                contact_heatmap / (np.max(contact_heatmap) + 1e-6), 0.0, 1.0
            )

            # Summarize structured contact table
            min_dist = float(np.min(dist_mat[a_contacts, b_contacts]))
            contact_records.append(
                {
                    "person_a_id": 0,
                    "person_b_id": 1,
                    "contact_points_count": int(len(a_contacts)),
                    "min_distance_cm": round(min_dist * 100, 2),
                    "confidence": round(
                        float(max(0.0, 1.0 - min_dist / dist_threshold_m)), 3
                    ),
                    "visible": True,
                }
            )

        return contact_heatmap, contact_records

    def generate_all_control_maps(
        self,
        person_list: List[Dict],
        camera: Camera,
    ) -> Dict[str, np.ndarray]:
        """Master generation entry point for multi-condition control inputs.

        person_list: each dict contains:
            "joints_3d": (J, 3)
            "vertices": (V, 3)
            "normals": (V, 3)
        """
        # 1. 2D Joints Skeleton
        joints_2d_list = []
        for p in person_list:
            j2d, _ = camera.project(p["joints_3d"])
            joints_2d_list.append(j2d)
        skeleton_map = self.render_skeleton(joints_2d_list)

        # 2. Depth & Normal Maps
        vertices_list = [p["vertices"] for p in person_list]
        normals_list = [p["normals"] for p in person_list]
        depth_map, normal_map = self.render_pointcloud_depth_normal(
            vertices_list, normals_list, camera
        )

        # 3. Two-person Contact
        contact_heatmap = np.zeros(
            (self.target_size[0], self.target_size[1], 1), dtype=np.float32
        )
        contact_table = []
        if len(person_list) >= 2:
            contact_heatmap, contact_table = self.compute_two_person_contacts(
                person_list[0]["vertices"], person_list[1]["vertices"], camera
            )

        # 4. Multi-channel stacked tensor for adapter injection (e.g. 8 channels)
        # Normal (3) + Depth (1) + Skeleton (3) + Contact (1)
        norm_float = normal_map.astype(np.float32) / 255.0
        skel_float = skeleton_map.astype(np.float32) / 255.0
        stacked_8ch = np.concatenate(
            [norm_float, depth_map, skel_float, contact_heatmap], axis=-1
        )  # (H, W, 8)

        return {
            "skeleton_rgb": skeleton_map,
            "depth_map": depth_map,
            "normal_rgb": normal_map,
            "contact_heatmap": contact_heatmap,
            "contact_table": contact_table,
            "stacked_control_8ch": stacked_8ch,
        }

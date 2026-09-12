from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch


# Fit3D 官方 dataset_util.py 使用的 25-joint 连边。
# 不重新定义人体拓扑，保持与官方可视化代码一致。
FIT3D_LIMBS = (
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (11, 12),
    (14, 15),
    (12, 13),
    (15, 16),
    (8, 7),
    (7, 0),
    (0, 1),
    (0, 4),
    (1, 2),
    (4, 5),
    (2, 3),
    (5, 6),
    (13, 21),
    (13, 22),
    (16, 23),
    (16, 24),
    (3, 17),
    (3, 18),
    (6, 19),
    (6, 20),
)

FIT3D_ROOT_JOINT = 8
FIT3D_IMAGE_SIZE = 900


@lru_cache(maxsize=64)
def _load_joints_sequence(path_str: str) -> np.ndarray:
    with open(path_str, "r", encoding="utf-8") as f:
        joints = np.asarray(
            json.load(f)["joints3d_25"],
            dtype=np.float32,
        )

    if (
        joints.ndim != 3
        or joints.shape[1:] != (25, 3)
    ):
        raise ValueError(
            "Fit3D joints3d_25 应为 [T,25,3]，"
            f"实际为 {joints.shape}"
        )

    return joints


@lru_cache(maxsize=64)
def _load_camera(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        camera = json.load(f)

    for group in camera:
        for key in camera[group]:
            camera[group][key] = np.asarray(
                camera[group][key],
                dtype=np.float32,
            )

    return camera


def world_to_camera(
    points_world: np.ndarray,
    camera: dict,
) -> np.ndarray:
    """
    Fit3D 官方坐标转换：

        X_camera = (X_world - T) @ R^T

    与官方 SMPL-X renderer 完全一致。
    """
    # Fit3D JSON 中 R 通常为 [3,3]，
    # T 实际保存为 [1,3]。
    # 这里显式整理形状，避免 NumPy 将
    # [25,3] 广播成 [1,25,3]。
    rotation = np.asarray(
        camera["extrinsics"]["R"],
        dtype=np.float32,
    ).reshape(3, 3)

    translation = np.asarray(
        camera["extrinsics"]["T"],
        dtype=np.float32,
    ).reshape(3)

    return (
        points_world - translation
    ) @ rotation.T


def project_wo_distortion(
    points_camera: np.ndarray,
    camera: dict,
) -> np.ndarray:
    """
    使用 Fit3D 官方无畸变透视投影：

        xy = XY / Z
        pixel = f * xy + c
    """
    z = points_camera[:, 2:3]

    if np.any(z <= 1e-6):
        raise ValueError(
            "存在位于相机后方或过近的 Fit3D 关节"
        )

    xy = points_camera[:, :2] / z

    intrinsics = camera[
        "intrinsics_wo_distortion"
    ]

    return (
        intrinsics["f"] * xy
        + intrinsics["c"]
    )


def normalized_relative_depth(
    points_camera: np.ndarray,
) -> np.ndarray:
    """
    深度只表达姿态中的前后关系，不表达：

    - 相机绝对距离
    - 人体绝对尺寸

    以 pelvis/root(8) 为中心，
    再用当前骨架 3D extent 做尺度归一化。
    """
    root = points_camera[
        FIT3D_ROOT_JOINT
    ]

    relative = points_camera - root[None, :]

    body_scale = np.linalg.norm(
        relative,
        axis=1,
    ).max()

    if body_scale < 1e-6:
        raise ValueError(
            "Fit3D 骨架尺度异常，无法归一化"
        )

    depth = (
        relative[:, 2]
        / body_scale
    )

    return np.clip(
        depth,
        -1.0,
        1.0,
    )


def _paint_disk(
    occupancy: np.ndarray,
    depth_map: np.ndarray,
    x: float,
    y: float,
    depth: float,
    radius: int,
):
    height, width = occupancy.shape

    cx = int(round(x))
    cy = int(round(y))

    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.ogrid[
        y0:y1,
        x0:x1,
    ]

    mask = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= radius**2
    )

    occ_region = occupancy[
        y0:y1,
        x0:x1,
    ]
    depth_region = depth_map[
        y0:y1,
        x0:x1,
    ]

    occ_region[mask] = 1.0

    # 两根骨骼发生 2D 重叠时，
    # 保留更靠近相机的那一根。
    depth_region[mask] = np.minimum(
        depth_region[mask],
        depth,
    )


def _rasterize_motion(
    points_2d: np.ndarray,
    relative_depth: np.ndarray,
    output_size: int,
    line_radius: int,
    use_depth: bool,
) -> np.ndarray:
    occupancy = np.zeros(
        (output_size, output_size),
        dtype=np.float32,
    )

    depth_map = np.full(
        (output_size, output_size),
        np.inf,
        dtype=np.float32,
    )

    scale = (
        output_size
        / float(FIT3D_IMAGE_SIZE)
    )

    points = points_2d * scale

    for joint_a, joint_b in FIT3D_LIMBS:
        p0 = points[joint_a]
        p1 = points[joint_b]

        z0 = relative_depth[joint_a]
        z1 = relative_depth[joint_b]

        steps = (
            int(
                max(
                    abs(float(p1[0] - p0[0])),
                    abs(float(p1[1] - p0[1])),
                )
            )
            + 1
        )

        steps = max(steps, 2)

        for alpha in np.linspace(
            0.0,
            1.0,
            steps,
            dtype=np.float32,
        ):
            point = (
                p0 * (1.0 - alpha)
                + p1 * alpha
            )

            depth = float(
                z0 * (1.0 - alpha)
                + z1 * alpha
            )

            _paint_disk(
                occupancy=occupancy,
                depth_map=depth_map,
                x=float(point[0]),
                y=float(point[1]),
                depth=depth,
                radius=line_radius,
            )

    # 再画关节点，防止短肢体端点过弱。
    for point, depth in zip(
        points,
        relative_depth,
    ):
        _paint_disk(
            occupancy=occupancy,
            depth_map=depth_map,
            x=float(point[0]),
            y=float(point[1]),
            depth=float(depth),
            radius=line_radius + 1,
        )

    depth_map[
        ~np.isfinite(depth_map)
    ] = 0.0

    if not use_depth:
        depth_map.fill(0.0)

    motion = np.zeros(
        (4, output_size, output_size),
        dtype=np.float32,
    )

    # 前 3 通道保持 V4 RGB skeleton 的接口形式。
    # 当前使用白色骨架，不人为加入额外语义颜色。
    motion[0] = occupancy
    motion[1] = occupancy
    motion[2] = occupancy

    # 第 4 通道：
    # root-relative、body-scale-normalized depth。
    motion[3] = depth_map

    return motion


@lru_cache(maxsize=1024)
def _build_cached(
    joints_path_str: str,
    camera_path_str: str,
    frame_index: int,
    output_size: int,
    line_radius: int,
    use_depth: bool,
) -> np.ndarray:
    joints_sequence = (
        _load_joints_sequence(
            joints_path_str
        )
    )

    if not (
        0
        <= frame_index
        < len(joints_sequence)
    ):
        raise IndexError(
            f"frame_index={frame_index} 越界，"
            f"序列长度={len(joints_sequence)}"
        )

    camera = _load_camera(
        camera_path_str
    )

    points_world = (
        joints_sequence[frame_index]
    )

    points_camera = world_to_camera(
        points_world=points_world,
        camera=camera,
    )

    points_2d = project_wo_distortion(
        points_camera=points_camera,
        camera=camera,
    )

    relative_depth = (
        normalized_relative_depth(
            points_camera
        )
    )

    return _rasterize_motion(
        points_2d=points_2d,
        relative_depth=relative_depth,
        output_size=output_size,
        line_radius=line_radius,
        use_depth=use_depth,
    )


def build_fit3d_motion_condition(
    joints_path: Path,
    camera_path: Path,
    frame_index: int,
    output_size: int = 512,
    line_radius: int = 3,
    use_depth: bool = True,
) -> torch.Tensor:
    """
    构造 V5-A motion condition：

        channel 0-2:
            GT 3D joints 投影得到的 2D skeleton

        channel 3:
            root-relative normalized depth

    输出：
        [4,H,W] float32

    注意：
        不读取 SMPL-X betas；
        不读取 target mesh；
        不把 target body shape 作为显式条件。
    """
    motion = _build_cached(
        str(Path(joints_path).resolve()),
        str(Path(camera_path).resolve()),
        int(frame_index),
        int(output_size),
        int(line_radius),
        bool(use_depth),
    )

    return torch.from_numpy(
        motion.copy()
    )

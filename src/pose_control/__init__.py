"""DeepGen 姿态控制模块。"""

from .deepgen_pose_controlnet_v3 import (
    DeepGenPoseControlNetV3,
)
from .deepgen_pose_controlnet_v4 import (
    DeepGenPoseControlNetV4,
)

__all__ = [
    "DeepGenPoseControlNetV3",
    "DeepGenPoseControlNetV4",
]

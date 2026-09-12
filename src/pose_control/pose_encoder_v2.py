import math

import torch
import torch.nn as nn


class PoseEncoderV2(nn.Module):
    """
    基于 MimicMotion 官方 PoseNet 卷积主干的姿态编码器。

    512×512 输入：
        512×512
        → 256×256
        → 128×128
        → 64×64

    当前只负责提取姿态空间特征，不直接接入 DeepGen。
    """

    def __init__(
        self,
        input_channels: int = 3,
    ):
        super().__init__()

        self.output_channels = 128

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                input_channels,
                16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        与 MimicMotion PoseNet 保持一致：
        卷积权重使用 He 初始化，bias 全部初始化为 0。
        """
        for module in self.conv_layers:
            if not isinstance(module, nn.Conv2d):
                continue

            fan_in = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
            )

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / fan_in),
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pose_map: torch.Tensor,
    ) -> torch.Tensor:
        if pose_map.ndim != 4:
            raise ValueError(
                "pose_map 必须为 [B,C,H,W]，"
                f"实际为 {tuple(pose_map.shape)}"
            )

        return self.conv_layers(pose_map)

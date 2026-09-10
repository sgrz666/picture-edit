import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    """
    将目标二维骨架图编码为 DeepGen 可以使用的空间姿态特征。

    对于 512×512 输入：
        512×512
        → 256×256
        → 128×128
        → 64×64
        → 32×32

    DeepGen 在 512×512 下的图像 token 网格也是 32×32，
    因此两者可以保持空间位置一一对应。

    设计参考 Champ GuidanceEncoder 的卷积逐级下采样思想，
    但只保留单图姿态控制需要的 2D 卷积部分。
    """

    def __init__(
        self,
        input_channels: int = 3,
        channels=(16, 32, 96, 256, 256),
    ):
        super().__init__()

        self.output_channels = channels[-1]

        self.conv_in = nn.Sequential(
            nn.Conv2d(
                input_channels,
                channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )

        self.down_blocks = nn.ModuleList()

        for in_channels, out_channels in zip(
            channels[:-1],
            channels[1:],
        ):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.SiLU(),
                )
            )

    def forward(
        self,
        pose_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pose_map:
                [B, C, H, W] 的目标骨架图。

        Returns:
            姿态特征 [B, 256, H/16, W/16]。
            512×512 输入时为 [B, 256, 32, 32]。
        """

        if pose_map.ndim != 4:
            raise ValueError(
                "pose_map 必须是 [B,C,H,W]，"
                f"实际为 {tuple(pose_map.shape)}"
            )

        x = self.conv_in(pose_map)

        for block in self.down_blocks:
            x = block(x)

        return x

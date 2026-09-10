from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .pose_encoder import PoseEncoder
from .token_alignment import align_pose_residuals


def zero_init(
    module: nn.Module,
) -> nn.Module:
    """
    将控制输出层初始化为 0。

    这样 Adapter 刚接入 DeepGen 时：
        control residual = 0

    因此初始行为仍接近原始 DeepGen，
    不会因为随机控制权重立即破坏预训练模型。
    """

    nn.init.zeros_(module.weight)

    if module.bias is not None:
        nn.init.zeros_(module.bias)

    return module


class DeepGenPoseAdapter(nn.Module):
    """
    DeepGen 的轻量姿态控制分支。

    Target Skeleton
        ↓
    PoseEncoder
        ↓
    Zero-initialized residual heads
        ↓
    Target pose tokens
        ↓
    Token Alignment
        ↓
    [Pose residual | Source zeros | Padding zeros]
        ↓
    DeepGen block_controlnet_hidden_states
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 1536,
        num_residual_heads: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_residual_heads = (
            num_residual_heads
        )

        self.pose_encoder = PoseEncoder(
            input_channels=input_channels,
        )

        pose_channels = (
            self.pose_encoder.output_channels
        )

        # 6 不是写死在 DeepGen 里的要求，
        # 只是当前 V1 默认值，可以以后通过实验调整。
        #
        # 每个 head 都输出一组独立 residual，
        # DeepGen 自己负责把这些 residual
        # 映射到它的 24 个 Transformer blocks。
        self.residual_heads = nn.ModuleList(
            [
                zero_init(
                    nn.Conv2d(
                        pose_channels,
                        hidden_dim,
                        kernel_size=1,
                    )
                )
                for _ in range(
                    num_residual_heads
                )
            ]
        )

    def forward(
        self,
        pose_map: torch.Tensor,
        target_latents: torch.Tensor,
        cond_hidden_states: Optional[Sequence],
        patch_size: int = 2,
    ) -> List[torch.Tensor]:
        """
        Args:
            pose_map:
                目标骨架图 [B,3,H,W]。

            target_latents:
                DeepGen 当前 noisy target latent。

            cond_hidden_states:
                DeepGen 原生 source/reference latents。

            patch_size:
                DeepGen Transformer 的 patch size。

        Returns:
            与 DeepGen 完整 image token 序列对齐后的
            block_controlnet_hidden_states。
        """

        pose_feature = self.pose_encoder(
            pose_map
        )

        residuals = []

        for head in self.residual_heads:
            residual = head(
                pose_feature
            )

            # [B,D,H,W]
            # →
            # [B,H*W,D]
            residual = (
                residual
                .flatten(2)
                .transpose(1, 2)
                .contiguous()
            )

            residuals.append(
                residual
            )

        return align_pose_residuals(
            residuals=residuals,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=patch_size,
        )

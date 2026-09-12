import torch

from .deepgen_pose_controlnet_v4 import (
    DeepGenPoseControlNetV4,
)
from .pose_encoder_v2 import PoseEncoderV2


class DeepGenMotionControlNetV5(
    DeepGenPoseControlNetV4
):
    """
    DeepGen Motion ControlNet V5-A。

    V4 已验证的部分全部保持不变：

        DeepGen frozen backbone
        + SD3 ControlNet
        + Source latent
        + Target-only residual injection

    唯一核心变化：

        V4:
            3ch 2D skeleton

        V5-A:
            4ch motion map
            = 3ch GT skeleton
            + 1ch relative depth

    因此该实验可以直接判断：
    加入真实 3D motion information 后，
    姿态控制精度是否提高。
    """

    def __init__(
        self,
        deepgen_transformer,
        num_layers: int = 6,
    ):
        super().__init__(
            deepgen_transformer=deepgen_transformer,
            num_layers=num_layers,
        )

        # PoseEncoderV2 本身支持可配置输入通道。
        # 这里只将 3ch 改为 4ch，
        # 后续 128ch 输出和 ControlNet 完全不变。
        self.pose_encoder = PoseEncoderV2(
            input_channels=4,
        )

    def forward(
        self,
        target_latents: torch.Tensor,
        motion_map: torch.Tensor,
        cond_hidden_states,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: int,
        conditioning_scale: float = 1.0,
    ):
        if (
            motion_map.ndim != 4
            or motion_map.shape[1] != 4
        ):
            raise ValueError(
                "V5 motion_map 必须为 "
                "[B,4,H,W]，实际为 "
                f"{tuple(motion_map.shape)}"
            )

        # 直接复用 V4 已验证的 ControlNet forward。
        # V4 内部会：
        #   4ch motion -> PoseEncoder -> 128ch
        #   128ch + 16ch Source latent -> 144ch
        #   -> SD3 ControlNet
        #   -> Target-only residual
        return super().forward(
            target_latents=target_latents,
            pose_map=motion_map,
            cond_hidden_states=cond_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            patch_size=patch_size,
            conditioning_scale=conditioning_scale,
        )

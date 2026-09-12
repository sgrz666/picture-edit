import torch
import torch.nn as nn
import torch.nn.functional as F

from .deepgen_pose_adapter import zero_init
from .pose_encoder_v2 import PoseEncoderV2
from .token_alignment import align_pose_residuals


class DeepGenPoseAdapterV2(nn.Module):
    """
    DeepGen 单人姿态控制 Adapter V2。

    路径：
        Target Skeleton
            ↓
        PoseEncoderV2
            ↓
        64×64 pose feature
            ↓
        2×2 average pooling
            ↓
        32×32 pose feature
            ↓
        6 个 zero-init residual heads
            ↓
        Target pose tokens
            ↓
        [Pose residual | Source zeros | Padding zeros]

    V2 相比 V1 只改变：
        1. PoseEncoder 换成已经通过敏感性测试的 V2；
        2. 用无参数 2×2 average pooling 将 64×64
           对齐到 DeepGen 的 32×32 token 网格。

    residual heads 和 Target-only token 对齐逻辑保持不变。
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 1536,
        num_residual_heads: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_residual_heads = num_residual_heads

        self.pose_encoder = PoseEncoderV2(
            input_channels=input_channels,
        )

        pose_channels = (
            self.pose_encoder.output_channels
        )

        # 保持 V1 的 residual 接口不变。
        # 每个输出层严格 zero-init，
        # 因此训练开始时 residual=0。
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
        cond_hidden_states,
        patch_size: int,
    ):
        """
        Args:
            pose_map:
                [B,3,512,512] Target Skeleton。

            target_latents:
                DeepGen 当前 noisy target latent。

            cond_hidden_states:
                DeepGen Source/reference 条件，
                用于确定完整 image token 长度。

            patch_size:
                DeepGen Transformer patch size。

        Returns:
            与 DeepGen image token 对齐后的
            residual list。
        """

        pose_feature = self.pose_encoder(
            pose_map
        )

        latent_h = target_latents.shape[-2]
        latent_w = target_latents.shape[-1]

        if (
            latent_h % patch_size != 0
            or latent_w % patch_size != 0
        ):
            raise ValueError(
                "Target latent 尺寸不能被 patch_size 整除："
                f"{latent_h}x{latent_w}, "
                f"patch_size={patch_size}"
            )

        target_h = latent_h // patch_size
        target_w = latent_w // patch_size

        # 当前 512×512 DeepGen：
        #
        # PoseEncoderV2 -> 64×64
        # DeepGen target token grid -> 32×32
        #
        # 这一步没有可学习参数。
        # 前面的 sensitivity test 已验证，
        # 2×2 average pooling 后仍能明显区分不同姿态。
        if pose_feature.shape[-2:] != (
            target_h * 2,
            target_w * 2,
        ):
            raise ValueError(
                "PoseEncoderV2 输出尺寸与 DeepGen "
                "Target token 网格不符合当前 2× 对齐关系："
                f"{tuple(pose_feature.shape[-2:])} vs "
                f"{target_h}x{target_w}"
            )

        pose_feature = F.avg_pool2d(
            pose_feature,
            kernel_size=2,
            stride=2,
        )

        if pose_feature.shape[-2:] != (
            target_h,
            target_w,
        ):
            raise RuntimeError(
                "Pose feature 对齐失败："
                f"{tuple(pose_feature.shape[-2:])} vs "
                f"{target_h}x{target_w}"
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

        # 完全复用 V1 已验证过的 token 对齐：
        #
        # [Target residual |
        #  Source/reference zeros |
        #  Padding zeros]
        return align_pose_residuals(
            residuals=residuals,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=patch_size,
        )

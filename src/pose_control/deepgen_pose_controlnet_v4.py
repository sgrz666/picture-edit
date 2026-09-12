import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import SD3ControlNetModel

from .pose_encoder_v2 import PoseEncoderV2
from .token_alignment import align_pose_residuals


class DeepGenPoseControlNetV4(nn.Module):
    """
    DeepGen Pose ControlNet V4。

    与 V3 的核心区别只有一个：

        V3:
            Skeleton -> DeepGen VAE -> 16ch pose latent

        V4:
            Skeleton -> MimicMotion-style PoseEncoder
                     -> 128ch / 64x64 pose feature

    后面的 SD3 ControlNet Transformer、DeepGen residual
    注入方式、Source token 保护全部保持不变。

    目的：
        提高 sparse skeleton 的空间细节保留能力，
        改善手腕、肩膀、膝盖等关节对齐精度。
    """

    def __init__(
        self,
        deepgen_transformer,
        num_layers: int = 6,
    ):
        super().__init__()

        config = deepgen_transformer.config

        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_layers = num_layers

        if self.in_channels != 16:
            raise ValueError(
                "当前 V4 按 DeepGen 16-channel latent 设计，"
                f"实际为 {self.in_channels}"
            )

        if (
            num_layers < 1
            or num_layers > config.num_layers
        ):
            raise ValueError(
                f"num_layers={num_layers} 不合法，"
                f"DeepGen 总层数={config.num_layers}"
            )

        # --------------------------------------------------
        # 1. Sparse Pose Encoder
        #
        # 直接复用已经通过 sensitivity test 的
        # MimicMotion-style PoseEncoderV2。
        #
        # 512x512
        #   -> 64x64 / 128 channels
        #
        # 不使用 MimicMotion 的 zero-init final projection：
        # SD3 ControlNet 自身的 pos_embed_input 已经 zero-init。
        # --------------------------------------------------

        self.pose_encoder = PoseEncoderV2(
            input_channels=3,
        )

        self.pose_channels = (
            self.pose_encoder.output_channels
        )

        if self.pose_channels != 128:
            raise ValueError(
                "当前 V4 期望 PoseEncoder 输出 128 channels，"
                f"实际={self.pose_channels}"
            )

        # --------------------------------------------------
        # 2. 官方 SD3 ControlNet 主体
        #
        # control condition:
        #
        #   Source latent       16 ch
        #   Pose feature       128 ch
        #                     -------
        #                     144 ch
        #
        # SD3ControlNetModel 的输入规则：
        #
        #   channels =
        #       in_channels
        #       + extra_conditioning_channels
        #
        # 因此这里 extra = 128。
        # --------------------------------------------------

        dual_attention_layers = tuple(
            layer
            for layer in getattr(
                config,
                "dual_attention_layers",
                (),
            )
            if layer < num_layers
        )

        self.controlnet = SD3ControlNetModel(
            sample_size=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            num_layers=num_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=config.joint_attention_dim,
            caption_projection_dim=config.caption_projection_dim,
            pooled_projection_dim=config.pooled_projection_dim,
            out_channels=config.out_channels,
            pos_embed_max_size=config.pos_embed_max_size,

            # 16 Source latent + 128 Pose feature
            # controlnet_cond 总共 144 channels。
            extra_conditioning_channels=128,

            dual_attention_layers=dual_attention_layers,
            qk_norm=getattr(
                config,
                "qk_norm",
                None,
            ),
        )

        # --------------------------------------------------
        # 3. 按官方 ControlNet 思路，
        #    从预训练 DeepGen Transformer 复制表示。
        #
        # 注意：
        # pos_embed_input 和 controlnet_blocks
        # 保留 SD3 ControlNet 自己的 zero-init，
        # 不能覆盖。
        # --------------------------------------------------

        self.controlnet.pos_embed.load_state_dict(
            deepgen_transformer.pos_embed.state_dict()
        )

        self.controlnet.time_text_embed.load_state_dict(
            deepgen_transformer.time_text_embed.state_dict()
        )

        self.controlnet.context_embedder.load_state_dict(
            deepgen_transformer.context_embedder.state_dict()
        )

        for index in range(num_layers):
            incompatible = (
                self.controlnet
                .transformer_blocks[index]
                .load_state_dict(
                    deepgen_transformer
                    .transformer_blocks[index]
                    .state_dict(),
                    strict=False,
                )
            )

            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "复制 DeepGen Transformer block "
                    "出现 unexpected keys："
                    f"{incompatible.unexpected_keys}"
                )

    @staticmethod
    def _get_source_latents(
        cond_hidden_states,
        target_latents: torch.Tensor,
    ):
        """
        取 DeepGen 已经编码好的第一张 Source/reference latent。

        当前单人验证每个 sample 只有一个 Source。
        """

        batch_size = target_latents.shape[0]

        if len(cond_hidden_states) != batch_size:
            raise ValueError(
                "cond_hidden_states batch 不一致："
                f"{len(cond_hidden_states)} vs {batch_size}"
            )

        source_latents = []

        for refs in cond_hidden_states:
            if len(refs) == 0:
                raise ValueError(
                    "当前 sample 没有 Source/reference latent"
                )

            ref = refs[0]

            if ref.ndim == 4:
                if ref.shape[0] != 1:
                    raise ValueError(
                        "Source latent 无法解析："
                        f"{tuple(ref.shape)}"
                    )
                ref = ref[0]

            if ref.ndim != 3:
                raise ValueError(
                    "Source latent 应为 [C,H,W]，实际："
                    f"{tuple(ref.shape)}"
                )

            source_latents.append(ref)

        source_latents = torch.stack(
            source_latents,
            dim=0,
        )

        if (
            source_latents.shape[-2:]
            != target_latents.shape[-2:]
        ):
            source_latents = F.interpolate(
                source_latents.float(),
                size=target_latents.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return source_latents

    def forward(
        self,
        target_latents: torch.Tensor,
        pose_map: torch.Tensor,
        cond_hidden_states,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: int,
        conditioning_scale: float = 1.0,
    ):
        """
        Args:
            target_latents:
                当前 noisy target latent。

            pose_map:
                [B,3,512,512] Skeleton RGB，范围 [0,1]。

        Returns:
            可直接送入 DeepGen
            block_controlnet_hidden_states 的 residual list。
        """

        if patch_size != self.patch_size:
            raise ValueError(
                f"patch_size 不一致："
                f"{patch_size} vs {self.patch_size}"
            )

        # --------------------------------------------------
        # Skeleton -> sparse-aware pose feature
        #
        # 512x512 -> 128x64x64
        # --------------------------------------------------

        pose_feature = self.pose_encoder(
            pose_map.float()
        )

        source_latents = self._get_source_latents(
            cond_hidden_states=cond_hidden_states,
            target_latents=target_latents,
        )

        target_size = target_latents.shape[-2:]

        if pose_feature.shape[-2:] != target_size:
            pose_feature = F.interpolate(
                pose_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        if source_latents.shape[-2:] != target_size:
            source_latents = F.interpolate(
                source_latents.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        control_dtype = next(
            self.controlnet.parameters()
        ).dtype

        # --------------------------------------------------
        # DeepGen-specific condition
        #
        # Source spatial state
        # +
        # Target sparse-pose feature
        #
        # 不再让 VAE 压缩 Skeleton。
        # --------------------------------------------------

        control_cond = torch.cat(
            [
                source_latents.float(),
                pose_feature,
            ],
            dim=1,
        ).to(
            device=target_latents.device,
            dtype=control_dtype,
        )

        if control_cond.shape[1] != 144:
            raise RuntimeError(
                "V4 control condition channel 错误："
                f"{control_cond.shape[1]} vs 144"
            )

        control_samples = self.controlnet(
            hidden_states=target_latents.to(
                dtype=control_dtype
            ),
            controlnet_cond=control_cond,
            encoder_hidden_states=encoder_hidden_states.to(
                dtype=control_dtype
            ),
            pooled_projections=pooled_projections.to(
                dtype=control_dtype
            ),
            timestep=timestep,
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )[0]

        # --------------------------------------------------
        # DeepGen image token:
        #
        # [Target | Source/reference | Padding]
        #
        # Control residual 仍只写 Target。
        # Source/reference token 严格保持 0。
        # --------------------------------------------------

        return align_pose_residuals(
            residuals=control_samples,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=patch_size,
        )

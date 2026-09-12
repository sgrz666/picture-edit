import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import SD3ControlNetModel

from .token_alignment import align_pose_residuals


class DeepGenPoseControlNetV3(nn.Module):
    """
    DeepGen Pose ControlNet V3。

    核心不是 Skeleton -> 静态 residual，
    而是使用官方 SD3 ControlNet 结构，让控制分支同时看到：

        1. 当前 noisy target latent；
        2. 当前 diffusion timestep；
        3. DeepGen 的 SCB semantic condition；
        4. Target pose latent；
        5. Source/reference latent。

    因此控制分支学习的是：

        当前 Source 人
        + 当前目标 Pose
        + 当前去噪状态
        -> 应怎样修改 Target tokens

    而不是 V2 的：

        Skeleton
        -> residual

    Source/reference token 本身仍严格不直接写 residual。
    """

    def __init__(
        self,
        deepgen_transformer,
        num_layers: int = 6,
    ):
        super().__init__()

        config = deepgen_transformer.config

        self.num_layers = num_layers
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size

        if self.in_channels != 16:
            raise ValueError(
                "当前 V3 按 DeepGen 16-channel latent 设计，"
                f"实际 in_channels={self.in_channels}"
            )

        if num_layers < 1 or num_layers > config.num_layers:
            raise ValueError(
                f"num_layers={num_layers} 不合法，"
                f"DeepGen 总层数={config.num_layers}"
            )

        # --------------------------------------------------
        # 官方 SD3 ControlNet 主体
        #
        # controlnet_cond:
        #
        #   Target Pose latent : 16 channels
        #   Source latent      : 16 channels
        #                       -------------
        #                       32 channels
        #
        # SD3 ControlNet 自身已经有：
        #   - zero-init condition projection
        #   - Transformer control blocks
        #   - zero-init block output projection
        #
        # 不再自己设计额外 residual heads。
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
            extra_conditioning_channels=config.in_channels,
            dual_attention_layers=dual_attention_layers,
            qk_norm=getattr(
                config,
                "qk_norm",
                None,
            ),
        )

        # --------------------------------------------------
        # 按官方 ControlNet 思路：
        # 从冻结的 DeepGen Transformer 复制预训练表示。
        #
        # zero-init 的 condition/output projection
        # 保持 ControlNet 自己的初始化，不覆盖。
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
                self.controlnet.transformer_blocks[index]
                .load_state_dict(
                    deepgen_transformer
                    .transformer_blocks[index]
                    .state_dict(),
                    strict=False,
                )
            )

            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "复制 DeepGen Transformer block 时"
                    "出现 unexpected keys："
                    f"{incompatible.unexpected_keys}"
                )

    @staticmethod
    def _get_source_latents(
        cond_hidden_states,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 DeepGen 原生 cond_hidden_states 中取出
        Source/reference image latent。

        当前单人实验只有一张 Source，
        因此每个 sample 使用第一张 reference latent。
        """

        batch_size = target_latents.shape[0]

        if len(cond_hidden_states) != batch_size:
            raise ValueError(
                "cond_hidden_states batch 与 target batch 不一致："
                f"{len(cond_hidden_states)} vs {batch_size}"
            )

        source_latents = []

        for refs in cond_hidden_states:
            if len(refs) == 0:
                raise ValueError(
                    "当前样本没有 Source/reference latent"
                )

            ref = refs[0]

            # DeepGen 当前格式通常为 [C,H,W]。
            if ref.ndim == 4:
                if ref.shape[0] != 1:
                    raise ValueError(
                        "无法解析 Source latent："
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

        if source_latents.shape[-2:] != target_latents.shape[-2:]:
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
        pose_latents: torch.Tensor,
        cond_hidden_states,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: int,
        conditioning_scale: float = 1.0,
    ):
        """
        返回能够直接送入 DeepGen
        block_controlnet_hidden_states 的 residual list。
        """

        if patch_size != self.patch_size:
            raise ValueError(
                f"patch_size 不一致："
                f"{patch_size} vs {self.patch_size}"
            )

        source_latents = self._get_source_latents(
            cond_hidden_states=cond_hidden_states,
            target_latents=target_latents,
        )

        if pose_latents.shape[-2:] != target_latents.shape[-2:]:
            pose_latents = F.interpolate(
                pose_latents.float(),
                size=target_latents.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if pose_latents.shape[1] != self.in_channels:
            raise ValueError(
                "Pose latent channel 错误："
                f"{pose_latents.shape[1]} "
                f"vs {self.in_channels}"
            )

        if source_latents.shape[1] != self.in_channels:
            raise ValueError(
                "Source latent channel 错误："
                f"{source_latents.shape[1]} "
                f"vs {self.in_channels}"
            )

        control_dtype = next(
            self.controlnet.parameters()
        ).dtype

        # --------------------------------------------------
        # DeepGen-specific adaptation
        #
        # 官方 SD3 ControlNet：
        #   noisy target + control latent
        #
        # 我们：
        #   noisy target
        #   + [target pose latent | source latent]
        #
        # 这样控制分支明确知道：
        #   “旧的人在哪里”
        #   “新姿态应该在哪里”
        # --------------------------------------------------

        control_cond = torch.cat(
            [
                pose_latents,
                source_latents,
            ],
            dim=1,
        ).to(
            device=target_latents.device,
            dtype=control_dtype,
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
        # 官方 ControlNet 输出只有 Target token。
        #
        # DeepGen 的 image token 顺序为：
        # [Target | Source/reference | Padding]
        #
        # 因此继续复用我们已经验证过的对齐逻辑：
        # [Control residual | 0 | 0]
        #
        # Source token 本身不直接修改。
        # --------------------------------------------------

        return align_pose_residuals(
            residuals=control_samples,
            target_latents=target_latents,
            cond_hidden_states=cond_hidden_states,
            patch_size=patch_size,
        )

from typing import List, Optional, Sequence

import torch


def latent_token_count(
    latent: torch.Tensor,
    patch_size: int,
) -> int:
    """
    根据 latent 空间尺寸计算经过 DeepGen PatchEmbed 后的 token 数。
    """

    if latent.ndim not in (3, 4):
        raise ValueError(
            "latent 必须是 [C,H,W] 或 [B,C,H,W]，"
            f"实际为 {tuple(latent.shape)}"
        )

    height, width = latent.shape[-2:]

    if (
        height % patch_size != 0
        or width % patch_size != 0
    ):
        raise ValueError(
            f"latent 尺寸 {(height, width)} "
            f"不能被 patch_size={patch_size} 整除"
        )

    return (
        height // patch_size
    ) * (
        width // patch_size
    )


def align_pose_residuals(
    residuals: Sequence[torch.Tensor],
    target_latents: torch.Tensor,
    cond_hidden_states: Optional[Sequence],
    patch_size: int,
) -> List[torch.Tensor]:
    """
    将只属于 Target 的姿态 residual 对齐到 DeepGen 完整图像 token。

    DeepGen 的真实 token 顺序已经通过运行时验证：

        [Target tokens | Source/reference tokens | Padding]

    因此姿态 residual 必须构造成：

        [Pose residual | 0 | 0]

    Source/reference 部分严格补零，避免姿态控制直接修改参考图 token。
    """

    if target_latents.ndim != 4:
        raise ValueError(
            "target_latents 必须为 [B,C,H,W]"
        )

    transformer_batch = target_latents.shape[0]

    target_tokens = latent_token_count(
        target_latents,
        patch_size,
    )

    # 默认没有参考图时，总长度就是 Target token 数。
    full_tokens = target_tokens

    if cond_hidden_states is not None:
        if len(cond_hidden_states) != transformer_batch:
            raise ValueError(
                "cond_hidden_states 与 Transformer batch 不一致："
                f"{len(cond_hidden_states)} "
                f"vs {transformer_batch}"
            )

        sample_lengths = []

        for refs in cond_hidden_states:
            sample_tokens = target_tokens

            for ref in refs:
                sample_tokens += latent_token_count(
                    ref,
                    patch_size,
                )

            sample_lengths.append(sample_tokens)

        # DeepGen 会把不同长度样本右侧 padding 到同一长度。
        full_tokens = max(sample_lengths)

    aligned = []

    for residual in residuals:
        if residual.ndim != 3:
            raise ValueError(
                "Pose residual 必须为 [B,N,D]"
            )

        pose_batch, pose_tokens, hidden_dim = residual.shape

        if pose_tokens != target_tokens:
            raise ValueError(
                "Pose token 数和 Target token 数不一致："
                f"{pose_tokens} vs {target_tokens}"
            )

        if transformer_batch % pose_batch != 0:
            raise ValueError(
                "Pose batch 无法对齐到 Transformer batch："
                f"{pose_batch} -> {transformer_batch}"
            )

        repeat_factor = (
            transformer_batch // pose_batch
        )

        # DeepGen 开启 CFG 时会把 target batch 扩展。
        # 姿态条件同步扩展到相同 batch。
        if repeat_factor > 1:
            residual = torch.cat(
                [residual] * repeat_factor,
                dim=0,
            )

        zero_tokens = (
            full_tokens - target_tokens
        )

        if zero_tokens > 0:
            zeros = residual.new_zeros(
                transformer_batch,
                zero_tokens,
                hidden_dim,
            )

            residual = torch.cat(
                [residual, zeros],
                dim=1,
            )

        aligned.append(residual)

    return aligned

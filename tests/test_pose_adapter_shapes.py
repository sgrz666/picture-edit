import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(
    0,
    str(SRC),
)

from pose_control.deepgen_pose_adapter import (
    DeepGenPoseAdapter,
)
from pose_control.token_alignment import (
    align_pose_residuals,
    latent_token_count,
)


def run_shape_test():
    """
    第一阶段：
    不加载 DeepGen，只验证 Adapter 自身的 shape contract。
    """

    print(
        "===== 1. POSE ADAPTER SHAPE TEST ====="
    )

    adapter = DeepGenPoseAdapter(
        input_channels=3,
        hidden_dim=1536,
        num_residual_heads=6,
    )

    # 一张目标 Skeleton。
    pose_map = torch.randn(
        1,
        3,
        512,
        512,
    )

    # 模拟 DeepGen 开启 CFG 后的 target latent。
    target_latents = torch.zeros(
        2,
        16,
        64,
        64,
    )

    # CFG 两个 sample，
    # 每个 sample 各有一张 source/reference image。
    cond_hidden_states = [
        [
            torch.zeros(
                16,
                64,
                64,
            )
        ],
        [
            torch.zeros(
                16,
                64,
                64,
            )
        ],
    ]

    residuals = adapter(
        pose_map=pose_map,
        target_latents=target_latents,
        cond_hidden_states=cond_hidden_states,
        patch_size=2,
    )

    assert len(residuals) == 6

    for index, residual in enumerate(
        residuals
    ):
        assert residual.shape == (
            2,
            2048,
            1536,
        ), (
            f"residual[{index}] shape 错误："
            f"{tuple(residual.shape)}"
        )

        # Zero-init 后初始输出必须严格为 0。
        assert (
            torch.count_nonzero(
                residual
            ).item()
            == 0
        )

    print(
        "Adapter output:",
        len(residuals),
        "x",
        tuple(
            residuals[0].shape
        ),
    )

    # 单独测试 TokenAlignment：
    # Target residual 设置为 1，
    # Source 部分必须严格保持 0。
    raw_residual = [
        torch.ones(
            1,
            1024,
            8,
        )
    ]

    aligned = align_pose_residuals(
        residuals=raw_residual,
        target_latents=target_latents,
        cond_hidden_states=cond_hidden_states,
        patch_size=2,
    )[0]

    assert aligned.shape == (
        2,
        2048,
        8,
    )

    assert torch.all(
        aligned[:, :1024] == 1
    )

    assert torch.all(
        aligned[:, 1024:] == 0
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}",
    )

    print(
        "Target tokens : 1024"
    )
    print(
        "Source tokens : 1024"
    )
    print(
        "Full tokens   : 2048"
    )

    print(
        "Source residual strictly zero: PASS"
    )

    print(
        "Shape test: PASS"
    )


def clone_cond_hidden_states(
    cond_hidden_states,
):
    """
    深拷贝 DeepGen 的 source/reference latent 列表。
    """

    if cond_hidden_states is None:
        return None

    return [
        [
            ref.detach().clone()
            for ref in refs
        ]
        for refs in cond_hidden_states
    ]


def clone_if_tensor(value):
    if torch.is_tensor(value):
        return (
            value
            .detach()
            .clone()
        )

    return value


def run_real_deepgen_test(
    model_path: str,
    input_image: str,
):
    """
    第二阶段：
    从一次真实 DeepGen image-editing forward 中
    捕获真实 Tensor，然后把 Pose Adapter residual
    送进真实 DeepGen Transformer。

    本测试不修改 DeepGen 源码和权重。
    """

    from PIL import Image
    from diffusers import DiffusionPipeline

    print()
    print(
        "===== 2. REAL DEEPGEN FORWARD TEST ====="
    )

    device = "cuda"
    dtype = torch.bfloat16

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )

    pipe.to(device)

    captured = {}

    original_forward = (
        pipe.transformer.forward
    )

    def capture_forward(
        *args,
        **kwargs,
    ):
        # 只记录第一次真实 denoising forward。
        if not captured:
            hidden_states = kwargs.get(
                "hidden_states"
            )

            if (
                hidden_states is None
                and len(args) > 0
            ):
                hidden_states = args[0]

            captured["hidden_states"] = (
                hidden_states
                .detach()
                .clone()
            )

            captured[
                "cond_hidden_states"
            ] = clone_cond_hidden_states(
                kwargs.get(
                    "cond_hidden_states"
                )
            )

            for name in (
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
            ):
                captured[name] = (
                    clone_if_tensor(
                        kwargs.get(name)
                    )
                )

            joint_kwargs = kwargs.get(
                "joint_attention_kwargs"
            )

            captured[
                "joint_attention_kwargs"
            ] = (
                dict(joint_kwargs)
                if isinstance(
                    joint_kwargs,
                    dict,
                )
                else joint_kwargs
            )

        return original_forward(
            *args,
            **kwargs,
        )

    pipe.transformer.forward = (
        capture_forward
    )

    source_image = Image.open(
        input_image
    ).convert("RGB")

    # 用户侧不使用文本控制。
    # 这里只提供固定中性字符串，
    # 用来保持 DeepGen 原始预训练计算路径。
    # 它不包含任何目标姿态信息。
    neutral_prompt = (
        "Edit the image according "
        "to the provided condition."
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            pipe(
                prompt=neutral_prompt,
                image=source_image,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=1,
                guidance_scale=4.0,
                seed=42,
            )
    finally:
        # 恢复原始 forward，
        # 避免测试 hook 影响后续运行。
        pipe.transformer.forward = (
            original_forward
        )

    if not captured:
        raise RuntimeError(
            "没有捕获到真实 DeepGen Transformer 输入"
        )

    target_latents = captured[
        "hidden_states"
    ]

    cond_hidden_states = captured[
        "cond_hidden_states"
    ]

    patch_size = (
        pipe.transformer.config.patch_size
    )

    target_tokens = latent_token_count(
        target_latents,
        patch_size,
    )

    print(
        "Real target latent:",
        tuple(
            target_latents.shape
        ),
    )

    adapter = DeepGenPoseAdapter(
        input_channels=3,
        hidden_dim=(
            pipe.transformer.config.num_attention_heads
            * pipe.transformer.config.attention_head_dim
        ),
        num_residual_heads=6,
    ).to(
        device=device,
        dtype=dtype,
    )

    # 当前只验证接口，
    # 不验证 Skeleton 的语义质量。
    pose_map = torch.zeros(
        1,
        3,
        512,
        512,
        device=device,
        dtype=dtype,
    )

    residuals = adapter(
        pose_map=pose_map,
        target_latents=target_latents,
        cond_hidden_states=cond_hidden_states,
        patch_size=patch_size,
    )

    print(
        "Pose residual:",
        len(residuals),
        "x",
        tuple(
            residuals[0].shape
        ),
    )

    assert len(residuals) == 6

    assert (
        residuals[0].shape[0]
        == target_latents.shape[0]
    )

    assert (
        residuals[0].shape[-1]
        == 1536
    )

    # Zero-init Adapter 初始输出应严格为零。
    assert (
        torch.count_nonzero(
            residuals[0]
        ).item()
        == 0
    )

    # --------------------------------------------------
    # 真实验证 block_controlnet_hidden_states
    #
    # 不能只把“全零 residual”送进去，
    # 否则只能证明 shape 没报错。
    #
    # 因此测试中人为给 Target token
    # 一个极小非零 residual。
    # Source 部分继续严格为零。
    # --------------------------------------------------

    probe_residuals = [
        residual.clone()
        for residual in residuals
    ]

    probe_residuals[0][
        :,
        :target_tokens,
        :,
    ] = 1e-3

    assert torch.all(
        probe_residuals[0][
            :,
            target_tokens:,
            :,
        ]
        == 0
    )

    common_kwargs = dict(
        hidden_states=target_latents,
        cond_hidden_states=cond_hidden_states,
        encoder_hidden_states=captured[
            "encoder_hidden_states"
        ],
        pooled_projections=captured[
            "pooled_projections"
        ],
        timestep=captured[
            "timestep"
        ],
        joint_attention_kwargs=captured[
            "joint_attention_kwargs"
        ],
        return_dict=False,
    )

    # 同一组真实 DeepGen 输入：
    # 一次不用控制，一次加入非零 Pose residual。
    with torch.inference_mode():
        output_without_control = (
            original_forward(
                **common_kwargs,
                block_controlnet_hidden_states=None,
            )[0]
        )

        output_with_control = (
            original_forward(
                **common_kwargs,
                block_controlnet_hidden_states=probe_residuals,
            )[0]
        )

    print(
        "DeepGen output:",
        tuple(
            output_with_control.shape
        ),
    )

    assert (
        output_with_control.shape
        == output_without_control.shape
    )

    assert torch.isfinite(
        output_with_control
    ).all()

    max_difference = (
        (
            output_with_control.float()
            - output_without_control.float()
        )
        .abs()
        .max()
        .item()
    )

    print(
        "Control-induced max difference:",
        max_difference,
    )

    if max_difference <= 0:
        raise AssertionError(
            "加入非零 Pose residual 后 "
            "DeepGen 输出没有发生任何变化"
        )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        "Peak VRAM:",
        f"{peak_vram:.3f} GB",
    )

    print(
        "Source residual remains zero: PASS"
    )

    print(
        "Non-zero control changes DeepGen output: PASS"
    )

    print(
        "Real DeepGen forward test: PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deepgen",
        action="store_true",
        help="额外执行真实 DeepGen forward 测试",
    )

    parser.add_argument(
        "--model-path",
        default=(
            "/home/lvjy/projects/picture-edit/"
            "models/DeepGen-1.0-diffusers"
        ),
    )

    parser.add_argument(
        "--input-image",
        default=(
            "/home/lvjy/projects/picture-edit/"
            "inputs/person1.jpg"
        ),
    )

    args = parser.parse_args()

    run_shape_test()

    if args.deepgen:
        run_real_deepgen_test(
            model_path=args.model_path,
            input_image=args.input_image,
        )


if __name__ == "__main__":
    main()

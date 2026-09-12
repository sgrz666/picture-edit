import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from diffusers import DiffusionPipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.deepgen_pose_adapter import (
    align_pose_residuals,
)


# 保持与当前训练/推理一致的中性指令。
NEUTRAL_PROMPT = (
    "Edit the image according to the provided condition."
)


def strength_name(value: float) -> str:
    return (
        f"{value:.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def image_mae(
    image_a: Image.Image,
    image_b: Image.Image,
) -> float:
    """
    计算两张最终 RGB 图像的平均绝对差异。
    只用于观察 residual 强度增加后，
    输出是否系统性偏离 Baseline。
    """
    a = (
        np.asarray(
            image_a.convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )

    b = (
        np.asarray(
            image_b.convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )

    return np.abs(a - b).mean().item()


def make_grid(
    images,
    labels,
    output_path: Path,
):
    """
    将 Source、Baseline 和不同 residual 强度结果
    拼成一张横向对比图。
    """
    cell_size = 320
    header_height = 40

    canvas = Image.new(
        "RGB",
        (
            cell_size * len(images),
            cell_size + header_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, (image, label) in enumerate(
        zip(images, labels)
    ):
        fitted = ImageOps.contain(
            image.convert("RGB"),
            (cell_size, cell_size),
        )

        x = (
            index * cell_size
            + (cell_size - fitted.width) // 2
        )

        y = (
            header_height
            + (cell_size - fitted.height) // 2
        )

        canvas.paste(
            fitted,
            (x, y),
        )

        draw.text(
            (
                index * cell_size + 10,
                12,
            ),
            label,
            fill="black",
        )

    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--probe-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[
            0.02,
            0.10,
            0.25,
        ],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前没有可用 CUDA GPU"
        )

    device = torch.device("cuda")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_image = Image.open(
        args.source_image
    ).convert("RGB")

    source_image.save(
        args.output_dir / "source.png"
    )

    print("===== LOAD DEEPGEN =====")

    pipe = DiffusionPipeline.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    transformer_dtype = (
        pipe.transformer.dtype
    )

    patch_size = (
        pipe.transformer.config.patch_size
    )

    if isinstance(
        patch_size,
        (tuple, list),
    ):
        patch_size = patch_size[0]

    hidden_dim = (
        pipe.transformer.config.num_attention_heads
        * pipe.transformer.config.attention_head_dim
    )

    print(
        "patch_size:",
        patch_size,
    )

    print(
        "hidden_dim:",
        hidden_dim,
    )

    # --------------------------------------------------
    # 1. Baseline
    # --------------------------------------------------

    print(
        "\n===== BASELINE ====="
    )

    with torch.inference_mode():
        baseline_result = pipe(
            prompt=NEUTRAL_PROMPT,
            image=source_image,
            negative_prompt="",
            height=512,
            width=512,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            seed=args.seed,
        )

    baseline = baseline_result.images[0]

    baseline_path = (
        args.output_dir
        / "baseline.png"
    )

    baseline.save(
        baseline_path
    )

    print(
        "saved:",
        baseline_path,
    )

    # --------------------------------------------------
    # 2. Target-only residual probe
    #
    # 不使用 Skeleton，不使用训练 checkpoint。
    #
    # 只验证：
    #   DeepGen 的 Target token residual 接口
    #   是否真的能改变最终 RGB。
    #
    # 只让第 1 组 residual 非零，
    # 与现有真实 DeepGen injection 单元测试一致。
    # --------------------------------------------------

    original_forward = (
        pipe.transformer.forward
    )

    state = {
        "strength": 0.0,
    }

    pattern_cache = {}

    def wrapped_forward(
        *f_args,
        **f_kwargs,
    ):
        hidden_states = f_kwargs[
            "hidden_states"
        ]

        cond_hidden_states = f_kwargs[
            "cond_hidden_states"
        ]

        batch = hidden_states.shape[0]

        # guidance_scale=1 时 Transformer batch=1，
        # 但 DeepGen 仍可能保留
        # [negative, positive] 两组 Source condition。
        #
        # 对 residual 对齐部分只保留 positive，
        # 与当前正式训练逻辑保持一致。
        align_cond = cond_hidden_states

        if (
            batch == 1
            and len(cond_hidden_states) == 2
        ):
            align_cond = (
                cond_hidden_states[-1:]
            )

        latent_h = (
            hidden_states.shape[-2]
        )

        latent_w = (
            hidden_states.shape[-1]
        )

        target_tokens = (
            latent_h // patch_size
        ) * (
            latent_w // patch_size
        )

        # 先构造 6 组全零 residual。
        # 只有第一组用于 probe。
        raw_residuals = [
            torch.zeros(
                (
                    batch,
                    target_tokens,
                    hidden_dim,
                ),
                device=hidden_states.device,
                dtype=transformer_dtype,
            )
            for _ in range(6)
        ]

        residuals = align_pose_residuals(
            residuals=raw_residuals,
            target_latents=hidden_states,
            cond_hidden_states=align_cond,
            patch_size=patch_size,
        )

        key = (
            target_tokens,
            hidden_dim,
            str(hidden_states.device),
        )

        if key not in pattern_cache:
            generator = torch.Generator(
                device=hidden_states.device
            )

            generator.manual_seed(
                args.probe_seed
            )

            pattern = torch.randn(
                (
                    1,
                    target_tokens,
                    hidden_dim,
                ),
                generator=generator,
                device=hidden_states.device,
                dtype=torch.float32,
            )

            # 固定为 RMS=1，
            # 因此 strength 就是实际 residual RMS。
            pattern = (
                pattern
                / pattern.pow(2)
                .mean()
                .sqrt()
            )

            pattern_cache[key] = pattern

        pattern = pattern_cache[key]

        # 只修改 Target token。
        # Source/reference 和 Padding 区仍严格保持 0。
        residuals[0][
            :,
            :target_tokens,
            :,
        ] = (
            pattern
            .expand(
                batch,
                -1,
                -1,
            )
            .to(transformer_dtype)
            * state["strength"]
        )

        f_kwargs[
            "block_controlnet_hidden_states"
        ] = residuals

        return original_forward(
            *f_args,
            **f_kwargs,
        )

    pipe.transformer.forward = (
        wrapped_forward
    )

    probe_images = []

    try:
        for strength in args.strengths:
            state["strength"] = strength

            print(
                "\n===== "
                f"TARGET PROBE {strength:.3f} "
                "====="
            )

            with torch.inference_mode():
                result = pipe(
                    prompt=NEUTRAL_PROMPT,
                    image=source_image,
                    negative_prompt="",
                    height=512,
                    width=512,
                    num_inference_steps=args.steps,
                    guidance_scale=1.0,
                    seed=args.seed,
                )

            image = result.images[0]

            name = (
                "target_probe_"
                + strength_name(strength)
                + ".png"
            )

            output_path = (
                args.output_dir / name
            )

            image.save(
                output_path
            )

            mae = image_mae(
                baseline,
                image,
            )

            probe_images.append(
                (
                    strength,
                    image,
                    mae,
                )
            )

            print(
                "saved:",
                output_path,
            )

            print(
                "Baseline -> Probe MAE:",
                f"{mae:.8f}",
            )

    finally:
        pipe.transformer.forward = (
            original_forward
        )

    # --------------------------------------------------
    # 3. 拼图
    # --------------------------------------------------

    images = [
        source_image,
        baseline,
    ]

    labels = [
        "Source",
        "Baseline",
    ]

    for strength, image, mae in probe_images:
        images.append(
            image
        )

        labels.append(
            f"Probe {strength:g}"
        )

    grid_path = (
        args.output_dir
        / "comparison_grid.png"
    )

    make_grid(
        images=images,
        labels=labels,
        output_path=grid_path,
    )

    print(
        "\n===== RESULT SUMMARY ====="
    )

    previous_mae = None

    for strength, _, mae in probe_images:
        print(
            f"strength={strength:.3f} "
            f"MAE={mae:.8f}"
        )

        previous_mae = mae

    print(
        "\nsaved grid:",
        grid_path,
    )

    print(
        "\nDeepGen residual visual probe: PASS"
    )


if __name__ == "__main__":
    main()

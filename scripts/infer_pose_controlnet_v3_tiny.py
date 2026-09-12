import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.deepgen_pose_controlnet_v3 import (
    DeepGenPoseControlNetV3,
)


NEUTRAL_PROMPT = (
    "Edit the image according to the provided condition."
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pose_tensor(
    path: Path,
    device: torch.device,
):
    image = Image.open(path).convert("RGB")

    array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )


def select_condition_batch(
    value,
    batch_size: int,
):
    """
    DeepGen 在 CFG 情况下可能保存
    [negative, positive] 两组 condition。

    guidance=1 的当前验证只使用 positive branch。
    """

    if torch.is_tensor(value):
        if (
            batch_size == 1
            and value.shape[0] == 2
        ):
            return value[-1:]

        return value

    if (
        batch_size == 1
        and len(value) == 2
    ):
        return value[-1:]

    return value


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
        "--pose-image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--control-scale",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA GPU")

    device = torch.device("cuda")

    set_seed(args.seed)

    print("===== LOAD DEEPGEN =====")

    pipe = DiffusionPipeline.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)

    transformer_dtype = pipe.transformer.dtype
    patch_size = pipe.transformer.config.patch_size

    print("===== BUILD CONTROLNET V3 =====")

    controlnet_v3 = DeepGenPoseControlNetV3(
        deepgen_transformer=pipe.transformer,
        num_layers=6,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    controlnet_v3.load_state_dict(
        checkpoint["controlnet_v3"],
        strict=True,
    )

    controlnet_v3.eval()

    print(
        "checkpoint step:",
        checkpoint.get("step"),
    )

    # --------------------------------------------------
    # Target Skeleton -> DeepGen VAE latent
    # --------------------------------------------------

    pose_map = load_pose_tensor(
        path=args.pose_image,
        device=device,
    )

    with torch.inference_mode():
        pose_pixels = (
            pose_map * 2.0 - 1.0
        ).to(
            dtype=transformer_dtype
        )

        pose_latents = pipe.pixels_to_latents(
            pose_pixels
        )

    source_image = Image.open(
        args.source_image
    ).convert("RGB")

    original_forward = (
        pipe.transformer.forward
    )

    def wrapped_forward(
        *forward_args,
        **forward_kwargs,
    ):
        hidden_states = forward_kwargs[
            "hidden_states"
        ]

        batch_size = hidden_states.shape[0]

        cond_hidden_states = (
            select_condition_batch(
                forward_kwargs[
                    "cond_hidden_states"
                ],
                batch_size,
            )
        )

        encoder_hidden_states = (
            select_condition_batch(
                forward_kwargs[
                    "encoder_hidden_states"
                ],
                batch_size,
            )
        )

        pooled_projections = (
            select_condition_batch(
                forward_kwargs[
                    "pooled_projections"
                ],
                batch_size,
            )
        )

        current_pose_latents = pose_latents

        if pose_latents.shape[0] != batch_size:
            current_pose_latents = (
                pose_latents.repeat(
                    batch_size,
                    1,
                    1,
                    1,
                )
            )

        residuals = controlnet_v3(
            target_latents=hidden_states,
            pose_latents=current_pose_latents,
            cond_hidden_states=cond_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=forward_kwargs[
                "timestep"
            ],
            patch_size=patch_size,
            conditioning_scale=args.control_scale,
        )

        residuals = [
            residual.to(
                dtype=hidden_states.dtype
            )
            for residual in residuals
        ]

        forward_kwargs[
            "block_controlnet_hidden_states"
        ] = residuals

        return original_forward(
            *forward_args,
            **forward_kwargs,
        )

    pipe.transformer.forward = wrapped_forward

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        print("===== INFER =====")

        with torch.inference_mode():
            result = pipe(
                prompt=NEUTRAL_PROMPT,
                image=source_image,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

        result.images[0].save(
            args.output
        )

    finally:
        pipe.transformer.forward = (
            original_forward
        )

    print(
        "saved:",
        args.output,
    )

    print(
        "Pose ControlNet V3 inference: PASS"
    )


if __name__ == "__main__":
    main()

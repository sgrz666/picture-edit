import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.deepgen_pose_adapter_v2 import DeepGenPoseAdapterV2

NEUTRAL_PROMPT = "Edit the image according to the provided condition."


def load_pose_tensor(path: Path, device: torch.device):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--pose-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="不传则运行 baseline DeepGen",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前没有可用 CUDA GPU")

    device = torch.device("cuda")

    print("===== LOAD PIPELINE =====")
    pipe = DiffusionPipeline.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to(device)

    source_image = Image.open(args.source_image).convert("RGB")
    pose_map = load_pose_tensor(args.pose_image, device)

    patch_size = pipe.transformer.config.patch_size
    hidden_dim = (
        pipe.transformer.config.num_attention_heads
        * pipe.transformer.config.attention_head_dim
    )

    adapter = None
    if args.checkpoint is not None:
        print("===== LOAD ADAPTER =====")
        ckpt = torch.load(
            args.checkpoint,
            map_location="cpu",
        )

        adapter = DeepGenPoseAdapterV2(
            input_channels=3,
            hidden_dim=hidden_dim,
            num_residual_heads=6,
        ).to(device=device, dtype=torch.float32)

        adapter.load_state_dict(ckpt["adapter"], strict=True)
        adapter.eval()

        print("checkpoint:", args.checkpoint)
        print("step      :", ckpt.get("step", "unknown"))
        print("identity  :", ckpt.get("identity", "unknown"))

    original_forward = pipe.transformer.forward

    if adapter is not None:
        def wrapped_forward(*f_args, **f_kwargs):
            hidden_states = f_kwargs["hidden_states"]
            cond_hidden_states = f_kwargs["cond_hidden_states"]

            # 训练时只使用 DeepGen CFG 中的 positive source branch。
            # CFG=1 推理时 Transformer batch=1，但 DeepGen pipeline
            # 仍可能保留 [negative, positive] 两组 source condition。
            # 因此 Adapter 侧必须与训练保持一致：只取 positive branch。
            adapter_cond_hidden_states = cond_hidden_states
            if (
                hidden_states.shape[0] == 1
                and len(cond_hidden_states) == 2
            ):
                adapter_cond_hidden_states = cond_hidden_states[-1:]

            local_pose = pose_map
            if local_pose.shape[0] != hidden_states.shape[0]:
                local_pose = local_pose.expand(
                    hidden_states.shape[0], -1, -1, -1
                ).contiguous()

            residuals = adapter(
                pose_map=local_pose,
                target_latents=hidden_states,
                cond_hidden_states=adapter_cond_hidden_states,
                patch_size=patch_size,
            )

            residuals = [
                residual.to(dtype=pipe.transformer.dtype)
                for residual in residuals
            ]

            f_kwargs["block_controlnet_hidden_states"] = residuals
            return original_forward(*f_args, **f_kwargs)

        pipe.transformer.forward = wrapped_forward

    try:
        print("===== INFER =====")
        torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            result = pipe(
                prompt=NEUTRAL_PROMPT,
                image=source_image,
                negative_prompt="",
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(args.output)

        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_resv = torch.cuda.max_memory_reserved(device) / 1024**3

        print("saved:", args.output)
        print(f"peak allocated: {peak_alloc:.2f} GB")
        print(f"peak reserved : {peak_resv:.2f} GB")
        print("Inference finished: PASS")
    finally:
        pipe.transformer.forward = original_forward


if __name__ == "__main__":
    main()

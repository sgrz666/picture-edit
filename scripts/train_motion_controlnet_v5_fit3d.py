import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.deepgen_motion_controlnet_v5 import (
    DeepGenMotionControlNetV5,
)
from pose_control.fit3d_motion_condition import (
    build_fit3d_motion_condition,
)# 用户侧不使用文本控制。
# 固定中性 prompt 只用于保留 DeepGen 原生 image-editing 条件路径。
NEUTRAL_PROMPT = (
    "Edit the image according to the provided condition."
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb_tensor(
    path: Path,
    device: torch.device,
    normalize_vae: bool,
):
    """
    读取已经准备好的 512x512 RGB。

    target RGB:
        [0,255] -> [-1,1]，供 VAE 使用。

    pose map:
        [0,255] -> [0,1]，供 PoseEncoder 使用。
    """
    image = Image.open(path).convert("RGB")

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )

    if normalize_vae:
        tensor = tensor * 2.0 - 1.0

    return tensor


def clone_cond_hidden_states(cond_hidden_states):
    return [
        [
            ref.detach().clone()
            for ref in refs
        ]
        for refs in cond_hidden_states
    ]


def capture_source_conditioning(
    pipe,
    source_path: Path,
):
    """
    使用已经验证过的真实 DeepGen image-editing 路径，
    提取固定 Source 对应的条件。

    这里不重新实现 Qwen2.5-VL / SCB / connector，
    而是让真实 DeepGen Pipeline 自己完成它们。

    每个 identity 只执行一次，之后训练直接复用缓存。
    """
    captured = {}

    original_forward = pipe.transformer.forward

    def capture_forward(*args, **kwargs):
        if not captured:
            encoder_hidden_states = kwargs[
                "encoder_hidden_states"
            ]

            pooled_projections = kwargs[
                "pooled_projections"
            ]

            cond_hidden_states = kwargs[
                "cond_hidden_states"
            ]

            captured["encoder_hidden_states"] = (
                encoder_hidden_states
                .detach()
                .clone()
            )

            captured["pooled_projections"] = (
                pooled_projections
                .detach()
                .clone()
            )

            captured["cond_hidden_states"] = (
                clone_cond_hidden_states(
                    cond_hidden_states
                )
            )

        return original_forward(
            *args,
            **kwargs,
        )

    pipe.transformer.forward = capture_forward

    source_image = Image.open(
        source_path
    ).convert("RGB")

    try:
        with torch.inference_mode():
            pipe(
                prompt=NEUTRAL_PROMPT,
                image=source_image,
                negative_prompt="",
                height=512,
                width=512,
                num_inference_steps=1,
                guidance_scale=4.0,
                seed=42,
            )
    finally:
        pipe.transformer.forward = original_forward

    if not captured:
        raise RuntimeError(
            f"未捕获到 Source condition：{source_path}"
        )

    # DeepGen CFG forward 中 Transformer batch=2：
    #
    #   [negative, positive]
    #
    # 正式训练不做 CFG，因此这里只保留 positive 分支。
    encoder_hidden_states = (
        captured["encoder_hidden_states"][-1:]
        .contiguous()
    )

    pooled_projections = (
        captured["pooled_projections"][-1:]
        .contiguous()
    )

    cond_hidden_states = (
        captured["cond_hidden_states"][-1:]
    )

    return {
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "cond_hidden_states": cond_hidden_states,
    }



def build_pose_region_weight(
    pose_map: torch.Tensor,
    spatial_size,
    pose_weight: float = 4.0,
    region_kernel: int = 61,
):
    """
    根据 Target Skeleton 构造姿态区域权重。

    思路来自 HumanSD 的 heatmap-guided denoising loss
    和 MimicMotion 的 regional loss amplification。

    DeepGen 适配：
        Skeleton 512x512
            ↓
        二值姿态区域
            ↓
        膨胀，覆盖骨架附近人体区域
            ↓
        下采样到 DeepGen flow prediction 空间
            ↓
        背景权重 1，姿态区域权重 pose_weight

    最后将空间权重归一化到 mean=1，
    只改变 loss 的空间关注分布，
    不整体放大 loss 或有效学习率。
    """
    if pose_weight < 1.0:
        raise ValueError(
            "pose_weight 必须 >= 1"
        )

    if (
        region_kernel < 1
        or region_kernel % 2 == 0
    ):
        raise ValueError(
            "region_kernel 必须是正奇数"
        )

    # 彩色 Skeleton → 单通道。
    pose_strength = pose_map.float().amax(
        dim=1,
        keepdim=True,
    )

    # 黑背景为 0；
    # skeleton 像素构成初始姿态区域。
    region = (
        pose_strength > 0.05
    ).float()

    # Skeleton 本身过于稀疏。
    # 将骨架附近扩展成连续人体姿态区域。
    region = F.max_pool2d(
        region,
        kernel_size=region_kernel,
        stride=1,
        padding=region_kernel // 2,
    )

    # DeepGen 的 loss 工作在 latent/flow 空间，
    # 因此把 512x512 区域映射到 model_pred 尺寸。
    region = F.interpolate(
        region,
        size=spatial_size,
        mode="area",
    )

    region = region.clamp(
        0.0,
        1.0,
    )

    spatial_weight = (
        1.0
        + (pose_weight - 1.0) * region
    )

    # 保持每张图的平均 loss scale 不变。
    spatial_weight = spatial_weight / (
        spatial_weight.mean(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-6)
    )

    return spatial_weight, region


def make_flow_match_sample(
    pipe,
    x0: torch.Tensor,
):
    batch_size = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    noise = torch.randn_like(x0)

    timestep_sampling = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=batch_size,
        device=device,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    )

    indices = (
        timestep_sampling
        * pipe.scheduler.config.num_train_timesteps
    ).long().clamp(
        0,
        pipe.scheduler.config.num_train_timesteps - 1,
    )

    timesteps = pipe.scheduler.timesteps[
        indices
    ].to(device=device)

    sigmas = pipe.scheduler.sigmas[
        indices
    ].to(
        device=device,
        dtype=dtype,
    )

    while sigmas.ndim < x0.ndim:
        sigmas = sigmas.unsqueeze(-1)

    noisy_latents = (
        (1.0 - sigmas) * x0
        + sigmas * noise
    )

    target_velocity = noise - x0

    weighting = compute_loss_weighting_for_sd3(
        weighting_scheme="cosmap",
        sigmas=sigmas,
    )

    return (
        noisy_latents,
        timesteps,
        target_velocity,
        weighting,
    )


def load_samples(
    data_root: Path,
    identity: str,
):
    manifest_path = (
        data_root / "manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    samples = manifest["samples"]

    if identity != "all":
        samples = [
            sample
            for sample in samples
            if sample["identity"] == identity
        ]

    if not samples:
        raise ValueError(
            f"没有找到 identity={identity} 的样本"
        )

    return samples


def save_checkpoint(
    adapter,
    optimizer,
    step,
    output_dir,
    identity,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"controlnet_v5_step_{step:06d}.pt"
    )

    torch.save(
        {
            "step": step,
            "identity": identity,
            "controlnet_v5": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

    print(
        f"[checkpoint] {path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/lvjy/projects/iper_data/"
            "tiny_pose_adapter_v1"
        ),
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/home/lvjy/projects/picture-edit/"
            "models/DeepGen-1.0-diffusers"
        ),
    )

    parser.add_argument(
        "--identity",
        default="s05",
        help=(
            "Stage 0 使用 008_1；"
            "Stage 1 使用 all"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--pose-loss-weight",
        type=float,
        default=4.0,
        help=(
            "Target Skeleton 附近区域的相对 loss 权重；"
            "1.0 等价于原始全图 loss"
        ),
    )

    parser.add_argument(
        "--pose-region-kernel",
        type=int,
        default=61,
        help=(
            "512x512 Skeleton 区域膨胀核大小，"
            "必须为奇数"
        ),
    )

    parser.add_argument(
        "--pose-encoder-lr",
        type=float,
        default=3e-4,
        help="MimicMotion-style PoseEncoder 学习率",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/lvjy/projects/picture-edit/"
            "repro_outputs/pose_adapter_tiny"
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="控制台打印间隔；CSV 仍然每步记录",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前没有可用 CUDA GPU"
        )

    device = torch.device("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    set_seed(args.seed)

    samples = load_samples(
        data_root=args.data_root,
        identity=args.identity,
    )

    identities = sorted(
        {
            sample["identity"]
            for sample in samples
        }
    )

    print(
        "===== TRAIN CONFIG ====="
    )
    print(
        "identity  :", args.identity
    )
    print(
        "samples   :", len(samples)
    )
    print(
        "identities:", identities
    )
    print(
        "steps     :", args.max_steps
    )
    print(
        "lr        :", args.lr
    )

    # --------------------------------------------------
    # DeepGen
    # --------------------------------------------------

    pipe = DiffusionPipeline.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe.to(device)

    # DeepGen 主干严格冻结。
    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()

    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    transformer_dtype = (
        pipe.transformer.dtype
    )

    patch_size = (
        pipe.transformer.config.patch_size
    )

    hidden_dim = (
        pipe.transformer.config.num_attention_heads
        * pipe.transformer.config.attention_head_dim
    )

    print(
        "DeepGen hidden_dim:",
        hidden_dim,
    )
    print(
        "DeepGen patch_size:",
        patch_size,
    )

    # --------------------------------------------------
    # Source condition cache
    # 一个 identity 的所有 target 共用同一个 source，
    # 所以没有理由每个训练 step 都重新跑 VLM。
    # --------------------------------------------------

    source_cache = {}

    print(
        "\n===== CACHE SOURCE CONDITIONS ====="
    )

    for identity in identities:
        sample = next(
            item
            for item in samples
            if item["identity"] == identity
        )

        source_path = (
            args.data_root
            / sample["source"]
        )

        print(
            f"cache {identity}: "
            f"{source_path}"
        )

        source_cache[
            identity
        ] = capture_source_conditioning(
            pipe=pipe,
            source_path=source_path,
        )

        torch.cuda.empty_cache()

    # --------------------------------------------------
    # 记录 DeepGen 加载 + Source condition 缓存峰值。
    # --------------------------------------------------
    init_peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        / 1024**3
    )
    init_peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        / 1024**3
    )
    current_allocated = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )
    current_reserved = (
        torch.cuda.memory_reserved(device)
        / 1024**3
    )

    print(
        "\n===== VRAM AFTER SOURCE CACHE ====="
    )
    print(
        f"Current allocated : "
        f"{current_allocated:.2f} GB"
    )
    print(
        f"Current reserved  : "
        f"{current_reserved:.2f} GB"
    )
    print(
        f"Init/cache peak allocated : "
        f"{init_peak_allocated:.2f} GB"
    )
    print(
        f"Init/cache peak reserved  : "
        f"{init_peak_reserved:.2f} GB"
    )

    # --------------------------------------------------
    # 恢复训练 scheduler
    #
    # Source condition cache 会调用一次 DeepGen 推理，
    # num_inference_steps=1 会修改 scheduler.timesteps。
    #
    # 后续 flow-matching 训练需要完整训练时间表，
    # 因此这里从原配置重新创建 scheduler，
    # 避免 timestep/sigma 索引越界。
    # --------------------------------------------------
    pipe.scheduler = (
        pipe.scheduler.__class__.from_config(
            pipe.scheduler.config
        )
    )

    # scheduler 重新创建后默认在 CPU。
    # 训练中的 timestep index 位于 GPU，
    # 因此将训练所需的时间表放到同一设备。
    pipe.scheduler.timesteps = (
        pipe.scheduler.timesteps.to(device)
    )

    pipe.scheduler.sigmas = (
        pipe.scheduler.sigmas.to(device)
    )

    print(
        "Training scheduler timesteps:",
        len(pipe.scheduler.timesteps),
    )

    print(
        "Training scheduler sigmas:",
        len(pipe.scheduler.sigmas),
    )

    # 从这里重新统计真正训练阶段的显存峰值。
    torch.cuda.reset_peak_memory_stats(device)

    # Source condition 的提取会消耗随机数，
    # 训练开始前重新固定 seed。
    set_seed(args.seed)

    # Pose Adapter

    adapter = DeepGenMotionControlNetV5(
        deepgen_transformer=pipe.transformer,
        num_layers=6,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )

    print(
        "\nMotion ControlNet V5 trainable params:",
        f"{trainable_params:,}",
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter.controlnet.parameters(),
                "lr": args.lr,
            },
            {
                "params": adapter.pose_encoder.parameters(),
                "lr": args.pose_encoder_lr,
            },
        ],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    output_dir = (
        args.output_dir
        / args.identity
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_csv = output_dir / "train_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "identity",
            "target",
            "loss",
            "grad_norm",
        ])

    adapter.train()

    print(
        "\n===== TRAIN START ====="
    )

    for step in range(
        1,
        args.max_steps + 1,
    ):
        sample = random.choice(
            samples
        )

        identity = sample["identity"]

        target_path = (
            args.data_root
            / sample["target"]
        )
        joints_path = Path(sample["joints3d"])
        camera_path = Path(sample["camera"])

        if not joints_path.is_absolute():
            joints_path = (
                args.data_root / joints_path
            )

        if not camera_path.is_absolute():
            camera_path = (
                args.data_root / camera_path
            )

        frame_index = int(
            sample["frame_index"]
        )

        target_name = Path(sample["target"]).stem

        # Target RGB → VAE 输入 [-1,1]
        target_pixels = load_rgb_tensor(
            path=target_path,
            device=device,
            normalize_vae=True,
        ).to(
            dtype=transformer_dtype
        )

        # Pose map 保持 [0,1]，
        # Adapter 本身使用 FP32 参数训练
        motion_map = (
            build_fit3d_motion_condition(
                joints_path=joints_path,
                camera_path=camera_path,
                frame_index=frame_index,
                output_size=512,
            )
            .unsqueeze(0)
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        # VAE 冻结
        with torch.inference_mode():
            x0 = pipe.pixels_to_latents(
                target_pixels
            )

        (
            noisy_latents,
            timesteps,
            target_velocity,
            weighting,
        ) = make_flow_match_sample(
            pipe=pipe,
            x0=x0,
        )

        condition = source_cache[
            identity
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        # residual 只作用在 Target tokens；
        # --------------------------------------------------
        # Pose ControlNet V3
        #
        # Control branch 同时看到：
        #   noisy target
        #   timestep
        #   SCB semantic condition
        #   target pose latent
        #   source latent
        # --------------------------------------------------
        residuals = adapter(
            target_latents=noisy_latents,
            motion_map=motion_map,
            cond_hidden_states=condition[
                "cond_hidden_states"
            ],
            encoder_hidden_states=condition[
                "encoder_hidden_states"
            ],
            pooled_projections=condition[
                "pooled_projections"
            ],
            timestep=timesteps,
            patch_size=patch_size,
            conditioning_scale=1.0,
        )

        # Transformer 是 BF16，
        # Adapter 保持 FP32 优化；
        residuals = [
            residual.to(
                dtype=transformer_dtype
            )
            for residual in residuals
        ]

        # Transformer 参数冻结
        model_pred = pipe.transformer(
            hidden_states=noisy_latents,
            cond_hidden_states=condition[
                "cond_hidden_states"
            ],
            encoder_hidden_states=condition[
                "encoder_hidden_states"
            ],
            pooled_projections=condition[
                "pooled_projections"
            ],
            timestep=timesteps,
            block_controlnet_hidden_states=residuals,
            return_dict=False,
        )[0]

        if model_pred.ndim != 4:
            raise RuntimeError(
                "Pose-region loss 当前要求 "
                "model_pred 为 [B,C,H,W]，实际为 "
                f"{tuple(model_pred.shape)}"
            )

        # --------------------------------------------------
        # Pose-region weighted flow-matching loss
        #
        # DeepGen 原始 flow target / timestep / sigma
        # 全部保持不变。
        #
        # 只借鉴 HumanSD / MimicMotion：
        # 提高 Target Skeleton 附近区域的监督权重。
        # --------------------------------------------------
        pose_region_weight, pose_region = (
            build_pose_region_weight(
                pose_map=motion_map[:, :3],
                spatial_size=model_pred.shape[-2:],
                pose_weight=args.pose_loss_weight,
                region_kernel=args.pose_region_kernel,
            )
        )

        pose_region_weight = (
            pose_region_weight.to(
                device=model_pred.device,
                dtype=torch.float32,
            )
        )

        squared_error = (
            model_pred.float()
            - target_velocity.float()
        ) ** 2

        loss = (
            weighting.float()
            * pose_region_weight
            * squared_error
        ).mean()

        if step == 1:
            print(
                "model_pred shape:",
                tuple(model_pred.shape),
            )

            print(
                "pose region coverage:",
                f"{pose_region.mean().item():.4f}",
            )

            print(
                "pose spatial weight:"
                f" min={pose_region_weight.min().item():.4f}"
                f" max={pose_region_weight.max().item():.4f}"
                f" mean={pose_region_weight.mean().item():.4f}"
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"step={step} 出现非有限 loss："
                f"{loss.item()}"
            )

        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_norm=args.grad_clip,
            )
        )

        optimizer.step()

        # 每一步都写 CSV：后续画密集曲线/统计表用这个
        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                identity,
                target_name,
                float(loss.item()),
                float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
            ])

        # 控制台按间隔打印，避免太吵
        if (
            step == 1
            or step % args.print_every == 0
            or step == args.max_steps
        ):
            print(
                f"step={step:04d} "
                f"identity={identity} "
                f"target={target_name} "
                f"loss={loss.item():.6f} "
                f"grad={float(grad_norm):.6f}"
            )

        if step % args.save_every == 0:
            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                step=step,
                output_dir=output_dir,
                identity=args.identity,
            )

    train_peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        / 1024**3
    )
    train_peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        / 1024**3
    )
    final_allocated = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )
    final_reserved = (
        torch.cuda.memory_reserved(device)
        / 1024**3
    )

    print(
        "\n===== VRAM SUMMARY ====="
    )
    print(
        f"Init/cache peak allocated : "
        f"{init_peak_allocated:.2f} GB"
    )
    print(
        f"Init/cache peak reserved  : "
        f"{init_peak_reserved:.2f} GB"
    )
    print(
        f"Training peak allocated   : "
        f"{train_peak_allocated:.2f} GB"
    )
    print(
        f"Training peak reserved    : "
        f"{train_peak_reserved:.2f} GB"
    )
    print(
        f"Final allocated           : "
        f"{final_allocated:.2f} GB"
    )
    print(
        f"Final reserved            : "
        f"{final_reserved:.2f} GB"
    )

    print("\nTraining finished: PASS")


if __name__ == "__main__":
    main()

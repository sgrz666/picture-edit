import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.deepgen_pose_adapter import DeepGenPoseAdapter


# 用户侧不使用文本控制。
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
        "encoder_hidden_states":
            encoder_hidden_states,

        "pooled_projections":
            pooled_projections,

        "cond_hidden_states":
            cond_hidden_states,
    }


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
):
    """
    Diffusers FlowMatch scheduler 使用的标准动态 shift。
    只有 scheduler 开启 dynamic shifting 时才会进入此分支。
    """
    slope = (
        max_shift - base_shift
    ) / (
        max_seq_len - base_seq_len
    )

    intercept = (
        base_shift
        - slope * base_seq_len
    )

    return (
        image_seq_len * slope
        + intercept
    )


def get_sigmas(
    scheduler,
    timesteps,
    n_dim,
    device,
    dtype,
):
    """
    与 DeepGen 官方训练代码保持一致，
    从 scheduler 中取当前 timestep 对应 sigma。
    """
    sigmas = scheduler.sigmas.to(
        device=device,
        dtype=dtype,
    )

    schedule_timesteps = (
        scheduler.timesteps.to(device)
    )

    timesteps = timesteps.to(device)

    step_indices = [
        (
            schedule_timesteps == timestep
        )
        .nonzero()
        .item()
        for timestep in timesteps
    ]

    sigma = sigmas[
        step_indices
    ].flatten()

    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma


def make_flow_match_sample(
    pipe,
    x0,
):
    """
    完全按 DeepGen 官方 diff_loss 构造：

        z_t = (1-sigma) * x0 + sigma * noise

    监督目标：

        velocity = noise - x0
    """
    scheduler = pipe.scheduler

    batch_size = x0.shape[0]

    use_dynamic_shifting = bool(
        getattr(
            scheduler,
            "use_dynamic_shifting",
            False,
        )
    )

    weighting_scheme = (
        "logit_normal"
        if use_dynamic_shifting
        else "none"
    )

    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        logit_mean=0.0,
        logit_std=1.0,
    )

    if use_dynamic_shifting:
        patch_size = (
            pipe.transformer.config.patch_size
        )

        image_seq_len = (
            math.prod(x0.shape[-2:])
            // patch_size**2
        )

        config = scheduler.config

        mu = calculate_shift(
            image_seq_len=image_seq_len,
            base_seq_len=config.get(
                "base_image_seq_len",
                256,
            ),
            max_seq_len=config.get(
                "max_image_seq_len",
                4096,
            ),
            base_shift=config.get(
                "base_shift",
                0.5,
            ),
            max_shift=config.get(
                "max_shift",
                1.15,
            ),
        )

        if config.get(
            "time_shift_type",
            "exponential",
        ) == "exponential":
            shift = math.exp(mu)

        elif config.get(
            "time_shift_type"
        ) == "linear":
            shift = mu

        else:
            raise ValueError(
                "不支持的 time_shift_type："
                f"{config.get('time_shift_type')}"
            )

        sigmas = u.to(
            device=x0.device,
            dtype=x0.dtype,
        )

        sigmas = (
            shift * sigmas
            / (
                1.0
                + (shift - 1.0) * sigmas
            )
        )

        num_train_timesteps = (
            scheduler.config.num_train_timesteps
        )

        timesteps = (
            sigmas
            * num_train_timesteps
        )

        sigmas = sigmas.view(
            batch_size,
            1,
            1,
            1,
        )

    else:
        num_train_timesteps = (
            scheduler.config.num_train_timesteps
        )

        indices = (
            u
            * num_train_timesteps
        ).long()

        indices = indices.clamp(
            0,
            len(scheduler.timesteps) - 1,
        )

        timesteps = (
            scheduler.timesteps[
                indices
            ]
            .to(x0.device)
        )

        sigmas = get_sigmas(
            scheduler=scheduler,
            timesteps=timesteps,
            n_dim=x0.ndim,
            device=x0.device,
            dtype=x0.dtype,
        )

    noise = torch.randn_like(x0)

    noisy_latents = (
        (1.0 - sigmas) * x0
        + sigmas * noise
    )

    target_velocity = (
        noise - x0
    )

    weighting = (
        compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme,
            sigmas=sigmas,
        )
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
        / f"adapter_step_{step:06d}.pt"
    )

    torch.save(
        {
            "step": step,
            "identity": identity,
            "adapter": adapter.state_dict(),
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
        default="008_1",
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
        default=3e-4,
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

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前没有可用 CUDA GPU"
        )

    device = torch.device("cuda")

    # 从这里开始记录模型加载和 Source condition 缓存的显存峰值。
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
    #
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

    # 从这里重新统计真正训练阶段的显存峰值。
    torch.cuda.reset_peak_memory_stats(device)

    # Source condition 的提取会消耗随机数，
    # 训练开始前重新固定 seed。
    set_seed(args.seed)

    # --------------------------------------------------
    # Pose Adapter
    # --------------------------------------------------

    adapter = DeepGenPoseAdapter(
        input_channels=3,
        hidden_dim=hidden_dim,
        num_residual_heads=6,
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
        "\nPose Adapter trainable params:",
        f"{trainable_params:,}",
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    output_dir = (
        args.output_dir
        / args.identity
    )

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

        pose_path = (
            args.data_root
            / sample["pose"]
        )

        # Target RGB → VAE 输入 [-1,1]
        target_pixels = load_rgb_tensor(
            path=target_path,
            device=device,
            normalize_vae=True,
        ).to(
            dtype=transformer_dtype
        )

        # Pose map 保持 [0,1]，
        # Adapter 本身使用 FP32 参数训练。
        pose_map = load_rgb_tensor(
            path=pose_path,
            device=device,
            normalize_vae=False,
        ).to(
            dtype=torch.float32
        )

        # VAE 冻结，不需要构建梯度图。
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

        # --------------------------------------------------
        # Pose Adapter
        #
        # residual 只作用在 Target tokens；
        # Source/reference token 区严格补 0。
        # --------------------------------------------------

        residuals = adapter(
            pose_map=pose_map,
            target_latents=noisy_latents,
            cond_hidden_states=condition[
                "cond_hidden_states"
            ],
            patch_size=patch_size,
        )

        # Transformer 是 BF16，
        # Adapter 保持 FP32 优化；
        # 这里只在接口处 cast，不切断梯度。
        residuals = [
            residual.to(
                dtype=transformer_dtype
            )
            for residual in residuals
        ]

        # --------------------------------------------------
        # DeepGen Transformer
        #
        # 注意：
        # Transformer 参数虽然冻结，
        # 这里绝不能使用 torch.no_grad()。
        #
        # loss 必须经过 Transformer 运算
        # 回传到 Pose residual 和 Adapter。
        # --------------------------------------------------

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

        loss = (
            weighting.float()
            * (
                model_pred.float()
                - target_velocity.float()
            ) ** 2
        ).mean()

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

        if (
            step == 1
            or step % 10 == 0
            or step == args.max_steps
        ):
            print(
                f"step={step:04d} "
                f"identity={identity} "
                f"target={sample['target_frame']} "
                f"loss={loss.item():.6f} "
                f"grad={float(grad_norm):.6f}"
            )

        if (
            step % args.save_every == 0
            or step == args.max_steps
        ):
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

    print(
        "\nTraining finished: PASS"
    )


if __name__ == "__main__":
    main()

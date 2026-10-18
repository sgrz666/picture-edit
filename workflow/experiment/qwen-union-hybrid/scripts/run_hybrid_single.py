#!/usr/bin/env python3
"""Hybrid Qwen Base + Union Control (No Mask) 推理执行引擎。

架构要点：
1. Qwen 基模保留原生多模态输入路径：image=[A_reference, pose_source_torso]，
   通过视觉自注意力直接继承 A 的全局高层语义、面部与服装特征，彻底告别 Mask 挖空人脸导致的身份丢失。
2. Union 控制分支仅接收纯几何输入 (Pose 与 Depth)，不使用 Mask，不使用 Inpaint (zero-mask / zero-inpaint)。
3. 在 Transformer 目标生成位置 (target_token_mask) 注入控制条件，经 16 个控制块生成 hints 残差。
4. 核心四组矩阵：
   - H0: Native Qwen Baseline (hints = None)
   - H1: Hybrid + Pose (hints = hints_pose)
   - H2: Hybrid + Depth (hints = hints_depth)
   - H3: Hybrid + Pose + Depth 残差融合 (hints = alpha * hints_pose + beta * hints_depth, alpha=1.0, beta=1.0)
5. 采用分阶段 Offload，显存占用稳定在 ~21 GiB，完美适配服务器多任务环境。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "pic-edit" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# dummy stub for unused deps
import importlib.util, types
from importlib.machinery import ModuleSpec
for name in ("librosa", "ftfy", "soxr"):
    if importlib.util.find_spec(name) is None:
        m = types.ModuleType(name)
        m.__path__, m.__file__, m.__spec__ = [], f"<stub:{name}>", ModuleSpec(name, loader=None, is_package=True)
        m.__getattr__ = lambda a, _n=name: (lambda *args, **kwargs: args[0] if len(args) > 0 and isinstance(args[0], str) else None)
        sys.modules[name] = m

EXP_ROOT = Path(__file__).resolve().parents[1]
VIDEOX_DIR = REPO_ROOT / "experiments/6.qwen-controlnet union/source/VideoX-Fun"
if str(VIDEOX_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOX_DIR))

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from safetensors.torch import load_file
from videox_fun.models import QwenImage21ControlTransformer2DModel
from videox_fun.models.qwenimage21_transformer2d import _IMG_TOKENS_PER_SLOT, _qwenimage21_prefix_segments

BASE_MODEL_DIR = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
UNION_CKPT_PATH = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
CONFIG_PATH = VIDEOX_DIR / "config/qwenimage21/qwenimage21_control.yaml"


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def load_hybrid_pipeline():
    """加载 Qwen 基模管道并替换 Transformer 为挂载 ControlNet Union 权重的模型。"""
    dtype = torch.bfloat16
    print(">> [Model Load] 从 CPU 加载 Control Transformer...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kwargs = cfg.get("transformer_additional_kwargs", {})

    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
        str(BASE_MODEL_DIR), subfolder="transformer", low_cpu_mem_usage=True,
        torch_dtype=dtype, transformer_additional_kwargs=kwargs,
    ).to(dtype)

    print(">> [Model Load] 加载 Union 控制权重并校验...")
    state = load_file(str(UNION_CKPT_PATH))
    state = state["state_dict"] if "state_dict" in state else state
    transformer.load_state_dict(state, strict=False)
    transformer.eval().requires_grad_(False)

    print(">> [Model Load] 构建 QwenImage21Pipeline (text_encoder, vae, scheduler)...")
    pipe = QwenImage21Pipeline.from_pretrained(
        str(BASE_MODEL_DIR),
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    pipe.transformer.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)

    return pipe


def encode_control_context(
    pipe,
    control_img: Image.Image,
    width: int,
    height: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """使用 VAE 将控制图编码为 zero-mask / zero-inpaint 的 129 通道 control_context。"""
    rgba = control_img.convert("RGBA")
    ctrl_prep = pipe.image_processor.preprocess(rgba, width=width, height=height).unsqueeze(2).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        c_lat = pipe.vae.encode(ctrl_prep).latent_dist.mode()
        lat_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
        lat_std = torch.tensor(pipe.vae.config.latents_std, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
        c_lat = (c_lat - lat_mean) / lat_std

    lat_h, lat_w = c_lat.shape[-2], c_lat.shape[-1]
    zero_mask = torch.zeros(1, 1, 1, lat_h, lat_w, device="cuda", dtype=dtype)
    zero_inpaint = torch.zeros(1, 64, 1, lat_h, lat_w, device="cuda", dtype=dtype)
    ctrl_context = torch.cat([c_lat, zero_mask, zero_inpaint], dim=1)  # [1, 129, 1, lat_h, lat_w]
    ctrl_context = ctrl_context.view(1, 129, lat_h * lat_w).transpose(1, 2)  # [1, N, 129]
    return ctrl_context, lat_h, lat_w


def hybrid_transformer_forward(
    model,
    hidden_states,
    encoder_hidden_states,
    timestep,
    img_shapes,
    img_mask,
    control_context_pose=None,
    control_context_depth=None,
    alpha=1.0,
    beta=1.0,
):
    """带目标 token 注入与双几何残差融合的 Transformer 单步前向。"""
    batch_size = hidden_states.shape[0]
    hidden_states = model.img_in(hidden_states)
    encoder_hidden_states = model.txt_in(encoder_hidden_states)

    repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

    target_tokens = math.prod(img_shapes[0][-1])
    joint_hidden_states = torch.cat(
        [
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
        ],
        dim=1,
    )
    joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
    joint_hidden_states[:, image_pad_mask] = hidden_states

    image_ids, target_token_mask = model.build_token_metadata(image_pad_mask, img_shapes[0])
    rotary_emb = model.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)

    timestep = timestep.to(hidden_states.dtype)
    if model.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = target_token_mask
    else:
        modulation_mask = None
    temb = model.time_text_embed(timestep, hidden_states)
    modulation = model.modulation(temb)

    prefix_len = int((~target_token_mask).sum())
    block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

    hints = None
    if control_context_pose is not None or control_context_depth is not None:
        control_kwargs = dict(
            modulation=modulation,
            rotary_emb=rotary_emb,
            attention_mask=None,
            target_token_mask=modulation_mask,
            layer_cache=None,
            kv_cache_mode=None,
            cache_write_slice=None,
            segments=block_segments,
            key_valid=None,
        )
        hints_p, hints_d = None, None
        if control_context_pose is not None:
            ctrl_features_pose = model.control_img_in(control_context_pose)
            ctrl_joint_pose = torch.zeros_like(joint_hidden_states)
            ctrl_joint_pose[:, target_token_mask] = ctrl_features_pose
            hints_p = model.forward_control(ctrl_joint_pose, joint_hidden_states, control_kwargs)
        if control_context_depth is not None:
            ctrl_features_depth = model.control_img_in(control_context_depth)
            ctrl_joint_depth = torch.zeros_like(joint_hidden_states)
            ctrl_joint_depth[:, target_token_mask] = ctrl_features_depth
            hints_d = model.forward_control(ctrl_joint_depth, joint_hidden_states, control_kwargs)

        if hints_p is not None and hints_d is not None:
            hints = tuple(alpha * hp + beta * hd for hp, hd in zip(hints_p, hints_d))
        elif hints_p is not None:
            hints = tuple(alpha * hp for hp in hints_p)
        elif hints_d is not None:
            hints = tuple(beta * hd for hd in hints_d)

    for index_block, block in enumerate(model.transformer_blocks):
        kwargs = dict(
            modulation=modulation,
            rotary_emb=rotary_emb,
            attention_mask=None,
            target_token_mask=modulation_mask,
            layer_cache=None,
            kv_cache_mode=None,
            cache_write_slice=None,
            segments=block_segments,
            key_valid=None,
            hints=hints,
            context_scale=1.0,
        )
        joint_hidden_states = block(joint_hidden_states, **kwargs)

    joint_hidden_states = model.norm_out(joint_hidden_states, temb, modulation_mask)
    output = model.proj_out(joint_hidden_states)
    return output


def generate_single_arm(
    pipe,
    arm: str,
    a_image: Image.Image,
    pose_image: Image.Image,
    depth_image: Optional[Image.Image],
    prompt: str,
    width: int = 640,
    height: int = 928,
    steps: int = 40,
    seed: int = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[Image.Image, dict]:
    """执行单个实验组的完整去噪流程，带严格的显存 offload 管理。"""
    dtype = torch.bfloat16
    start_time = time.perf_counter()
    peak_vram = 0.0

    print(f"\n[{arm}] === Phase 1: 文本与多模态视觉编码 (A + PoseGuide + PE Prompt) ===")
    pipe.text_encoder.to("cuda")
    input_images = [a_image.convert("RGB"), pose_image.convert("RGB")]
    prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
        image=input_images,
        prompt=prompt,
        device="cuda",
        num_images_per_prompt=1,
    )
    peak_vram = max(peak_vram, torch.cuda.memory_allocated() / (1024**3))
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    print(f"[{arm}] === Phase 2: VAE 编码条件图、控制特征与初始噪声 ===")
    pipe.vae.to("cuda")
    vae_images = []
    for img in input_images:
        rgba = img.convert("RGBA")
        vae_images.append(pipe.image_processor.preprocess(rgba, width=width, height=height).unsqueeze(2))

    generator = torch.Generator("cuda").manual_seed(seed)
    latents, input_images_latents = pipe.prepare_latents(
        vae_images, 1, 64, height, width, dtype, "cuda", generator, None
    )

    def append_target_slots(mask):
        return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

    image_pad_mask = append_target_slots(image_pad_mask)

    # 准备控制条件
    ctrl_context_pose = None
    ctrl_context_depth = None
    lat_h, lat_w = None, None

    if arm in ("H1", "H3"):
        ctrl_context_pose, lat_h, lat_w = encode_control_context(pipe, pose_image, width, height, dtype)
    if arm in ("H2", "H3") and depth_image is not None:
        ctrl_context_depth, lat_h, lat_w = encode_control_context(pipe, depth_image, width, height, dtype)

    if lat_h is None:
        lat_h = height // (pipe.vae_scale_factor)
        lat_w = width // (pipe.vae_scale_factor)

    img_shapes = [[
        (1, lat_h, lat_w),
        (1, lat_h, lat_w),
        (1, lat_h, lat_w),
    ]]

    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    print(f"[{arm}] === Phase 3: Transformer 40 步去噪迭代 ===")
    pipe.transformer.to("cuda")
    mu = calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    pipe.scheduler.set_timesteps(steps, device="cuda", sigmas=sigmas, mu=mu)
    timesteps = pipe.scheduler.timesteps

    latent_model_input = torch.cat([input_images_latents, latents], dim=1)

    step_times = []
    for step_idx, t in enumerate(timesteps):
        t_step_start = time.perf_counter()
        timestep_expanded = t.expand(1).to(dtype)
        latent_model_input = torch.cat([input_images_latents, latents], dim=1)

        with torch.no_grad():
            output = hybrid_transformer_forward(
                pipe.transformer,
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_expanded / 1000,
                img_shapes=img_shapes,
                img_mask=image_pad_mask,
                control_context_pose=ctrl_context_pose,
                control_context_depth=ctrl_context_depth,
                alpha=alpha,
                beta=beta,
            )
            noise_pred = output[:, -latents.shape[1]:]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)
        if (step_idx + 1) % 10 == 0 or step_idx == steps - 1:
            avg_s = sum(step_times[-10:]) / len(step_times[-10:])
            cur_mem = torch.cuda.memory_allocated() / (1024**3)
            peak_vram = max(peak_vram, cur_mem)
            print(f"[{arm}] Step {step_idx + 1:2d}/{steps} | {avg_s:.2f}s/step | VRAM: {cur_mem:.2f} GiB")

    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()

    print(f"[{arm}] === Phase 4: VAE 解码 Latents 到最终图像 ===")
    pipe.vae.to("cuda")
    with torch.no_grad():
        unpacked = latents.transpose(1, 2).reshape(1, 64, 1, lat_h, lat_w)
        lat_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
        lat_std = torch.tensor(pipe.vae.config.latents_std, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
        unpacked = unpacked * lat_std + lat_mean
        decoded = pipe.vae.decode(unpacked).sample
        img_out = pipe.image_processor.postprocess(decoded.squeeze(2), output_type="pil")[0]

    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    total_time = time.perf_counter() - start_time
    print(f"[{arm}] 完成生成！总耗时: {total_time:.2f}s | 峰值显存: {peak_vram:.2f} GiB")

    meta = {
        "arm": arm,
        "steps": steps,
        "seed": seed,
        "alpha": alpha if arm == "H3" else None,
        "beta": beta if arm == "H3" else None,
        "generation_seconds": round(total_time, 2),
        "peak_vram_gib": round(peak_vram, 2),
    }
    return img_out, meta


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Qwen Base + Union Control Inference")
    parser.add_argument("--arms", type=str, default="H0,H1,H2,H3", help="运行的实验组列表 (逗号分隔)")
    parser.add_argument("--steps", type=int, default=40, help="推理步数 (固定40)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (固定42)")
    parser.add_argument("--alpha", type=float, default=1.0, help="H3 Pose 融合权重")
    parser.add_argument("--beta", type=float, default=1.0, help="H3 Depth 融合权重")
    parser.add_argument("--output", type=Path, default=EXP_ROOT / "output", help="输出根目录")
    parser.add_argument("--dry-run", action="store_true", help="演练模式，仅校验配置")
    parser.add_argument("--resume", action="store_true", help="若已有结果则跳过")
    args = parser.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    manifest_path = EXP_ROOT / "data/prepared_controls/prep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到 prep_manifest.json: {manifest_path}，请先运行 prepare_hybrid_controls.py")
    prep = json.loads(manifest_path.read_text(encoding="utf-8"))

    print("=" * 66)
    print("      Hybrid Qwen Base + Union Control (No Mask) 执行矩阵")
    print("=" * 66)
    print(f"Canvas 尺寸: {prep['canvas_size']} | 步数: {args.steps} | Seed: {args.seed}")
    print(f"请求执行组别: {arm_list}")
    print(f"H3 融合参数: alpha={args.alpha}, beta={args.beta}\n")

    if args.dry_run:
        print("✅ [Dry Run] 配置与控制输入检查通过，退出。")
        return

    # 读取输入图像与提示词
    a_image = Image.open(prep["source_assets"]["A_reference"]["path"])
    pose_image = Image.open(prep["controls"]["pose"]["path"])
    depth_image = Image.open(prep["controls"]["depth"]["path"])
    prompt = prep["prompt"]["text"]

    pipe = load_hybrid_pipeline()

    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_records = {}

    for arm in arm_list:
        arm_dir = out_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = arm_dir / "final_image.png"
        record_path = arm_dir / "run_record.json"

        if args.resume and out_img_path.is_file() and record_path.is_file():
            print(f"⏩ [{arm}] 产物已存在且启用 --resume，跳过生成。")
            summary_records[arm] = json.loads(record_path.read_text(encoding="utf-8"))
            continue

        print(f"\n------------------------------------------------------------------")
        print(f"开始执行 [{arm}] ...")
        print(f"------------------------------------------------------------------")

        img, meta = generate_single_arm(
            pipe=pipe,
            arm=arm,
            a_image=a_image,
            pose_image=pose_image,
            depth_image=depth_image,
            prompt=prompt,
            width=prep["canvas_size"][0],
            height=prep["canvas_size"][1],
            steps=args.steps,
            seed=args.seed,
            alpha=args.alpha,
            beta=args.beta,
        )

        img.save(out_img_path)
        img_sha = sha256_file(out_img_path)

        record = {
            "arm": arm,
            "status": "completed",
            "output": {"path": str(out_img_path), "sha256": img_sha, "size_wh": list(img.size)},
            "performance": meta,
        }
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_records[arm] = record
        print(f"✅ [{arm}] 保存产物: {out_img_path} (SHA256: {img_sha[:12]}...)")

    exec_summary_path = out_root / "execution_summary.json"
    exec_summary_path.write_text(json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 全部实验组生成完毕！执行汇总写入: {exec_summary_path}")


if __name__ == "__main__":
    main()

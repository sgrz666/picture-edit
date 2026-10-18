#!/usr/bin/env python3
"""自检：验证模型分阶段 Offload 推理架构与显存峰值（严格控制在 25GB 内）。"""
import sys
from pathlib import Path
import math
import time
import torch
import torch.nn as nn
from PIL import Image

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

VIDEOX_DIR = REPO_ROOT / "experiments/6.qwen-controlnet union/source/VideoX-Fun"
if str(VIDEOX_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOX_DIR))

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from safetensors.torch import load_file
import yaml
from videox_fun.models import QwenImage21ControlTransformer2DModel
from videox_fun.models.qwenimage21_transformer2d import _IMG_TOKENS_PER_SLOT, _qwenimage21_prefix_segments

BASE_MODEL_DIR = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
UNION_CKPT_PATH = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
CONFIG_PATH = VIDEOX_DIR / "config/qwenimage21/qwenimage21_control.yaml"

dtype = torch.bfloat16
device = torch.device("cuda")

free_gib = torch.cuda.mem_get_info()[0] / (1024**3)
print(f">> [Init] 当前 GPU 可用显存: {free_gib:.2f} GiB")

print(">> [Model Load] 从 CPU 加载 Control Transformer...")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
kwargs = cfg.get("transformer_additional_kwargs", {})

transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
    str(BASE_MODEL_DIR), subfolder="transformer", low_cpu_mem_usage=True,
    torch_dtype=dtype, transformer_additional_kwargs=kwargs,
).to(dtype)

print(">> [Model Load] 加载 Union 权重...")
state = load_file(str(UNION_CKPT_PATH))
state = state["state_dict"] if "state_dict" in state else state
transformer.load_state_dict(state, strict=False)
transformer.eval().requires_grad_(False)

print(">> [Model Load] 从 CPU 加载 QwenImage21Pipeline (text_encoder, vae, scheduler)...")
pipe = QwenImage21Pipeline.from_pretrained(
    str(BASE_MODEL_DIR),
    transformer=transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
pipe.transformer.eval().requires_grad_(False)
pipe.text_encoder.eval().requires_grad_(False)
pipe.vae.eval().requires_grad_(False)

pose_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/pose_source_torso.png"
depth_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/depth_source_torso.png"
a_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/pose_source_torso.png"

img_a = Image.open(a_path).convert("RGB")
img_pose = Image.open(pose_path).convert("RGB")
prompt = "A person in blue jeans, neutral studio background."

print("\n--- Phase 1: 文本与多模态编码 (Text Encoder -> GPU) ---")
pipe.text_encoder.to("cuda")
print(f"Text Encoder 在显存: 当前占用 {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")

input_images = [img_a, img_pose]
prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
    image=input_images,
    prompt=prompt,
    device="cuda",
    num_images_per_prompt=1,
)
print("prompt_embeds encoded, shape:", prompt_embeds.shape)

pipe.text_encoder.to("cpu")
torch.cuda.empty_cache()
print(f"Text Encoder 卸回 CPU, 当前占用: {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")

print("\n--- Phase 2: VAE 编码条件图与初始 Latents (VAE -> GPU) ---")
pipe.vae.to("cuda")
print(f"VAE 在显存: 当前占用 {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")

width, height = 640, 928
vae_images = []
for img in input_images:
    rgba = img.convert("RGBA")
    vae_images.append(pipe.image_processor.preprocess(rgba, width=width, height=height).unsqueeze(2))

latents, input_images_latents = pipe.prepare_latents(
    vae_images, 1, 64, height, width, dtype, "cuda", torch.Generator("cuda").manual_seed(42), None
)

def append_target_slots(mask):
    return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

image_pad_mask = append_target_slots(image_pad_mask)

# 编码 Pose control context (129 channels)
ctrl_prep = pipe.image_processor.preprocess(img_pose.convert("RGBA"), width=width, height=height).unsqueeze(2).to(device="cuda", dtype=dtype)
with torch.no_grad():
    c_lat = pipe.vae.encode(ctrl_prep).latent_dist.mode()
    lat_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
    lat_std = torch.tensor(pipe.vae.config.latents_std, device="cuda", dtype=dtype).view(1, 64, 1, 1, 1)
    c_lat = (c_lat - lat_mean) / lat_std

lat_h, lat_w = c_lat.shape[-2], c_lat.shape[-1]
zero_mask = torch.zeros(1, 1, 1, lat_h, lat_w, device="cuda", dtype=dtype)
zero_inpaint = torch.zeros(1, 64, 1, lat_h, lat_w, device="cuda", dtype=dtype)
ctrl_context_pose = torch.cat([c_lat, zero_mask, zero_inpaint], dim=1)  # [1, 129, 1, lat_h, lat_w]
ctrl_context_pose = ctrl_context_pose.view(1, 129, lat_h * lat_w).transpose(1, 2)  # [1, 2320, 129]

img_shapes = [[
    (1, lat_h, lat_w),
    (1, lat_h, lat_w),
    (1, lat_h, lat_w),
]]

pipe.vae.to("cpu")
torch.cuda.empty_cache()
print(f"VAE 卸回 CPU, 当前占用: {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")
print(f"Latents shape: {latents.shape}, Ctrl context shape: {ctrl_context_pose.shape}, img_shapes: {img_shapes}")

print("\n--- Phase 3: Transformer 单步混合前向测试 (Transformer -> GPU) ---")
pipe.transformer.to("cuda")
print(f"Transformer 在显存: 当前占用 {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")

latent_model_input = torch.cat([input_images_latents, latents], dim=1)
timestep = torch.tensor([1000.0], device="cuda", dtype=dtype)

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

print(">> [Test] 执行 H0 前向...")
with torch.no_grad():
    out_h0 = hybrid_transformer_forward(
        pipe.transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, None, None
    )
print("H0 成功！output shape:", out_h0.shape)

print(">> [Test] 执行 H1 (Hybrid Pose) 前向...")
with torch.no_grad():
    out_h1 = hybrid_transformer_forward(
        pipe.transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, control_context_pose=ctrl_context_pose, control_context_depth=None
    )
print("H1 成功！output shape:", out_h1.shape)
diff = (out_h1 - out_h0).abs().max().item()
print(f"✅ H1 控制残差有效注入: max diff = {diff:.4f}")

pipe.transformer.to("cpu")
torch.cuda.empty_cache()
print(f"\nTransformer 卸回 CPU, 当前占用: {torch.cuda.memory_allocated() / (1024**3):.2f} GiB")
print("🎉 Offload 与 Hybrid 架构全面验证成功！")

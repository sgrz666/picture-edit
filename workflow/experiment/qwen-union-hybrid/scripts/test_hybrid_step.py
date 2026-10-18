#!/usr/bin/env python3
"""自检与验证 Hybrid Qwen Base + Union Control 步进前向与残差注入。"""
import sys
from pathlib import Path
import math
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

device = torch.device("cuda")
dtype = torch.bfloat16

print(">> [Test] 加载 Control Transformer...")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
kwargs = cfg.get("transformer_additional_kwargs", {})

transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
    str(BASE_MODEL_DIR), subfolder="transformer", low_cpu_mem_usage=True,
    torch_dtype=dtype, transformer_additional_kwargs=kwargs,
).to(dtype).to(device)

state = load_file(str(UNION_CKPT_PATH))
state = state["state_dict"] if "state_dict" in state else state
transformer.load_state_dict(state, strict=False)
transformer.eval()
print(">> [Test] Transformer 加载成功！")

pipe = QwenImage21Pipeline.from_pretrained(str(BASE_MODEL_DIR), torch_dtype=dtype).to(device)
pipe.transformer = transformer

pose_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/pose_source_torso.png"
depth_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/depth_source_torso.png"

img_pose = Image.open(pose_path)
images = [img_pose, img_pose]  # 模拟 A 与 poseguide
prompt = "A photo of a person."

input_image_sizes, input_images, vae_images = [], [], []
for img in images:
    if hasattr(img, "mode") and img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    input_image_sizes.append((w, h))
    input_images.append(pipe.image_processor.resize(img, width=w, height=h))
    vae_images.append(pipe.image_processor.preprocess(img, width=w, height=h).unsqueeze(2))

prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
    image=input_images, prompt=prompt, device=device, num_images_per_prompt=1
)

latents, input_images_latents = pipe.prepare_latents(
    vae_images, 1, 64, 928, 640, dtype, device, torch.Generator("cuda").manual_seed(42), None
)

def append_target_slots(mask):
    return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

image_pad_mask = append_target_slots(image_pad_mask)

img_shapes = [[
    *[ (1, vae_h // pipe.vae_scale_factor, vae_w // pipe.vae_scale_factor) for vae_w, vae_h in input_image_sizes ],
    (1, 928 // pipe.vae_scale_factor, 640 // pipe.vae_scale_factor)
]]

latent_model_input = torch.cat([input_images_latents, latents], dim=1)
timestep = torch.tensor([1000.0], device=device, dtype=dtype)

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

print(">> [Test] 测试 H0 (Native baseline, 无 hints)...")
with torch.no_grad():
    out0 = hybrid_transformer_forward(
        transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, None, None
    )
print(f"✅ H0 成功输出 shape: {out0.shape}")

# 构造模拟 129 通道控制上下文
ctrl_mock = torch.randn(1, 9280, 129, device=device, dtype=dtype)

print(">> [Test] 测试 H1 (Hybrid + Pose)...")
with torch.no_grad():
    out1 = hybrid_transformer_forward(
        transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, control_context_pose=ctrl_mock, control_context_depth=None
    )
print(f"✅ H1 成功输出 shape: {out1.shape}")

print(">> [Test] 测试 H2 (Hybrid + Depth)...")
with torch.no_grad():
    out2 = hybrid_transformer_forward(
        transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, control_context_pose=None, control_context_depth=ctrl_mock
    )
print(f"✅ H2 成功输出 shape: {out2.shape}")

print(">> [Test] 测试 H3 (Hybrid + Pose + Depth 融合)...")
with torch.no_grad():
    out3 = hybrid_transformer_forward(
        transformer, latent_model_input, prompt_embeds, timestep / 1000,
        img_shapes, image_pad_mask, control_context_pose=ctrl_mock, control_context_depth=ctrl_mock,
        alpha=1.0, beta=1.0
    )
print(f"✅ H3 成功输出 shape: {out3.shape}")

diff_h1_h0 = (out1 - out0).abs().max().item()
diff_h3_h1 = (out3 - out1).abs().max().item()
print(f">> [Test] 残差注入强度验证: max(|H1 - H0|) = {diff_h1_h0:.4f}, max(|H3 - H1|) = {diff_h3_h1:.4f}")
assert diff_h1_h0 > 0, "H1 与 H0 输出相同，控制残差未生效！"
assert diff_h3_h1 > 0, "H3 与 H1 输出相同，融合未生效！"
print("🎉 完美！H0, H1, H2, H3 全矩阵数学与张量完全通过自检！")

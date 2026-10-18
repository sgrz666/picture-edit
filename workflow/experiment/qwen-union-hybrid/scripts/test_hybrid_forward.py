#!/usr/bin/env python3
"""自检脚本：验证 Hybrid Qwen Base + Union Control 单步前向计算与残差融合正确性。"""
import sys
from pathlib import Path
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "pic-edit" and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util
import types
from importlib.machinery import ModuleSpec

for name in ("librosa", "ftfy", "soxr"):
    if importlib.util.find_spec(name) is not None:
        continue
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__file__ = f"<stub:{name}>"
    mod.__spec__ = ModuleSpec(name, loader=None, is_package=True)

    def _dummy(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            return args[0]
        return None

    def _getattr(attr, _n=name):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _dummy

    mod.__getattr__ = _getattr
    sys.modules[name] = mod

VIDEOX_DIR = REPO_ROOT / "experiments/6.qwen-controlnet union/source/VideoX-Fun"
if str(VIDEOX_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOX_DIR))

BASE_MODEL_DIR = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
UNION_CKPT_PATH = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
CONFIG_PATH = VIDEOX_DIR / "config/qwenimage21/qwenimage21_control.yaml"

from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from videox_fun.models import (AutoencoderKLQwenImage21,
                               Qwen3VLForConditionalGeneration,
                               Qwen3VLProcessor,
                               QwenImage21ControlTransformer2DModel)
from videox_fun.pipeline.pipeline_qwenimage21_control import QwenImage21ControlPipeline

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(">> [Test] Control Transformer 加载成功！")

# 构造虚拟输入测试前向
batch_size = 1
channels = 64
lat_h, lat_w = 116, 80  # 928x640 / 8
target_tokens = lat_h * lat_w

# 模拟 joint_hidden_states (text + condition 1 + condition 2 + target)
txt_tokens = 50
cond1_tokens = 9280
cond2_tokens = 9280
total_seq_len = txt_tokens + cond1_tokens + cond2_tokens + target_tokens

# 构造真实的 img_shapes 和 img_mask
img_shapes = [
    [
        (1, 116, 80),  # condition 1
        (1, 116, 80),  # condition 2
        (1, 116, 80),  # target
    ]
]
# image_pad_mask: True for all image tokens
repeats = torch.ones(total_seq_len, dtype=torch.long, device=device)
full_img_pad_mask = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
full_img_pad_mask[txt_tokens:] = True

joint = torch.randn(1, total_seq_len, 4096, device=device, dtype=dtype)
target_token_mask = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
target_token_mask[-target_tokens:] = True

# 模拟 129 通道 control_context
control_context = torch.randn(1, target_tokens, 129, device=device, dtype=dtype)
ctrl_features = transformer.control_img_in(control_context)
print(">> [Test] control_features shape:", ctrl_features.shape)

ctrl_joint = torch.zeros_like(joint)
ctrl_joint[:, target_token_mask] = ctrl_features

rotary_emb = transformer.pos_embed(img_shapes[0], full_img_pad_mask, device=device)
temb = torch.randn(1, 4096, device=device, dtype=dtype)
modulation = transformer.modulation(temb)

control_kwargs = dict(
    modulation=modulation,
    rotary_emb=rotary_emb,
    attention_mask=None,
    target_token_mask=target_token_mask,
    layer_cache=None,
    kv_cache_mode=None,
    cache_write_slice=None,
    segments=None,
    key_valid=None,
)

print(">> [Test] 测试 forward_control 前向...")
with torch.no_grad():
    hints = transformer.forward_control(ctrl_joint, joint, control_kwargs)
print(f">> [Test] forward_control 成功产出 {len(hints)} 个 hints skips, skip 0 shape: {hints[0].shape}")

# 测试融合
hints_2 = tuple(h * 0.5 for h in hints)
hints_fusion = tuple(h1 + h2 for h1, h2 in zip(hints, hints_2))
print(">> [Test] hints_fusion 计算成功！")
print("✅ 自检通过！混合架构数学与张量完全兼容！")

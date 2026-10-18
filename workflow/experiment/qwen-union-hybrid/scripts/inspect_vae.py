import sys
from pathlib import Path
import torch
from PIL import Image

REPO_ROOT = Path("/home/shangguanrz/project/pic-edit")
VIDEOX_DIR = REPO_ROOT / "experiments/6.qwen-controlnet union/source/VideoX-Fun"
if str(VIDEOX_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOX_DIR))

# dummy stub
import importlib.util, types
from importlib.machinery import ModuleSpec
for name in ("librosa", "ftfy", "soxr"):
    if importlib.util.find_spec(name) is None:
        m = types.ModuleType(name)
        m.__path__, m.__file__, m.__spec__ = [], f"<stub:{name}>", ModuleSpec(name, loader=None, is_package=True)
        m.__getattr__ = lambda a, _n=name: (lambda *args, **kwargs: args[0] if len(args) > 0 and isinstance(args[0], str) else None)
        sys.modules[name] = m

from diffusers import QwenImage21Pipeline
BASE_MODEL_DIR = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
pipe = QwenImage21Pipeline.from_pretrained(str(BASE_MODEL_DIR), torch_dtype=torch.bfloat16)

pose_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/pose_source_torso.png"
img_pose = Image.open(pose_path).convert("RGBA")
width, height = 640, 928

ctrl_prep = pipe.image_processor.preprocess(img_pose, width=width, height=height)
print("ctrl_prep shape:", ctrl_prep.shape)

pipe.vae.to("cuda")
with torch.no_grad():
    t = ctrl_prep.unsqueeze(2).to("cuda", torch.bfloat16)
    dist = pipe.vae.encode(t).latent_dist
    c_lat = dist.mode()
    print("c_lat shape:", c_lat.shape)

latents, _ = pipe.prepare_latents([t], 1, 64, height, width, torch.bfloat16, "cuda", torch.Generator("cuda").manual_seed(42), None)
print("latents shape:", latents.shape)

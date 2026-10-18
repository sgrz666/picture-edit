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
pipe.vae.to("cuda")

pose_path = "/home/shangguanrz/project/pic-edit/workflow/experiment/qwen-union-hybrid/data/prepared_controls/pose_source_torso.png"
img_pose = Image.open(pose_path).convert("RGBA")
width, height = 640, 928

ctrl_prep = pipe.image_processor.preprocess(img_pose, width=width, height=height).unsqueeze(2).to("cuda", torch.bfloat16)

with torch.no_grad():
    c_lat = pipe.vae.encode(ctrl_prep).latent_dist.mode()
    decoded = pipe.vae.decode(c_lat).sample
    print("decoded shape:", decoded.shape)
    img_out = pipe.image_processor.postprocess(decoded.squeeze(2), output_type="pil")[0]
    print("img_out size:", img_out.size)
    assert img_out.size == (640, 928), f"Expected (640, 928) but got {img_out.size}"
    print("✅ VAE Decode 验证通过！")

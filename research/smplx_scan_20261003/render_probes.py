"""Render saved camera-space meshes with their native cameras; no estimator reload."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
args = parser.parse_args()
directory = Path(args.dir).resolve()
repo = Path("/home/shangguanrz/project/pic-edit/add/tools/multi-hmr")
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"
sys.path.insert(0, str(repo))
os.chdir(repo)
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from utils.render import render_meshes

faces = np.load(repo / "models/smplx/SMPLX_NEUTRAL.npz")["f"]
panels = []
for name, title in [("smplestx_single", "SMPLest-X / single person"), ("multihmr_hi4d", "Multi-HMR / two people, shared camera")]:
    metadata = json.loads((directory / f"{name}.json").read_text())
    data = np.load(directory / f"{name}.npz")
    image = Image.open(metadata["image"]).convert("RGB")
    if name == "multihmr_hi4d":
        size = metadata["canvas_size"]
        image = ImageOps.pad(ImageOps.contain(image, (size, size)), (size, size))
        meshes = [data[f"person_{i}_v3d"] for i in range(metadata["person_count"])]
        K = data["K_canvas"][0]
        colors = [(0.2, 0.7, 1.0), (1.0, 0.45, 0.2)]
    else:
        meshes = [data["smplx_mesh_cam"][0]]
        K = data["K_image"]
        colors = [(0.2, 0.7, 1.0)]
    rendered = render_meshes(np.asarray(image), meshes, [faces] * len(meshes),
                             {"focal": [K[0, 0], K[1, 1]], "princpt": [K[0, 2], K[1, 2]]},
                             color=colors, alpha=0.65)
    overlay = Image.fromarray(rendered.astype(np.uint8))
    overlay.save(directory / f"{name}_overlay.png")
    left = ImageOps.pad(image, (480, 480), color=(235, 235, 235))
    right = ImageOps.pad(overlay, (480, 480), color=(235, 235, 235))
    panel = Image.new("RGB", (960, 512), "white")
    panel.paste(left, (0, 32))
    panel.paste(right, (480, 32))
    ImageDraw.Draw(panel).text((12, 10), title + " | input (left), mesh overlay (right)", fill="black")
    panels.append(panel)
preview = Image.new("RGB", (960, 1024), "white")
for i, panel in enumerate(panels):
    preview.paste(panel, (0, i * 512))
preview.save(directory / "smoke_previews.jpg", quality=95)
print(str(directory / "smoke_previews.jpg"))

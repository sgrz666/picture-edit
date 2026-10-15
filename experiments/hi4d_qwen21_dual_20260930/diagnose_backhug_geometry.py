import numpy as np
import torch
import nvdiffrast.torch as dr
from scipy.signal import fftconvolve
from PIL import Image, ImageDraw

from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset, letterbox_array, letterbox_points, SIZE

d = Hi4DQwen21Dataset("datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_test.jsonl",
                      "/home/shangguanrz/nihaochen/qwen21_hi4d_controls")
row = d.rows[573]
with np.load(row["target_geometry_npz"]) as data:
    geo = {name: data[name] for name in data.files}
with np.load(row["target_smpl"]) as data:
    smpl = {name: data[name] for name in data.files}
with np.load(row["target_camera"]) as data:
    camera = {name: data[name] for name in data.files}
index = list(camera["ids"]).index(row["camera_id"])
extr = camera["extrinsics"][index].astype(np.float32)
K = geo["K_runtime"].astype(np.float32)
height, width = int(geo["height"]), int(geo["width"])
semantic = [letterbox_array((geo[f"semantic_{side}"] > 0).astype(np.uint8), nearest=True).astype(bool)
            for side in ("A", "B")]
smpl_model = __import__("smplx").create("models/smpl", model_type="smpl", gender="neutral")
faces = torch.as_tensor(smpl_model.faces.astype(np.int32), device="cuda")
context = dr.RasterizeCudaContext(device=torch.device("cuda"))
raster_masks = []
for person in range(2):
    verts = smpl["verts"][person]
    points = verts @ extr[:, :3].T + extr[:, 3]
    projection = points @ K.T
    xy = letterbox_points(projection[:, :2] / projection[:, 2:3], width, height)
    depth = points[:, 2]
    clip = np.stack([(xy[:, 0] * 2 / SIZE - 1) * depth,
                     (1 - xy[:, 1] * 2 / SIZE) * depth,
                     (2 * (depth - 0.1) / 20 - 1) * depth, depth], axis=1)
    raster, _ = dr.rasterize(context, torch.from_numpy(clip).to("cuda", dtype=torch.float32)[None],
                             faces, resolution=(SIZE, SIZE))
    raster_masks.append(raster[0, :, :, 3].cpu().numpy() > 0)

for i, mesh in enumerate(raster_masks):
    yy, xx = np.nonzero(mesh)
    print("mesh", i, "bbox", (xx.min(), yy.min(), xx.max(), yy.max()), "pixels", len(xx))
for j, target in enumerate(semantic):
    yy, xx = np.nonzero(target)
    print("semantic", j, "bbox", (xx.min(), yy.min(), xx.max(), yy.max()), "pixels", len(xx))
for i, mesh in enumerate(raster_masks):
    for j, target in enumerate(semantic):
        overlap = int((mesh & target).sum())
        flipped = int((mesh[::-1] & target).sum())
        corr = fftconvolve(target.astype(np.float32), mesh[::-1, ::-1].astype(np.float32), mode="full")
        index = np.unravel_index(np.argmax(corr), corr.shape)
        shift = (int(index[1] - SIZE + 1), int(index[0] - SIZE + 1))
        print("mesh", i, "semantic", j, "direct", overlap / max(1, target.sum()),
              "vertical_flip", flipped / max(1, target.sum()),
              "best_shift", shift, "best_coverage", float(corr[index] / target.sum()))

panel = Image.new("RGB", (SIZE * 2, SIZE * 2), (0, 0, 0))
draw = ImageDraw.Draw(panel)
for i, mask in enumerate(raster_masks + semantic):
    layer = Image.fromarray((mask.astype(np.uint8) * 255), "L").convert("RGB")
    panel.paste(layer, ((i % 2) * SIZE, (i // 2) * SIZE))
    draw.text(((i % 2) * SIZE + 8, (i // 2) * SIZE + 8),
              f"{'mesh' if i < 2 else 'semantic'} {i % 2}", fill="red")
panel.save("/tmp/backhug_geometry_573.png")

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import onnx2torch

FACE_EVAL_DIR = Path("/home/shangguanrz/project/pic-edit/models/face_eval")
FACE_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Identity Encoder (ArcFace GlintR100)
# -------------------------------------------------------------
print("Building 1/4: identity_encoder.ts ...")
class IdentityEncoderWrapper(nn.Module):
    def __init__(self, glintr100_module):
        super().__init__()
        self.glintr100 = glintr100_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is in [-1, 1], shape [B, 3, H, W]
        # GlintR100 expects [B, 3, 112, 112] in [-1, 1]
        x_112 = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        feat = self.glintr100(x_112)
        return F.normalize(feat, dim=-1)

glintr100_onnx = "/home/shangguanrz/project/pic-edit/models/antelopev2/glintr100.onnx"
glintr100_torch = onnx2torch.convert(glintr100_onnx).eval()
id_wrapper = IdentityEncoderWrapper(glintr100_torch).eval()

dummy_crop = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    out_id = id_wrapper(dummy_crop)
print("  Identity encoder test output shape:", out_id.shape)
assert out_id.shape == (1, 512)

id_traced = torch.jit.trace(id_wrapper, dummy_crop)
id_path = FACE_EVAL_DIR / "identity_encoder.ts"
id_traced.save(str(id_path))
print("  Saved:", id_path)

# -------------------------------------------------------------
# 2. Perceptual Encoder (VGG16 Features)
# -------------------------------------------------------------
print("Building 2/4: perceptual_encoder.ts ...")
class VGGPerceptualWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features.eval()
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)]) # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)]) # relu3_3
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # relu4_3
        # Register normalization mean and std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [-1, 1]
        x_norm = ((x + 1.0) * 0.5 - self.mean) / self.std
        h1 = self.slice1(x_norm)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        # Pool to fixed spatial size to handle variable input dimensions
        p1 = F.adaptive_avg_pool2d(h1, (4, 4)).flatten(1)
        p2 = F.adaptive_avg_pool2d(h2, (4, 4)).flatten(1)
        p3 = F.adaptive_avg_pool2d(h3, (4, 4)).flatten(1)
        p4 = F.adaptive_avg_pool2d(h4, (4, 4)).flatten(1)
        return torch.cat([p1, p2, p3, p4], dim=1)

perc_wrapper = VGGPerceptualWrapper().eval()
with torch.no_grad():
    out_perc = perc_wrapper(dummy_crop)
print("  Perceptual encoder test output shape:", out_perc.shape)

perc_traced = torch.jit.trace(perc_wrapper, dummy_crop)
perc_path = FACE_EVAL_DIR / "perceptual_encoder.ts"
perc_traced.save(str(perc_path))
print("  Saved:", perc_path)

# -------------------------------------------------------------
# 3. LPIPS Evaluator (VGG-based Pairwise Distance)
# -------------------------------------------------------------
print("Building 3/4: lpips.ts ...")
import lpips
class LPIPSWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net='vgg', verbose=False).eval()

    def forward(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
        # in0 and in1 are in [-1, 1]
        dist = self.lpips_model(in0, in1)
        return dist.flatten(1).mean(dim=1)

lpips_wrapper = LPIPSWrapper().eval()
with torch.no_grad():
    out_lpips = lpips_wrapper(dummy_crop, dummy_crop)
print("  LPIPS test output shape:", out_lpips.shape, "val:", out_lpips.item())

lpips_traced = torch.jit.trace(lpips_wrapper, (dummy_crop, dummy_crop))
lpips_path = FACE_EVAL_DIR / "lpips.ts"
lpips_traced.save(str(lpips_path))
print("  Saved:", lpips_path)

# -------------------------------------------------------------
# 4. Landmark Evaluator (2D 106-point Face Alignment)
# -------------------------------------------------------------
print("Building 4/4: landmark_encoder.ts ...")
class LandmarkWrapper(nn.Module):
    def __init__(self, landmark_module):
        super().__init__()
        self.landmark_net = landmark_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [-1, 1], shape [B, 3, H, W]
        # 2d106det expects [B, 3, 192, 192]
        x_192 = F.interpolate(x, size=(192, 192), mode="bilinear", align_corners=False)
        # 2d106det uses standard BGR or RGB normalized input
        out = self.landmark_net(x_192)
        return out

landmark_onnx = "/home/shangguanrz/project/pic-edit/models/antelopev2/2d106det.onnx"
landmark_torch = onnx2torch.convert(landmark_onnx).eval()
landmark_wrapper = LandmarkWrapper(landmark_torch).eval()

with torch.no_grad():
    out_lm = landmark_wrapper(dummy_crop)
print("  Landmark encoder test output shape:", out_lm.shape)
assert out_lm.shape == (1, 212)

landmark_traced = torch.jit.trace(landmark_wrapper, dummy_crop)
landmark_path = FACE_EVAL_DIR / "landmark_encoder.ts"
landmark_traced.save(str(landmark_path))
print("  Saved:", landmark_path)

print("\nAll 4 evaluation models built and saved successfully to:", FACE_EVAL_DIR)

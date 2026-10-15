import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("/home/shangguanrz/project/pic-edit")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.precompute_iper_face_features_v65 import LocalFaceFeatureExtractor

print("Initializing LocalFaceFeatureExtractor on cuda...")
extractor = LocalFaceFeatureExtractor(
    insightface_root="/home/shangguanrz/project/pic-edit/models/antelopev2",
    dino_model="/home/shangguanrz/project/pic-edit/models/dinov2-giant",
    device="cuda",
)
print("Extractor initialized successfully!")
print("Arcface ID:", extractor.arcface_model_id)
print("DINO ID:", extractor.dino_model_id)

dummy_tight = np.zeros((112, 112, 3), dtype=np.uint8)
dummy_expanded = np.zeros((224, 224, 3), dtype=np.uint8)

# Test DINO forward pass directly
from PIL import Image
inputs = extractor._processor(images=Image.fromarray(dummy_expanded), return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    out = extractor._dino(**inputs).last_hidden_state
print("DINO output shape:", out.shape)
assert tuple(out.shape[1:]) == (257, 1536), f"Expected (257, 1536), got {out.shape}"
print("DINOv2-G forward pass verified on CUDA!")

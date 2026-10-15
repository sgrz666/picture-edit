import torch
from pathlib import Path

p = Path("/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_step_003000.pt")
ckpt = torch.load(p, map_location="cpu", weights_only=False)
sd = ckpt.get("adapter_state_dict", ckpt)

for k in sorted(sd):
    if "face_adapter.zero_heads" in k:
        print(k, sd[k].shape, sd[k].norm().item())

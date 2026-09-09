from safetensors.torch import load_file
weights = load_file("/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/connector/model.safetensors")
for k, v in weights.items():
    if "projector" in k:
        print(f"{k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}, std={v.std().item():.4f}")
print("meta_queries:", weights["meta_queries"].shape, weights["meta_queries"].min().item(), weights["meta_queries"].max().item(), weights["meta_queries"].std().item())
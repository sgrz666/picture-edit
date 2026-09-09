with open("/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/transformer/diffusion_pytorch_model.safetensors", "rb") as f:
    chunk0 = f.read(1024*1024)
    print("Chunk 0 first 1MB:", len(chunk0), "all zeros?", all(b == 0 for b in chunk0))
    f.seek(308714605)
    chunk1 = f.read(1024*1024)
    print("Chunk 1 first 1MB:", len(chunk1), "all zeros?", all(b == 0 for b in chunk1))
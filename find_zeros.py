with open("/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/transformer/diffusion_pytorch_model.safetensors", "rb") as f:
    idx = 0
    zero_chunks = []
    non_zero = 0
    while chunk := f.read(1024 * 1024):
        if all(b == 0 for b in chunk):
            zero_chunks.append(idx)
        else:
            non_zero += 1
        idx += 1
    print(f"Total 1MB chunks: {idx}")
    print(f"Non-zero chunks: {non_zero}")
    print(f"Zero chunks count: {len(zero_chunks)}")
    if zero_chunks:
        print("First 10 zero chunk indices:", zero_chunks[:10])
        print("Last 10 zero chunk indices:", zero_chunks[-10:])
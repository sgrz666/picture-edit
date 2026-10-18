import torch
free, total = torch.cuda.mem_get_info()
print(f"CUDA memory: Free = {free / 1024**3:.2f} GiB, Total = {total / 1024**3:.2f} GiB")

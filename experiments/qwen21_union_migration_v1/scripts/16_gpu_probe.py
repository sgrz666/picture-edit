#!/usr/bin/env python
"""探针：矩阵为什么停在等待显存——实测 free/total、小张量 CUDA 运算、以及谁在占卡。

不用 nvidia-smi（本机 NVML 与驱动版本不匹配，报 Failed to initialize NVML），
以 torch 的实际 CUDA 查询与运算为准，占卡进程用 /proc 里的 nvidia 设备句柄与命令行判断。
"""
import os
import pathlib
import subprocess

import torch

free, total = torch.cuda.mem_get_info()
print(f"cuda_available={torch.cuda.is_available()} free_GiB={free/2**30:.2f} total_GiB={total/2**30:.2f} "
      f"reserved_GiB={torch.cuda.memory_reserved()/2**30:.2f}")
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
torch.cuda.synchronize()
print(f"bf16 matmul ok, sum={y:.4f}")
print(f"allocated_GiB={torch.cuda.memory_allocated()/2**30:.2f}")

print("--- GPU 设备持有者 ---")
try:
    out = subprocess.run(["fuser", "-v", "/dev/nvidia0"], capture_output=True, text=True, timeout=30).stdout
    print(out.strip()[:1500] or "(fuser 无输出)")
except Exception as e:
    print("fuser 失败:", e)

print("--- 机器上的 python/torch 进程 ---")
try:
    ps = subprocess.run(["ps", "-eo", "pid,user,etime,time,%cpu,rss,cmd", "--sort=-rss"],
                        capture_output=True, text=True, timeout=30).stdout.splitlines()
    hits = [l for l in ps if ("python" in l and "grep" not in l)]
    print("\n".join(hits[:12]))
except Exception as e:
    print("ps 失败:", e)

print("--- 我自己名下的进程内存 ---")
me = os.environ.get("USER", "?")
print("user:", me)
print("本进程 rss_GiB:",
      round(int(pathlib.Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30, 2))

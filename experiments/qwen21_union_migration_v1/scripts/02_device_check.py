#!/usr/bin/env python
"""设备检查：用真实 CUDA/BF16 运算判定推理可行性，NVML 异常单独记录。

nvidia-smi 在这台机器上因驱动/NVML 版本不匹配而失败，因此不把它当作结论依据；
以实际小张量运算 + torch.cuda.mem_get_info() 为准。
"""
import json
import subprocess
import sys
import time

out = {"python": sys.version.split()[0]}

import torch

out["torch"] = torch.__version__
out["cuda_available"] = torch.cuda.is_available()
out["driver_version_from_torch"] = torch.version.cuda

# NVML 路径单独记录，不与可行性结论混在一起
try:
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                       capture_output=True, text=True, timeout=30)
    out["nvidia_smi"] = {"returncode": p.returncode, "stdout": p.stdout.strip()[:400],
                         "stderr": p.stderr.strip()[:400]}
except Exception as e:
    out["nvidia_smi"] = {"error": repr(e)}

if not torch.cuda.is_available():
    out["verdict"] = "CUDA 不可用，停在环境诊断"
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(2)

out["device_name"] = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
out["mem_get_info"] = {"free_GiB": round(free / 2**30, 2), "total_GiB": round(total / 2**30, 2)}

# 小张量 BF16 matmul + 同步 + 一个稍大的分配，验证真实算力与显存分配
res = {}
try:
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    res["bf16_matmul_1024"] = {"ok": True, "ms": round((time.time() - t0) * 1000, 1),
                               "finite": bool(torch.isfinite(c).all().item()),
                               "mean": round(float(c.float().mean()), 4)}
    del a, b, c

    x = torch.empty(int(8 * 2**30), dtype=torch.uint8, device="cuda")
    res["alloc_8GiB"] = {"ok": True, "allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 2)}
    del x
    torch.cuda.empty_cache()

    # attention 形状的最小前向（与主干 head_dim=128 一致）
    q = torch.randn(1, 1024, 32, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = q.clone()
    v = q.clone()
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    res["sdpa_bf16_1024tok"] = {"ok": True, "out_shape": list(o.shape),
                                "finite": bool(torch.isfinite(o).all().item())}
    res["peak_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out["cuda_checks"] = res
    out["verdict"] = "CUDA/BF16 实际运算通过，可以进入权重加载与推理"
except Exception as e:
    out["cuda_checks"] = res
    out["verdict"] = f"CUDA 运算失败，停在环境诊断：{type(e).__name__}: {e}"
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(3)

print(json.dumps(out, indent=1, ensure_ascii=False))

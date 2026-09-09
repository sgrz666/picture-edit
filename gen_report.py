import os
import subprocess
import torch

pip_freeze = subprocess.check_output(
    ["/home/shangguanrz/miniconda3/envs/deepgen/bin/pip", "list", "--format=freeze"],
    text=True,
)
with open("/home/shangguanrz/project/pic-edit/requirements_lock.txt", "w", encoding="utf-8") as f:
    f.write(pip_freeze)

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
gpu_cap = str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "N/A"
gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB" if torch.cuda.is_available() else "N/A"

git_commit = subprocess.check_output(
    ["git", "-C", "/home/shangguanrz/project/pic-edit/deepgen", "rev-parse", "HEAD"],
    text=True,
).strip()
lfs_ver = subprocess.check_output(
    ["/home/shangguanrz/.local/bin/git-lfs", "--version"],
    text=True,
).strip()

content = f"""# 系统与硬件配置锁定报告

- **主机名**: node01
- **操作系统**: Ubuntu 24.04.2 LTS
- **系统内存**: 125 GiB (可用: 113 GiB)
- **GPU 型号**: {gpu_name}
- **GPU 计算能力**: {gpu_cap}
- **GPU 总显存**: {gpu_mem}
- **PyTorch 版本**: {torch.__version__}
- **PyTorch 配套 CUDA**: {torch.version.cuda}
- **DeepGen 代码 Commit**: {git_commit}
- **Git-LFS 版本**: {lfs_ver}
- **环境依赖总数**: {len(pip_freeze.splitlines())} 个包
"""

with open("/home/shangguanrz/project/pic-edit/system_environment_lock.md", "w", encoding="utf-8") as f:
    f.write(content)

print(content)
print("Environment lock file written to /home/shangguanrz/project/pic-edit/requirements_lock.txt")

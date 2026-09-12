import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_control.pose_encoder import PoseEncoder
from pose_control.pose_encoder_v2 import PoseEncoderV2


def load_pose(
    path: Path,
    device: torch.device,
):
    image = Image.open(path).convert("RGB")

    array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def rms(x):
    return (
        x.float()
        .pow(2)
        .mean()
        .sqrt()
        .item()
    )


def mean_diff(a, b):
    return (
        (a.float() - b.float())
        .abs()
        .mean()
        .item()
    )


def max_diff(a, b):
    return (
        (a.float() - b.float())
        .abs()
        .max()
        .item()
    )


def spatial_std(x):
    return (
        x.float()
        .std(dim=(-2, -1))
        .mean()
        .item()
    )


def report(
    name,
    pose_a,
    pose_b,
    zero,
):
    signal = mean_diff(
        pose_a,
        pose_b,
    )

    magnitude = rms(pose_a)

    relative = (
        signal / magnitude
        if magnitude > 0
        else 0.0
    )

    print(f"\n===== {name} =====")
    print("shape:", tuple(pose_a.shape))

    print(
        f"RMS pose A        : {magnitude:.8e}"
    )

    print(
        f"RMS zero          : {rms(zero):.8e}"
    )

    print(
        f"spatial std A     : "
        f"{spatial_std(pose_a):.8e}"
    )

    print(
        f"mean diff A-B     : "
        f"{signal:.8e}"
    )

    print(
        f"max diff A-B      : "
        f"{max_diff(pose_a, pose_b):.8e}"
    )

    print(
        f"mean diff A-zero  : "
        f"{mean_diff(pose_a, zero):.8e}"
    )

    print(
        f"relative A-B      : "
        f"{relative:.8e}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose-a",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pose-b",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前没有可用 CUDA GPU"
        )

    device = torch.device("cuda")

    pose_a = load_pose(
        args.pose_a,
        device,
    )

    pose_b = load_pose(
        args.pose_b,
        device,
    )

    zero = torch.zeros_like(
        pose_a
    )

    print("===== INPUT =====")

    print(
        "nonzero ratio A:",
        (pose_a.abs() > 1e-6)
        .float()
        .mean()
        .item()
    )

    print(
        "nonzero ratio B:",
        (pose_b.abs() > 1e-6)
        .float()
        .mean()
        .item()
    )

    print(
        "input diff A-B:",
        mean_diff(
            pose_a,
            pose_b,
        )
    )

    # -------------------------
    # V1
    # -------------------------

    torch.manual_seed(args.seed)

    encoder_v1 = PoseEncoder().to(
        device=device,
        dtype=torch.float32,
    )

    encoder_v1.eval()

    with torch.no_grad():
        v1_a = encoder_v1(pose_a)
        v1_b = encoder_v1(pose_b)
        v1_zero = encoder_v1(zero)

    report(
        "V1 / 32x32",
        v1_a,
        v1_b,
        v1_zero,
    )

    # -------------------------
    # V2
    # -------------------------

    torch.manual_seed(args.seed)

    encoder_v2 = PoseEncoderV2().to(
        device=device,
        dtype=torch.float32,
    )

    encoder_v2.eval()

    with torch.no_grad():
        v2_a = encoder_v2(pose_a)
        v2_b = encoder_v2(pose_b)
        v2_zero = encoder_v2(zero)

    report(
        "V2 / 64x64",
        v2_a,
        v2_b,
        v2_zero,
    )

    # 这里只做接口可行性诊断，不写入正式模型。
    # 检查 64×64 特征压到 DeepGen 32×32 后，
    # 姿态差异还能保留多少。
    v2_a_32 = F.avg_pool2d(
        v2_a,
        kernel_size=2,
        stride=2,
    )

    v2_b_32 = F.avg_pool2d(
        v2_b,
        kernel_size=2,
        stride=2,
    )

    v2_zero_32 = F.avg_pool2d(
        v2_zero,
        kernel_size=2,
        stride=2,
    )

    report(
        "V2 / diagnostic 32x32",
        v2_a_32,
        v2_b_32,
        v2_zero_32,
    )

    if v2_zero.abs().max().item() > 1e-7:
        raise RuntimeError(
            "V2 的全零输入产生了非零特征，"
            "zero-bias 初始化没有生效"
        )

    print(
        "\nPoseEncoder V2 sensitivity test: PASS"
    )


if __name__ == "__main__":
    main()

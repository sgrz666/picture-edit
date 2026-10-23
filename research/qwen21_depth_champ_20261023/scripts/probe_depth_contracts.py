"""Small CPU mechanism probes, not image-quality or training validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--official-weight", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(23)
    torch.set_num_threads(2)
    result = {"torch": torch.__version__, "device": "cpu", "scope": "mechanisms only"}

    # A foreground/background boundary: a background sentinel must not halve real Z.
    z = torch.tensor([[[[4., 0.], [4., 0.]]]])
    valid = (z > 0).float()
    naive = F.avg_pool2d(z, 2)
    coverage = F.avg_pool2d(valid, 2)
    masked = F.avg_pool2d(z * valid, 2) / coverage.clamp_min(1e-8)
    assert masked.item() == 4. and naive.item() == 2.
    result["masked_pool"] = {"naive": naive.item(), "mask_aware": masked.item(),
                              "coverage": coverage.item()}

    # Zero gate * zero projection is a dead product even if the downstream head is trained.
    gate = nn.Parameter(torch.zeros(()))
    projection = nn.Linear(3, 4, bias=False)
    nn.init.zeros_(projection.weight)
    x = torch.randn(2, 3)
    (gate * projection(x)).sum().backward()
    assert gate.grad.item() == 0. and projection.weight.grad.abs().max().item() == 0.
    result["double_zero_product"] = {"gate_grad": gate.grad.item(),
                                      "projection_grad_max": projection.weight.grad.abs().max().item()}

    # A single zero projection with a nonzero multiplier receives a learning signal.
    projection.zero_grad(set_to_none=True)
    projection(x).sum().backward()
    assert projection.weight.grad.norm().item() > 0
    result["single_zero_projection"] = {"projection_grad_norm": projection.weight.grad.norm().item()}

    # Two serial zero heads plus a nonzero inherited path cause a delay, not permanent death.
    trunk = nn.Linear(3, 4, bias=False)
    depth_head = nn.Linear(4, 4, bias=False)
    body_head = nn.Linear(4, 2, bias=False)
    nn.init.zeros_(depth_head.weight)
    nn.init.zeros_(body_head.weight)
    optimizer = torch.optim.SGD([*trunk.parameters(), *depth_head.parameters(), *body_head.parameters()], lr=.05)
    g0, target = torch.randn(2, 4), torch.randn(2, 2)
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = body_head(g0 + depth_head(trunk(x)))
        F.mse_loss(prediction, target).backward()
        steps.append({"step": step, "body_head_grad": body_head.weight.grad.norm().item(),
                      "depth_head_grad": depth_head.weight.grad.norm().item(),
                      "depth_trunk_grad": trunk.weight.grad.norm().item()})
        optimizer.step()
    assert steps[0]["depth_head_grad"] == 0 and steps[1]["depth_head_grad"] > 0
    assert steps[1]["depth_trunk_grad"] == 0 and steps[2]["depth_trunk_grad"] > 0
    result["serial_zero_heads"] = steps

    if args.official_weight:
        state = torch.load(args.official_weight, map_location="cpu", weights_only=True)
        w, b = state["conv_in.weight"], state["conv_in.bias"]
        gray = torch.rand(1, 1, 32, 48)
        original = F.conv2d(gray.expand(-1, 3, -1, -1), w, b, padding=1)
        w2 = torch.cat((w.sum(1, keepdim=True), torch.zeros_like(w[:, :1])), dim=1)
        converted = F.conv2d(torch.cat((gray, torch.ones_like(gray)), dim=1), w2, b, padding=1)
        incorrect_mean = F.conv2d(gray, w.mean(1, keepdim=True), b, padding=1)
        error = (original-converted).abs().max().item()
        assert torch.allclose(original, converted, atol=2e-6, rtol=1e-5)
        result["rgb_gray_to_depth_valid_conv"] = {"original_shape": list(w.shape),
                "converted_shape": list(w2.shape), "sum_conversion_max_abs": error,
                "incorrect_mean_max_abs": (original-incorrect_mean).abs().max().item(),
                "scope": "first convolution equivalence only; no encoder or Qwen quality claim"}

    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

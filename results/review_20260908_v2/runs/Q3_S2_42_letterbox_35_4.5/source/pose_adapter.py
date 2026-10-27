from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock2D(nn.Module):

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.norm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, padding=1, bias=False
    )
    self.norm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, padding=1, bias=False
    )
    self.act = nn.SiLU()

    if in_channels != out_channels:
      self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    else:
      self.shortcut = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.conv1(self.act(self.norm1(x)))
    h = self.conv2(self.act(self.norm2(h)))
    return h + self.shortcut(x)


class Downsample2D(nn.Module):

  def __init__(self, channels: int):
    super().__init__()
    self.conv = nn.Conv2d(
        channels, channels, kernel_size=3, stride=2, padding=1
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class PoseConditionAdapter(nn.Module):
  """Lightweight condition adapter for DeepGen DiT backbone.

  - Parameter budget: ~160M parameters (within the 100M - 300M design target)
  - Input: Multi-channel control map (B, C_in, H, W), e.g., C_in=8
  - Output: Token residuals of dimension hidden_dim (1536) matching DiT patch
  tokens
  - Zero-conv initialization: Outputs zero at initialization so baseline
  behavior is preserved
  """

  def __init__(
      self,
      in_channels: int = 8,
      base_channels: int = 256,
      hidden_dim: int = 1536,  # DeepGen DiT hidden size
      num_blocks: int = 4,
      num_injection_layers: int = 6,  # Number of DiT blocks receiving residuals
  ):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_dim = hidden_dim
    self.num_injection_layers = num_injection_layers

    # Stage 1: Initial projection from pixel space
    self.init_conv = nn.Conv2d(
        in_channels, base_channels, kernel_size=3, padding=1
    )

    # Downsampling stages: H/W -> H/2 -> H/4 -> H/8 -> H/16
    # For 512x512 input: 512 -> 256 -> 128 -> 64 -> 32 (matching VAE /8 and patch /2 = /16)
    self.down1 = nn.Sequential(
        ResnetBlock2D(base_channels, base_channels), Downsample2D(base_channels)
    )  # -> /2

    self.down2 = nn.Sequential(
        ResnetBlock2D(base_channels, base_channels * 2),
        Downsample2D(base_channels * 2),
    )  # -> /4

    self.down3 = nn.Sequential(
        ResnetBlock2D(base_channels * 2, base_channels * 2),
        Downsample2D(base_channels * 2),
    )  # -> /8

    self.down4 = nn.Sequential(
        ResnetBlock2D(base_channels * 2, base_channels * 4),
        Downsample2D(base_channels * 4),
    )  # -> /16

    mid_channels = base_channels * 4  # 1024
    self.mid_blocks = nn.ModuleList(
        [ResnetBlock2D(mid_channels, mid_channels) for _ in range(num_blocks)]
    )

    # Output projection to DiT token embedding space (1536)
    self.out_proj = nn.Conv2d(mid_channels, hidden_dim, kernel_size=1)

    # Multi-layer residual injectors with ZERO initialization
    self.zero_convs = nn.ModuleList([
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        for _ in range(num_injection_layers)
    ])

    self._init_zero_weights()

  def _init_zero_weights(self):
    """Zero-initialize the final projection layers so initial adapter output is strictly zero."""
    for zc in self.zero_convs:
      nn.init.zeros_(zc.weight)
      if zc.bias is not None:
        nn.init.zeros_(zc.bias)

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, control_map: torch.Tensor) -> List[torch.Tensor]:
    """control_map: (B, C_in, H, W) e.g.

    (B, 8, 512, 512) Returns:
      residuals: List of (B, N_tokens, hidden_dim) where N_tokens = (H/16)*(W/16)
    """
    x = self.init_conv(control_map)
    x = self.down1(x)
    x = self.down2(x)
    x = self.down3(x)
    x = self.down4(x)

    for block in self.mid_blocks:
      x = block(x)

    feat = self.out_proj(x)  # (B, hidden_dim, H/16, W/16)

    # Generate multi-layer residuals
    residuals = []
    B, C, H_tok, W_tok = feat.shape
    for zc in self.zero_convs:
      res_map = zc(feat)  # (B, hidden_dim, H_tok, W_tok)
      # Reshape to token sequence: (B, N_tokens, hidden_dim)
      res_tokens = res_map.flatten(2).transpose(1, 2)
      residuals.append(res_tokens)

    return residuals


if __name__ == "__main__":
  adapter = PoseConditionAdapter()
  num_params = adapter.count_parameters()
  print(
      f"PoseConditionAdapter initialized. Trainable parameters:"
      f" {num_params / 1e6:.2f} M"
  )

  dummy_input = torch.randn(2, 8, 512, 512)
  residuals = adapter(dummy_input)
  print(f"Input shape: {dummy_input.shape}")
  print(f"Number of residual layers: {len(residuals)}")
  print(f"Layer 0 residual token shape: {residuals[0].shape}")
  print(
      f"Zero-init check (max abs value):"
      f" {residuals[0].abs().max().item():.6f}"
  )

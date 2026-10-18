"""One recorded image-space transform for TikTok RGB and geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from PIL import Image

VERSION = 'letterbox_v1'
NATIVE_VERSION = 'native_pad_v1'



@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    side: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    @classmethod
    def from_size(cls, size, side):
        width, height = map(int, size)
        if min(width, height, side) <= 0:
            raise ValueError('Image and canvas dimensions must be positive')
        scale = side / max(width, height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        return cls(width, height, side, rw, rh, (side-rw)//2, (side-rh)//2)

    def metadata(self):
        return {'version': VERSION, **asdict(self), 'valid_rect': list(self.rect)}

    @property
    def rect(self):
        return self.left, self.top, self.left+self.resized_width, self.top+self.resized_height

    def image(self, image, *, nearest=False, fill=0):
        """Condition maps share the native image FoV even if stored at 512."""
        method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        resized = image.resize((self.resized_width, self.resized_height), method)
        canvas = Image.new(image.mode, (self.side, self.side), fill)
        canvas.paste(resized, (self.left, self.top))
        return canvas

    def valid_mask(self):
        mask = torch.zeros(1, self.side, self.side, dtype=torch.float32)
        x0,y0,x1,y1 = self.rect
        mask[:,y0:y1,x0:x1] = 1
        return mask

    def points(self, points):
        is_numpy = isinstance(points, np.ndarray)
        out = points.copy() if is_numpy else points.clone()
        out[..., 0] = points[..., 0] * (self.resized_width / self.side) + self.left / self.side
        out[..., 1] = points[..., 1] * (self.resized_height / self.side) + self.top / self.side
        return out

    def global_camera(self, vector):
        if vector.shape[-1] != 26:
            raise ValueError('Expected the legacy 26D SMPL-X global condition')
        out = vector.clone()
        # 10 betas, 6 rot6d, 3 translation, then normalized fx,fy,cx,cy,tx,ty,tz.
        out[...,19] *= self.resized_width/self.side
        out[...,20] *= self.resized_height/self.side
        out[...,21] = vector[...,21] * (self.resized_width/self.side) + self.left/self.side
        out[...,22] = vector[...,22] * (self.resized_height/self.side) + self.top/self.side
        return out

    def unpad(self, image):
        if torch.is_tensor(image):
            cropped = image[..., self.top:self.top + self.resized_height, self.left:self.left + self.resized_width]
            return torch.nn.functional.interpolate(cropped, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return image.crop(self.rect).resize((self.width, self.height), Image.Resampling.LANCZOS)

    def unpad_points(self, points):
        is_numpy = isinstance(points, np.ndarray)
        out = points.copy() if is_numpy else points.clone()
        out[..., 0] = (points[..., 0] - self.left / self.side) * (self.side / self.resized_width)
        out[..., 1] = (points[..., 1] - self.top / self.side) * (self.side / self.resized_height)
        return out


@dataclass(frozen=True)
class NativePad:
    width: int
    height: int
    padded_width: int
    padded_height: int
    left: int
    top: int
    stride: int = 16

    @classmethod
    def from_size(cls, size, stride: int = 16, pad_mode: str = 'edge'):
        width, height = map(int, size)
        if min(width, height, stride) <= 0:
            raise ValueError('Image and canvas dimensions must be positive')
        # Qwen VAE downsamples 16x and diffusers _unpack_latents uses 32x (2 * (dim // 32)),
        # while DiT VLM slots require total tokens to be divisible by 4.
        min_step = max(stride, 32) if stride == 16 else stride
        pw = ((width + min_step - 1) // min_step) * min_step
        ph = ((height + min_step - 1) // min_step) * min_step
        while ((pw // stride) * (ph // stride)) % 4 != 0:
            ph += stride
        if pad_mode == 'edge':
            left, top = 0, 0
        elif pad_mode == 'center':
            left, top = (pw - width) // 2, (ph - height) // 2
        else:
            raise ValueError(f'Unsupported pad_mode: {pad_mode}')
        return cls(width, height, pw, ph, left, top, stride)

    @classmethod
    def from_record(cls, record: Mapping):
        return cls(
            width=int(record['width']),
            height=int(record['height']),
            padded_width=int(record.get('padded_width', record['width'])),
            padded_height=int(record.get('padded_height', record['height'])),
            left=int(record.get('left', 0)),
            top=int(record.get('top', 0)),
            stride=int(record.get('stride', 16)),
        )

    def metadata(self):
        return {'version': NATIVE_VERSION, **asdict(self), 'valid_rect': list(self.rect)}

    @property
    def rect(self):
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.padded_height // self.stride, self.padded_width // self.stride

    @property
    def token_count(self) -> int:
        h, w = self.grid_shape
        return h * w

    def image(self, image, *, nearest=False, fill=0):
        """Preserves unscaled native RGB pixel dimensions, padding only to stride multiple."""
        if image.size != (self.width, self.height):
            method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
            img = image.resize((self.width, self.height), method)
        else:
            img = image
        canvas = Image.new(image.mode, (self.padded_width, self.padded_height), fill)
        canvas.paste(img, (self.left, self.top))
        return canvas

    def unpad(self, image):
        if torch.is_tensor(image):
            return image[..., self.top:self.top + self.height, self.left:self.left + self.width]
        return image.crop(self.rect)

    def valid_mask(self):
        mask = torch.zeros(1, self.padded_height, self.padded_width, dtype=torch.float32)
        x0, y0, x1, y1 = self.rect
        mask[:, y0:y1, x0:x1] = 1.0
        return mask

    def points(self, points):
        is_numpy = isinstance(points, np.ndarray)
        out = points.copy() if is_numpy else points.clone()
        out[..., 0] = points[..., 0] * (self.width / self.padded_width) + self.left / self.padded_width
        out[..., 1] = points[..., 1] * (self.height / self.padded_height) + self.top / self.padded_height
        return out

    def unpad_points(self, points):
        is_numpy = isinstance(points, np.ndarray)
        out = points.copy() if is_numpy else points.clone()
        out[..., 0] = (points[..., 0] - self.left / self.padded_width) * (self.padded_width / self.width)
        out[..., 1] = (points[..., 1] - self.top / self.padded_height) * (self.padded_height / self.height)
        return out

    def global_camera(self, vector):
        if vector.shape[-1] != 26:
            raise ValueError('Expected the legacy 26D SMPL-X global condition')
        out = vector.clone()
        out[..., 19] *= self.width / self.padded_width
        out[..., 20] *= self.height / self.padded_height
        out[..., 21] = vector[..., 21] * (self.width / self.padded_width) + self.left / self.padded_width
        out[..., 22] = vector[..., 22] * (self.height / self.padded_height) + self.top / self.padded_height
        return out



def valid_body_velocity_loss(squared, human, valid=None):
    """Human weight 3; fractional token coverage excludes synthetic borders."""
    weight = 1 + 2*human
    if valid is not None:
        weight = weight * valid
    denominator = weight.sum()*squared.shape[-1]
    if not bool(denominator > 0):
        raise ValueError('Body loss has no valid image tokens')
    return (squared*weight).sum()/denominator

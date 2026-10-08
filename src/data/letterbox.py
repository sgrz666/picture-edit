"""One recorded image-space transform for TikTok RGB and geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from PIL import Image

VERSION = 'letterbox_v1'


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
        out = points.clone()
        out[...,0] = points[...,0] * (self.resized_width/self.side) + self.left/self.side
        out[...,1] = points[...,1] * (self.resized_height/self.side) + self.top/self.side
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


def valid_body_velocity_loss(squared, human, valid=None):
    """Human weight 3; fractional token coverage excludes synthetic borders."""
    weight = 1 + 2*human
    if valid is not None:
        weight = weight * valid
    denominator = weight.sum()*squared.shape[-1]
    if not bool(denominator > 0):
        raise ValueError('Body loss has no valid image tokens')
    return (squared*weight).sum()/denominator

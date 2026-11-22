"""Pixel background rewards on explicitly corresponding Source canvases."""

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageFilter

from src.data.canvas_contract import CanvasSpec
from ..native_v62.training.io import verify


@dataclass(frozen=True)
class SourceBackground:
    pixels: np.ndarray
    support: np.ndarray

    def score(self, image):
        pixels = np.asarray(image, dtype=np.float32) / 255
        if pixels.shape != self.pixels.shape:
            raise ValueError('Generated background requires the bound Source Canvas')
        mse = float(np.square(pixels[self.support] - self.pixels[self.support]).mean())
        return mse, math.exp(-mse / .02)


def prepare_background(pair, source, target, target_foreground):
    """Fix pixel support before generation, including a Source/GT alignment check."""
    alignment = pair.get('background_alignment')
    if alignment is None:
        return None, 'alignment_unverified'
    src = CanvasSpec.from_record(pair['source_canvas'])
    dst = CanvasSpec.from_record(pair['target_canvas'])
    if src.fingerprint() != dst.fingerprint():
        return None, 'canvas_transform_mismatch'
    refs = {name: pair.get(name) for name in ('source_rgb', 'target_rgb', 'source_mask')}
    if (not all(ref and ref.get('sha256') for ref in refs.values()) or alignment != {
            'method': 'identity', 'canvas_id': dst.fingerprint(),
            **{name + '_sha256': refs[name]['sha256'] for name in ('source_rgb', 'target_rgb')}}):
        return None, 'alignment_provenance_mismatch'
    bound = {name: verify(ref) for name, ref in refs.items()}
    with Image.open(bound['source_mask']['path']) as mask:
        source_foreground = np.asarray(src.image(mask.convert('L'), nearest=True)) > 0
    valid = dst.valid_mask()[0].numpy() > 0
    target_foreground = np.asarray(target_foreground, dtype=bool)
    if target_foreground.shape != valid.shape:
        raise ValueError('Target foreground and Canvas differ')
    # Require a measured person support on both frames.
    if not source_foreground.any() or not target_foreground.any():
        return None, 'foreground_support_missing'
    foreground = Image.fromarray((source_foreground | target_foreground).astype(np.uint8) * 255)
    support = valid & (np.asarray(foreground.filter(ImageFilter.MaxFilter(33))) == 0)
    if support.sum() < .1 * valid.sum():
        return None, 'background_support_insufficient'
    pixels, target_pixels = (np.asarray(image.convert('RGB'), dtype=np.float32) / 255 for image in (source, target))
    if pixels.shape != (*valid.shape, 3) or target_pixels.shape != pixels.shape:
        raise ValueError('Background alignment images do not match the bound Canvas')
    if np.abs(pixels[support] - target_pixels[support]).mean() > .035:
        return None, 'background_pixels_inconsistent'
    return SourceBackground(pixels, support), None

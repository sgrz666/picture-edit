"""Source A pixel scoring on the immutable output Canvas."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from src.data.canvas_contract import CanvasSpec
from ..native_v62.training.io import verify


BACKGROUND_METRIC = 'masked_source_l1_v1'
BACKGROUND_SCALE = .05


@dataclass(frozen=True)
class SourceBackground:
    pixels: np.ndarray
    support: np.ndarray
    mapping: str = 'identity'

    def score(self, image):
        pixels = np.asarray(image)
        if pixels.shape != self.pixels.shape:
            raise ValueError('Generated background requires the bound output Canvas')
        difference = np.subtract(pixels[self.support], self.pixels[self.support], dtype=np.float32) / 255.
        l1 = float(np.abs(difference).mean())
        return {'background_l1': l1, 'background_mse': float(np.square(difference).mean()),
                'background_reward': math.exp(-l1 / BACKGROUND_SCALE)}


def _mapping(source, output):
    if source.fingerprint() == output.fingerprint():
        return 'identity'
    sh, sw = source.original_hw
    oh, ow = output.original_hw
    # Accept only the same upright content aspect, with at most one source
    # and output pixel of combined raster rounding. No crop is inferred.
    if abs(sw * oh - ow * sh) <= max(sh, oh):
        return 'content_resize'
    return None


def prepare_source_background(source, source_canvas, output_canvas, source_foreground, target_foreground):
    """Establish Source A -> output support without evaluating Target RGB."""
    mapping = _mapping(source_canvas, output_canvas)
    if mapping is None:
        return None, 'content_aspect_mismatch'
    if source_foreground is None or target_foreground is None:
        return None, 'foreground_support_missing'
    source_foreground = np.asarray(source_foreground, dtype=bool)
    target_foreground = np.asarray(target_foreground, dtype=bool)
    if (source.size != source_canvas.canvas_hw[::-1] or source_foreground.shape != source_canvas.canvas_hw
            or target_foreground.shape != output_canvas.canvas_hw):
        raise ValueError('Source RGB and foreground masks must use their bound Canvases')
    if not source_foreground.any() or not target_foreground.any():
        return None, 'foreground_support_missing'
    source = source.convert('RGB')
    if mapping == 'content_resize':
        content = source.crop(source_canvas.valid_rect).resize(output_canvas.resized_hw[::-1], Image.Resampling.BILINEAR)
        mapped = Image.new('RGB', output_canvas.canvas_hw[::-1])
        mapped.paste(content, output_canvas.offset_xy)
        source = mapped
        mask = Image.fromarray(source_foreground).crop(source_canvas.valid_rect)
        mask = mask.resize(output_canvas.resized_hw[::-1], Image.Resampling.NEAREST)
        mapped_mask = Image.new('1', output_canvas.canvas_hw[::-1])
        mapped_mask.paste(mask, output_canvas.offset_xy)
        source_foreground = np.asarray(mapped_mask, dtype=bool)
    valid = output_canvas.valid_mask()[0].numpy() > 0
    union = (source_foreground | target_foreground).astype(np.uint8)
    expanded = cv2.dilate(union, np.ones((33, 33), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    support = valid & (expanded == 0)
    if support.sum() < .1 * valid.sum():
        return None, 'background_support_insufficient'
    return SourceBackground(np.asarray(source), support, mapping), None


def source_person_foreground(sample):
    """Union all Source A slots on its Canvas, excluding independent Source B."""
    local = sample['local_condition_v4']
    bank = sample.get('source_bank_v5')
    if bank is None:
        slots = sorted(set(local.source_indices[0, local.person_valid[0]].tolist()))
        masks = sample['source_person_mask'][slots, 0].bool().cpu().numpy()
    else:
        primary = bank.sources[0]
        if primary.qwen_image_index != 0 or primary.rgb_sha256 != local.metadata[0]['source_rgb_sha256']:
            raise ValueError('Dual Source background must use the first reference image')
        masks = primary.person_mask[0, :, 0].bool().cpu().numpy()
    if not all(mask.any() for mask in masks) or not len(masks):
        return None
    return masks.any(axis=0)


def regional_mask_failure(entries, source_key, target_key):
    for role, key in (('source', source_key), ('target', target_key)):
        if entries.get(key, {}).get('metadata', {}).get('segmentation_accepted') is False:
            return f'{role}_foreground_unqualified'
    return None


def prepare_background(pair, source, target_foreground, *, source_foreground=None):
    """Read optional SHA-bound Source A mask evidence and score on the output grid."""
    src = CanvasSpec.from_record(pair['source_canvas'])
    dst = CanvasSpec.from_record(pair['target_canvas'])
    alignment = pair.get('background_alignment')
    ref = pair.get('source_mask')
    if pair.get('source_mask_accepted') is False:
        return None, 'source_foreground_unqualified'
    if alignment is not None:
        expected = {'method': _mapping(src, dst), 'canvas_id': dst.fingerprint(),
                    **{name + '_sha256': pair.get(name, {}).get('sha256') for name in ('source_rgb', 'target_rgb')}}
        if (_mapping(src, dst) is None or alignment != expected or not ref or not ref.get('sha256')
                or not expected['source_rgb_sha256'] or not expected['target_rgb_sha256']):
            return None, 'alignment_provenance_mismatch'
    if ref is not None:
        if not ref.get('sha256'):
            raise ValueError('Source background mask must have a pinned SHA256')
        with Image.open(verify(ref)['path']) as mask:
            verified_mask = np.asarray(src.image(mask.convert('L'), nearest=True)) > 0
        source_foreground = (verified_mask if source_foreground is None else (source_foreground | verified_mask))
    return prepare_source_background(source, src, dst, source_foreground, target_foreground)


def native_background(sample, condition, source, target_masks, entry=None):
    """Bind background to ordered Qwen image 0 with Native Source/target geometry."""
    meta = sample['local_condition_v4'].metadata[0]
    primary = condition['images'][0]
    if (condition['primary_source_index'] != 0 or primary['role'] not in ('source', 'source_a')
            or primary['file']['sha256'] != meta['source_rgb_sha256']
            or primary['canvas_id'] != meta['source_canvas_id']
            or condition['target_canvas_id'] != meta['target_canvas_id']):
        raise ValueError('Native background must use the bound primary Source A and output Canvas')
    if entry is not None and (entry['source_rgb']['sha256'] != meta['source_rgb_sha256']
            or entry['target_rgb']['sha256'] != meta['driver_rgb_sha256']):
        raise ValueError('Background index RGB differs from the bound Native pair')
    pair = {**(entry or {}), 'source_canvas': sample['source_canvas'], 'target_canvas': sample['target_canvas']}
    target = np.logical_or.reduce(target_masks) if all(mask.any() for mask in target_masks) else None
    return prepare_background(pair, source, target, source_foreground=source_person_foreground(sample))

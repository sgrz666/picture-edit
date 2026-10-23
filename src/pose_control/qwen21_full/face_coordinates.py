"""Apply the RGB letterbox transform to native-normalized Face coordinates."""
from __future__ import annotations

import json
from typing import Mapping, Sequence

import torch

from src.data.letterbox import Letterbox, VERSION, NativePad, NATIVE_VERSION


def transform_from_record(record: Mapping) -> Letterbox | NativePad:
    version = record.get('version')
    if version == NATIVE_VERSION:
        transform = NativePad.from_record(record)
        if transform.metadata() != dict(record):
            raise ValueError('Face transform differs from the RGB transform')
        return transform
    if version != VERSION:
        raise ValueError('Face coordinates require recorded letterbox_v1 or native_pad_v1 transform')
    transform = Letterbox.from_size((record['width'], record['height']), record['side'])
    if transform.metadata() != dict(record):
        raise ValueError('Face transform differs from the RGB transform')
    return transform


def batch_alignment_records(batch, count):
    if 'alignment_meta' not in batch:
        return None
    values = batch['alignment_meta']
    if isinstance(values, str):
        values = [values]
    if len(values) != count:
        raise ValueError('One source/target Face transform is required per sample')
    return [json.loads(item) if isinstance(item, str) else item for item in values]


def transform_face_condition(condition, alignments: Sequence[Mapping] | None):
    """No changes to identity, texture, confidence, jaw, expression or slot binding."""
    if alignments is None:
        return condition
    if len(alignments) != condition.face_valid.shape[0]:
        raise ValueError('Face transform batch mismatch')
    condition.landmarks = condition.landmarks.clone()
    condition.source_boxes = condition.source_boxes.clone()
    condition.target_boxes = condition.target_boxes.clone()
    for index, record in enumerate(alignments):
        source = transform_from_record(record['source'])
        target = transform_from_record(record['target'])
        for slot in range(condition.face_valid.shape[1]):
            if not bool(condition.face_valid[index, slot]):
                continue
            condition.landmarks[index, slot] = target.points(condition.landmarks[index, slot])
            condition.source_boxes[index, slot] = source.points(
                condition.source_boxes[index, slot].reshape(2, 2)).reshape(4)
            condition.target_boxes[index, slot] = target.points(
                condition.target_boxes[index, slot].reshape(2, 2)).reshape(4)
    return condition


def valid_image_tokens(batch, grid, device):
    if 'valid_image_mask' not in batch:
        return None
    if isinstance(grid, (tuple, list)):
        size = (int(grid[0]), int(grid[1]))
    else:
        size = (int(grid), int(grid))
        canvas = batch.get('tgt_image')
        if canvas is not None and size != (canvas.shape[-2] // 16, canvas.shape[-1] // 16):
            raise ValueError(
                f"square token grid {size} does not match the real canvas grid "
                f"{(canvas.shape[-2] // 16, canvas.shape[-1] // 16)}; pass the rectangular "
                "target_token_hw instead of one integer")
    return torch.nn.functional.interpolate(
        batch['valid_image_mask'].to(device=device, dtype=torch.float32),
        size=size, mode='area').flatten(2).transpose(1, 2)

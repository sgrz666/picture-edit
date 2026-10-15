import numpy as np
from PIL import Image
import pytest

from scripts.eval.eval_qwen21_joint_pilot import compute_bg_l1


def test_background_requires_both_original_masks():
    source = Image.new('RGB', (64, 64))
    mask = Image.new('L', source.size)
    value, reason = compute_bg_l1(source, source, mask, None)
    assert value is None and 'A/B' in reason
    value, reason = compute_bg_l1(source, source, mask, mask.resize((32, 32)))
    assert value is None and 'dimensions' in reason


def test_background_excludes_union_dilation_and_canvas_boundary():
    source = Image.new('RGB', (64, 64))
    a, b = np.zeros((64, 64), dtype=np.uint8), np.zeros((64, 64), dtype=np.uint8)
    a[20, 20] = b[40, 40] = 255
    changed = np.zeros((64, 64, 3), dtype=np.uint8)
    changed[:8] = changed[-8:] = 255
    changed[:, :8] = changed[:, -8:] = 255
    changed[12:29, 12:29] = changed[32:49, 32:49] = 255
    value, reason = compute_bg_l1(source, Image.fromarray(changed), Image.fromarray(a), Image.fromarray(b))
    assert reason is None and value == 0
    changed[10, 10] = 255
    value, reason = compute_bg_l1(source, Image.fromarray(changed), Image.fromarray(a), Image.fromarray(b))
    assert reason is None and value > 0


def test_background_full_union_has_no_score():
    image = Image.new('RGB', (64, 64))
    value, reason = compute_bg_l1(image, image, Image.new('L', image.size, 255), Image.new('L', image.size))
    assert value is None and 'background pixels' in reason

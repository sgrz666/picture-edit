import numpy as np

from src.data.hi4d_qwen21_dataset import SIZE, _orient_parts_to_semantic


def test_vertical_orientation_uses_geometry_semantic_area():
    parts = np.zeros((SIZE, SIZE), np.uint8)
    parts[100:180, 150:190] = 7
    semantic = np.zeros((SIZE, SIZE), bool)
    semantic[SIZE - 180:SIZE - 100, 150:190] = True
    aligned, coverage, flipped = _orient_parts_to_semantic(parts, semantic)
    assert flipped
    assert coverage == 1.0
    assert np.array_equal(aligned[semantic], np.full(80 * 40, 7, np.uint8))


def test_orientation_keeps_aligned_raster_unchanged():
    parts = np.zeros((SIZE, SIZE), np.uint8)
    parts[100:180, 150:190] = 7
    semantic = np.zeros((SIZE, SIZE), bool)
    semantic[100:180, 150:190] = True
    aligned, coverage, flipped = _orient_parts_to_semantic(parts, semantic)
    assert not flipped
    assert coverage == 1.0
    assert np.array_equal(aligned, parts)

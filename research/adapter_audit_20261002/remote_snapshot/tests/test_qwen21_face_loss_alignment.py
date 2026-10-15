from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pose_control.qwen21_full.face_loss_alignment import (
    align_face_rgb, build_arcface_alignment_grid,
)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def five_points():
    # A translated, enlarged canonical face inside a non-square original image.
    from insightface.utils.face_align import arcface_dst
    return torch.tensor(arcface_dst.copy())[None] * 1.6 + torch.tensor([25., 7.])


def textured_image():
    yy, xx = torch.meshgrid(torch.arange(192), torch.arange(240), indexing="ij")
    return torch.stack((xx / 239., yy / 191., .5 + .4 * torch.sin(xx.float() * .31)
                        * torch.cos(yy.float() * .23)), dim=0)[None]


def test_alignment_matches_opencv_geometry_with_interpolation_tolerance():
    cv2 = pytest.importorskip("cv2")
    from insightface.utils.face_align import norm_crop

    image = textured_image()
    landmarks = five_points()
    aligned = align_face_rgb(image, landmarks)
    expected = norm_crop(image[0].permute(1, 2, 0).numpy(),
                         landmark=landmarks[0].numpy(), image_size=112)
    expected = torch.from_numpy(expected).permute(2, 0, 1)[None]
    error = (aligned - expected).abs()
    # cv2 INTER_LINEAR quantizes interpolation fractions to 1/32. PyTorch uses
    # continuous bilinear fractions; forcing a 1e-6 image tolerance is invalid.
    assert float(error.mean()) < .002
    assert float(error.max()) < .01
    assert aligned.shape == (1, 3, 112, 112)


def test_fixed_grid_shared_for_prediction_target_and_gradients():
    landmarks = five_points().requires_grad_(True)
    grid = build_arcface_alignment_grid(landmarks, (192, 240))
    assert not grid.requires_grad
    target = textured_image()
    prediction = (target * .9).clone().requires_grad_(True)
    target_crop = align_face_rgb(target, grid=grid)
    prediction_crop = align_face_rgb(prediction, grid=grid)
    torch.testing.assert_close(target_crop * .9, prediction_crop, rtol=1e-5, atol=1e-6)
    (prediction_crop - target_crop).square().mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert landmarks.grad is None
    # The crop is local, so the unused image corner receives no image gradient.
    assert torch.count_nonzero(prediction.grad[..., :20, :20]) == 0


def test_batched_landmarks_use_separate_alignment_without_channel_swap():
    points = five_points()
    second = points + torch.tensor([10., 4.])
    grid = build_arcface_alignment_grid(torch.cat((points, second)), (192, 240))
    images = textured_image().repeat(2, 1, 1, 1)
    aligned = align_face_rgb(images, grid=grid)
    assert not torch.equal(aligned[0], aligned[1])
    assert float((aligned[1, 0] - aligned[0, 0]).mean()) == pytest.approx(10 / 239, abs=1e-6)
    assert float((aligned[1, 1] - aligned[0, 1]).mean()) == pytest.approx(4 / 191, abs=1e-6)


def test_half_precision_sampling_promotes_and_keeps_gradient():
    image = textured_image().to(torch.bfloat16).requires_grad_(True)
    aligned = align_face_rgb(image, five_points())
    assert aligned.dtype == torch.float32
    aligned.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


@pytest.mark.parametrize("failure", ["nan", "infinity", "outside", "repeated", "collinear", "shape", "empty"])
def test_invalid_five_points_are_explicitly_rejected(failure):
    points = five_points()
    if failure == "nan":
        points[0, 0, 0] = float("nan")
    elif failure == "infinity":
        points[0, 0, 0] = float("inf")
    elif failure == "outside":
        points[0, 0, 0] = -1
    elif failure == "repeated":
        points[0, 1] = points[0, 0]
    elif failure == "collinear":
        points = torch.arange(5).float()[None, :, None].repeat(1, 1, 2) + 20
    elif failure == "shape":
        points = points[:, :4]
    elif failure == "empty":
        points = points[:0]
    with pytest.raises(ValueError):
        build_arcface_alignment_grid(points, (192, 240))


def test_input_contracts_and_grid_are_checked():
    image = textured_image()
    points = five_points()
    grid = build_arcface_alignment_grid(points, (192, 240))
    with pytest.raises(ValueError, match="exactly one"):
        align_face_rgb(image)
    with pytest.raises(ValueError, match="exactly one"):
        align_face_rgb(image, points, grid=grid)
    with pytest.raises(ValueError, match="batch sizes"):
        align_face_rgb(image.repeat(2, 1, 1, 1), points)
    with pytest.raises(ValueError, match="grid must have shape"):
        align_face_rgb(image, grid=grid[:, :100])
    with pytest.raises(ValueError, match="RGB in"):
        align_face_rgb(torch.ones_like(image) * 1.1, points)
    with pytest.raises(ValueError, match="non-finite"):
        align_face_rgb(torch.full_like(image, float("nan")), points)
    with pytest.raises(ValueError, match="multiple"):
        build_arcface_alignment_grid(points, (192, 240), output_size=100)

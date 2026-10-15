import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import pytest

from src.data.letterbox import NativePad, NATIVE_VERSION, valid_body_velocity_loss
from src.pose_control.qwen21_full.face_coordinates import transform_from_record, valid_image_tokens
from src.pose_control.qwen21_full.face_supervision import detection_record
from src.pose_control.qwen21_full.qwen_runtime import transformer_inputs
from src.pose_control.qwen21_full.joint_training import joint_velocity_loss


def test_native_pad_stride_and_dimensions():
    # TikTok native aspect ratio 604x1080
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    assert pad.width == 604
    assert pad.height == 1080
    assert pad.padded_width == 608
    assert pad.padded_height == 1088
    assert pad.left == 0
    assert pad.top == 0
    assert pad.rect == (0, 0, 604, 1080)
    assert pad.grid_shape == (68, 38)
    assert pad.token_count == 68 * 38
    assert pad.token_count % 4 == 0, "token count must be divisible by 4 for Qwen21 VLM slots"


def test_native_pad_image_unpad_preserves_pixels():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    # Create deterministic non-trivial image
    np.random.seed(42)
    orig_array = np.random.randint(0, 256, (1080, 604, 3), dtype=np.uint8)
    orig_img = Image.fromarray(orig_array)

    padded_img = pad.image(orig_img, fill=0)
    assert padded_img.size == (608, 1088)
    padded_array = np.asarray(padded_img)

    # Within valid rect, pixels must be exactly identical
    assert np.array_equal(padded_array[:1080, :604], orig_array)
    # Outside valid rect, pixels must be fill (0)
    assert np.all(padded_array[1080:, :] == 0)
    assert np.all(padded_array[:, 604:] == 0)

    # Unpad should recover exact original image
    unpadded_img = pad.unpad(padded_img)
    assert unpadded_img.size == (604, 1080)
    assert np.array_equal(np.asarray(unpadded_img), orig_array)


def test_native_pad_points_reversibility():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    # Normalized keypoints in [0, 1] relative to (604, 1080)
    pts = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.25, 0.75],
    ], dtype=torch.float32)

    mapped = pad.points(pts)
    # Since left=0, top=0, mapped coordinates must be scaled by width/padded_width and height/padded_height
    assert torch.allclose(mapped[:, 0], pts[:, 0] * (604.0 / 608.0))
    assert torch.allclose(mapped[:, 1], pts[:, 1] * (1080.0 / 1088.0))

    # Unpad points must recover exact original coordinates
    recovered = pad.unpad_points(mapped)
    assert torch.allclose(recovered, pts, atol=1e-6)


def test_native_pad_valid_mask():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    mask = pad.valid_mask()
    assert mask.shape == (1, 1088, 608)
    assert (mask[:, :1080, :604] == 1.0).all()
    assert (mask[:, 1080:, :] == 0.0).all()
    assert (mask[:, :, 604:] == 0.0).all()


def test_native_pad_metadata_roundtrip():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    meta = pad.metadata()
    assert meta['version'] == NATIVE_VERSION
    assert meta['width'] == 604
    assert meta['height'] == 1080
    assert meta['padded_width'] == 608
    assert meta['padded_height'] == 1088

    restored = NativePad.from_record(meta)
    assert restored == pad
    # Also verify transform_from_record accepts it
    tf = transform_from_record(meta)
    assert tf == pad


def test_valid_image_tokens_rectangular():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    mask = pad.valid_mask().unsqueeze(0)  # [1, 1, 1088, 608]
    batch = {'valid_image_mask': mask}

    tokens = valid_image_tokens(batch, pad.grid_shape, device='cpu')
    assert tokens is not None
    assert tokens.shape == (1, pad.token_count, 1)

    # In token space, the bottom row (row index 67) and right column (col index 37) contain pad regions
    tokens_2d = tokens.view(68, 38)
    # Interior tokens are fully valid (1.0)
    assert (tokens_2d[:67, :37] == 1.0).all()
    # Bottom-right token overlaps pad region, so coverage is < 1.0
    assert float(tokens_2d[67, 37]) < 1.0


def test_transformer_inputs_rectangular_shapes():
    pair = {
        'source_latent': torch.zeros(1, 2584, 64),
        'prompt_embeds': torch.zeros(1, 10, 4096),
        'prompt_mask': torch.ones(1, 10, dtype=torch.bool),
        'img_mask': torch.ones(1, 656, dtype=torch.bool),
        'resolution': 'native',
        'target_token_hw': (68, 38),
        'source_token_hw': (68, 38),
    }
    noisy_target = torch.randn(1, 2584, 64)
    sigma = torch.tensor([0.5])

    inputs = transformer_inputs(pair, noisy_target, sigma)
    assert inputs['hidden_states'].shape == (1, 2584 * 2, 64)
    assert inputs['img_shapes'] == [[(1, 68, 38), (1, 68, 38)]]


def test_pad_tokens_zero_gradient_in_loss():
    # Construct a 4x4 token grid where bottom row is pad (valid = 0)
    B, H, W, D = 1, 4, 4, 32
    valid = torch.ones(B, H, W, 1)
    valid[:, -1, :, :] = 0.0  # last row is completely pad
    valid_tokens = valid.flatten(1, 2)

    human = torch.ones(B, H * W, 1) * 0.5
    face_mask = torch.zeros(B, H * W, 1)
    face_mask[:, 0, :] = 1.0  # first token is face

    # Prediction with requires_grad
    prediction = torch.randn(B, H * W, D, requires_grad=True)
    velocity = torch.zeros(B, H * W, D)

    squared = (prediction - velocity).square()
    loss, parts = joint_velocity_loss(squared, human, valid_tokens, face_mask, face_active=True)

    loss.backward()

    grad = prediction.grad.view(B, H, W, D)
    # Tokens in valid region must have nonzero gradient
    assert float(grad[:, :3, :, :].abs().sum()) > 0.0
    # Tokens in completely padded region must have EXACTLY ZERO gradient
    assert float(grad[:, -1, :, :].abs().sum()) == 0.0


def test_detection_record_native():
    record = detection_record(
        box=[100, 100, 200, 250, 0.95],
        keypoints=None,
        geometry_box=[90, 90, 210, 260],
        image_hw=(1080, 604),
        score_threshold=0.5,
        iou_threshold=0.2,
        align_mode='native',
    )
    assert record['score'] == 0.95
    assert record['geometric_match'] is True
    assert 'native' in record['bbox_output_px']
    assert 'native' in record['short_side_output_px']
    assert record['short_side_output_px']['native'] == pytest.approx(100.0)
    assert record['bbox_output_px']['native'] is not None


def test_native_pad_odd_multiple_of_16_pads_to_32():
    # 1072 is 16 * 67 (odd multiple of 16).
    # If padded only to 16, height would remain 1072.
    # But QwenImage21Pipeline._unpack_latents requires height = 2 * (1072 // 32) = 66,
    # which implies 66 * 16 = 1056, causing latent reshaping mismatch.
    # NativePad now ensures padded_height is a multiple of 32 (1088).
    pad = NativePad.from_size((608, 1072), stride=16, pad_mode='edge')
    assert pad.padded_height % 32 == 0
    assert pad.padded_height == 1088
    assert pad.padded_width % 32 == 0
    assert pad.padded_width == 608
    assert pad.token_count % 4 == 0
    assert pad.grid_shape == (1088 // 16, 608 // 16)


def test_native_pad_unpad_torch_tensor():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    # 3D tensor: [C, H, W] = [3, 1088, 608]
    t3 = torch.randn(3, 1088, 608)
    unpadded3 = pad.unpad(t3)
    assert unpadded3.shape == (3, 1080, 604)
    assert torch.equal(unpadded3, t3[:, :1080, :604])

    # 4D tensor: [B, C, H, W] = [2, 3, 1088, 608]
    t4 = torch.randn(2, 3, 1088, 608)
    unpadded4 = pad.unpad(t4)
    assert unpadded4.shape == (2, 3, 1080, 604)
    assert torch.equal(unpadded4, t4[:, :, :1080, :604])

    # 2D tensor: [H, W] = [1088, 608]
    t2 = torch.randn(1088, 608)
    unpadded2 = pad.unpad(t2)
    assert unpadded2.shape == (1080, 604)
    assert torch.equal(unpadded2, t2[:1080, :604])


def test_native_pad_numpy_points_and_unpad():
    pad = NativePad.from_size((604, 1080), stride=16, pad_mode='edge')
    pts_np = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.25, 0.75],
    ], dtype=np.float32)

    mapped_np = pad.points(pts_np)
    assert isinstance(mapped_np, np.ndarray)
    assert mapped_np.shape == (4, 2)
    assert np.allclose(mapped_np[:, 0], pts_np[:, 0] * (604.0 / 608.0))
    assert np.allclose(mapped_np[:, 1], pts_np[:, 1] * (1080.0 / 1088.0))

    unpadded_np = pad.unpad_points(mapped_np)
    assert isinstance(unpadded_np, np.ndarray)
    assert np.allclose(unpadded_np, pts_np, atol=1e-6)


def test_pose_metric_compare_scaled_native_diagonal():
    from src.evaluation.joint_pilot_metrics import PoseMetric
    # 25 body keypoints in normalized coordinates
    target = np.zeros((25, 3), dtype=np.float32)
    target[:, 2] = 1.0  # all visible
    target[:, :2] = 0.5  # centered

    # detected points in pixel coordinates: 17 COCO joints
    # For native size (604, 1080), center is (302, 540)
    native_pts = np.full((17, 2), [302.0, 540.0], dtype=np.float32)
    # Shift joint 0 by 50 pixels
    native_pts[0, 0] += 50.0
    scores = np.ones(17, dtype=np.float32)

    detected = (native_pts, scores)
    res_native = PoseMetric.compare(detected, target, image_size=(604, 1080))
    res_default = PoseMetric.compare(detected, target, image_size=(512, 512))

    diag_native = math.hypot(604.0, 1080.0)
    expected_nme_native = (50.0 / diag_native) / 17.0

    assert abs(res_native['pose_nme_diagonal'] - expected_nme_native) < 1e-4
    assert res_native['pose_nme_diagonal'] != res_default['pose_nme_diagonal']


def test_prepare_fallback_token_hw(monkeypatch):
    from unittest.mock import MagicMock
    from src.pose_control.qwen21_full.execution import prepare

    monkeypatch.setattr('src.pose_control.qwen21_full.execution.build_condition_bundle', lambda *args, **kwargs: MagicMock())

    mock_adapter = MagicMock()
    mock_adapter.parameters.return_value = iter([torch.tensor([0.0])])
    mock_adapter.legacy = MagicMock()
    mock_adapter.prepare_conditioning.return_value = "prepared"

    batch = {
        'tgt_image': torch.zeros(1, 3, 1088, 608),
        'src_image': torch.zeros(1, 3, 1088, 608),
    }
    pair = {
        'resolution': 'native',
        'source_latent': torch.zeros(1, 10, 64),
    }
    # No target_token_hw or source_token_hw in pair, with_face=False
    result = prepare(mock_adapter, batch, pair, [], None, with_face=False)
    assert result == "prepared"
    _, kwargs = mock_adapter.prepare_conditioning.call_args
    assert kwargs['target_token_hw'] == (1088 // 16, 608 // 16)
    assert kwargs['source_token_hw'] == (1088 // 16, 608 // 16)


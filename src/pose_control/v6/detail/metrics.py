import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn

def compute_psnr(img1_np, img2_np, mask=None):
    """img1_np, img2_np in [0, 255] float32. mask is binary [H, W] or [H, W, 1]."""
    diff_sq = np.square(img1_np - img2_np)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, :, None]
        m_sum = np.sum(mask) * img1_np.shape[2]
        if m_sum == 0:
            return float("inf")
        mse = float(np.sum(diff_sq * mask) / m_sum)
    else:
        mse = float(np.mean(diff_sq))
    if mse <= 1e-10:
        return 99.0
    return float(-10.0 * np.log10(mse / (255.0 ** 2)))


def compute_ssim(img1_np, img2_np, mask=None):
    """Compute SSIM. If crop box is given or whole image."""
    # img1_np, img2_np uint8 or float32 in [0, 255]
    if img1_np.shape[0] < 7 or img1_np.shape[1] < 7:
        return 1.0
    win_size = min(7, min(img1_np.shape[0], img1_np.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    val = ssim_fn(img1_np, img2_np, channel_axis=2, data_range=255.0, win_size=win_size)
    return float(val)


def crop_roi(img, box_norm):
    """Crop image using normalized [x1, y1, x2, y2] box."""
    h, w = img.shape[:2]
    x1 = max(0, min(int(round(box_norm[0] * w)), w - 1))
    y1 = max(0, min(int(round(box_norm[1] * h)), h - 1))
    x2 = max(x1 + 1, min(int(round(box_norm[2] * w)), w))
    y2 = max(y1 + 1, min(int(round(box_norm[3] * h)), h))
    return img[y1:y2, x1:x2]


def evaluate_keypoint_metrics(det_kps, det_scores, gt_kps, gt_scores, gt_boxes):
    """
    Evaluates:
    - Face landmark NME
    - Left-hand PCK@0.1
    - Right-hand PCK@0.1
    """
    # det_kps: [134, 2] in pixel coords or normalized
    # gt_kps: [134, 2] in same scale
    # gt_boxes: face_box, lh_box, rh_box in pixel coords [x1, y1, x2, y2]
    results = {
        "face_nme": None,
        "face_detected": True,
        "lh_pck": 0.0,
        "lh_detected": True,
        "rh_pck": 0.0,
        "rh_detected": True,
    }
    if det_kps is None or len(det_kps) == 0:
        results["face_detected"] = False
        results["lh_detected"] = False
        results["rh_detected"] = False
        return results

    # 1. Face NME
    face_det = det_kps[24:92]
    face_gt = gt_kps[24:92]
    face_det_sc = det_scores[24:92]
    face_gt_sc = gt_scores[24:92]

    # Check if face was detected
    if np.mean(face_det_sc) < 0.15:
        results["face_detected"] = False
    else:
        # Face box scale as normalization factor
        fbox = gt_boxes[0]
        bbox_scale = max(fbox[2] - fbox[0], fbox[3] - fbox[1], 1.0)
        valid_face = face_gt_sc >= 0.30
        if np.any(valid_face):
            diffs = np.linalg.norm(face_det[valid_face] - face_gt[valid_face], axis=-1)
            results["face_nme"] = float(np.mean(diffs) / bbox_scale)

    # 2. Hand PCK@0.1
    for hand_idx, (name, s_idx, e_idx, box_idx) in enumerate([
        ("lh", 92, 113, 1),
        ("rh", 113, 134, 2),
    ]):
        h_det = det_kps[s_idx:e_idx]
        h_gt = gt_kps[s_idx:e_idx]
        h_det_sc = det_scores[s_idx:e_idx]
        h_gt_sc = gt_scores[s_idx:e_idx]

        if np.mean(h_det_sc) < 0.15:
            results[f"{name}_detected"] = False
            results[f"{name}_pck"] = 0.0
        else:
            hbox = gt_boxes[box_idx]
            hbox_scale = max(hbox[2] - hbox[0], hbox[3] - hbox[1], 1.0)
            threshold = 0.10 * hbox_scale
            valid_hand = h_gt_sc >= 0.30
            if np.any(valid_hand):
                diffs = np.linalg.norm(h_det[valid_hand] - h_gt[valid_hand], axis=-1)
                correct = np.sum(diffs <= threshold)
                results[f"{name}_pck"] = float(correct / np.sum(valid_hand))

    results["hands_avg_pck"] = float((results["lh_pck"] + results["rh_pck"]) * 0.5)
    return results

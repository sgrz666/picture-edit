"""Per-person appearance similarity: did each source person stay that same person?

DINOv2 features are an *appearance* proxy, not biometric identity. A regenerated or occluded
face lowers the score without proving the person was swapped, so the metric is reported
together with the best-similarity assignment: when the appearance-optimal pairing disagrees
with the pose-optimal pairing, that is the identity-swap signal, and when it agrees but the
score is low, that is appearance drift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MODEL_NAME = 'dinov2_vits14'
MIN_CROP_PIXELS = 32
# Cosine similarity per person that a better pairing must beat the pose pairing by. A total
# difference of 0.02 across two people is float noise, not evidence of a swap.
SWAP_MARGIN_PER_PERSON = 0.02


def crop_person(image: Image.Image, box, pad_ratio: float = 0.02) -> Image.Image | None:
    """Crop one person box with a small margin, clamped to the image."""
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    if not all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        return None
    pad_x, pad_y = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
    left, top = max(0, int(np.floor(x0 - pad_x))), max(0, int(np.floor(y0 - pad_y)))
    right, bottom = min(width, int(np.ceil(x1 + pad_x))), min(height, int(np.ceil(y1 + pad_y)))
    if right - left < 2 or bottom - top < 2:
        return None
    return image.crop((left, top, right, bottom))


def load_embedder(repo: Path, device: str = 'cuda'):
    """Load the local DINOv2 backbone, or return None with a reason when it is unavailable."""
    repo = Path(repo)
    if not (repo / 'hubconf.py').is_file():
        return None, f'no local DINOv2 repository at {repo}'
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        return None, f'torch unavailable: {error}'
    try:
        model = torch.hub.load(str(repo), MODEL_NAME, source='local', pretrained=True)
    except Exception as error:  # torch.hub raises many types depending on the failure
        return None, f'DINOv2 failed to load from {repo}: {type(error).__name__}: {error}'
    model = model.eval().requires_grad_(False).to(device)

    def embed(image: Image.Image) -> np.ndarray:
        import torch
        array = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC),
            dtype=np.float32) / 255.0
        value = torch.from_numpy(array).permute(2, 0, 1)[None]
        value = (value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) \
            / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.inference_mode():
            feature = model(value.to(device))[0].float().cpu().numpy()
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm else feature

    return embed, None


def appearance_matrix(source_features, generated_features) -> np.ndarray:
    """Cosine similarity between each source person and each generated person."""
    if not source_features or not generated_features:
        return np.zeros((len(source_features), len(generated_features)), dtype=float)
    source = np.stack(source_features)
    generated = np.stack(generated_features)
    return np.clip(source @ generated.T, -1.0, 1.0)


def best_assignment(similarity: np.ndarray):
    """The pairing that maximises total similarity, for two people only."""
    rows, columns = similarity.shape
    if rows != columns:
        return None, None
    if rows == 1:
        return [0], float(similarity[0, 0])
    direct = float(similarity[0, 0] + similarity[1, 1])
    swapped = float(similarity[0, 1] + similarity[1, 0])
    return ([0, 1] if direct >= swapped else [1, 0]), max(direct, swapped)


def judge_appearance(similarity: np.ndarray, pose_assignment) -> dict:
    """Compare the pose-based pairing with the appearance-optimal pairing.

    ``pose_assignment[i]`` is which generated person corresponds to source person i by body
    position. Disagreement between the two pairings is the identity-swap signal.
    """
    if similarity.size == 0 or not np.all(np.isfinite(similarity)):
        return {'available': False, 'reason': 'no usable appearance features'}
    rows, columns = similarity.shape
    if rows != columns:
        return {'available': False, 'reason': f'{rows} source people versus {columns} generated',
            'source_people': rows, 'generated_people': columns}
    pairs = best_assignment(similarity)
    optimal, total = pairs if pairs[0] is not None else (None, None)
    following_pose = [similarity[index, int(slot)] for index, slot in enumerate(pose_assignment)]
    margin_per_person = (total - float(sum(following_pose))) / rows if optimal is not None else None
    swapped = (optimal is not None and list(optimal) != list(pose_assignment)
        and margin_per_person > SWAP_MARGIN_PER_PERSON)
    return {'available': True, 'similarity_matrix': [[float(value) for value in row]
            for row in similarity],
        'identity_similarity': [float(value) for value in following_pose],
        'identity_mean': float(np.mean(following_pose)),
        'appearance_optimal_assignment': list(optimal) if optimal else None,
        'assignment_matches_pose': list(optimal) == list(pose_assignment) if optimal else None,
        'identity_swap_suspected': bool(swapped),
        'swap_margin_per_person': margin_per_person,
        'swap_margin_threshold': SWAP_MARGIN_PER_PERSON}

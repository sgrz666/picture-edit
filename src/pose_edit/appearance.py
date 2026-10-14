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


def crop_person(image: Image.Image, box, pad_ratio: float = 0.02,
    shape: str = 'tight') -> Image.Image | None:
    """Crop one person box with a small margin, clamped to the image.

    ``shape='tight'`` keeps the detector's own aspect ratio and is what every stored evaluation in
    both pipelines used. ``shape='square'`` grows the shorter side to match the longer one, which
    is the comparison the two-person import metric needs: the reference crops come from Hi4D
    silhouette mask boxes and the generated crops from detector boxes, so a tight crop puts a
    different amount of background around the two sides and can move a cosine similarity without
    any change in appearance. A square that no longer fits (a body against the image border) is
    shrunk to what fits rather than padded with invented pixels, so the caller can still tell that
    this crop lost area.
    """
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    if not all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        return None
    pad_x, pad_y = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
    left, top = max(0, int(np.floor(x0 - pad_x))), max(0, int(np.floor(y0 - pad_y)))
    right, bottom = min(width, int(np.ceil(x1 + pad_x))), min(height, int(np.ceil(y1 + pad_y)))
    if right - left < 2 or bottom - top < 2:
        return None
    if shape == 'square':
        centre_x, centre_y = (left + right) / 2.0, (top + bottom) / 2.0
        side = max(right - left, bottom - top)
        # One half-length for both axes keeps the result square; capping it by the free room on each
        # side keeps it inside the image. Near a border the square shrinks, which is recorded by
        # returning the smaller crop instead of padding with pixels that were never there.
        half = min(side / 2.0, centre_x, width - centre_x, centre_y, height - centre_y)
        left, right = int(round(centre_x - half)), int(round(centre_x + half))
        top, bottom = int(round(centre_y - half)), int(round(centre_y + half))
    elif shape != 'tight':
        raise ValueError(f"unknown crop shape {shape!r}; use 'tight' or 'square'")
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


def mask_box(mask) -> list | None:
    """Tight xyxy box around the nonzero part of a person mask, or None if the mask is empty.

    Lives here because both the identity metric and the framing audit need the same silhouette
    box; a second copy in a caller is how the two would drift apart.
    """
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(columns):
        return None
    return [float(columns[0]), float(rows[0]), float(columns[-1] + 1), float(rows[-1] + 1)]


def judge_import(embed, reference_crops, target_crops, generated_crops) -> dict:
    """Is an extra body in the result actually one of the target photograph's people?

    This is only informative when the source and the target show *different* people: in a
    same-identity pair A and B are the same actors in the same clothes, so "looks like B" and
    "looks like A" are the same statement and the caller must say so rather than report a number.
    """
    if not reference_crops or not target_crops or not generated_crops:
        return {'available': False, 'reason': 'need source, target and generated crops'}
    reference = [embed(crop) for crop in reference_crops]
    target = [embed(crop) for crop in target_crops]
    bodies = []
    for crop in generated_crops:
        feature = embed(crop)
        against_reference = [float(np.dot(feature, one)) for one in reference]
        against_target = [float(np.dot(feature, one)) for one in target]
        best_reference = max(against_reference)
        best_target = max(against_target)
        bodies.append({'best_source_similarity': best_reference,
            'best_target_similarity': best_target,
            'target_minus_source': best_target - best_reference,
            'looks_more_like_target': best_target - best_reference > SWAP_MARGIN_PER_PERSON})
    return {'available': True, 'bodies': bodies,
        'bodies_liking_target_more': sum(1 for body in bodies if body['looks_more_like_target']),
        'margin_per_body': SWAP_MARGIN_PER_PERSON,
        'basis': 'DINOv2 cosine similarity of each generated body against the source people and '
            'the target people, best match each way'}


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

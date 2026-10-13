"""Turn a multi-HMR fit into the DWPose-134 body block the two-person guides are drawn from.

Shared by ``fit_dual_report.py`` (which measures the fit against the ground truth) and
``dual_geometry.py`` (which can draw the guides from the fit instead of the Hi4D landmarks), so the
slot mapping lives in exactly one place: the report and the guide cannot disagree about which
joint a slot means because they read the same table.

Only the 13 body joints a mesh actually has are mapped. The nose, eyes and ears that DWPose also
reports stay NaN, so a joint the estimator cannot supply is never drawn and never scored.
"""
from __future__ import annotations

import numpy as np

# One table serves SMPL-24 (the Hi4D ground truth) and the SMPL-X 55-joint order the fit reports,
# because every name here exists in both.
DWPOSE_TO_JOINT = {1: 'neck', 2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist', 8: 'right_hip', 9: 'right_knee',
    10: 'right_ankle', 11: 'left_hip', 12: 'left_knee', 13: 'left_ankle'}
HIP_SLOTS = (8, 11)
SMPL24_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar',
    'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hand', 'right_hand']


def joint_indices(names, table=DWPOSE_TO_JOINT):
    lookup = {str(name): index for index, name in enumerate(names)}
    missing = sorted({name for name in table.values() if name not in lookup})
    if missing:
        raise KeyError(f'skeleton has no {missing}')
    return {slot: lookup[name] for slot, name in table.items()}


def hip_centre(points):
    hips = points[list(HIP_SLOTS)]
    centre = np.nanmean(hips, axis=0)
    return centre if np.isfinite(centre).all() else None


def person_from_joints(joints2d, names, size):
    """Put one skeleton's 2D joints into the DWPose-134 body block, in-box points only."""
    indices = joint_indices(names)
    points = np.full((134, 2), np.nan)
    for slot, index in indices.items():
        points[slot] = joints2d[index]
    usable = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < size[0])
        & (points[:, 1] >= 0) & (points[:, 1] < size[1]))
    return {'points': np.where(usable[:, None], points, np.nan),
        'scores': np.where(usable, 1.0, 0.0)}


def load_fit(npz_path, size, persons_required=2):
    """Per-person landmark sets from a fit worker npz, ordered left to right by hip centre.

    The guide numbers people by horizontal position, exactly the way the test set built its slots
    (``pair.json`` ``B.slot_order`` is hip-x order), so a fitted pair maps onto the same slots the
    ground-truth path would have produced.
    """
    with np.load(npz_path, allow_pickle=True) as arrays:
        names = [str(value) for value in arrays['joint_names']]
        j2d = np.asarray(arrays['j2d_native'], dtype=float)
    if len(j2d) != persons_required:
        raise ValueError(f'{npz_path} fitted {len(j2d)} people but the guide needs '
            f'{persons_required}; refusing to guess which body to drop')
    people = [person_from_joints(row, names, size) for row in j2d]
    centres = [hip_centre(person['points']) for person in people]
    if any(centre is None for centre in centres):
        raise ValueError(f'{npz_path} has a person without both hips, so it cannot be ordered')
    order = sorted(range(len(people)), key=lambda index: centres[index][0])
    return [people[index] for index in order], {'joint_order': names,
        'fit_person_count': int(len(people)), 'slot_order_by_hip_x': order,
        'source_npz': str(npz_path)}

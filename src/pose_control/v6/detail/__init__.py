"""Face/hand detail conditioning contracts and local trainable encoders."""

from .binder import DetailTokenBinder
from .conditions import (
    DetailReferenceBatch,
    DetailRegion,
    FaceHandDetailCondition,
    PreparedFaceHandDetailConditioning,
)
from .encoders import (
    DetailAppearanceEncoder,
    DetailSpatialEncoder,
    ReferenceFeatureProvider,
    canonicalize_left_hand_keypoints,
    canonicalize_left_hand_pose,
    restore_left_hand_keypoints,
    restore_left_hand_pose,
)
from .preparer import FaceHandDetailPreparer, soft_box_masks

__all__ = [
    "DetailAppearanceEncoder",
    "DetailReferenceBatch",
    "DetailRegion",
    "DetailSpatialEncoder",
    "DetailTokenBinder",
    "FaceHandDetailCondition",
    "FaceHandDetailPreparer",
    "PreparedFaceHandDetailConditioning",
    "ReferenceFeatureProvider",
    "canonicalize_left_hand_keypoints",
    "canonicalize_left_hand_pose",
    "restore_left_hand_keypoints",
    "restore_left_hand_pose",
    "soft_box_masks",
]

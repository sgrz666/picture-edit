"""V6.5 face-conditioning contracts and encoders."""

from .conditions import (
    FaceFineCondition,
    FaceReferenceFeatures,
    PreparedFaceConditioning,
)
from .content_encoder import (
    FaceContentEncoder,
    FacePerceiver,
    pool_arcface_references,
)
from .geometry import (
    FaceGeometryTokenizer,
    dwpose68_to_face72,
    normalize_face_landmarks_to_roi,
    smplx137_to_face72,
)
from .preparer import (
    FaceConditioningPreparer,
    FaceConditionPreparer,
    FacePreparer,
)
from .resampler import FeedForward, PerceiverAttention, Resampler
from .spatial import (
    FaceSpatialEncoder,
    build_face_heatmaps_and_mask,
    face_landmarks_to_heatmaps,
    render_face_landmark_heatmaps,
)

__all__ = [
    "FaceFineCondition",
    "FaceReferenceFeatures",
    "PreparedFaceConditioning",
    "FaceGeometryTokenizer",
    "smplx137_to_face72",
    "dwpose68_to_face72",
    "normalize_face_landmarks_to_roi",
    "PerceiverAttention",
    "FeedForward",
    "Resampler",
    "pool_arcface_references",
    "FacePerceiver",
    "FaceContentEncoder",
    "render_face_landmark_heatmaps",
    "face_landmarks_to_heatmaps",
    "build_face_heatmaps_and_mask",
    "FaceSpatialEncoder",
    "FaceConditioningPreparer",
    "FaceConditionPreparer",
    "FacePreparer",
]

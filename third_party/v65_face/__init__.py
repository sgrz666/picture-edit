"""Pinned, locally adapted building blocks used by the V6.5 face path.

See :mod:`third_party.v65_face.UPSTREAM_SOURCES.md` for exact source commits,
license status, and the interface changes made by this project.
"""

from .mcld import FacePoseGuider2D, safe_square_crop_normalized
from .stableanimator import FacePerceiver, FusionFaceId, InsightFaceArcFaceExtractor
from .visual_persona import (
    FeedForward,
    IPAttnProcessor2_0,
    PerceiverAttention,
    Resampler,
)
from .xdyna import add_optional_face_residual, repeat_condition_batch

__all__ = [
    "FacePoseGuider2D",
    "safe_square_crop_normalized",
    "FacePerceiver",
    "FusionFaceId",
    "InsightFaceArcFaceExtractor",
    "FeedForward",
    "IPAttnProcessor2_0",
    "PerceiverAttention",
    "Resampler",
    "add_optional_face_residual",
    "repeat_condition_batch",
]

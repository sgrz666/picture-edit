"""Unified single/dual-person SMPL-X Adapter V6."""

from .condition_injector import SMPLXConditionInjector
from .controlled_pipeline import ControlledDeepGenPipeline
from .conditions import (
    AdapterIdentityCondition,
    ConditionBundle,
    ContactRelationBatch,
    TaskType,
)
from .deepgen_adapter import UnifiedSMPLXAdapterV6
from .checkpoint import (
    ARCHITECTURE_VERSION,
    BRANCH_NAMES,
    DETAIL_SCHEMA_VERSION,
    FACE_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    build_v64_checkpoint,
    build_v65_checkpoint,
    freeze_for_detail_training,
    freeze_for_face_training,
    load_v63_checkpoint,
    load_v64_checkpoint,
    load_v65_checkpoint,
)
from .detail import HandDetailCondition
from .face import FaceFineCondition, FaceReferenceFeatures, PreparedFaceConditioning
from .interface import (
    BranchControlResiduals,
    DeepGenControlInterface,
    DeepGenControlOutput,
    PreparedControlConditioning,
    StrengthScheduleConfig,
)
from .reasoning import AdapterReasoningConfig, InternalControlState, SMPLXAdapterReasoner

__all__ = [
    "AdapterIdentityCondition",
    "AdapterReasoningConfig",
    "BranchControlResiduals",
    "ConditionBundle",
    "ContactRelationBatch",
    "ControlledDeepGenPipeline",
    "DeepGenControlInterface",
    "DeepGenControlOutput",
    "SMPLXConditionInjector",
    "SMPLXAdapterReasoner",
    "TaskType",
    "PreparedControlConditioning",
    "StrengthScheduleConfig",
    "UnifiedSMPLXAdapterV6",
    "InternalControlState",
    "DETAIL_SCHEMA_VERSION",
    "FACE_SCHEMA_VERSION",
    "ARCHITECTURE_VERSION",
    "BRANCH_NAMES",
    "PREPROCESSING_SCHEMA_VERSION",
    "build_v64_checkpoint",
    "build_v65_checkpoint",
    "freeze_for_detail_training",
    "freeze_for_face_training",
    "load_v63_checkpoint",
    "load_v64_checkpoint",
    "load_v65_checkpoint",
    "FaceFineCondition",
    "FaceReferenceFeatures",
    "PreparedFaceConditioning",
    "HandDetailCondition",
]

"""Compatibility exports, loaded on demand; Native imports never construct DeepGen."""
from importlib import import_module

_EXPORTS = {'SMPLXConditionInjector': ('condition_injector', 'SMPLXConditionInjector'),
 'ControlledDeepGenPipeline': ('controlled_pipeline',
                               'ControlledDeepGenPipeline'),
 'AdapterIdentityCondition': ('conditions', 'AdapterIdentityCondition'),
 'ConditionBundle': ('conditions', 'ConditionBundle'),
 'ContactRelationBatch': ('conditions', 'ContactRelationBatch'),
 'TaskType': ('conditions', 'TaskType'),
 'UnifiedSMPLXAdapterV6': ('deepgen_adapter', 'UnifiedSMPLXAdapterV6'),
 'ARCHITECTURE_VERSION': ('checkpoint', 'ARCHITECTURE_VERSION'),
 'BRANCH_NAMES': ('checkpoint', 'BRANCH_NAMES'),
 'DETAIL_SCHEMA_VERSION': ('checkpoint', 'DETAIL_SCHEMA_VERSION'),
 'FACE_SCHEMA_VERSION': ('checkpoint', 'FACE_SCHEMA_VERSION'),
 'HAND_SCHEMA_VERSION': ('checkpoint', 'HAND_SCHEMA_VERSION'),
 'PREPROCESSING_SCHEMA_VERSION': ('checkpoint', 'PREPROCESSING_SCHEMA_VERSION'),
 'build_v64_checkpoint': ('checkpoint', 'build_v64_checkpoint'),
 'build_v65_checkpoint': ('checkpoint', 'build_v65_checkpoint'),
 'build_v66_checkpoint': ('checkpoint', 'build_v66_checkpoint'),
 'freeze_for_detail_training': ('checkpoint', 'freeze_for_detail_training'),
 'freeze_for_face_training': ('checkpoint', 'freeze_for_face_training'),
 'freeze_for_hand_training': ('checkpoint', 'freeze_for_hand_training'),
 'load_v63_checkpoint': ('checkpoint', 'load_v63_checkpoint'),
 'load_v64_checkpoint': ('checkpoint', 'load_v64_checkpoint'),
 'load_v65_checkpoint': ('checkpoint', 'load_v65_checkpoint'),
 'load_v66_checkpoint': ('checkpoint', 'load_v66_checkpoint'),
 'HandDetailCondition': ('detail', 'HandDetailCondition'),
 'FaceFineCondition': ('face', 'FaceFineCondition'),
 'FaceReferenceFeatures': ('face', 'FaceReferenceFeatures'),
 'PreparedFaceConditioning': ('face', 'PreparedFaceConditioning'),
 'HandFineCondition': ('hand', 'HandFineCondition'),
 'HandReferenceFeatures': ('hand', 'HandReferenceFeatures'),
 'PreparedHandConditioning': ('hand', 'PreparedHandConditioning'),
 'BranchControlResiduals': ('interface', 'BranchControlResiduals'),
 'DeepGenControlInterface': ('interface', 'DeepGenControlInterface'),
 'DeepGenControlOutput': ('interface', 'DeepGenControlOutput'),
 'PreparedControlConditioning': ('interface', 'PreparedControlConditioning'),
 'StrengthScheduleConfig': ('interface', 'StrengthScheduleConfig'),
 'AdapterReasoningConfig': ('reasoning', 'AdapterReasoningConfig'),
 'InternalControlState': ('reasoning', 'InternalControlState'),
 'SMPLXAdapterReasoner': ('reasoning', 'SMPLXAdapterReasoner'),
 'NativeFullRuntimeV4': ('native_full_runtime', 'NativeFullRuntimeV4'),
 'NativeLocalConditionV4': ('native_local_condition', 'NativeLocalConditionV4'),
 'NativeLocalReferencesV4': ('native_local_condition',
                             'NativeLocalReferencesV4'),
 'PreparedNativeFullControlV4': ('native_local_condition',
                                 'PreparedNativeFullControlV4'),
 'install_native_full_v4': ('native_full_integration',
                            'install_native_full_v4')}
__all__ = ['AdapterIdentityCondition', 'AdapterReasoningConfig', 'BranchControlResiduals', 'ConditionBundle', 'ContactRelationBatch', 'ControlledDeepGenPipeline', 'DeepGenControlInterface', 'DeepGenControlOutput', 'SMPLXConditionInjector', 'SMPLXAdapterReasoner', 'TaskType', 'PreparedControlConditioning', 'StrengthScheduleConfig', 'UnifiedSMPLXAdapterV6', 'InternalControlState', 'DETAIL_SCHEMA_VERSION', 'FACE_SCHEMA_VERSION', 'HAND_SCHEMA_VERSION', 'ARCHITECTURE_VERSION', 'BRANCH_NAMES', 'PREPROCESSING_SCHEMA_VERSION', 'build_v64_checkpoint', 'build_v65_checkpoint', 'build_v66_checkpoint', 'freeze_for_detail_training', 'freeze_for_face_training', 'freeze_for_hand_training', 'load_v63_checkpoint', 'load_v64_checkpoint', 'load_v65_checkpoint', 'load_v66_checkpoint', 'FaceFineCondition', 'FaceReferenceFeatures', 'PreparedFaceConditioning', 'HandDetailCondition', 'HandFineCondition', 'HandReferenceFeatures', 'PreparedHandConditioning', 'NativeFullRuntimeV4', 'NativeLocalConditionV4', 'NativeLocalReferencesV4', 'PreparedNativeFullControlV4', 'install_native_full_v4']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module('.' + module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

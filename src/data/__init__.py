"""Lazy data compatibility exports; native Canvas imports do not load legacy models."""
from importlib import import_module

_EXPORTS = {'CanvasPolicyError': ('canvas_contract', 'CanvasPolicyError'),
 'CanvasSpec': ('canvas_contract', 'CanvasSpec'),
 'LegacyStretch': ('canvas_contract', 'LegacyStretch'),
 'NonUniformScaleError': ('canvas_contract', 'NonUniformScaleError'),
 'aspect_preserving': ('canvas_contract', 'aspect_preserving'),
 'canvas_for_mode': ('canvas_contract', 'canvas_for_mode'),
 'letterbox': ('canvas_contract', 'letterbox'),
 'native_pad': ('canvas_contract', 'native_pad'),
 'require_canonical': ('canvas_contract', 'require_canonical'),
 'PairContractError': ('geometry_pair_dataset_v2', 'PairContractError'),
 'assert_pair_contract': ('geometry_pair_dataset_v2', 'assert_pair_contract'),
 'bucket_by_canvas': ('geometry_pair_dataset_v2', 'bucket_by_canvas'),
 'collate_geometry_pairs': ('geometry_pair_dataset_v2',
                            'collate_geometry_pairs'),
 'IPERPoseDataset': ('iper_dataset', 'IPERPoseDataset'),
 'TikTokPoseDataset': ('tiktok_dataset', 'TikTokPoseDataset'),
 'TikTokV3PoseDataset': ('tiktok_dataset', 'TikTokV3PoseDataset'),
 'TikTokV65FaceOverfitLoader': ('tiktok_dataset', 'TikTokV65FaceOverfitLoader'),
 'TikTokV65OverfitInputs': ('tiktok_dataset', 'TikTokV65OverfitInputs'),
 'is_tiktok_v3_manifest': ('tiktok_dataset', 'is_tiktok_v3_manifest'),
 'load_pose_dataset': ('tiktok_dataset', 'load_pose_dataset')}
__all__ = ['IPERPoseDataset', 'CanvasSpec', 'CanvasPolicyError', 'NonUniformScaleError', 'LegacyStretch', 'canvas_for_mode', 'require_canonical', 'native_pad', 'letterbox', 'aspect_preserving', 'PairContractError', 'assert_pair_contract', 'collate_geometry_pairs', 'bucket_by_canvas', 'TikTokPoseDataset', 'TikTokV3PoseDataset', 'TikTokV65FaceOverfitLoader', 'TikTokV65OverfitInputs', 'is_tiktok_v3_manifest', 'load_pose_dataset']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module('.' + module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

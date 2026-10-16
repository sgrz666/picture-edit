"""Data modules for picture edit."""

from importlib import import_module

__all__ = [
    "IPERPoseDataset",
    "TikTokPoseDataset",
    "TikTokV3PoseDataset",
    "TikTokV65FaceOverfitLoader",
    "TikTokV65OverfitInputs",
    "is_tiktok_v3_manifest",
    "load_pose_dataset",
]


def __getattr__(name):
    """Load each dataset only when its public export is requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".iper_dataset" if name == "IPERPoseDataset" else ".tiktok_dataset"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

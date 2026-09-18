import os
import torch
import pytest
from pathlib import Path

from src.data.iper_dataset import IPERPoseDataset
from src.pose_control.v6.condition_injector import SMPLXConditionInjector
from src.pose_control.v6.conditions import TaskType


@pytest.fixture
def dataset_paths():
    # If running on server or local with access to datasets
    sampled_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt"
    assets_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"
    pairs_jsonl = f"{sampled_root}/splits/train_pairs.jsonl"
    return pairs_jsonl, sampled_root, assets_root


def test_v6_dataset_loading_and_injector_pass():
    sampled_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt"
    assets_root = "/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast"
    pairs_jsonl = f"{sampled_root}/splits/train_pairs.jsonl"
    
    if not os.path.exists(pairs_jsonl):
        pytest.skip(f"Data not found at {pairs_jsonl}")
        
    dataset = IPERPoseDataset(
        pairs_jsonl=pairs_jsonl,
        sampled_root=sampled_root,
        assets_root=assets_root,
        resolution=512,
        augment=False,
    )
    
    assert len(dataset) > 0, "Dataset is empty"
    
    # Load first sample
    item = dataset[0]
    
    # Check keys
    required_keys = [
        "src_image", "tgt_image", "control_map",
        "normal", "depth", "part_onehot", "pose_heatmap",
        "smplx_global", "human_mask", "task_id"
    ]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset output"
        
    # Check shapes and ranges
    assert item["src_image"].shape == (3, 512, 512)
    assert item["tgt_image"].shape == (3, 512, 512)
    assert item["control_map"].shape == (8, 512, 512)
    
    normal = item["normal"]
    assert normal.shape == (3, 512, 512)
    assert normal.amin() >= -1.0001 and normal.amax() <= 1.0001
    
    depth = item["depth"]
    assert depth.shape == (1, 512, 512)
    assert depth.amin() >= 0.0 and depth.amax() <= 1.0001
    
    part_onehot = item["part_onehot"]
    assert part_onehot.shape == (14, 512, 512)
    # Check one-hot property (each pixel at most one class)
    assert torch.all(part_onehot.sum(dim=0) <= 1.0001)
    # Check non-empty foreground
    assert part_onehot.sum() > 100, "Part map appears completely empty"
    
    pose_heatmap = item["pose_heatmap"]
    assert pose_heatmap.shape == (25, 512, 512)
    assert pose_heatmap.amin() >= 0.0 and pose_heatmap.amax() <= 1.0001
    assert pose_heatmap.sum() > 0, "Pose heatmap appears completely empty"
    
    smplx_global = item["smplx_global"]
    assert smplx_global.shape == (26,)
    
    mask = item["human_mask"]
    assert mask.shape == (1, 512, 512)
    assert mask.amin() >= 0.0 and mask.amax() <= 1.0001
    
    # Forward pass through SMPLXConditionInjector
    injector = SMPLXConditionInjector(use_depth=False)
    
    bundle = injector(
        normal_a=normal.unsqueeze(0),
        pose_heatmap_a=pose_heatmap.unsqueeze(0),
        part_onehot_a=part_onehot.unsqueeze(0),
        smplx_global_a=smplx_global.unsqueeze(0),
        human_mask_a=mask.unsqueeze(0),
        task_id=item["task_id"].unsqueeze(0),
    )
    
    assert bundle is not None
    assert bundle.person_a_spatial.shape == (1, 256, 64, 64)
    assert bundle.person_a_mask.shape == (1, 1, 64, 64)
    assert bundle.person_a_global_tokens.shape == (1, 4, 256)
    assert bundle.task_token.shape == (1, 1, 256)
    assert bundle.person_count.item() == 1
    print("V6 Condition Injector consumed bundle successfully!")

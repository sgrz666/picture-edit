#!/usr/bin/env python3
"""Split official iPER 79 training appearances into dev-train (69) and dev-val (10).

Ensures:
1. 20 official test appearances remain completely untouched.
2. Intersection between dev-train and dev-val appearances is 0.
3. Intersection with test appearances is 0.
4. Total pairs dev_train + dev_val == official train pairs (30,336).
"""

import json
import os
import random
import argparse
from pathlib import Path


def split_iper_train_val(
    splits_dir: str,
    val_appearance_count: int = 10,
    seed: int = 42,
):
    splits_path = Path(splits_dir)
    train_file = splits_path / "train_pairs.jsonl"
    test_file = splits_path / "test_pairs.jsonl"
    
    assert train_file.exists(), f"Train file not found: {train_file}"
    assert test_file.exists(), f"Test file not found: {test_file}"
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_pairs = [json.loads(line) for line in f if line.strip()]
        
    with open(test_file, "r", encoding="utf-8") as f:
        test_pairs = [json.loads(line) for line in f if line.strip()]
        
    train_appearances = sorted(list(set(p["appearance"] for p in train_pairs)))
    test_appearances = sorted(list(set(p["appearance"] for p in test_pairs)))
    
    print(f"Official train pairs: {len(train_pairs)}, appearances: {len(train_appearances)}")
    print(f"Official test pairs: {len(test_pairs)}, appearances: {len(test_appearances)}")
    
    # Assert zero leakage with test set
    test_intersection = set(train_appearances) & set(test_appearances)
    assert len(test_intersection) == 0, f"Leakage between train and test: {test_intersection}"
    assert len(train_appearances) == 79, f"Expected 79 train appearances, got {len(train_appearances)}"
    assert len(test_appearances) == 20, f"Expected 20 test appearances, got {len(test_appearances)}"
    
    # Deterministic split of train appearances
    rng = random.Random(seed)
    shuffled_apps = list(train_appearances)
    rng.shuffle(shuffled_apps)
    
    val_apps = set(shuffled_apps[:val_appearance_count])
    train_dev_apps = set(shuffled_apps[val_appearance_count:])
    
    assert len(val_apps) == val_appearance_count
    assert len(train_dev_apps) == len(train_appearances) - val_appearance_count
    assert len(val_apps & train_dev_apps) == 0
    assert len(val_apps & set(test_appearances)) == 0
    assert len(train_dev_apps & set(test_appearances)) == 0
    
    dev_train_pairs = [p for p in train_pairs if p["appearance"] in train_dev_apps]
    dev_val_pairs = [p for p in train_pairs if p["appearance"] in val_apps]
    
    assert len(dev_train_pairs) + len(dev_val_pairs) == len(train_pairs), "Mismatch in total pair count!"
    
    dev_train_file = splits_path / "dev_train_pairs.jsonl"
    dev_val_file = splits_path / "dev_val_pairs.jsonl"
    
    with open(dev_train_file, "w", encoding="utf-8") as f:
        for p in dev_train_pairs:
            f.write(json.dumps(p) + "\n")
            
    with open(dev_val_file, "w", encoding="utf-8") as f:
        for p in dev_val_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"\nSuccessfully generated splits:")
    print(f"  Dev-train: {len(dev_train_pairs)} pairs across {len(train_dev_apps)} appearances -> {dev_train_file}")
    print(f"  Dev-val:   {len(dev_val_pairs)} pairs across {len(val_apps)} appearances -> {dev_val_file}")
    print(f"  Official test set untouched: {len(test_pairs)} pairs across {len(test_appearances)} appearances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/home/shangguanrz/project/pic-edit/datasets/iPER/iper_sampled_6src64tgt/splits",
    )
    parser.add_argument("--val_appearance_count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    split_iper_train_val(args.splits_dir, args.val_appearance_count, args.seed)

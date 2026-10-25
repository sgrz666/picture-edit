#!/usr/bin/env python3
"""V5 training preparation CLI; default prepare never starts a model/optimizer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    from src.pose_control.qwen21_full.native_v5.training.cli import main as training_main
    return training_main()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""V6.2 training preparation CLI; default prepare never starts a model/optimizer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    from src.pose_control.qwen21_full.native_v62.training.cli import main as training_main
    return training_main()

if __name__=='__main__':main()

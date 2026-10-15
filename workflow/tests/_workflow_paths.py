"""Put the workflow directory and the repository root on sys.path for every runner.

``pytest`` gets this from ``conftest.py``; ``python -m unittest discover -s tests``,
which is the command the README documents, does not read conftest, so each test module
imports this file first instead.
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
WORKFLOW_ROOT = TESTS_DIR.parent
REPO_ROOT = WORKFLOW_ROOT.parent

for candidate in (str(REPO_ROOT), str(WORKFLOW_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

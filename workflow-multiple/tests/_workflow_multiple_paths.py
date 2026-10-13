"""Put the two-person workflow directory and the repository root on sys.path for every runner.

``pytest`` also benefits from the repository-root ``conftest.py``, but
``python -m unittest discover -s tests`` does not read conftest at all, so each test module
imports this file first. The name is workflow-specific on purpose: two suites both calling a
helper ``_workflow_multiple_paths`` makes pytest import only one of them and the other suite then fails to
find its own modules.
"""
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
WORKFLOW_ROOT = TESTS_DIR.parent
REPO_ROOT = WORKFLOW_ROOT.parent

for candidate in (str(REPO_ROOT), str(WORKFLOW_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

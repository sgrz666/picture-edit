"""Only the missing pure-Python SAM dependencies may be overlaid."""
import importlib.util
from pathlib import Path

import pytest


def module():
    p=Path(__file__).parents[1]/'experiments/xiaohongshu-region-sdpose-sft-rl/scripts/prepare_sam_dependencies.py'
    spec=importlib.util.spec_from_file_location('sam_dependency_overlay',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_dependency_allowlist_excludes_training_frameworks():
    m=module()
    assert {x[0] for x in m.PACKAGES}=={'hydra','omegaconf','antlr4','iopath','portalocker'}
    assert not {'torch','diffusers','transformers'}&{x[0] for x in m.PACKAGES}


def test_binary_dependency_is_rejected_before_copying(tmp_path):
    m=module();(tmp_path/'__init__.py').write_text('')
    m.require_pure_python(tmp_path)
    (tmp_path/'native.so').write_bytes(b'binary')
    with pytest.raises(ValueError,match='binary'):m.require_pure_python(tmp_path)

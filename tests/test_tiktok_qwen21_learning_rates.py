import pytest
from torch import nn

from src.pose_control.qwen21_full.execution import parameter_groups


def test_face_learning_rate_scale_keeps_branch_ratios():
    adapter = nn.Module()
    adapter.output_bridges = nn.ModuleDict({'face': nn.Linear(4, 4)})
    adapter.legacy = nn.Module()
    adapter.legacy.face_preparer = nn.Linear(4, 4)
    adapter.legacy.control_interface = nn.Module()
    adapter.legacy.control_interface.face_adapter = nn.Linear(4, 4)
    groups, _ = parameter_groups(adapter, 'face', lr_scale=.01)
    rates = {group['name']: group['lr'] for group in groups}
    assert rates == pytest.approx({'heads_and_interface': 1e-6, 'encoders': 5e-7, 'control_core': 1e-7})


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_learning_rate_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='lr_scale'):
        parameter_groups(nn.Linear(1, 1), 'face', lr_scale=scale)

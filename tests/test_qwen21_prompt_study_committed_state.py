import random
import numpy as np
import pytest
import torch
from src.pose_control.qwen21_full.prompt_study.protocol import validate_committed_training_state


def saved_state():
    return dict(adapter_state_dict={'body.weight': torch.ones(2, 2)},
                optimizer_state_dict={'state': {0: {'step': torch.tensor(1024)}},
                                      'param_groups': [{'params': [0]}]},
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=[torch.get_rng_state()],
                python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def test_actual_saved_training_state_is_accepted():
    validate_committed_training_state(saved_state(), [{'loss': .2, 'grad_norm': .4}])


@pytest.mark.parametrize('key,value', [('adapter_state_dict', {}), ('optimizer_state_dict', {}),
                                      ('torch_rng_state', None), ('cuda_rng_state', []),
                                      ('python_rng_state', None), ('numpy_rng_state', None)])
def test_header_without_training_state_is_rejected(key, value):
    checkpoint = saved_state()
    checkpoint[key] = value
    with pytest.raises(ValueError, match='committed'):
        validate_committed_training_state(checkpoint, [{'loss': .2, 'grad_norm': .4}])


@pytest.mark.parametrize('key,value', [('loss', float('nan')), ('loss', float('inf')),
                                     ('grad_norm', float('nan')), ('grad_norm', 0)])
def test_nonfinite_or_no_gradient_update_is_rejected(key, value):
    row = dict(loss=.2, grad_norm=.4)
    row[key] = value
    with pytest.raises(ValueError, match=key):
        validate_committed_training_state(saved_state(), [row])


def test_nonfinite_adapter_weight_is_rejected():
    checkpoint = saved_state()
    checkpoint['adapter_state_dict']['body.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Adapter tensor'):
        validate_committed_training_state(checkpoint, [{'loss': .2, 'grad_norm': .4}])

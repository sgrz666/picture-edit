import pytest
import torch
from scripts.inherit_qwen21_full_weights import verify_state


def fixture():
    receiver = {'embedding': torch.arange(12.).reshape(3, 4), 'preserved': torch.ones(2)}
    source = {'e': torch.zeros(3, 4)}
    entries = [{'source': 'e', 'target': 'embedding', 'rows': [0]}]
    actual = {name: value.clone() for name, value in receiver.items()}
    actual['embedding'][0] = source['e'][0]
    return actual, receiver, source, entries


def test_saved_state_retains_unselected_rows_and_parameters():
    assert verify_state(*fixture())['passed']


@pytest.mark.parametrize('fault', ['outside', 'unselected_row', 'selected_row', 'schema', 'dtype', 'nonfinite'])
def test_saved_state_detects_corruption(fault):
    actual, receiver, source, entries = fixture()
    if fault == 'outside': actual['preserved'][0] = 3
    if fault == 'unselected_row': actual['embedding'][1, 0] = 3
    if fault == 'selected_row': actual['embedding'][0, 0] = 3
    if fault == 'schema': actual['extra'] = torch.zeros(1)
    if fault == 'dtype': actual['preserved'] = actual['preserved'].double()
    if fault == 'nonfinite': actual['embedding'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        verify_state(actual, receiver, source, entries)

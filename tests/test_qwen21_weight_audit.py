import pytest
import torch
from torch import nn
from scripts.audit_qwen21_weight_compatibility import apply_mapping_in_memory


def test_explicit_rows_preserve_unselected_state():
    model = nn.Sequential(nn.Embedding(3, 4), nn.Linear(4, 2))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {'source': torch.ones(3, 4), 'unused': torch.ones(1)}
    apply_mapping_in_memory(model, state, [{'source': 'source', 'target': '0.weight', 'rows': [0]}], ['unused'])
    assert torch.equal(model[0].weight[0], state['source'][0])
    assert torch.equal(model[0].weight[1:], before['0.weight'][1:])
    assert torch.equal(model[1].weight, before['1.weight'])
    assert torch.equal(model[1].bias, before['1.bias'])


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'shape', 'undeclared', 'nonfinite', 'bad_rows'])
def test_invalid_mapping_fails_before_any_mutation(fault):
    model = nn.Linear(4, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    source = {'w': torch.zeros(2, 4), 'b': torch.zeros(2)}
    mapping = [{'source': 'w', 'target': 'weight'}, {'source': 'b', 'target': 'bias'}]
    if fault == 'missing': mapping[1]['target'] = 'missing'
    if fault == 'duplicate': mapping[1]['target'] = 'weight'
    if fault == 'shape': source['b'] = torch.zeros(3)
    if fault == 'undeclared': source['extra'] = torch.zeros(1)
    if fault == 'nonfinite': source['b'][0] = float('nan')
    if fault == 'bad_rows': mapping[0]['rows'] = [0, 0]
    with pytest.raises(ValueError):
        apply_mapping_in_memory(model, source, mapping, [])
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])

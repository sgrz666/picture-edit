import importlib.util
from pathlib import Path
import pytest
import torch


def driver():
    path=Path(__file__).parents[1]/'experiments/bilibili-single-spatial-train-20261107/region-sdpose-sft-rl/scripts/rl.py'
    spec=importlib.util.spec_from_file_location('bilibili_region_rl_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_policy_state_copy_preserves_non_tensor_architecture_headers():
    module=driver();state={'weight':torch.ones(2),'_extra_state':{'schema':'native_full_v7','config':[1,2]}}
    copied=module.clone_policy_state(state)
    copied['weight'][0]=9;copied['_extra_state']['config'][0]=8
    assert state['weight'][0]==1 and state['_extra_state']['config'][0]==1


def test_reference_weight_context_restores_parameter_objects_on_failure():
    module=driver();net=torch.nn.Linear(2,1);current=module.clone_policy_state(net.state_dict())
    reference={name:torch.zeros_like(value) for name,value in current.items()}
    ids=[id(p) for p in net.parameters()]
    with pytest.raises(RuntimeError):
        with module.weights(net,reference,current):
            assert net.weight.count_nonzero()==0
            raise RuntimeError('controlled reference-probe failure')
    assert [id(p) for p in net.parameters()]==ids
    assert torch.equal(net.weight,current['weight']) and torch.equal(net.bias,current['bias'])


def test_failed_reference_loading_also_restores_live_weights():
    module=driver();net=torch.nn.Linear(2,1);current=module.clone_policy_state(net.state_dict())
    with pytest.raises(RuntimeError):
        with module.weights(net,{'weight':torch.zeros_like(net.weight)},current):pass
    assert torch.equal(net.weight,current['weight']) and torch.equal(net.bias,current['bias'])

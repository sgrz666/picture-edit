from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_qwen21_dual_v8_context import fixture


@pytest.mark.parametrize('cfg',[1.,4.])
def test_actual_transformer_cfg_observer_preserves_scheduler_and_records_both_branches(cfg):
    from src.pose_control.qwen21_full.dual.diagnostics import GenerationDiagnostics
    model,inputs,_=fixture()
    class Scheduler:
        def step(self,velocity,timestep,sample,return_dict=False):
            return (sample-.1*velocity,)
    scheduler=Scheduler()
    pipe=SimpleNamespace(transformer=model,scheduler=scheduler)
    adapter=SimpleNamespace(last_diagnostics={3:{'total':{'rms':.01}}})
    original=scheduler.step
    with torch.no_grad(),GenerationDiagnostics(pipe,adapter,cfg=cfg,target_tokens=4) as observer:
        cond=model(**inputs)[0][:,-4:]
        velocity=cond
        if cfg>1:
            adapter.last_diagnostics={3:{'total':{'rms':.02}}}
            uncond=model(**dict(inputs,encoder_hidden_states=torch.zeros_like(inputs['encoder_hidden_states'])))[0][:,-4:]
            velocity=uncond+cfg*(cond-uncond)
        sample=torch.ones_like(velocity)
        expected=original(velocity,torch.tensor(.6),sample)
        actual=scheduler.step(velocity,torch.tensor(.6),sample)
        torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    assert scheduler.step==original and 'step' not in scheduler.__dict__
    assert not model._forward_hooks
    item=observer.records[0]
    assert item['cfg']==cfg and item['combined']['finite'] and item['combined_matches_native_formula']
    assert item['cond']['residuals'][3]['total']['rms']==.01
    if cfg>1:
        assert item['uncond']['residuals'][3]['total']['rms']==.02
    else:
        assert item['uncond'] is None


def test_cfg_observer_restores_existing_step_on_generation_error():
    from src.pose_control.qwen21_full.dual.diagnostics import GenerationDiagnostics
    model,_,_=fixture()
    scheduler=SimpleNamespace(step=lambda *args:args)
    original=scheduler.step
    pipe=SimpleNamespace(transformer=model,scheduler=scheduler)
    with pytest.raises(RuntimeError,match='abort'):
        with GenerationDiagnostics(pipe,None,cfg=1,target_tokens=4):
            raise RuntimeError('abort')
    assert scheduler.step is original and not model._forward_hooks

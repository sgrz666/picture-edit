"""Equivalence tests; CPU or explicit CUDA gradient buffers, no optimizer steps."""
from types import SimpleNamespace
import os
import pytest
import torch


def backend_for(gradients,device='cpu'):
    params={};rows=[]
    for i,gradient in enumerate(gradients):
        name='p'+str(i);p=torch.nn.Parameter(torch.zeros(gradient.shape if gradient is not None else (3,),device=device))
        p.grad=None if gradient is None else gradient.to(device)
        params[name]=p;rows.append(SimpleNamespace(name=name,runtime_group='body',role='encoder'))
    active=set(params)
    return SimpleNamespace(active=active,catalog=SimpleNamespace(rows=rows,parameters=params,assert_flags=lambda a:None),
                           base_parameters={},evaluators=[],pipe=SimpleNamespace(vae=None,text_encoder=None))


def legacy_audit(backend,require_any=True):
    # Fixed pre-optimization numerical reference; the frozen checks are tested separately.
    rows={};nonzero=False
    for r in backend.catalog.rows:
        p=backend.catalog.parameters[r.name];g=p.grad
        if p.dtype!=torch.float32:raise ValueError('Adapter parameters must be FP32')
        if g is not None and not torch.isfinite(g).all():raise FloatingPointError('Nonfinite gradient: '+r.name)
        status='not_called_or_no_gradient' if g is None else 'zero' if not g.count_nonzero() else 'nonzero'
        if status=='nonzero':nonzero=True
        rows[r.name]={'status':status,'norm':None if g is None else float(g.norm()),'active':r.name in backend.active,'group':r.runtime_group,'role':r.role}
    if require_any and not nonzero:raise RuntimeError('No active finite nonzero gradient')
    return rows


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_packed_audit_is_exact_and_does_not_modify_gradients(device,monkeypatch):
    if device=='cuda' and (os.environ.get('NATIVE_TRAINING_SPEED_CUDA')!='1' or not torch.cuda.is_available()):
        pytest.skip('Explicit isolated CUDA audit required')
    from src.pose_control.qwen21_full.native_v5.training import validation as v
    b=backend_for([None,torch.zeros(7),torch.arange(15,dtype=torch.float32).reshape(3,5),
                   torch.tensor([1e-30,-1e-30]),torch.empty(0)],device)
    expected=legacy_audit(b)
    before={n:None if p.grad is None else p.grad.clone() for n,p in b.catalog.parameters.items()}
    calls=[];original=v._gradient_metrics_to_cpu
    def transfer(value):calls.append(value.device);return original(value)
    monkeypatch.setattr(v,'_gradient_metrics_to_cpu',transfer)
    assert v.audit_gradients(b)==expected
    assert calls==[next(iter(b.catalog.parameters.values())).device]
    for name,p in b.catalog.parameters.items():
        assert p.grad is None if before[name] is None else torch.equal(p.grad,before[name])


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_packed_audit_retains_finite_failure_and_parameter_name(bad):
    from src.pose_control.qwen21_full.native_v5.training.validation import audit_gradients
    b=backend_for([torch.ones(2),torch.tensor([bad])])
    with pytest.raises(FloatingPointError,match='p1'):audit_gradients(b)


def test_packed_audit_retains_zero_missing_dtype_and_frozen_guards():
    from src.pose_control.qwen21_full.native_v5.training.validation import audit_gradients
    b=backend_for([None,torch.zeros(3)])
    assert audit_gradients(b,require_any=False)==legacy_audit(b,False)
    with pytest.raises(RuntimeError,match='nonzero'):audit_gradients(b)
    b.catalog.parameters['p0']=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='FP32'):audit_gradients(b,require_any=False)
    b=backend_for([torch.ones(2)]);b.active.clear()
    with pytest.raises(RuntimeError,match='Frozen'):audit_gradients(b)


def test_parameter_binding_audit_enumerates_once_and_rejects_replacement(monkeypatch):
    from src.pose_control.qwen21_full.native_v5.training.parameters import ParameterCatalog
    runtime=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Linear(3,3))
    catalog=object.__new__(ParameterCatalog);catalog.runtime=runtime;catalog.parameters=dict(runtime.named_parameters())
    original=runtime.named_parameters;calls=[]
    def counted(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(runtime,'named_parameters',counted)
    catalog.assert_flags(set(catalog.parameters))
    assert len(calls)==1
    runtime[0].weight=torch.nn.Parameter(torch.zeros_like(runtime[0].weight))
    with pytest.raises(RuntimeError,match='objects changed'):catalog.assert_flags(set(catalog.parameters))

"""Opt-in scalar diagnostics; no Tensor retention, RNG, or module mutation."""
from contextlib import contextmanager
from contextvars import ContextVar

from ..native_v5.observation import ResidualObserver

_observer=ContextVar('native_v62_diagnostics',default=None)


class DiagnosticsV62:
    def __init__(self):
        self.records=[]
        self.legacy=ResidualObserver()

    @contextmanager
    def capture(self):
        token=_observer.set(self)
        try:
            with self.legacy.capture():yield self
        finally:_observer.reset(token)

    def report(self):
        return {'schema':'native_v62_diagnostics_v1','tensor_retention':False,
                'records':list(self.records),'branch_residuals':self.legacy.report()}


def observe_permissions(allowed,query_ids,key_ids,*,layer):
    observer=_observer.get()
    if observer is None:return
    hard=query_ids>=0
    cross=hard[...,None] & (key_ids[:,None]!=query_ids[...,None])
    observer.records.append({'kind':'routing','layer':int(layer),
        'hard_queries':int(hard.sum()),'soft_queries':int((~hard).sum()),
        'cross_person_permissions':int((allowed&cross).sum()),
        'no_legal_key_queries':int((~allowed.any(-1)).sum())})


def observe_branch(branch,delta,hidden,*,layer,metadata):
    observer=_observer.get()
    if observer is None:return
    import torch
    with torch.no_grad():
        d=delta.detach().float();h=hidden.detach().float()
        observer.records.append({'kind':'branch','branch':branch,'layer':int(layer),
            'rms':float(d.square().mean().sqrt()),
            'relative_l2':float(d.norm()/(h.norm()+1e-6)),
            'gate_reasons':[str(row.get(branch+'_gate_reason','existing_owned_roi')) for row in metadata]})

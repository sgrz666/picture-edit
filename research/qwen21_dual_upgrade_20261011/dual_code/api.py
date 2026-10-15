"""Explicit dispatch; callers retain their original single-person context."""
from contextlib import nullcontext

from .context import DualV8ControlContext


def select_control_context(mode,transformer,*,generation=None,adapter=None,prepared=None,
                           sigma=None,strengths=None,single_context_factory=None,
                           single_context=None,legacy_dual_context_factory=None):
    if single_context is not None and single_context_factory is not None:
        raise ValueError('supply either the original single context or its factory')
    def original_single():
        if single_context is not None:return single_context
        if single_context_factory is not None:return single_context_factory()
        raise ValueError('supply the original single context or context factory')
    if mode=='native':return nullcontext()
    if mode=='single_v6':
        return original_single()
    if mode=='legacy_dual_lite':
        if legacy_dual_context_factory is None:raise ValueError('supply the original legacy dual context factory')
        return legacy_dual_context_factory()
    if mode!='dual_v8':raise ValueError(f'unknown adapter mode {mode!r}')
    if generation is None:raise ValueError('dual_v8 dispatch requires a typed generation input')
    generation.validate()
    dual=generation.geometry.person_valid.all(-1)
    if not bool(dual.all()):
        if bool(dual.any()):raise ValueError('mixed single/dual batches must be dispatched separately')
        return original_single()
    if adapter is None or prepared is None or sigma is None:
        raise ValueError('dual_v8 requires its adapter, prepared state, and actual flow sigma')
    return DualV8ControlContext(transformer,adapter,prepared,sigma=sigma,strengths=strengths)

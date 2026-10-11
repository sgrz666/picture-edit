"""Observe native conditional, unconditional and scheduler input velocities."""
from contextlib import AbstractContextManager
import copy
import torch


def velocity_stats(value):
    value=value.detach().float()
    return dict(rms=float(value.square().mean().sqrt()),max_abs=float(value.abs().max()),
                finite=bool(torch.isfinite(value).all()))


class GenerationDiagnostics(AbstractContextManager):
    """Read-only hooks and a restored scheduler wrapper; native math is retained.

    QwenImage21 calls cond, then uncond when CFG is enabled, then scheduler.step.
    The latter exposes the actual combined velocity, not an inferred branch RMS.
    """
    def __init__(self,pipe,adapter,*,cfg,target_tokens):
        self.pipe,self.adapter,self.cfg,self.target_tokens=pipe,adapter,float(cfg),target_tokens
        self.records=[]
        self.pending=[]
        self.handle=None
        self.original_step=None

    def _capture(self,module,args,output):
        value=output[0] if isinstance(output,(tuple,list)) else output.sample
        velocity=value[:,-self.target_tokens:].detach().clone()
        stats=velocity_stats(velocity)
        if not stats['finite']:
            raise FloatingPointError('nonfinite native CFG branch velocity')
        stats['residuals']=copy.deepcopy(getattr(self.adapter,'last_diagnostics',{}))
        self.pending.append((velocity,stats))

    def _step(self,velocity,timestep,sample,*args,**kwargs):
        expected_calls=2 if self.cfg>1 else 1
        if len(self.pending)!=expected_calls:
            raise RuntimeError('native CFG call order differs from the inspected QwenImage21 pipeline')
        cond,uncond=self.pending[0],self.pending[1] if self.cfg>1 else None
        expected=cond[0] if uncond is None else uncond[0]+self.cfg*(cond[0]-uncond[0])
        if expected.shape!=velocity.shape:
            raise ValueError('observed native target velocity shape differs from scheduler input')
        combined=velocity_stats(velocity)
        if not combined['finite']:
            raise FloatingPointError('nonfinite combined native CFG velocity')
        difference=float((velocity.detach().float()-expected.float()).abs().max())
        self.records.append(dict(step=len(self.records),timestep=float(torch.as_tensor(timestep).flatten()[0]),
            cfg=self.cfg,cond=cond[1],uncond=None if uncond is None else uncond[1],combined=combined,
            combined_matches_native_formula=torch.equal(velocity,expected),combined_formula_max_abs=difference))
        self.pending=[]
        return self.original_step(velocity,timestep,sample,*args,**kwargs)

    def __enter__(self):
        if getattr(self.pipe,'_dual_v8_diagnostics_active',False):
            raise RuntimeError('another generation diagnostics observer is active')
        self.pipe._dual_v8_diagnostics_active=True
        self.had_instance_step='step' in vars(self.pipe.scheduler)
        self.original_step=self.pipe.scheduler.step
        self.handle=self.pipe.transformer.register_forward_hook(self._capture)
        self.pipe.scheduler.step=self._step
        return self

    def __exit__(self,*exception):
        if self.handle is not None:self.handle.remove()
        if self.original_step is not None:
            if self.had_instance_step:self.pipe.scheduler.step=self.original_step
            else:del self.pipe.scheduler.step
        self.pending=[]
        self.pipe._dual_v8_diagnostics_active=False
        return False

    @property
    def summary(self):
        return dict(observer_version='native_cfg_velocity_v1',cfg=self.cfg,steps=self.records)

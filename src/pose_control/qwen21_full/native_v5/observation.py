"""Optional scalar-only observation. No state, parameters, RNG or tensor mutation."""
from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar('native_v5_residual_observer', default=None)


class ResidualObserver:
    def __init__(self):
        self.records = []

    @contextmanager
    def capture(self):
        token = _observer.set(self)
        try:
            yield self
        finally:
            _observer.reset(token)

    def report(self):
        by_branch = {}
        for row in self.records:
            item = by_branch.setdefault(row['branch'], dict(observations=0, nonzero=0, max_rms=0.))
            item['observations'] += 1
            item['nonzero'] += int(row['rms'] > 0)
            item['max_rms'] = max(item['max_rms'], row['rms'])
        return dict(schema='native_v5_residual_observation_v1', tensor_retention=False,
                    branch_summary=by_branch, records=self.records)


def observe_residual(branch, value, *, layer, sigma, gain=None, region=None,
                     condition_valid=True, space='injected_hidden', row=None):
    observer = _observer.get()
    if observer is None:
        return
    import torch
    with torch.no_grad():
        rms = 0. if value is None or value.numel() == 0 else float(value.detach().float().square().mean().sqrt())
        observer.records.append(dict(branch=branch, layer=int(layer), sigma=sigma.detach().float().cpu().tolist(),
            gain=None if gain is None else gain.detach().float().cpu().tolist(), rms=rms,
            region=region, row=row, space=space, condition_valid=bool(condition_valid),
            elements=0 if value is None else value.numel()))

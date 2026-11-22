"""Immutable optimizer-update checkpoints for Native RL."""

import json
import os
from pathlib import Path

import torch

from ..native_v62.training.checkpointing import capture_rng, restore_rng
from ..native_v62.training.config import digest
from ..native_v62.training.io import file_sha, verify, write_json
from .runner import RunCounters


SCHEMA = 'native_rl_commit_v3'


def save_checkpoint(directory, payload, optimizer, selector, counters, *, identity, selection, pending, resumable):
    directory = Path(directory).resolve()
    path = directory / f'update-{counters.optimizer_updates:06d}-batch-{counters.rollout_batches:06d}.pt'
    state = {'schema': SCHEMA, 'identity': identity, 'counters': counters.record(),
             'rng': capture_rng(), 'sampler': selector.getstate(), 'optimizer': optimizer.state_dict(),
             'selection': selection, 'pending': pending, 'resumable': resumable}
    with path.open('xb') as handle:
        torch.save({**payload, 'native_rl_state': state}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    ref = {'path': str(path), 'sha256': file_sha(path)}
    write_json(path.with_suffix('.commit.json'), _marker(ref, payload['architecture'], state))
    return ref


def _marker(ref, architecture, state):
    return {'schema': SCHEMA, 'checkpoint': ref, 'architecture': architecture,
            'identity_sha256': digest(state['identity']), 'counters': state['counters'],
            'resumable': state['resumable']}


def load_checkpoint(reference):
    path = Path(reference['path']).resolve()
    marker_path = path.with_suffix('.commit.json')
    if not marker_path.is_file():
        raise ValueError('Native RL resume requires a completed commit record')
    marker = json.loads(marker_path.read_text(encoding='utf8'))
    if marker.get('schema') != SCHEMA:
        raise ValueError('This checkpoint predates Native RL optimizer resume')
    ref = verify(reference)
    if marker.get('checkpoint') != ref:
        raise ValueError('Native RL checkpoint path or SHA differs from its commit record')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = payload.get('native_rl_state', {})
    if state.get('schema') != SCHEMA or marker != _marker(ref, payload['architecture'], state):
        raise ValueError('Native RL checkpoint identity differs from its commit record')
    if not state['resumable']:
        raise ValueError('Native RL checkpoint has a terminal stop decision; use a resumable checkpoint')
    return payload, ref


def restore_checkpoint(payload, runtime, optimizer, selector, identity):
    state = payload['native_rl_state']
    if digest(state['identity']) != digest(identity):
        raise ValueError('Native RL resume identity differs (policy, reference, data, code, evaluator or schedule)')
    counters = RunCounters(**state['counters'])
    if any(type(v) is not int or v < 0 for v in counters.record().values()):
        raise ValueError('Invalid Native RL counters')
    pending = state['pending']
    if pending is not None and not (0 < pending['cursor'] < pending['epochs'] and pending['groups']):
        raise ValueError('Invalid Native RL replay cursor')
    runtime.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    selector.setstate(state['sampler'])
    restore_rng(state['rng'])
    return counters, pending

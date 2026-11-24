import random

import pytest
import torch

from src.pose_control.qwen21_full.native_rl.runner import FrozenReference, Replay, RunCounters, update_policy


def test_immutable_resume_restores_pending_replay_and_optimizer(tmp_path):
    from src.pose_control.qwen21_full.native_rl import checkpointing as ckpt

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    selector = random.Random(42)
    reference = FrozenReference(model.parameters())
    counters = RunCounters()
    counters.rollout(2)
    counters.updated(2)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = torch.ones(1, 2)
    replay = Replay(state, state + 1, torch.tensor([.2]), 700., .7, .6)
    pending = {'cursor': 1, 'epochs': 2, 'groups': [dict(key=('train', 0), condition='sha',
                  rows=[vars(replay)], advantage=torch.tensor([1.]))]}
    identity = {'sft_reference': {'sha256': '1' * 64}, 'algorithm': {'clip': .001}}
    payload = {'architecture': 'native_full_v7', 'state_dict': model.state_dict()}
    ref = ckpt.save_checkpoint(tmp_path, payload, optimizer, selector, counters, identity=identity,
                               selection={'baseline': {'pose': .5}}, pending=pending, resumable=True)
    expected_seed = selector.randrange(2**63)
    expected_rng = torch.rand(2)
    saved, bound = ckpt.load_checkpoint(ref)
    assert bound == ref
    other = torch.nn.Linear(2, 1, bias=False)
    resumed = torch.optim.AdamW(other.parameters(), lr=.01)
    restored, in_progress = ckpt.restore_checkpoint(saved, other, resumed, selector, identity)
    assert restored.record() == counters.record()
    assert in_progress['cursor'] == 1 and in_progress['epochs'] == 2
    torch.testing.assert_close(in_progress['groups'][0]['rows'][0]['action'], replay.action)
    torch.testing.assert_close(other.weight, model.weight, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state[other.weight]['exp_avg'], optimizer.state[model.weight]['exp_avg'], rtol=0, atol=0)
    assert selector.randrange(2**63) == expected_seed
    torch.testing.assert_close(torch.rand(2), expected_rng, rtol=0, atol=0)
    assert not torch.equal(reference.weights[0], model.weight)
    with pytest.raises(ValueError, match='identity'):
        ckpt.restore_checkpoint(saved, other, resumed, selector, {**identity, 'algorithm': {'clip': .1}})
    with open(ref['path'], 'ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='SHA'):
        ckpt.load_checkpoint(ref)


def test_optimizer_rejects_frozen_members_and_filter_preserves_moments():
    from src.pose_control.qwen21_full.native_rl.runner import restrict_optimizer

    active = torch.nn.Parameter(torch.tensor(1.))
    frozen = torch.nn.Parameter(torch.tensor(2.))
    optimizer = torch.optim.AdamW([{'params': [active, frozen], 'names': ['active', 'frozen']}])
    (active + frozen).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    frozen.requires_grad_(False)
    with pytest.raises(RuntimeError, match='freeze mask'):
        update_policy(optimizer, [], [active], [frozen], beta=0.)
    moment = optimizer.state[active]['exp_avg'].clone()
    restrict_optimizer(optimizer, [active])
    assert optimizer.param_groups[0]['names'] == ['active'] and frozen not in optimizer.state
    torch.testing.assert_close(optimizer.state[active]['exp_avg'], moment)


@pytest.mark.parametrize('with_spatial,with_adaface', [(False, False), (True, False), (True, True)])
def test_cli_resume_mid_batch_matches_ten_continuous_updates(tmp_path, monkeypatch, with_spatial, with_adaface):
    from contextlib import ExitStack
    import json
    from pathlib import Path
    from types import SimpleNamespace as Box

    from PIL import Image
    from src.evaluation import native_kprpe
    from src.pose_control.qwen21_full.native_rl import cli, pose_worker, rewards, spatial_worker
    from src.pose_control.qwen21_full.native_rl.objective import transition
    from src.pose_control.qwen21_full.native_v62.training import checkpointing, data

    parent = {'path': str(tmp_path / 'sft.pt'), 'sha256': '0' * 64}
    route = Box(name='body', poselink_strength=0.)
    cfg = Box(initialization=parent, schema='v7', seed=42, generation_seed=47, diagnostics=False,
              qwen_condition_profile='single_2ref', experiment_mode='standard', signature=lambda: 'bound-config')
    records = [dict(source_id=source, index=0, pair_id=source, role=role, eligible=True)
               for source, role in (('a', 'train'), ('b', 'train'), ('c', 'validation'))]
    source = Box(records=records, signature='bound-data', datasets={r['source_id']: Box(rows=[r]) for r in records})
    payload = {'architecture': 'native_full_v7', 'training_state': {'stage_update': 8},
               'state_dict': {'native': torch.tensor(.1)}}

    class Runtime(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = torch.nn.Parameter(torch.tensor(.1))

        def checkpoint_payload(self, *, metadata):
            return {'architecture': 'native_full_v7', 'state_dict': self.state_dict(), 'metadata': metadata}

    class Backend:
        def __init__(self, config, dataset, *, saved, purpose):
            self.config, self.device = config, torch.device('cpu')
            self.runtime = Runtime()
            self.runtime.load_state_dict(saved['state_dict'])
            self.identity = {'code': 'fake', 'qwen': 'frozen'}
            self.pipe = Box(vae=Box(to=lambda _: None), transformer=torch.nn.Module())
            self.catalog = Box(parameters=dict(self.runtime.named_parameters()), assert_flags=lambda _: None,
                               rows=[Box(name='native', role='head', runtime_group='native_body')])

    class Worker:
        def __init__(self, *args, **kwargs):
            self.identity = {'environment': 'test'}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class Bank:
        def __init__(self, *args, **kwargs):
            self.spatial = kwargs.get('spatial')
            self.adaface = kwargs.get('identity') is not None

        def score(self, target, image):
            assert self.adaface == target.adaface
            value = image.getpixel((0, 0))[0] / 255.
            return ({'pose': value, 'identity': .8, 'hands': .6} if with_spatial else {'pose': value}), {
                'pck_002': value, 'id_eligible': 0}

    calls = []

    class VisualWorker(Worker):
        @classmethod
        def from_args(cls, args):
            return cls()

        def score_group(self, target, paths):
            calls.append(len(paths))
            results = []
            for i, path in enumerate(paths):
                with Image.open(path) as image:
                    results.append({'id': i, 'reward': (image.getpixel((0, 0))[0] % 37) / 36.})
            return results

    class Rollout:
        def __init__(self, backend, key):
            self.backend = backend
            self.pair = {'condition_sha256': key[0]}

        def predict(self, row):
            return row.state * self.backend.runtime.native

        def generate(self, seed, *, stochastic=False, **_):
            image = Image.new('RGB', (1, 1), (seed % 256, 0, 0))
            if not stochastic:
                return image, None
            init = torch.ones(1, 4)
            action, log_prob, _, _ = transition(init, init * self.backend.runtime.native.detach(), .7, .5,
                                               generator=torch.Generator().manual_seed(seed))
            return image, Replay(init, action, log_prob, 700., .7, .5)

    monkeypatch.setattr(cli, '_settings', lambda args: (cfg, route))
    monkeypatch.setattr(cli, '_backend_for', lambda *a: (lambda _: source, Backend, 'native_full_v7'))
    monkeypatch.setattr(cli, '_generator', lambda backend, route, key, args: Rollout(backend, key))
    monkeypatch.setattr(cli, '_targets', lambda bank, data, chosen, background: (
        {(r['source_id'], 0): (Box(identities=(True,) if with_spatial else (), adaface=with_adaface and r['role'] == 'train',
                                 contacts=(), background_status='absent',
                                 spatial=Box(prompt_sha256='fixed-prompt', record=lambda: {'prompt_sha256': 'fixed-prompt'})),)
         for r in chosen},
        [], {(r['source_id'], 0): {'driver_rgb_sha256': r['source_id']} for r in chosen}))
    monkeypatch.setattr(rewards, 'evaluator_assets', lambda _: [])
    monkeypatch.setattr(native_kprpe, 'recognition_assets', lambda *_: [])
    monkeypatch.setattr(native_kprpe, 'KPRPEIdentityMetric', lambda *_: object())
    monkeypatch.setattr(rewards, 'ImageRewardBank', Bank)
    monkeypatch.setattr(pose_worker, 'SDPoseWorker', Worker)
    monkeypatch.setattr(spatial_worker, 'SpatialWorker', VisualWorker)
    monkeypatch.setattr(checkpointing, 'environment', lambda: {'torch': 'test'})
    monkeypatch.setattr(checkpointing, 'load_committed', lambda *a, **k: (payload, parent, None))
    monkeypatch.setattr(data, 'route_options', lambda *a: (Box(body_strength=1.), Box()))

    def run(name, *, pause=None, checkpoint=None):
        reward_flags = ['--spatial-model', 'simulated-model'] if with_spatial else ['--pose-only']
        if with_adaface:
            reward_flags.extend(('--adaface-model', 'adaface-model', '--adaface-rpe-extension', 'rpe-extension'))
        args = cli._arguments(['check-data', '--config', 'dummy.json', '--updates', '10', '--group-size', '2',
                               '--groups-per-update', '2', '--replay-epochs', '2', '--validate-every', '3', *reward_flags])
        args.output, args.stop_after_updates = tmp_path / name, pause
        args.action = 'resume' if checkpoint else 'train'
        args.checkpoint = Path(checkpoint['path']) if checkpoint else None
        with ExitStack() as stack:
            cli._run(args, stack)
        return json.loads((args.output / 'summary.json').read_text())

    continuous = run('continuous')
    continuous_calls = calls.copy()
    calls.clear()
    first = run('paused', pause=5)
    second = run('resumed', checkpoint=first['checkpoint'])
    keys = ['optimizer_updates', 'rollout_batches', 'generated_images', 'replayed_transitions']
    assert first['stop'] == 'paused' and first['optimizer_updates'] == 5
    # The spatial fixture also exercises a zero-advantage batch where the
    # equally weighted Body and SpatialReward completely cancel.
    expected = [10, 6, 24, 40] if with_spatial else [10, 5, 20, 40]
    assert [second[k] for k in keys] == [continuous[k] for k in keys] == expected
    assert second['baseline'] == continuous['baseline']
    clean, restored = [torch.load(r['checkpoint']['path'], weights_only=True) for r in (continuous, second)]
    torch.testing.assert_close(clean['state_dict']['native'], restored['state_dict']['native'], rtol=0, atol=0)
    torch.testing.assert_close(clean['native_rl_state']['optimizer']['state'], restored['native_rl_state']['optimizer']['state'], rtol=0, atol=0)
    assert clean['native_rl_state']['sampler'] == restored['native_rl_state']['sampler']

    def rollouts(*dirs):
        return [json.loads(path.read_text())['groups'] for name in dirs
                for path in sorted((tmp_path / name).glob('rollout-*/rollout.json'))]

    assert rollouts('continuous') == rollouts('paused', 'resumed')
    assert calls == continuous_calls
    if with_spatial:
        manifest = json.loads((tmp_path / 'continuous' / 'run.json').read_text(encoding='utf8'))
        assert manifest['weights']['spatial'] == .25 and 'spatial' not in manifest['selection_weights']
        assert manifest['identity_backends'] == {'rollout': 'adaface_kprpe' if with_adaface else 'glintr100',
                                                  'evaluation': 'glintr100'}
        assert calls.count(2) == expected[1] * 2
        group = rollouts('resumed')[0][0]
        assert all('spatial' in row['reward'] for row in group['candidates'])

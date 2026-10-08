import importlib
import importlib.util
import hashlib
import json

import pytest
import torch


def joint_module():
    assert importlib.util.find_spec('src.pose_control.qwen21_full.joint_training') is not None, 'joint training path is missing'
    return importlib.import_module('src.pose_control.qwen21_full.joint_training')


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_bridges = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ['geometry', 'face', 'hand']})
        self.latent_bridge = torch.nn.Linear(2, 2)
        self.legacy = torch.nn.Module()
        self.legacy.face_preparer = torch.nn.Linear(2, 2)
        self.legacy.hand_preparer = torch.nn.Linear(2, 2)
        self.legacy.condition_injector = torch.nn.Module()
        self.legacy.condition_injector.global_encoder = torch.nn.Linear(2, 2)


def test_joint_optimizer_updates_both_branches_and_preserves_hand():
    module = joint_module()
    adapter = TinyAdapter()
    groups, names = module.joint_parameter_groups(adapter, body_scale=.25, face_scale=.5)
    module.enable_joint_training(adapter, names)
    flat = [p for group in groups for p in group['params']]
    assert len(flat) == len({id(p) for p in flat})
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    x = torch.ones(1, 2)
    loss = adapter.output_bridges['geometry'](x).sum() + adapter.output_bridges['face'](x).sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before['output_bridges.geometry.weight'], adapter.output_bridges['geometry'].weight)
    assert not torch.equal(before['output_bridges.face.weight'], adapter.output_bridges['face'].weight)
    assert torch.equal(before['output_bridges.hand.weight'], adapter.output_bridges['hand'].weight)
    assert not adapter.legacy.hand_preparer.weight.requires_grad
    assert adapter.legacy.condition_injector.global_encoder.weight.requires_grad
    assert adapter.legacy.face_preparer.weight.requires_grad
    rates = {group['name']: group['lr'] for group in groups}
    assert rates['body_heads_and_interface'] == pytest.approx(2.5e-5)
    assert rates['face_heads_and_interface'] == pytest.approx(5e-5)


def test_joint_loss_masks_padding_and_upweights_small_faces():
    module = joint_module()
    squared = torch.tensor([[[1.], [3.], [1000.], [1000.]]], requires_grad=True)
    human = torch.ones(1, 4, 1)
    valid = torch.tensor([[[1.], [1.], [0.], [0.]]])
    face = torch.tensor([[[0.], [1.], [0.], [0.]]])
    total, parts = module.joint_velocity_loss(squared, human, valid, face, face_active=True, face_weight=.5)
    assert parts['body'].item() == pytest.approx(2.)
    assert parts['face'].item() == pytest.approx(3.)
    assert total.item() == pytest.approx(3.5)
    total.backward()
    assert squared.grad[0, 2:].abs().sum() == 0
    assert squared.grad[0, 1] > squared.grad[0, 0]


def test_joint_high_noise_does_not_require_an_active_face():
    module = joint_module()
    squared = torch.ones(1, 4, 2, requires_grad=True)
    human = valid = torch.ones(1, 4, 1)
    total, parts = module.joint_velocity_loss(squared, human, valid, torch.zeros_like(valid), face_active=False)
    assert parts['face'].item() == 0
    total.backward()
    assert squared.grad.abs().sum() > 0


def test_joint_resume_rejects_changed_lora_or_sample_sequence():
    module = joint_module()
    signature = {'lora_sha256': 'abc', 'train_sha256': 'pairs', 'seed': 42}
    module.validate_joint_signature(signature, dict(signature))
    with pytest.raises(ValueError, match='lora_sha256'):
        module.validate_joint_signature(signature, {**signature, 'lora_sha256': 'def'})
    with pytest.raises(ValueError, match='train_sha256'):
        module.validate_joint_signature(signature, {**signature, 'train_sha256': 'other'})


def test_freezing_a_loaded_scriptmodule_keeps_input_gradients(tmp_path):
    module = joint_module()
    path = tmp_path / 'encoder.ts'
    torch.jit.trace(torch.nn.Linear(2, 2), torch.ones(1, 2)).save(str(path))
    evaluator = torch.jit.load(str(path))
    module.freeze_evaluator(evaluator)
    x = torch.ones(1, 2, requires_grad=True)
    evaluator(x).square().sum().backward()
    assert x.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in evaluator.parameters())


def test_preflight_rejects_disconnected_group_and_changed_initialization():
    module = joint_module()
    with pytest.raises(AssertionError, match='body_encoders'):
        module.validate_group_gradients({'body_heads': 2., 'body_encoders': 0., 'face_heads': 1.})
    with pytest.raises(AssertionError):
        module.validate_group_gradients({'body_heads': float('nan')}, branches=('body',))
    gate = {'passed': True, 'joint_signature': {'seed': 42}, 'initialization_sha256': 'original'}
    module.validate_preflight(gate, {'seed': 42}, 'original')
    with pytest.raises(ValueError, match='initialization'):
        module.validate_preflight(gate, {'seed': 42}, 'changed')


def test_locked_split_rejects_unapproved_manifest_and_validation_overlap(tmp_path):
    module = joint_module()
    common = [{'pair_id': f'c{i}', 'sequence_id': f'c{i}', 'target_stem': '1', 'source_video_group': 'train'} for i in range(256)]
    branch = [{'pair_id': f'b{i}', 'sequence_id': f'b{i}', 'target_stem': '1', 'source_video_group': 'train'} for i in range(256)]
    val = [{'pair_id': f'v{i}', 'source_video_group': 'val'} for i in range(24)]
    def write(name, rows):
        path = tmp_path / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path
    paths = [write('common256_pairs.jsonl', common), write('branch256_pairs.jsonl', branch),
             write('train512_pairs.jsonl', common + branch), write('val24_pairs.jsonl', val)]
    lock = {'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (tmp_path / 'lock.json').write_text(json.dumps(lock))
    assert module.validate_locked_split(paths[0], paths[3])['role'] == 'common'
    with pytest.raises(ValueError, match='role'):
        module.validate_locked_split(paths[2], paths[3])
    val[0]['source_video_group'] = 'train'
    write('val24_pairs.jsonl', val)
    lock['sha256']['val24_pairs.jsonl'] = hashlib.sha256(paths[3].read_bytes()).hexdigest()
    (tmp_path / 'lock.json').write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='overlap'):
        module.validate_locked_split(paths[0], paths[3])


def test_resume_state_requires_exact_optimizer_tensors_and_rng():
    module = joint_module()
    state = {'step': torch.tensor(2.), 'moments': [torch.tensor([.2, .3])], 'seed': (42, None)}
    module.assert_exact_state(state, {'step': state['step'].clone(), 'moments': [state['moments'][0].clone()], 'seed': (42, None)})
    with pytest.raises(AssertionError, match='moments'):
        module.assert_exact_state(state, {**state, 'moments': [torch.tensor([.2, .4])]})

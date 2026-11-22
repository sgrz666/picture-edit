import numpy as np
import pytest
import torch

from tests.test_native_v62_integration import control_case, installed
from tests.test_native_v7_integration import installed as installed_v7, installed62
from src.pose_control.qwen21_full.native_rl.objective import advantages, clipped_loss, transition
from src.pose_control.qwen21_full.native_rl.rewards import ImageRewardBank, PersonPose, RewardTarget, score_candidate
from src.pose_control.qwen21_full.native_rl.runner import FrozenReference, NativeRollout, Replay, update_policy


def test_transition_replays_stored_bf16_action():
    x = torch.randn(1, 8, 4).bfloat16()
    velocity = torch.randn(x.shape, requires_grad=True)
    sample, old, _, _ = transition(x, velocity.detach(), .7, .5, generator=torch.Generator().manual_seed(3))
    _, new, _, _ = transition(x, velocity, .7, .5, action=sample)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    new.backward()
    assert velocity.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='interior'):
        transition(x, velocity, 1., .5, action=sample)


def test_gdpo_masks_eligibility_and_records_low_variance():
    values, stats = advantages(
        [{'pose': [.2, .8], 'identity': [.5001, .5002]}, {'pose': [.6, .2]}],
        {'pose': .5, 'identity': .5}, std_floor=.001,
    )
    torch.testing.assert_close(values, torch.tensor([[-.5, .5], [1., -1.]]) / .625**.5)
    assert stats[0]['identity']['low_variance'] and stats[1]['identity']['applicable'] is False
    baseline, _ = advantages([{'pose': [.2, .8], 'identity': [.5001, .5002]}, {'pose': [.6, .2]}],
                             {'pose': .5, 'identity': .5}, mode='gdpo', std_floor=.001)
    assert baseline[0, 1] > baseline[1, 0]
    with pytest.raises(ValueError, match='finite'):
        advantages([{'pose': [.2, float('nan')]}], {'pose': 1.})


def test_clipping_and_frozen_reference_penalty():
    new = torch.tensor([.4, -.4], requires_grad=True)
    mean = torch.ones(2, 4, requires_grad=True)
    loss, diagnostics = clipped_loss(new, torch.zeros(2), torch.tensor([1., -1.]), mean, torch.zeros_like(mean), .5, clip=.1, beta=.001)
    assert diagnostics['clip_fraction'] == 1.
    loss.backward()
    assert new.grad.abs().sum() == 0 and mean.grad.abs().sum() > 0


def test_dual_identity_swap_and_missing_person_penalty():
    def pose(x):
        points = np.column_stack((np.linspace(x - 8, x + 8, 17), np.linspace(20, 80, 17)))
        return PersonPose(points, np.ones(17, dtype=bool), np.array([x - 10, 15, x + 10, 85]))

    right, left = pose(75), pose(25)
    a, b = np.eye(2)
    target = RewardTarget((right, left), (a, b), (100, 100))
    correct, details = score_candidate(target, (left, right), (b, a))
    swapped, _ = score_candidate(target, (left, right), (a, b))
    missing, miss = score_candidate(target, (right,), (None,))
    assert correct['pose'] == pytest.approx(1)
    assert correct['binding'] > swapped['binding']
    assert details['id_swap'] is False
    assert missing['pose'] == pytest.approx(.35)
    assert missing['identity'] == 0 and miss['id_missing'] == 2


def test_reward_evaluators_see_composited_qwen_rgba():
    from PIL import Image

    bank = ImageRewardBank.__new__(ImageRewardBank)
    pose = PersonPose(np.zeros((17, 2)), np.ones(17, dtype=bool), np.array([0, 0, 32, 32]))

    def detect(image, canvas):
        assert image.mode == 'RGB' and image.getpixel((0, 0)) == (128, 128, 128)
        return ()

    bank._poses = detect
    bank._faces = lambda image, boxes: ()
    reward, _ = bank.score(RewardTarget((pose,), (None,), (32, 32)), None,
                           Image.new('RGBA', (32, 32), (255, 0, 0, 0)))
    assert reward['pose'] == 0


def test_dual_hands_and_people_are_reported_in_target_slots():
    def person_at(x):
        whole = np.tile([x, 40.], (134, 1))
        return PersonPose(whole[:17], np.ones(17, bool), np.array([x - 9, 20, x + 9, 60]),
                          whole, np.ones(134, bool))

    left, right = person_at(20.), person_at(80.)
    target = RewardTarget((left, right), (None, None), (100, 100))
    _, both = score_candidate(target, (right, left), (None, None))
    _, missing = score_candidate(target, (right,), (None,))
    assert both['two_person_detected'] and not missing['two_person_detected']
    assert both['person0_hand_pck_002'] == both['person1_hand_pck_002'] == 1.
    assert missing['person0_hand_pck_002'] == 0. and missing['person1_hand_pck_002'] == 1.
    assert both['occlusion_consistency'] is None


def test_native_reward_bank_scores_only_eligible_source_background():
    from PIL import Image
    from src.pose_control.qwen21_full.native_rl.background import SourceBackground

    bank = ImageRewardBank.__new__(ImageRewardBank)
    pose = PersonPose(np.zeros((17, 2)), np.ones(17, bool), np.array([0, 0, 32, 32]))
    bank._poses = lambda *args: (pose,)
    bank._faces = lambda *args: (None,)
    target = RewardTarget((pose,), (None,), (32, 32))
    black = Image.new('RGB', (32, 32))
    assert 'background' not in bank.score(target, None, black)[0]
    source = SourceBackground(np.zeros((32, 32, 3)), np.ones((32, 32), bool))
    target = RewardTarget((pose,), (None,), (32, 32), background=source, background_status='aligned')
    assert bank.score(target, None, black)[0]['background'] == 1.
    scores, details = bank.score(target, None, Image.new('RGB', (32, 32), (255, 255, 255)))
    assert scores['background'] < 1e-6 and details['source_background_l1'] == details['source_background_mse'] == 1.


@pytest.mark.parametrize('scene, expected', [('single', .15), ('dual', .10)])
def test_masked_l1_default_weights(scene, expected):
    from src.pose_control.qwen21_full.native_rl.cli import _reward_weights

    weights = _reward_weights(scene, pose_only=False)
    assert weights['background'] == expected and sum(weights.values()) == pytest.approx(1.)
    assert 'background' not in _reward_weights(scene, pose_only=False, background_weight=0.)


def test_background_source_content_mapping_and_two_person_exclusion():
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_rl.background import prepare_source_background

    source, output = native_pad((64, 64)), native_pad((128, 128))
    a = Image.new('RGB', (64, 64), (10, 20, 30))
    mask_a = np.zeros((64, 64), bool)
    mask_a[30:34, 30:34] = True
    target = np.zeros((128, 128), bool)
    target[35:41, 20:25] = True
    target[80:86, 95:100] = True
    reward, reason = prepare_source_background(a, source, output, mask_a, target)
    assert reason is None and reward.mapping == 'content_resize'
    generated = np.asarray(a.resize((128, 128))).copy()
    generated[60:68, 60:68] = 255
    generated[target] = 255
    assert reward.score(generated)['background_l1'] == 0
    generated[0:10, 0:10] = 255
    assert reward.score(generated)['background_l1'] > 0
    assert prepare_source_background(a, source, native_pad((128, 64)), mask_a,
                                     np.ones((128, 64), bool))[0] is None


@pytest.mark.parametrize('profile', ['single_2ref', 'dual_4ref', 'dual_source_5ref'])
def test_native_background_is_bound_to_source_a_for_each_profile(profile):
    from types import SimpleNamespace as Box
    from PIL import Image
    from src.data.canvas_contract import native_pad
    from src.pose_control.qwen21_full.native_rl.background import native_background

    canvas = native_pad((128, 128))
    meta = {'source_rgb_sha256': 'a', 'driver_rgb_sha256': 'b', 'source_canvas_id': canvas.fingerprint(),
            'target_canvas_id': canvas.fingerprint()}
    dual = profile != 'single_2ref'
    local = Box(metadata=[meta], person_valid=torch.tensor([[True, dual]]), source_indices=torch.tensor([[1, 0]] if dual else [[0, -1]]))
    source = torch.zeros(2, 1, 128, 128, dtype=torch.bool)
    source[0, 0, 20:25, 20:25] = True
    source[1, 0, 90:96, 90:96] = True
    targets = [np.zeros((128, 128), bool) for _ in range(1 + dual)]
    targets[0][20:25, 90:95] = True
    if dual:
        targets[1][90:95, 20:25] = True
    condition = {'primary_source_index': 0, 'target_canvas_id': canvas.fingerprint(),
                 'images': [{'role': 'source_a' if profile == 'dual_source_5ref' else 'source',
                             'canvas_id': canvas.fingerprint(), 'file': {'sha256': 'a'}}]}
    sample = {'source_person_mask': source, 'source_canvas': canvas.metadata(), 'target_canvas': canvas.metadata(),
              'local_condition_v4': local}
    if profile == 'dual_source_5ref':
        sample['source_bank_v5'] = Box(sources=(Box(qwen_image_index=0, rgb_sha256='a', person_mask=source[None, :1]),
                                                Box(qwen_image_index=1, rgb_sha256='other', person_mask=source[None, 1:])))
    a = Image.new('RGB', (128, 128), (15, 30, 90))
    bg, reason = native_background(sample, condition, a, targets)
    assert reason is None and bg.score(a)['background_reward'] == 1.
    assert not bg.support[22, 22] and not bg.support[22, 92]
    assert bool(bg.support[92, 92]) == (profile != 'dual_4ref')
    if dual:
        assert not bg.support[92, 22]
    with pytest.raises(ValueError, match='primary Source A'):
        native_background(sample, {**condition, 'primary_source_index': 1}, a, targets)


def test_policy_updates_only_selected_parameter_and_rejects_bad_replay():
    active = torch.nn.Parameter(torch.tensor(.1))
    frozen = torch.nn.Parameter(torch.tensor(.4), requires_grad=False)
    x = torch.ones(1, 4)
    actions = []
    for seed in (1, 2):
        y, old, _, _ = transition(x, x * (active.detach() + frozen), .7, .5, generator=torch.Generator().manual_seed(seed))
        actions.append(Replay(x, y, old, 700., .7, .5))
    optimizer = torch.optim.AdamW([active], lr=.001)

    def predict(row):
        return row.state * (active + frozen)

    before = active.detach().clone()
    metrics = update_policy(optimizer, [(actions, torch.tensor([1., -1.]), predict)], [active], [frozen], beta=0.)
    assert active != before and frozen == .4
    assert metrics['max_ratio_error'] == 0 and metrics['grad_norm'] > 0
    with pytest.raises(RuntimeError, match='Replay parity'):
        update_policy(optimizer, [(actions, torch.tensor([1., -1.]), predict)], [active], [frozen], beta=0.)


@pytest.mark.parametrize('architecture', ['installed', 'installed_v7'])
def test_native_scheduler_rollout_replay_with_installed_qwen(architecture, request, monkeypatch):
    import os
    from types import SimpleNamespace

    pytest.importorskip('transformers')
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from PIL import Image
    from src.pose_control.qwen21_full.native_v62.generation import generate_native

    _, _, model, _, runtime = request.getfixturevalue(architecture)
    roles = ('source', 'target_pose')
    _, control, options, inputs = control_case(roles)
    device = torch.device(os.getenv('NATIVE_RL_TEST_DEVICE', 'cpu'))
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    # Real packed Qwen, V6.2 binding and diffusion scheduler with a lightweight deterministic decoder.
    class Decoder(torch.nn.Linear):
        config = SimpleNamespace(latents_mean=[0.] * 64, latents_std=[1.] * 64)

        @property
        def dtype(self):
            return self.weight.dtype

        def decode(self, latent, **kwargs):
            return (latent[:, :3].clamp(-1, 1),)

    def postprocess(value, **kwargs):
        rgb = ((value[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).astype(np.uint8)
        return [Image.fromarray(rgb)]

    model.to(device)
    if device.type == 'cuda':
        for p in model.parameters():
            if id(p) not in {id(q) for q in runtime.parameters()}:
                p.data = p.data.to(dtype)
    model.enable_gradient_checkpointing()
    prepared = control.to(device=device)
    monkeypatch.setattr(runtime, 'encode', lambda *args, **kwargs: prepared)
    pipe = SimpleNamespace(transformer=model, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
                           vae=Decoder(1, 1), image_processor=SimpleNamespace(postprocess=postprocess),
                           vae_scale_factor=16, _pack_latents=QwenImage21Pipeline._pack_latents,
                           _unpack_latents=QwenImage21Pipeline._unpack_latents)
    pair = dict(qwen_condition_profile='single_2ref', reference_latents=inputs['hidden_states'][:, :16].to(dtype),
                reference_roles=roles, reference_canvas_ids=('canvas', 'canvas'), target_canvas_id='canvas',
                target_token_hw=(2, 4), source_token_hw=(2, 4), reference_token_hws=((2, 4), (2, 4)),
                source_latent=inputs['hidden_states'][:, :8].to(dtype), img_mask=inputs['img_mask'],
                prompt_embeds=inputs['encoder_hidden_states'].to(dtype), prompt_mask=inputs['encoder_hidden_states_mask'])
    batch = dict(local_condition_v4=SimpleNamespace(metadata=[{'target_canvas_id': 'canvas'}]),
                 valid_image_mask=control.spatial.valid_pixels)
    base = SimpleNamespace(identity=False)
    rollout = NativeRollout(SimpleNamespace(pipe=pipe, runtime=runtime, device=device), batch, pair,
                            SimpleNamespace(poselink_strength=1.), options, base, steps=3, sde_step=1, noise_level=.7)
    image, _ = rollout.generate(3)
    native = generate_native(pipe, runtime, batch, pair, options=options, base_options=base, seed=3, steps=3)
    assert np.array_equal(np.asarray(image), np.asarray(native))
    rows = [rollout.generate(seed, stochastic=True)[1] for seed in (3, 4)]
    active = list(runtime.poselink.output_heads.parameters())
    for p in model.parameters():
        p.requires_grad_(any(p is a for a in active))
    frozen = [p for p in model.parameters() if not p.requires_grad]
    sft = FrozenReference(active)
    with torch.no_grad(), sft.use():
        means = {id(r): transition(r.state, rollout.predict(r), r.sigma, r.next_sigma, action=r.action)[2] for r in rows}
    optim = torch.optim.AdamW(active, lr=.003)
    replay = [(rows, torch.tensor([-1., 1.]), rollout.predict)]
    first = update_policy(optim, replay, active, frozen, reference_means=means)
    assert first['max_ratio_error'] == 0 and first['grad_norm'] > 0 and first['kl'] == 0
    second = update_policy(optim, replay, active, frozen, reference_means=means, check_parity=False)
    assert second['kl'] > 0
    with torch.no_grad(), sft.use():
        assert all(torch.equal(p, ref) for p, ref in zip(active, sft.weights))
def test_sum_keeps_low_variance_components_before_normalization():
    result, _ = advantages([{'pose': [.0, .1, .2], 'face': [.5002, .5001, .5]}],
                           {'pose': .01, 'face': 20.}, mode='sum', std_floor=.001)
    assert result[0, 0] > 1 and result[0, 2] < -1


def test_rl_selects_matching_architecture_and_disables_sft_auxiliaries(tmp_path):
    import json
    from types import SimpleNamespace

    from src.pose_control.qwen21_full.native_rl.cli import _settings, _backend_for
    from src.pose_control.qwen21_full.native_v7.training import NativeBackendV7, TrainingConfigV7, stage_preset_v7

    config = TrainingConfigV7(stage=stage_preset_v7('S3'),
                              initialization={'path': '/data/parent.pt', 'sha256': '0' * 64})
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config.record()))
    rl, route = _settings(SimpleNamespace(config=path, action='check-data', route=None))
    assert _backend_for(rl)[1:] == (NativeBackendV7, 'native_full_v7')
    assert not rl.additional_evaluators and route.loss.identity == 0


def test_replay_epochs_stop_on_optimizer_update_budget():
    from src.pose_control.qwen21_full.native_rl.runner import RunCounters

    counters = RunCounters()
    counters.rollout(4)
    assert counters.epochs(optimizer_budget=1, replay_epochs=2) == 1
    counters.updated(4)
    assert counters.epochs(optimizer_budget=1, replay_epochs=2) == 0
    assert counters.record() == dict(optimizer_updates=1, rollout_batches=1, generated_images=4, replayed_transitions=4)

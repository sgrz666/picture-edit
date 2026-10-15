"""Read-only adapter audit probes. Run from the remote project with its Qwen environment.

No optimizer step, pretrained model load, or project/checkpoint write is performed.
Only the cache probe writes into an automatically removed temporary directory.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("audit_bridge_fixture", ROOT / "tests/test_qwen21_full_bridge.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

from src.pose_control.qwen21_full.execution import parameter_groups
from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullAdapter, Qwen21FullControlContext
from src.pose_control.qwen21_full.qwen_runtime import Qwen21IPERCache

torch.set_num_threads(2)
torch.manual_seed(731)
report = {}


class NearIdentityQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(64, 16)
        self.transformer_blocks = nn.ModuleList(nn.Linear(16, 16) for _ in range(32))
        for block in self.transformer_blocks:
            nn.init.eye_(block.weight)
            nn.init.zeros_(block.bias)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, img_shapes, **kwargs):
        value = self.img_in(hidden_states)
        for block in self.transformer_blocks:
            value = checkpoint(block, value, use_reentrant=False) if self.gradient_checkpointing else block(value)
        return (value,)


def compare_checkpoint_recomputation():
    model = fixture.make_model()
    nn.init.normal_(model.output_bridges['geometry'].weight, std=.02)
    initial = copy.deepcopy(model.state_dict())
    backbone = NearIdentityQwen().requires_grad_(False)
    backbone_initial = copy.deepcopy(backbone.state_dict())
    first = fixture.fake_inputs()
    second = {key: value.clone() if torch.is_tensor(value) else value for key, value in first.items()}
    second['hidden_states'] = torch.randn_like(first['hidden_states'])
    second['timestep'] = torch.tensor([.2])
    second['encoder_hidden_states'] = torch.randn_like(first['encoder_hidden_states'])

    def run(checkpointing, two_forwards):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        backbone.load_state_dict(backbone_initial)
        backbone.gradient_checkpointing = checkpointing
        torch.manual_seed(812)
        prepared = fixture.make_prepared(model)
        with set_checkpoint_early_stop(False), Qwen21FullControlContext(backbone, model, prepared):
            a = backbone(**first)[0][:, -16:]
            loss = a.square().mean()
            if two_forwards:
                b = backbone(**second)[0][:, -16:]
                loss = loss + b.square().mean()
            loss.backward()
        return float(loss.detach()), model.output_bridges['geometry'].weight.grad.detach().clone()

    for count in (1, 2):
        eager_loss, eager_gradient = run(False, count == 2)
        try:
            checkpoint_loss, checkpoint_gradient = run(True, count == 2)
            error = float((checkpoint_gradient - eager_gradient).abs().max())
            report[f'checkpoint_{count}_forward'] = {
                'eager_loss': eager_loss, 'checkpoint_loss': checkpoint_loss,
                'gradient_max_abs_difference': error,
                'gradient_allclose': bool(torch.allclose(checkpoint_gradient, eager_gradient, rtol=1e-4, atol=1e-6)),
            }
        except Exception as exc:
            report[f'checkpoint_{count}_forward'] = {'error': str(exc), 'exception': type(exc).__name__}


def cache_identity_probe():
    image = Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))
    with tempfile.TemporaryDirectory(prefix='qwen_adapter_audit_') as folder:
        cache = Qwen21IPERCache(folder)
        cached = {'reference_latent': torch.full((1, 1024, 64), 123.)}
        cache._store(cache._path(image, 'source'), cached)
        # A cache hit never inspects the newly selected model's configuration,
        # revision, encoder normalization, or actual encoder tensors.
        selected_other_model = SimpleNamespace(config=SimpleNamespace(_name_or_path='different-model'))
        result = cache.source(selected_other_model, image)
        report['cache_different_model'] = {
            'selected_model': 'different-model', 'returned_previous_cache': bool(torch.equal(result['reference_latent'], cached['reference_latent'])),
            'cached_pipeline_accessed': False,
        }


def mixed_optional_branches_probe():
    model = fixture.make_model(face=True)
    prepared = fixture.make_prepared(model, batch=2)
    face = fixture.prepared_face().expand_to_batch(2)
    prepared = replace(prepared, face=face)
    args = dict(noisy_target=torch.randn(2, 16, 64), prepared=prepared,
                prompt_embeds=torch.randn(2, 5, 24), prompt_mask=torch.ones(2, 5, dtype=torch.bool),
                sigma=torch.tensor([.8, .2]), target_token_hw=(4, 4))
    result = model.build_residuals(**args)
    report['mixed_face_activity'] = {'face_active_rows': result.diagnostics['face_active_rows'],
                                     'first_row_raw_face_max_abs': float(result.raw_residuals.face[0][0].detach().abs().max()),
                                     'second_row_raw_face_max_abs': float(result.raw_residuals.face[0][1].detach().abs().max())}


def native_transformer_probe():
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    legacy = fixture.make_model().legacy
    adapter = Qwen21FullAdapter(legacy, qwen_hidden_size=32, qwen_text_dim=24, pooled_input_dim=20)
    nn.init.normal_(adapter.output_bridges['geometry'].weight, std=.02)
    native = QwenImage21Transformer2DModel(in_channels=64, out_channels=64,
                                         num_attention_heads=2, attention_head_dim=16,
                                         context_in_dim=24, num_layers=32, axes_dims_rope=(4, 6, 6),
                                         causal_condition=True).eval().requires_grad_(False)
    inputs = dict(hidden_states=torch.randn(1, 32, 64), timestep=torch.tensor([.5]),
                  encoder_hidden_states=torch.randn(1, 7, 24), encoder_hidden_states_mask=None,
                  img_shapes=[[(1, 4, 4), (1, 4, 4)]],
                  img_mask=torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool),
                  return_dict=False)
    prepared = fixture.make_prepared(adapter)
    with torch.no_grad():
        baseline = native(**inputs)[0]
        with Qwen21FullControlContext(native, adapter, prepared):
            controlled = native(**inputs)[0]
        cache = QwenImage21KVCache(32)
        with Qwen21FullControlContext(native, adapter, prepared):
            native(**inputs, kv_cache=cache, kv_cache_mode='extract')
            cached = native(**inputs, kv_cache=cache, kv_cache_mode='cached')[0]
    native.enable_gradient_checkpointing()
    adapter.zero_grad(set_to_none=True)
    with Qwen21FullControlContext(native, adapter, prepared):
        native(**inputs)[0][:, -16:].square().mean().backward()
    report['native_tiny_qwen32'] = {
        'prefix_exactly_preserved': bool(torch.equal(baseline[:, :-16], controlled[:, :-16])),
        'controlled_target_changed': bool(not torch.equal(baseline[:, -16:], controlled[:, -16:])),
        'cached_uncached_target_max_abs': float((cached[:, -16:] - controlled[:, -16:]).abs().max()),
        'checkpoint_output_bridge_gradient_norm': float(adapter.output_bridges['geometry'].weight.grad.norm()),
    }


def defaults_and_supervision_probe():
    from src.pose_control.qwen21_full.execution import face_velocity_loss
    training_spec = importlib.util.spec_from_file_location('audit_train_entry', ROOT / 'train_iper_qwen21_full.py')
    training = importlib.util.module_from_spec(training_spec)
    training_spec.loader.exec_module(training)
    previous = sys.argv
    try:
        sys.argv = ['train_iper_qwen21_full.py', '--branch', 'face']
        args = training.arguments()
    finally:
        sys.argv = previous
    mask = torch.zeros(1, 1024, 1)
    mask[:, :6] = 1
    squared = torch.ones(1, 1024, 64, requires_grad=True)
    global_objective = face_velocity_loss(squared, mask, 'global')[0]
    roi_objective = face_velocity_loss(squared, mask, 'roi')[0]
    global_gradient = torch.autograd.grad(global_objective, squared, retain_graph=True)[0][0, 0, 0]
    roi_gradient = torch.autograd.grad(roi_objective, squared)[0][0, 0, 0]
    report['face_training_defaults'] = {
        'evaluator_range': args.face_evaluator_range,
        'flow_loss': args.face_flow_loss, 'alignment': args.face_aux_alignment,
        'auxiliary_every': args.face_aux_every,
        'example_face_token_fraction': float(mask.mean()),
        'roi_vs_global_face_gradient_ratio': float(roi_gradient / global_gradient),
    }
    perceptual = torch.jit.load(str(ROOT / 'models/face_eval/perceptual_encoder.ts'), map_location='cpu').eval()
    identity = torch.jit.load(str(ROOT / 'models/face_eval/identity_encoder.ts'), map_location='cpu').eval()
    report['saved_evaluator_forward_code'] = {'identity': identity.code, 'perceptual': perceptual.code}


def schedule_probe():
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    from src.pose_control.qwen21_full.interface.strength import StrengthScheduleConfig
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(ROOT / 'models/Qwen-Image-2.1'), subfolder='scheduler')
    mu = calculate_shift(1024, scheduler.config.base_image_seq_len, scheduler.config.max_image_seq_len,
                         scheduler.config.base_shift, scheduler.config.max_shift)
    scheduler.set_timesteps(30, device='cpu', sigmas=np.linspace(1., 1 / 30, 30), mu=mu)
    model = fixture.make_model(face=True)
    schedule = model.legacy.control_interface.strength_controller.face_schedule
    rows = []
    for index in range(10, 15):
        sigma = scheduler.timesteps[index] / 1000
        progress = torch.tensor(index / 29)
        rows.append({'index': index, 'sigma': float(sigma), 'train_progress': float(1 - sigma),
                     'default_inference_progress': float(progress),
                     'train_face_multiplier': float(schedule.multiplier(1 - sigma)),
                     'default_inference_face_multiplier': float(schedule.multiplier(progress))})
    report['schedule_512_30_steps'] = rows


def dual_and_hand_bridge_probe():
    from src.pose_control.qwen21_full.conditions import ConditionBundle
    from src.pose_control.qwen21_full.deepgen_adapter import UnifiedSMPLXAdapterV6
    from src.pose_control.qwen21_full.hand.conditions import HandFineCondition, HandReferenceFeatures
    sys.path.insert(0, str(ROOT / 'tests'))
    import test_adapter_reasoning_v1 as reasoning_fixture
    import test_v66_hand_core as hand_fixture
    legacy = UnifiedSMPLXAdapterV6.build_tiny(hidden_dim=32, geometry_channels=16,
                                            context_input_dim=24, pooled_input_dim=20,
                                            num_heads=4, enable_hand_adapter=True, face_dim=32)
    for head in legacy.control_core.geometry_zero_heads:
        nn.init.normal_(head.weight, std=.02)
    for head in legacy.control_core.interaction_zero_heads:
        nn.init.normal_(head.weight, std=.02)
    for head in legacy.control_interface.hand_adapter.zero_heads:
        nn.init.normal_(head.weight, std=.02)
    model = Qwen21FullAdapter(legacy, qwen_hidden_size=16, qwen_text_dim=24, pooled_input_dim=20)
    for name in ('geometry', 'interaction', 'hand'):
        nn.init.normal_(model.output_bridges[name].weight, std=.02)
    bundle = ConditionBundle(**vars(reasoning_fixture._make_bundle(counts=(2,))))
    source = torch.randn(1, 16, 64)
    error = None
    try:
        model.prepare_conditioning(condition_bundle=bundle, source_latent=source, target_token_hw=(4, 4))
    except ValueError as exc:
        error = str(exc)
    prepared = model.prepare_conditioning(condition_bundle=bundle, source_latent=source,
                                          source_person_latents=torch.randn(1, 2, 16, 64),
                                          target_token_hw=(4, 4))
    dual = model.build_residuals(noisy_target=torch.randn(1, 16, 64), prepared=prepared,
                                prompt_embeds=torch.randn(1, 5, 24), prompt_mask=None,
                                sigma=torch.tensor([.2]), target_token_hw=(4, 4), hand_mode='off')
    hand_condition, hand_references = hand_fixture._inputs()
    hand_condition = HandFineCondition(**vars(hand_condition))
    hand_references = HandReferenceFeatures(**vars(hand_references))
    hand_prepared = model.legacy.hand_preparer(hand_condition, hand_references)
    single_prepared = replace(fixture.make_prepared(model), hand=hand_prepared)
    hand_args = dict(noisy_target=torch.randn(1, 16, 64), prepared=single_prepared,
                     prompt_embeds=torch.randn(1, 5, 24), prompt_mask=None,
                     sigma=torch.tensor([.2]), target_token_hw=(4, 4))
    on = model.build_residuals(**hand_args, hand_mode='v66')
    off = model.build_residuals(**hand_args, hand_mode='off')
    report['full_bridge_dual_and_hand_synthetic'] = {
        'dual_missing_person_latents_rejected': error,
        'dual_interaction_valid': bool(prepared.interaction_valid.item()),
        'dual_raw_interaction_max_abs': float(dual.raw_residuals.interaction[0].detach().abs().max()),
        'hand_active_rows': on.diagnostics['hand_active_rows'],
        'hand_enabled_vs_off_residual_max_abs': max(float((a - b).detach().abs().max()) for a, b in zip(on.residuals, off.residuals)),
        'trained_quality_claim': False,
    }


compare_checkpoint_recomputation()
cache_identity_probe()
mixed_optional_branches_probe()
native_transformer_probe()
defaults_and_supervision_probe()
schedule_probe()
dual_and_hand_bridge_probe()
print(json.dumps(report, indent=2))

"""Immutable scope and config builder for the explicitly authorized single-person run."""
from copy import deepcopy
from pathlib import Path

DEFAULT_RUN = '/data/shangguanrz/pic-edit-experiments/bilibili-single-spatial-train-20261107-v1'
PHASES = ('S0', 'S1', 'S2', 'S3')
BUDGETS = (10000, 2000, 4000, 4000)
# These are known Native V5 groups whose inputs are absent in single-person data.
INACTIVE_GROUPS = ('geometry_dual', 'relation_raster', 'relation_tokens',
                   'interaction_controller', 'interaction_heads', 'strength_interaction')


def make_plan(run_root, repo_root, authorization):
    if (not isinstance(authorization, dict) or authorization.get('source') != 'direct_user_message'
            or not isinstance(authorization.get('text'), str) or not authorization['text'].strip()):
        raise ValueError('Record the actual direct user authorization text before launch')
    return {'schema': 'bilibili_single_supervisor_plan_v1', 'run_root': str(run_root),
            'repo_root': str(repo_root), 'authorization': deepcopy(authorization),
            'stages': [{'phase': p, 'budget_updates': b} for p, b in zip(PHASES, BUDGETS)],
            'total_updates': sum(BUDGETS), 'max_retries': 2, 'gpu_free_gib': 30,
            'host_available_gib': 10, 'resource_poll_seconds': 60,
            'counts': {'pairs': 12156, 'train': 9520, 'validation': 1490, 'test': 1146},
            'canvas_long_side': 512, 'seed': 42, 'initialization_kind': 'native_v5_init_v2',
            'geometry_version': 'sapiens_v1', 'hand_policy': 'visible_region_v2',
            'spatial_frames': 1960, 'technical_only': True}


def stage_config(plan, prepared, phase, *, initialization, cache_identity=None, body_gate=None):
    stage = next(s for s in plan['stages'] if s['phase'] == phase)
    branches = ('body',) if phase == 'S0' else ('joint',) if phase == 'S3' else ('face', 'hand', 'detail')
    loss = {'velocity': 1., 'face': 0., 'hand': 0., 'identity': 0., 'perceptual': 0.,
            'spatial_body': .5, 'spatial_joint': .5, 'spatial_hand': 1., 'spatial_max': 2.5}
    # Strictly retain only train/validation sources; the separate test declaration is never copied.
    sources = deepcopy(prepared['sources'])
    if not sources or any(s['role'] not in ('train', 'validation') or s['source_id'] == 'bilibili_test' for s in sources):
        raise ValueError('Training sources must contain only the declared train/validation split')
    return {'schema': 'native_v5_training_v1', 'stage': {
                'stage_id': phase+'_bilibili_single_spatial', 'phase': phase,
                'budget_updates': stage['budget_updates'], 'accumulation': 4,
                'unfreeze_geometry': False, 'exclude_groups': list(INACTIVE_GROUPS),
                'optimizer': {'preset': 'protect_inherited', 'multiplier': .1 if phase == 'S3' else 1.},
                'routes': [{'name': b, 'branch': b, 'weight': 1., 'loss': deepcopy(loss),
                            'controls': {'interaction': False},
                            'base_controls': {'interaction': False, 'raster': False, 'part_contact': False}}
                           for b in branches]},
            'sources': sources, 'initialization': deepcopy(initialization),
            'model_path': prepared['model_path'], 'model_manifest': deepcopy(prepared['model_manifest']),
            'cache_root': prepared['cache_root'], 'cache_identity': deepcopy(cache_identity),
            'output_root': str(Path(plan['run_root'])/'stages'/phase),
            'body_baseline_report': deepcopy(body_gate), 'seed': 42,
            'micro_batch': 1, 'num_workers': 0, 'world_size': 1,
            'save_every_updates': 100, 'probe_every_updates': 100, 'probe_samples': 10,
            'generation_every_updates': 1000, 'generation_seed': 42, 'generation_steps': 20,
            'qwen_condition_profile': prepared.get('bindings', {}).get('qwen_condition_profile', 'single_2ref'),
            'gradient_case_budget': None, 'validation_loss_limit_ratio': 2., 'validation_failure_patience': 3,
            'continuous_stage_authorization': deepcopy(plan['authorization'])}

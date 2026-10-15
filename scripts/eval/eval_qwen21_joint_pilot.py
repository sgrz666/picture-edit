#!/usr/bin/env python3
"""Sample one frozen Qwen/BFS base with the joint Body+Face controller.

One invocation loads exactly one adapter checkpoint and at most one LoRA.
B RGB is exported for offline comparison and is never passed to the pipeline.
Image paths in records are relative to the evaluation directory, so the full
output tree can be copied to another computer without rewriting manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN_ROOT = Path('/home/shangguanrz/nihaochen/qwen21_full')

from src.pose_control.qwen21_full.prompt_study.protocol import SCHEMA, ARMS, EVAL_STEPS
from src.pose_control.qwen21_full.prompt_study.prompts import PromptBank
from src.pose_control.qwen21_full.data_runtime import (
    pair_metadata, study_prompt_signature, study_asset_signature, study_code_signature, validate_study_cache,
)


def validate_study_inputs(args, dataset):
    from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock, validate_study_binding, require_eight_arms_completed
    if args.study_split == 'test':
        require_eight_arms_completed(args.study_lock.parent.parent)
    lock = load_study_lock(args.study_lock)
    split = lock['splits'][args.study_split]
    if args.pairs.resolve() != (args.study_lock.parent / split['path']).resolve():
        raise ValueError('Study evaluation pairs do not match the requested locked split')
    protocol = args.arm.split('_', 1)[1]
    run_dir = args.study_lock.parent.parent
    if args.prompt_manifest.resolve() != (run_dir / 'prompts' / protocol / f'{args.study_split}.jsonl').resolve():
        raise ValueError('Study evaluation prompt manifest does not match the requested split/protocol')
    if args.cache.resolve() != (run_dir / 'cache' / protocol).resolve():
        raise ValueError('Study evaluation cache does not match the protocol')
    validate_study_binding(lock, args.arm, digest_file(args.lora) if args.lora else None, args.model)
    bank = PromptBank(args.prompt_manifest, protocol, [pair_metadata(row)['pair_id'] for row in dataset.rows])
    return lock, bank


def validate_study_schedule(args, indices):
    from src.pose_control.qwen21_full.prompt_study.results import eval_jobs
    jobs = {(row['arm'], row['step'], row['split'], row['seed'], row['pair_index'])
            for row in eval_jobs(include_final_test=True)}
    if any((args.arm, args.study_step, args.study_split, args.seed, index) not in jobs for index in indices):
        raise ValueError('Study evaluation is outside the locked schedule')


def validate_study_checkpoint(args, lock, checkpoint, bank):
    from src.pose_control.qwen21_full.prompt_study.protocol import validate_study_binding
    from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION
    lock_sha = digest_file(args.study_lock)
    expected_header = {'study_schema': SCHEMA, 'arm': args.arm, 'step': args.study_step,
                       'study_lock_sha256': lock_sha, 'model_revision': MODEL_REVISION,
                       'joint_initialization_sha256': lock['initial_checkpoint_sha256'],
                       'next_update_index': args.study_step, 'protocol': bank.protocol,
                       'resolution': 'native', 'align_mode': 'native', 'max_updates': 1024,
                       'initial_history_steps': 256, 'branch': 'joint'}
    if any(checkpoint.get(key) != value for key, value in expected_header.items()):
        raise ValueError('Study checkpoint arm/step/schema/lock/model/initialization differs')
    if checkpoint.get('sample_order') != lock['epoch_order'][:args.study_step]:
        raise ValueError('Study checkpoint sample trajectory differs')
    lora_sha = digest_file(args.lora) if args.lora else None
    validate_study_binding(lock, args.arm, lora_sha, args.model)
    expected = {'schema': SCHEMA, 'study_schema': SCHEMA, 'arm': args.arm,
                'protocol': bank.protocol, 'study_lock_sha256': lock_sha,
                'max_updates': 1024, 'initial_history_steps': 256, 'model_revision': MODEL_REVISION,
                'model': str(args.model.resolve()), 'model_manifest_sha256': lock['artifacts']['model']['manifest_sha256'],
                'lora_sha256': lora_sha, 'strength': 1, 'resolution': 'native', 'align_mode': 'native',
                'preprocessing': 'native_v1', 'with_hand': False}
    expected.update(study_prompt_signature(bank))
    expected['instruction_sha256'] = digest_file(args.study_lock.parent.parent / 'prompts/instruction_v1.json')
    expected.update(study_asset_signature(args.features, args.study_lock.parent))
    expected.update(study_code_signature())
    signature = checkpoint.get('joint_signature', {})
    if any(signature.get(key) != value for key, value in expected.items()):
        raise ValueError('Study checkpoint training signature/template/runtime differs from evaluation')
    train_manifest = args.study_lock.parent.parent / 'prompts' / bank.protocol / 'train.jsonl'
    if signature.get('prompt_manifest_sha256') != digest_file(train_manifest):
        raise ValueError('Study checkpoint training prompt manifest differs')
    return expected


def validate_study_records(args, config, dataset, bank, records, indices):
    """Reusable outputs retain their exact locked pair, prompt and PNG bytes."""
    from src.pose_control.qwen21_full.prompt_study.results import validate_retained_artifacts
    for row in records:
        index = row['pair_index']
        if index not in indices:
            raise ValueError('Study result has an unexpected pair index')
        meta = pair_metadata(dataset.rows[index])
        expected = {'pair_id': meta['pair_id'], 'arm': args.arm, 'protocol': bank.protocol,
                    'study_step': args.study_step, 'study_split': args.study_split, 'seed': args.seed,
                    'checkpoint_sha256': config['checkpoint_sha256'], 'lora_sha256': config['lora_sha256'],
                    'prompt_sha256': bank.records[meta['pair_id']]['compiled_prompt_sha256']}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('Existing study result signature/pair/prompt differs')
        if any((row.get('metadata') or {}).get(key) != value for key, value in meta.items()):
            raise ValueError('Existing study result metadata differs from locked pair')
        if row.get('status') == 'complete':
            path = args.output / row['image']
            if not path.is_file() or digest_file(path) != row.get('image_sha256'):
                raise ValueError('Existing study result image is missing or changed')
            with Image.open(path) as saved:
                if [saved.width, saved.height] != row.get('image_size'):
                    raise ValueError('Existing study result image dimensions differ')
        validate_retained_artifacts(args.output, row)


def select_indices(selection, count):
    """Parse zero-based comma-separated indices and inclusive ranges unchanged."""
    if selection is None:
        return list(range(count))
    indices = []
    for item in selection.split(','):
        item = item.strip()
        if not re.fullmatch(r'\d+(?:-\d+)?', item):
            raise ValueError(f'Invalid index selection: {item!r}')
        parts = [int(value) for value in item.split('-')]
        if len(parts) == 2:
            if parts[0] > parts[1]:
                raise ValueError('Index ranges must be ascending')
            parts = list(range(parts[0], parts[1] + 1))
        indices.extend(parts)
    if len(indices) != len(set(indices)) or any(index >= count for index in indices):
        raise ValueError(f'Indices must be unique and in [0,{count})')
    return indices


def case_stem(index, metadata):
    """Stable readable filenames without platform-reserved path punctuation."""
    readable = '_'.join(str(metadata.get(key, 'unknown')) for key in
                        ('source_appearance', 'source_stem', 'target_stem'))
    readable = re.sub(r'[^A-Za-z0-9_-]+', '_', readable).strip('_')[:110] or 'pair'
    identity = {key: metadata.get(key) for key in ('pair_id', 'source_appearance', 'target_appearance', 'source_stem', 'target_stem')}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return f'{index:03d}_{readable}_{suffix}'


def crop_normalized(image, box, side=256):
    """Use normalized xyxy as supplied, without output-based detection/recentering."""
    if box is None or len(box) != 4:
        return None, None
    values = [float(value) for value in box]
    if (not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        return None, None
    pixels = (math.floor(values[0] * image.width), math.floor(values[1] * image.height),
              math.ceil(values[2] * image.width), math.ceil(values[3] * image.height))
    return image.crop(pixels).resize((side, side), Image.Resampling.LANCZOS), pixels


def image_strip(images, labels, side=256, missing='Unavailable'):
    canvas = Image.new('RGB', (side * len(images), side + 40), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for column, (image, label) in enumerate(zip(images, labels)):
        left = side * column
        draw.text((left + 6, 8), label, fill='#222222')
        if image is None:
            draw.rectangle((left, 40, left + side - 1, side + 39), fill='#48282a')
            draw.text((left + 8, side // 2), missing[:38], fill='white')
        else:
            tile = ImageOps.contain(image.convert('RGB'), (side, side), Image.Resampling.LANCZOS)
            canvas.paste(tile, (left + (side - tile.width) // 2, 40 + (side - tile.height) // 2))
    return canvas


def unpad_artifact_images(metadata, *, source, target, generated, normal, pose, target_mask):
    from src.pose_control.qwen21_full.face_coordinates import transform_from_record
    alignment = metadata.get('alignment', {})
    def original(image, side):
        return transform_from_record(alignment[side]).unpad(image) if image is not None and side in alignment else image
    return (original(source, 'source'), original(target, 'target'), original(generated, 'target'),
            original(normal, 'target'), original(pose, 'target'), original(target_mask, 'target'))


def load_source_person_mask(pair_row, metadata, source):
    """Read the original locked A mask for offline scoring without resizing."""
    path = pair_row.get('source_person_mask')
    if not path:
        return None
    alignment = metadata.get('alignment', {}).get('source')
    if alignment is not None:
        from src.pose_control.qwen21_full.face_coordinates import transform_from_record
        transform = transform_from_record(alignment)
        original_size = (transform.width, transform.height)
    else:
        original_size = source.size
    with Image.open(path) as image:
        if image.size != original_size:
            raise ValueError(f'Source A mask dimensions {image.size} differ from original A {original_size}')
        return image.convert('L')


def save_conditioning_artifacts(output, stem, *, metadata, normal, pose, body25):
    output = Path(output)
    (output / 'conditioning').mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, image in (('normal', normal), ('body25_preview', pose)):
        if image is not None:
            path = f'conditioning/{stem}_{key}.png'
            image.save(output / path)
            paths[f'conditioning_{key}'] = path
    if body25 is not None:
        path = f'conditioning/{stem}_body25.json'
        write_json(output / path, body25)
        paths['conditioning_body25'] = path
    for key, field in (('depth', 'target_depth_vis'), ('dwpose_json', 'target_dwpose_json'),
                       ('dwpose_original', 'target_dwpose_png'), ('dwpose_filtered', 'filtered_dwpose')):
        source = Path(metadata[field]) if metadata.get(field) else None
        if source is not None and source.is_file():
            path = f'conditioning/{stem}_{key}{source.suffix}'
            shutil.copyfile(source, output / path)
            paths[f'conditioning_{key}'] = path
    return paths


def save_case_artifacts(output, stem, *, source, target, generated, normal, pose,
                        source_box, target_box,
                        source_hand_box=None, target_hand_box=None,
                        target_lh_box=None, target_rh_box=None, failure=None,
                        target_mask=None, source_mask=None):
    """Export original aligned RGBs, controls, whole panels and fixed-ROI crops."""
    output = Path(output)
    paths = {}
    for folder in ('images', 'panels', 'faces', 'hands'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for key, image, suffix in (('source', source, 'source'), ('target', target, 'target'),
                               ('image', generated, 'on'), ('normal', normal, 'normal'), ('pose', pose, 'pose')):
        paths[key] = f'images/{stem}_{suffix}.png' if image is not None else None
        if image is not None:
            image.convert('RGB').save(output / paths[key])
    if target_mask is not None:
        paths['target_mask'] = f'images/{stem}_mask.png'
        target_mask.convert('L').save(output / paths['target_mask'])
    if source_mask is not None:
        paths['source_mask'] = f'images/{stem}_source_mask.png'
        source_mask.convert('L').save(output / paths['source_mask'])
    crops = []
    for key, image, box in (('source', source, source_box), ('target', target, target_box),
                            ('output', generated, target_box)):
        crop = crop_normalized(image, box)[0] if image is not None else None
        paths[f'face_{key}'] = f'faces/{stem}_{key}.png' if crop is not None else None
        if crop is not None:
            crop.save(output / paths[f'face_{key}'])
        crops.append(crop)
    paths['comparison'] = f'panels/{stem}_whole.jpg'
    image_strip([source, target, normal, pose, generated],
                ['Source A', 'Target B (offline)', 'B normal', 'B pose', 'Generated'],
                missing=failure or 'Unavailable').save(output / paths['comparison'], quality=92)
    paths['face_comparison'] = f'panels/{stem}_face.jpg'
    image_strip(crops, ['A source ROI', 'B target ROI (offline)', 'Output at B target ROI'],
                missing=failure or 'No valid face ROI').save(output / paths['face_comparison'], quality=92)

    # Hand crops
    hand_box = target_hand_box or target_lh_box or target_rh_box
    if hand_box is not None:
        hand_crops = []
        for key, image, b in (('source', source, source_hand_box or hand_box),
                              ('target', target, hand_box),
                              ('output', generated, hand_box)):
            hc = crop_normalized(image, b)[0] if image is not None else None
            paths[f'hand_{key}'] = f'hands/{stem}_{key}.png' if hc is not None else None
            if hc is not None:
                hc.save(output / paths[f'hand_{key}'])
            hand_crops.append(hc)
        paths['hand_comparison'] = f'panels/{stem}_hand.jpg'
        image_strip(hand_crops, ['A source hand ROI', 'B target hand ROI (offline)', 'Output at B target hand ROI'],
                    missing=failure or 'No valid hand ROI').save(output / paths['hand_comparison'], quality=92)
    for side, box in (('left', target_lh_box), ('right', target_rh_box)):
        if target_lh_box is None and target_rh_box is None:
            continue
        side_crops = []
        for key, image in (('target', target), ('output', generated)):
            crop = crop_normalized(image, box)[0] if image is not None else None
            name = f'hand_{side}_{key}'
            paths[name] = f'hands/{stem}_{side}_{key}.png' if crop is not None else None
            if crop is not None:
                crop.save(output / paths[name])
            side_crops.append(crop)
        paths[f'hand_{side}_comparison'] = f'panels/{stem}_{side}_hand.jpg'
        image_strip(side_crops, [f'B {side} hand ROI (offline)', f'Output at B {side} hand ROI'],
                    missing=failure or 'No valid hand ROI').save(output / paths[f'hand_{side}_comparison'], quality=92)
    hashes = {key: digest_file(output / path) for key, path in paths.items() if path is not None}
    paths['artifact_sha256'] = hashes
    for key in ('source', 'target', 'source_mask', 'target_mask'):
        if key in hashes:
            paths[f'{key}_sha256'] = hashes[key]
    return paths


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--dataset-type', choices=('tiktok_v3', 'iper'))
    parser.add_argument('--align-mode', choices=('letterbox', 'resize', 'native'), default='letterbox')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--features', type=Path, help='Fingerprint-validated FaceConditionStore cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lora', type=Path)
    parser.add_argument('--label')
    parser.add_argument('--variant', help='Report column: init, J, Qwen, BFSBody, BFSHead, or a named ablation')
    parser.add_argument('--stage', choices=('initial', 'common', 'hotload', 'mid', 'final'))
    parser.add_argument('--indices', help='Zero-based indices, e.g. 0,2,4-11. No outcome-based replacement.')
    parser.add_argument('--resolution', default='512', help='512, 1024, or native')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--geometry-strength', type=float, default=1.0, help='Set 0 for Face-only diagnosis')
    parser.add_argument('--face-strength', type=float, default=1.0, help='Set 0 for Body-only diagnosis')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen-Image-2.1')
    parser.add_argument('--tiktok-raw-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_dataset/TikTok_dataset')
    parser.add_argument('--tiktok-assets-root', type=Path, default=ROOT / 'datasets/TikTokDataset/TikTok_3d_assets_native')
    parser.add_argument('--sampled-root', type=Path, default=ROOT / 'datasets/iPER/iper_sampled_6src64tgt')
    parser.add_argument('--assets-root', type=Path, default=ROOT / 'datasets/iPER/iper_assets_512_nvdiffrast')
    parser.add_argument('--progress-mode', choices=('step', 'sigma'), default='step')
    parser.add_argument('--with-hand', action='store_true', help='Enable Hand branch joint conditioning')
    parser.add_argument('--hand-strength', type=float, default=1.0, help='Set 0 for Hand-off diagnosis')
    parser.add_argument('--hand-mode', default='v66', choices=('v66', 'off'))
    parser.add_argument('--use-kv-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--score-only', action='store_true', help='CPU trusted IdentityMetric/PoseMetric on saved outputs; no pipeline load')
    parser.add_argument('--metric-models-root', type=Path, help='Optional directory containing DWPose/ and antelopev2/ for CPU scoring')
    parser.add_argument('--study-lock', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--prompt-manifest', type=Path)
    parser.add_argument('--study-step', type=int)
    parser.add_argument('--study-split', choices=('train', 'val', 'diagnostic', 'test'))
    args = parser.parse_args(argv)
    res_str = str(args.resolution).strip()
    args.resolution = int(res_str) if res_str.isdigit() else res_str
    if args.study_lock is not None:
        missing = [name for name in ('arm', 'prompt_manifest', 'study_step', 'study_split') if getattr(args, name) is None]
        if missing:
            parser.error('Study evaluation requires ' + ', '.join('--' + name.replace('_', '-') for name in missing))
        if args.study_step not in EVAL_STEPS:
            parser.error('Study evaluation step must be 0,128,...,1024')
        if args.resolution != 'native' or args.align_mode != 'native' or args.steps != 30:
            parser.error('Study evaluation requires native/native and 30 sampling steps')
        if args.with_hand or args.geometry_strength != 1 or args.face_strength != 1:
            parser.error('Study evaluation requires Body/Face strength 1 and Hand disabled')
        if not args.use_kv_cache or args.progress_mode != 'step':
            parser.error('Study evaluation requires KV caching and step progress')
        if args.seed not in (42, 43) or (args.study_split not in ('val', 'test') and args.seed != 42):
            parser.error('Study evaluation requires seed42, with seed43 allowed for full val/test')
        canonical_variant = 'BFSHead' if args.arm.startswith('head_') else 'Qwen'
        if args.variant not in (None, canonical_variant) or args.stage not in (None, 'final'):
            parser.error('Study evaluation requires canonical Qwen/BFSHead variant and final stage')
        if (args.arm.startswith('head_')) != bool(args.lora):
            parser.error('Study Head arms require Head LoRA; Qwen arms require no LoRA')
        args.variant = canonical_variant
        if args.label is None and args.arm is not None and args.study_step is not None:
            args.label = f"{args.arm}_step_{args.study_step}"
        if args.dataset_type is None:
            args.dataset_type = "tiktok_v3"
        if args.stage is None:
            args.stage = "final"
    if not args.score_only:
        missing = [name for name in ('checkpoint', 'pairs', 'dataset_type', 'cache', 'features', 'label') if getattr(args, name) is None]
        if missing:
            parser.error('Generation requires ' + ', '.join('--' + name.replace('_', '-') for name in missing))
    if args.steps < 1:
        parser.error('--steps must be positive')
    if any(not math.isfinite(value) or value < 0 for value in (args.geometry_strength, args.face_strength, args.hand_strength)):
        parser.error('Body/Face/Hand strengths must be finite and nonnegative')
    if args.dataset_type == 'iper' and args.align_mode == 'letterbox':
        parser.error('QwenFullDataset iPER implements resize only; pass --align-mode resize explicitly')
    return args


def infer_variant_stage(label, lora):
    normalized = label.lower().replace('-', '').replace('_', '')
    variant = 'BFH' if 'bfh' in normalized else 'BFSBody' if 'bfsbody' in normalized else 'BFSHead' if 'bfshead' in normalized else 'init' if 'init' in normalized else 'J' if normalized == 'j' or normalized.startswith(('common', 'j256')) else 'Qwen'
    if lora is not None and variant in ('init', 'J', 'Qwen', 'BFH'):
        raise ValueError('For LoRA runs, label must name BFSBody/BFSHead or pass --variant explicitly')
    stage = 'hotload' if 'hotload' in normalized else 'mid' if 'mid' in normalized or '128' in normalized else 'initial' if variant == 'init' else 'common' if variant in ('J', 'BFH') else 'final'
    return variant, stage


def controller_strengths(args):
    # Shared Detail is disabled by absent prepared.detail. This legacy gate must
    # remain enabled because the separate Face branch also reads it.
    return {'geometry_strength': args.geometry_strength, 'face_strength': args.face_strength,
            'interaction_strength': 0, 'detail_strength': 1.0,
            'hand_strength': args.hand_strength if getattr(args, 'with_hand', False) else 0,
            'hand_mode': getattr(args, 'hand_mode', 'v66') if getattr(args, 'with_hand', False) else 'off'}


def generation_arguments(source, args, *, prompt, generator, target_size=None):
    if args.resolution == 'native' or args.align_mode == 'native':
        width, height = target_size if target_size is not None else source.size
    else:
        width = height = int(args.resolution)
    return {'prompt': prompt, 'image': [source], 'width': width, 'height': height,
            'num_inference_steps': args.steps, 'true_cfg_scale': 1.0,
            'use_kv_cache': args.use_kv_cache, 'generator': generator}


def validate_base_provenance(variant, stage, lora_sha256, checkpoint):
    """Reject mislabeled bases and unintended prior changes after adaptation."""
    names = {'init': 'init', 'initial': 'init', 'j': 'J', 'common': 'J', 'qwen': 'Qwen',
             'bfsbody': 'BFSBody', 'bfshead': 'BFSHead', 'bfh': 'BFH'}
    canonical = names.get(re.sub(r'[-_ ]', '', variant).lower())
    if canonical in ('init', 'J', 'Qwen', 'BFH') and lora_sha256 is not None:
        raise ValueError(f'Canonical {canonical} requires plain Qwen without LoRA')
    if canonical in ('BFSBody', 'BFSHead') and lora_sha256 is None:
        raise ValueError(f'Canonical {canonical} requires a loaded BFS LoRA')
    signature = checkpoint.get('joint_signature')
    signed = isinstance(signature, dict) and 'lora_sha256' in signature
    trained_hash = signature['lora_sha256'] if signed else None
    if stage in ('mid', 'final'):
        if not signed:
            raise ValueError('Adapted mid/final evaluation requires checkpoint joint_signature.lora_sha256')
        if trained_hash != lora_sha256:
            raise ValueError('Checkpoint training LoRA differs from the inference LoRA')
        if lora_sha256 is not None and float(signature.get('strength', 1.0)) != 1.0:
            raise ValueError('Checkpoint training LoRA strength differs from pilot inference strength 1.0')
    elif signed and stage != 'hotload' and trained_hash != lora_sha256:
        raise ValueError('Checkpoint training LoRA differs from inference outside the explicit hotload stage')
    return {'canonical_variant': canonical, 'checkpoint_joint_signature': signature,
            'checkpoint_lora_sha256': trained_hash, 'inference_lora_sha256': lora_sha256,
            'base_matches_training': signed and trained_hash == lora_sha256,
            'hotload_override': stage == 'hotload' and signed and trained_hash != lora_sha256,
            'stage': stage, 'joint_signature_present': signed}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def tensor_visuals(sample):
    """Show actual normal conditioning and colorized Body25 heatmaps on their grid."""
    import numpy as np
    normal = sample['normal'].detach().float().cpu()
    normal = ((normal + 1) / 2).clamp(0, 1)
    mask = sample['human_mask'].detach().float().cpu().clamp(0, 1)
    normal = (normal * mask).permute(1, 2, 0).numpy()
    normal_image = Image.fromarray((normal * 255).round().astype(np.uint8))
    heatmap = sample['pose_heatmap'].detach().float().cpu().clamp(0, 1).numpy()
    palette = np.array([[((i * 83) % 191 + 64), ((i * 131) % 191 + 64), ((i * 47) % 191 + 64)]
                        for i in range(heatmap.shape[0])], dtype=np.float32) / 255
    pose = np.einsum('chw,cd->hwd', heatmap, palette).clip(0, 1)
    return normal_image, Image.fromarray((pose * 255).round().astype(np.uint8))


def target_keypoints(dataset, sample):
    """Read trusted Body25 geometry and use the same recorded RGB transform."""
    meta = sample['pair_metadata']
    base = dataset.base
    if dataset.dataset_type == 'tiktok_v3':
        bundle = base._get_v6_bundle(meta['target_appearance'])
    else:
        bundle = base._get_v6_conditions(meta['target_appearance'])
    if bundle is None:
        return None
    index = bundle['stem_to_idx'][meta['target_stem']]
    points = bundle['kps_25_coords'][index].float().clone()
    visible = (points[:, 2] > .05) & ((points[:, :2] >= 0) & (points[:, :2] <= 1)).all(-1)
    if 'alignment_meta' in sample:
        from src.pose_control.qwen21_full.face_coordinates import transform_from_record
        record = json.loads(sample['alignment_meta'])
        target_meta = record['target']
        if target_meta.get('version') != 'native_pad_v1':
            points = transform_from_record(target_meta).points(points)
    points[:, 2] = torch.where(visible, points[:, 2], torch.zeros_like(points[:, 2]))
    return points.tolist()


def compute_bg_l1(source, generated, mask_a=None, mask_b=None):
    """Original-pixel RGB L1 outside A/B mask union dilated 8px and 8px edges."""
    import numpy as np
    if source is None or generated is None:
        return None, 'Missing source or generated image'
    if source.size != generated.size:
        return None, f'Source {source.size} and output {generated.size} dimensions differ'
    if mask_a is None or mask_b is None:
        return None, 'Both original A/B human masks are required'

    w, h = source.size
    masks = []
    for m in (mask_a, mask_b):
        if m is not None:
            if not isinstance(m, Image.Image):
                m = Image.fromarray(np.asarray(m))
            if m.size != (w, h):
                return None, f'Original human mask dimensions {m.size} differ from image {(w, h)}'
            masks.append(np.asarray(m.convert('L')) > 127)

    if not masks:
        return None, 'No valid human mask available'

    union_mask = masks[0] if len(masks) == 1 else np.logical_or(masks[0], masks[1])
    union_img = Image.fromarray((union_mask.astype(np.uint8) * 255))
    dilated_img = union_img.filter(ImageFilter.MaxFilter(17))
    dilated_np = np.asarray(dilated_img) > 0
    bg_pixels = ~dilated_np
    bg_pixels[:8, :] = bg_pixels[-8:, :] = False
    bg_pixels[:, :8] = bg_pixels[:, -8:] = False

    if not np.any(bg_pixels):
        return None, 'No valid background pixels outside dilated mask'

    src_np = np.asarray(source.convert('RGB'), dtype=np.float32)
    gen_np = np.asarray(generated.convert('RGB'), dtype=np.float32)
    l1 = float(np.mean(np.abs(gen_np[bg_pixels] - src_np[bg_pixels]))) / 255.0
    return l1, None


def score_saved_outputs(output, models_root=None):
    """Reuse trusted CPU metrics; keep failures and detector multiplicity explicit."""
    import cv2
    import numpy as np
    from src.evaluation.joint_pilot_metrics import IdentityMetric, PoseMetric
    output = Path(output)
    scoring_sha = None
    config_path = output / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    if config.get('study_schema'):
        from src.pose_control.qwen21_full.prompt_study.protocol import load_study_lock
        from src.pose_control.qwen21_full.prompt_study.measurement import validate_scoring_runtime
        run = next((p for p in output.parents if (p / 'data/lock.json').is_file()), None)
        if run is None:
            raise ValueError('Study scoring requires the locked run directory')
        lock = load_study_lock(run / 'data/lock.json', verify_assets=False)
        scoring_sha = validate_scoring_runtime(lock, Path(models_root) if models_root else ROOT / 'models', ROOT)
    identity, pose = IdentityMetric(models_root), PoseMetric(models_root)
    rows = [json.loads(line) for line in (output / 'per_sample.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    scored = []
    for row in rows:
        score = {'pair_index': row['pair_index'], 'image_stem': row['image_stem'], 'status': row['status'],
                 'image': row.get('image'), 'failure': row.get('failure'), 'identity_A': None, 'face_count': None}
        score.update({key: row.get(key) for key in ('pair_id', 'arm', 'protocol', 'study_step', 'study_split',
                     'seed', 'prompt_sha256', 'image_sha256', 'image_size')})
        try:
            if row['status'] == 'complete':
                with Image.open(output / row['image']) as saved:
                    generated = saved.convert('RGB')
                with Image.open(output / row['source']) as saved:
                    source = saved.convert('RGB')
                if row.get('resolution') not in ('native', None):
                    res = int(row['resolution'])
                    generated = generated.resize((res, res), Image.Resampling.BILINEAR)
                    source = source.resize((res, res), Image.Resampling.BILINEAR)
                score.update(image_size=[generated.width, generated.height])
                score['image_sha256'] = digest_file(output / row['image'])
                source_boxes, _ = identity.det.detect(cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR))
                boxes, _ = identity.det.detect(cv2.cvtColor(np.asarray(generated), cv2.COLOR_RGB2BGR))
                score.update(source_face_count=len(source_boxes), face_count=len(boxes), face_unique=len(boxes) == 1)
                if len(source_boxes) == 1 and len(boxes) == 1:
                    score['identity_A'] = identity.cosine(identity.embedding(source), identity.embedding(generated))
                if row.get('target_body25') is not None:
                    detected = pose.detect(generated)
                    pose_scores = pose.compare(detected, np.asarray(row['target_body25'], dtype=np.float32),
                                               image_size=(generated.width, generated.height))
                    if (row.get('arm') or row.get('study_split')) and (
                            pose_scores.get('pose_detected') is False or pose_scores.get('pose_matched_joints') == 0):
                        for metric in ('pose_nme_diagonal', 'pose_pck_002', 'pose_pck_005', 'pose_joint_recall'):
                            pose_scores[metric] = None
                        pose_scores['pose_failure'] = (
                            'Output pose detector failed' if detected is None or pose_scores.get('pose_detected') is False
                            else 'No matched output and target pose joints')
                    score.update(pose_scores)
                else:
                    score['pose_failure'] = 'Missing original target Body25 geometry; no heatmap-derived substitute'

                # Background preservation error
                stem = row.get('image_stem')
                mask_a_path = output / row['source_mask'] if row.get('source_mask') and (output / row['source_mask']).is_file() else None
                mask_b_path = output / row['target_mask'] if row.get('target_mask') and (output / row['target_mask']).is_file() else None
                if mask_b_path is None and stem and (output / 'images' / f'{stem}_mask.png').is_file():
                    mask_b_path = output / 'images' / f'{stem}_mask.png'
                mask_a = Image.open(mask_a_path).convert('L') if mask_a_path else None
                mask_b = Image.open(mask_b_path).convert('L') if mask_b_path else None
                bg_l1, bg_reason = compute_bg_l1(source, generated, mask_a, mask_b)
                score['bg_l1_A'] = bg_l1
                if bg_reason is not None:
                    score['bg_l1_reason'] = bg_reason

                # Same-person pixel L1 to GT target (prohibited on cross-person diagnostic)
                split = row.get('study_split') or row.get('split') or (row.get('metadata') or {}).get('split')
                is_cross = bool(row.get('cross_person') or (row.get('metadata') or {}).get('cross_person')
                                or row.get('is_cross') or (row.get('metadata') or {}).get('is_cross') or split == 'diagnostic')
                target_path = output / row['target'] if row.get('target') and (output / row['target']).is_file() else None
                if not is_cross and target_path is not None:
                    with Image.open(target_path) as saved_target:
                        tgt = saved_target.convert('RGB')
                    if tgt.size == generated.size:
                        score['pixel_l1_GT'] = float(np.mean(np.abs(np.asarray(generated, dtype=np.float32) - np.asarray(tgt, dtype=np.float32)))) / 255.0
                    else:
                        score['pixel_l1_GT'] = None
                        score['pixel_l1_reason'] = 'Target and generated dimensions differ'
                elif is_cross:
                    score['pixel_l1_GT'] = None
                    score['pixel_l1_reason'] = 'Cross-person diagnostic prohibits B GT error'
        except Exception as exc:
            score['score_failure'] = f'{type(exc).__name__}: {exc}'
        scored.append(score)
    is_native = any(row.get('resolution') == 'native' for row in rows)
    write_json(output / 'scores.json', {'schema': 'qwen21_joint_pilot_scores_v2',
                                       'scoring_signature_sha256': scoring_sha,
                                       'config_sha256': digest_file(output / 'config.json') if (output / 'config.json').is_file() else None,
                                       'per_sample_sha256': digest_file(output / 'per_sample.jsonl'),
                                       'metric_resolution': 'per_sample_native' if is_native else 512,
                                       'identity_definition': 'unique-face ArcFace output versus Source A',
                                       'pose_definition': 'trusted PoseMetric versus transformed B Body25',
                                       'bg_l1_definition': 'original-pixel RGB mean L1 outside A/B mask union dilated 8 pixels (MaxFilter17), excluding all four 8-pixel canvas edges; missing/mismatched masks yield no score',
                                       'per_sample': scored})
    return scored


def main(argv=None):
    args = arguments(argv)
    if args.score_only:
        scores = score_saved_outputs(args.output, args.metric_models_root)
        print(json.dumps({'scored': len(scores), 'output': str(args.output / 'scores.json')}), flush=True)
        return
    import torch
    from diffusers import QwenImage21Pipeline
    from src.pose_control.qwen21_full.execution import (FaceConditionStore, QwenFullDataset, Qwen21IPERCache,
                                                       load_adapter_checkpoint, new_adapter, prepare, seed_all)
    from src.pose_control.qwen21_full.qwen_adapter import Qwen21FullControlContext
    from src.pose_control.qwen21_full.qwen_runtime import MODEL_REVISION, PROMPT, dataset_image_to_pil

    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(4)
    dataset = QwenFullDataset(args.pairs, resolution=args.resolution, dataset_type=args.dataset_type,
                              align_mode=args.align_mode, sampled_root=args.sampled_root, assets_root=args.assets_root,
                              tiktok_raw_root=args.tiktok_raw_root, tiktok_assets_root=args.tiktok_assets_root)
    indices = select_indices(args.indices, len(dataset))
    inferred_variant, inferred_stage = infer_variant_stage(args.label, args.lora if args.variant is None else None)
    variant = args.variant or inferred_variant
    if args.variant is None and args.geometry_strength == 0:
        variant += '+FaceOnly' if args.face_strength else '+NoControl'
    elif args.variant is None and args.face_strength == 0:
        variant += '+BodyOnly'
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ('score_only', 'metric_models_root')}
    if args.study_lock is not None:
        validate_study_schedule(args, indices)
        lock, bank = validate_study_inputs(args, dataset)
    else:
        bank = None
    target_b_note = (
        'Qwen RGB condition=Source A only; B may be read by PE for PB/PBG, numeric geometry, dimensions, offline scoring'
        if args.study_lock is not None
        else 'offline only'
    )
    config.update(schema='qwen21_joint_pilot_eval_v1', variant=variant, stage=args.stage or inferred_stage,
                  indices=indices, pairs_sha256=digest_file(args.pairs), checkpoint_sha256=digest_file(args.checkpoint),
                  lora_sha256=digest_file(args.lora) if args.lora else None, lora_strength=1.0 if args.lora else None,
                  prompt=PROMPT if bank is None else "per_sample", true_cfg_scale=1.0, generation_rgb_inputs=['Source A'], target_B_rgb=target_b_note,
                  geometry_strength=args.geometry_strength, face_strength=args.face_strength,
                  hand_strength=args.hand_strength if getattr(args, 'with_hand', False) else 0,
                  with_hand=getattr(args, 'with_hand', False), interaction_strength=0,
                  shared_detail_condition=False, legacy_detail_gate=1.0, runner_sha256=digest_file(__file__),
                  model_revision=MODEL_REVISION, model_config_sha256=digest_file(args.model / 'transformer/config.json'),
                  process_isolation='one base and one adapter checkpoint per invocation')
    if args.study_lock is not None:
        config.update(
            study_schema=SCHEMA,
            study_step=args.study_step,
            arm=args.arm,
            protocol=args.arm.split("_", 1)[1],
            prompt_manifest_sha256=digest_file(args.prompt_manifest),
            lock_sha=digest_file(args.study_lock),
            study_split=args.study_split,
        )
    # Validate the mmap CPU checkpoint header before allocating the frozen base.
    checkpoint_header = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if bank is not None:
        config['study_checkpoint_provenance'] = validate_study_checkpoint(args, lock, checkpoint_header, bank)
    config['base_provenance'] = validate_base_provenance(config['variant'], config['stage'], config['lora_sha256'], checkpoint_header)
    del checkpoint_header
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing evaluation configuration differs; use a new output directory')
    write_json(config_path, config)
    selected_rows = [dataset.rows[index] for index in indices]
    (args.output / 'selected_pairs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected_rows), encoding='utf-8')
    records = args.output / 'per_sample.jsonl'
    prior = [json.loads(line) for line in records.read_text(encoding='utf-8').splitlines() if line.strip()] if records.exists() else []
    if len({row['pair_index'] for row in prior}) != len(prior) or any(row['pair_index'] not in indices for row in prior):
        raise ValueError('Existing evaluation records have duplicate or unexpected pair indices')
    if bank is not None:
        validate_study_records(args, config, dataset, bank, prior, indices)
    completed = {row['pair_index']: row for row in prior}
    store = FaceConditionStore(args.features)
    cache = Qwen21IPERCache(args.cache, resolution=args.resolution, prompt=PROMPT,
                           preprocessing='native_v1' if args.align_mode == 'native' else ('letterbox_v1' if args.align_mode == 'letterbox' else 'resize_v1'))
    if bank is not None:
        from scripts.precompute_qwen21_prompt_study_cache import validate_cache_manifest
        validate_cache_manifest(args.study_lock.parent.parent, bank.protocol, [args.study_split])
        validate_study_cache(cache, dataset, bank, indices=indices, require_target=False)
    # Failed attempts remain frozen in this directory. Use a new output directory
    # for reruns so they cannot quietly disappear from the pilot's failure rate.
    pipe = QwenImage21Pipeline.from_pretrained(str(args.model), dtype=torch.bfloat16).to('cuda')
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        model.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    from src.pose_control.qwen21_full.frozen_bfs_lora import load_frozen_bfs
    lora_report = load_frozen_bfs(pipe, args.lora, strength=1.0)
    if lora_report['file_sha256'] != config['lora_sha256']:
        raise ValueError('Actual loaded LoRA file changed after evaluation configuration was fingerprinted')
    write_json(args.output / 'lora_loading.json', {
        **lora_report, 'checkpoint_sha256': config['checkpoint_sha256'],
    })
    for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError('Qwen/BFS parameters must be frozen')
    adapter = new_adapter('cuda').eval().requires_grad_(False)
    payload = load_adapter_checkpoint(adapter, args.checkpoint)
    if bank is not None:
        validate_study_checkpoint(args, lock, payload, bank)
    actual_provenance = validate_base_provenance(config['variant'], config['stage'], lora_report['file_sha256'], payload)
    if actual_provenance != config['base_provenance']:
        raise ValueError('Checkpoint signature changed after evaluation configuration was fingerprinted')
    write_json(args.output / 'checkpoint_metadata.json', {key: payload.get(key) for key in (
        'architecture', 'resolution', 'step', 'branch', 'model_revision', 'joint_signature',
        'joint_initialization_sha256', 'study_schema', 'arm', 'protocol', 'max_updates',
        'initial_history_steps', 'study_lock_sha256', 'align_mode', 'next_update_index')})
    write_json(args.output / 'base_provenance.json', {**actual_provenance, 'loaded_lora_report': lora_report,
              'checkpoint_sha256': config['checkpoint_sha256'], 'checkpoint': str(args.checkpoint)})
    del payload
    from src.pose_control.qwen21_full.data_runtime import HandConditionStore
    hand_store = HandConditionStore(
        raw_root=args.tiktok_raw_root,
        assets_root=args.tiktok_assets_root,
        align_mode=args.align_mode,
    ) if args.with_hand else None
    rows = []
    for index in indices:
        if index in completed:
            row = completed[index]
            if row['status'] == 'complete' and not (args.output / row['image']).is_file():
                raise FileNotFoundError(f'Completed output is missing for pair {index}; use a new run directory')
            rows.append(row)
            continue
        from src.pose_control.qwen21_full.data_runtime import pair_metadata
        initial_metadata = pair_metadata(dataset.rows[index], args.sampled_root)
        row = {'pair_index': index, 'pair_idx': index, 'mode': args.label, 'label': args.label, 'variant': config['variant'],
               'stage': config['stage'], 'resolution': args.resolution, 'status': 'failed', 'failure': None,
               'image': None, 'face_valid': None, 'source_face_box': None, 'target_face_box': None,
               'source': None, 'target': None, 'target_body25': None, 'metrics': None, 'metadata': initial_metadata}
        if bank is not None:
            row.update(pair_id=initial_metadata['pair_id'], seed=args.seed, arm=args.arm, protocol=bank.protocol,
                       study_step=args.study_step, study_split=args.study_split, split=args.study_split,
                       cross_person=initial_metadata['cross_person'],
                       checkpoint_sha256=config['checkpoint_sha256'], lora_sha256=config['lora_sha256'],
                       prompt_sha256=bank.records[initial_metadata['pair_id']]['compiled_prompt_sha256'])
        source = target = normal = pose = generated = None
        target_mask = source_mask = None
        artifacts_unpadded = False
        prepared = context = result = None
        target_hand_box = target_lh_box = target_rh_box = source_hand_box = None
        stem = case_stem(index, initial_metadata)
        phase = 'dataset'
        start = time.perf_counter()
        try:
            sample = dataset[index]
            batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else [value] for key, value in sample.items()}
            meta = dict(sample['pair_metadata'])
            if 'alignment_meta' in sample:
                meta['alignment'] = json.loads(sample['alignment_meta'])
            stem = case_stem(index, meta)
            row.update(metadata=meta, image_stem=stem, appearance=meta['source_appearance'])
            if bank is not None:
                compiled_prompt = bank.prompt_for(meta['pair_id'])
                prompt_record = bank.records[meta['pair_id']]
                row.update(
                    arm=args.arm, protocol=bank.protocol, study_step=args.study_step,
                    prompt_sha256=prompt_record['compiled_prompt_sha256'],
                    prompt=compiled_prompt,
                    prompt_record_pair_id=prompt_record['pair_id'],
                    initialization='fresh_gaussian',
                )
            else:
                compiled_prompt = PROMPT
            source, target = dataset_image_to_pil(sample['src_image']), dataset_image_to_pil(sample['tgt_image'])
            normal, pose = tensor_visuals(sample)
            if bank is not None:
                source_mask = load_source_person_mask(dataset.rows[index], meta, source)
            target_mask = None
            if 'human_mask' in sample:
                import numpy as np
                mask_t = sample['human_mask'].detach().float().cpu().clamp(0, 1)
                mask_arr = (mask_t[0].numpy() * 255 if mask_t.ndim == 3 else mask_t.squeeze().numpy() * 255).round().astype(np.uint8)
                target_mask = Image.fromarray(mask_arr, mode='L')
            row['target_body25'] = target_keypoints(dataset, sample)
            phase = 'conditioning'
            with torch.inference_mode():
                entry = cache.source(pipe, source, prompt=compiled_prompt)
                if args.resolution == 'native':
                    tgt_hw = (sample['tgt_image'].shape[1] // 16, sample['tgt_image'].shape[2] // 16)
                    src_hw = entry.get('token_hw', (sample['src_image'].shape[1] // 16, sample['src_image'].shape[2] // 16))
                    pair = {'source_latent': entry['reference_latent'].to('cuda'), 'resolution': args.resolution,
                            'target_token_hw': tgt_hw, 'source_token_hw': src_hw}
                else:
                    pair = {'source_latent': entry['reference_latent'].to('cuda'), 'resolution': args.resolution}
                prepared = prepare(adapter, batch, pair, [meta], store, with_face=True,
                                   with_hand=args.with_hand, hand_store=hand_store)
                if prepared.face is None:
                    raise AssertionError('Joint inference requires prepared Face conditioning')
                if prepared.detail is not None:
                    raise AssertionError('Joint pilot does not allow shared Detail conditioning')
                if not args.with_hand and prepared.hand is not None:
                    raise AssertionError('Joint Body/Face pilot must not prepare Hand conditioning')
                if args.with_hand and prepared.hand is None:
                    raise AssertionError('Joint Body/Face/Hand pilot requires prepared Hand conditioning')
                row['face_valid'] = bool(prepared.face.face_valid.any())
                row['face_slots_valid'] = prepared.face.face_valid.detach().cpu().tolist()[0]
                # Store boxes independently of source identity validity: a missing
                # A face must not remove the B ROI from the visual failure audit.
                from src.pose_control.qwen21_full.face_coordinates import transform_from_record
                source_cache = store._load(meta['source_appearance'])
                target_cache = store._load(meta['target_appearance'])
                for name, face_cache, frame in (('source', source_cache, meta['source_stem']), ('target', target_cache, meta['target_stem'])):
                    frame_index = face_cache['stem_to_idx'][frame]
                    valid_key = 'valid' if name == 'source' else 'geometry_valid'
                    valid = bool(face_cache[valid_key][frame_index])
                    row[f'{name}_face_geometry_valid'] = valid
                    box = face_cache['face_boxes'][frame_index].float().clone() if valid else None
                    row[f'{name}_face_box_original_normalized'] = box.tolist() if box is not None else None
                    if box is not None and 'alignment_meta' in sample:
                        box = transform_from_record(meta['alignment'][name]).points(box.reshape(2, 2)).reshape(4)
                    row[f'{name}_face_box_padded_normalized'] = box.tolist() if box is not None else None
                    row[f'{name}_face_box'] = box.tolist() if box is not None else None
                if getattr(args, 'study_lock', None) is not None:
                    from src.pose_control.qwen21_full.prompt_study.geometry import hand_boxes
                    dwpose_json = meta.get("target_dwpose_json")
                    if dwpose_json and Path(dwpose_json).is_file():
                        dw_rec = json.loads(Path(dwpose_json).read_text(encoding="utf-8"))
                        h_boxes = hand_boxes(dw_rec["keypoints_norm"], dw_rec["scores"])
                        target_lh_box = h_boxes.get("left")
                        target_rh_box = h_boxes.get("right")
                        target_hand_box = target_lh_box or target_rh_box
                        row['target_left_hand_box'] = target_lh_box
                        row['target_right_hand_box'] = target_rh_box
                        row['target_hand_box'] = target_hand_box
                elif args.with_hand and prepared.hand is not None:
                    lh_valid = bool(prepared.hand.hand_valid[0, 0, 0])
                    rh_valid = bool(prepared.hand.hand_valid[0, 0, 1])
                    row['left_hand_valid'] = lh_valid
                    row['right_hand_valid'] = rh_valid
                    if lh_valid:
                        target_lh_box = prepared.hand.target_boxes[0, 0, 0].detach().cpu().tolist()
                        row['target_left_hand_box'] = target_lh_box
                    if rh_valid:
                        target_rh_box = prepared.hand.target_boxes[0, 0, 1].detach().cpu().tolist()
                        row['target_right_hand_box'] = target_rh_box
                    target_hand_box = target_lh_box or target_rh_box
                    row['target_hand_box'] = target_hand_box
                    if target_hand_box is not None and hand_store is not None:
                        hand_cond, _ = hand_store.for_pairs([meta], device='cpu')
                        source_hand_box = hand_cond.source_boxes[0, 0, 0 if lh_valid else 1].detach().cpu().tolist()
                        row['source_hand_box'] = source_hand_box
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            phase = 'generation'
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                with Qwen21FullControlContext(pipe.transformer, adapter, prepared, **controller_strengths(args), progress_mode=args.progress_mode,
                                             num_inference_steps=args.steps) as context:
                    target_size = (sample['tgt_image'].shape[2], sample['tgt_image'].shape[1]) if args.resolution == 'native' else None
                    gen_kwargs = generation_arguments(
                        source, args, prompt=compiled_prompt,
                        generator=torch.Generator(device='cuda').manual_seed(args.seed),
                        target_size=target_size,
                    )
                    if gen_kwargs["image"] != [source] or "latents" in gen_kwargs or "target" in gen_kwargs:
                        raise AssertionError("Study generation violated source-only/noise initialization")
                    result = pipe(**gen_kwargs)
            torch.cuda.synchronize()
            row.update(seconds=time.perf_counter() - start, peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3,
                       control_forwards=context.forward_count, block_calls=context.call_counts,
                       last_control_diagnostics=context.output.diagnostics, sampling_schedule=context.step_diagnostics)
            if context.forward_count != args.steps or any(value != args.steps for value in context.call_counts.values()):
                raise AssertionError('Incomplete sampling or missing layer injection')
            diagnostics = context.output.diagnostics
            if diagnostics.get('detail_active_rows', 0):
                raise AssertionError('Disabled Detail branch was activated')
            if not args.with_hand and diagnostics.get('hand_active_rows', 0):
                raise AssertionError('Disabled Hand branch was activated')
            generated = Image.alpha_composite(Image.new('RGBA', result.images[0].size, 'white'), result.images[0].convert('RGBA')).convert('RGB')
            if 'alignment' in meta and 'target' in meta['alignment']:
                from src.pose_control.qwen21_full.face_coordinates import transform_from_record
                target_tf = transform_from_record(meta['alignment']['target'])
                generated = target_tf.unpad(generated)
                target = target_tf.unpad(target)
                if normal is not None:
                    normal = target_tf.unpad(normal)
                if pose is not None:
                    pose = target_tf.unpad(pose)
                if target_mask is not None:
                    target_mask = target_tf.unpad(target_mask)
            if 'alignment' in meta and 'source' in meta['alignment']:
                from src.pose_control.qwen21_full.face_coordinates import transform_from_record
                source_tf = transform_from_record(meta['alignment']['source'])
                source = source_tf.unpad(source)
            artifacts_unpadded = True
            if getattr(args, 'study_lock', None) is not None:
                for side in ('source', 'target'):
                    row[f'{side}_face_box'] = row.get(f'{side}_face_box_original_normalized')
            row['status'] = 'complete'
            row['image_size'] = [generated.width, generated.height]
        except Exception as exc:
            row.update(failure=f'{type(exc).__name__}: {exc}', failure_phase=phase,
                       failure_traceback=traceback.format_exc(), seconds=time.perf_counter() - start)
            # Preserve the fixed pair, including invalid conditions and OOM; never
            # replace a failed case with a different frame or successful output.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            row.setdefault('image_stem', stem)
            if getattr(args, 'study_lock', None) is not None:
                if not artifacts_unpadded:
                    source, target, generated, normal, pose, target_mask = unpad_artifact_images(
                        row['metadata'], source=source, target=target, generated=generated,
                        normal=normal, pose=pose, target_mask=target_mask)
                for side in ('source', 'target'):
                    orig_box = row.get(f'{side}_face_box_original_normalized')
                    if orig_box is not None:
                        row[f'{side}_face_box'] = orig_box
                condition_metadata = dict(row['metadata'])
                condition_metadata.setdefault('filtered_dwpose', str(args.study_lock.parent / 'geometry' / initial_metadata['pair_id'] / 'dwpose_filtered.png'))
                row.update(save_conditioning_artifacts(args.output, stem, metadata=condition_metadata,
                           normal=normal, pose=pose, body25=row.get('target_body25')))
            if source is not None and target is not None:
                row.update(save_case_artifacts(
                    args.output, stem, source=source, target=target, generated=generated,
                    normal=normal, pose=pose, source_box=row['source_face_box'], target_box=row['target_face_box'],
                    source_hand_box=source_hand_box, target_hand_box=target_hand_box,
                    target_lh_box=target_lh_box, target_rh_box=target_rh_box,
                    target_mask=target_mask, source_mask=source_mask, failure=row['failure']))
                if row.get('image'):
                    row['image_sha256'] = digest_file(args.output / row['image'])
            with records.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            rows.append(row)
            write_json(args.output / 'per_sample.json', rows)
            print(json.dumps({key: row.get(key) for key in ('pair_index', 'status', 'face_valid', 'seconds', 'failure')}, ensure_ascii=False), flush=True)
            del prepared, context, result
    summary = {'attempted': len(rows), 'complete': sum(row['status'] == 'complete' for row in rows),
               'failed': sum(row['status'] != 'complete' for row in rows),
               'face_condition_valid': sum(row.get('face_valid') is True for row in rows),
               'label': args.label, 'variant': config['variant'], 'stage': config['stage'],
               'evaluation_scope': f'locked study {args.study_split}' if bank is not None else 'pilot diagnostic pairs; not final held-out test'}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

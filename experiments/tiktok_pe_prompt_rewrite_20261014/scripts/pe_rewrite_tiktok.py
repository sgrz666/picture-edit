#!/usr/bin/env python3
"""Rewrite the per-pair task prompt for TikTok_v3 with the official PE-I2I model.

One record per pair: the PE sees A (source_rgb), B (target_rgb) and B's detected
DWPose skeleton (target_dwpose), and answers the official three-field JSON contract.
Two downstream prompt forms are compiled from that answer because the two consumers
of this bank do not receive the same images:

  training  -- the Qwen-Image-2.1 VLM sees only A (src/pose_control/qwen21/runtime.py
               encodes ``image=[vlm_image]``); the skeleton enters as adapter control.
               So every image reference must be resolved into role words.
  inference -- the workflow generator sees A + skeleton, so <image3> maps to <image2>.

Usage (on sg, qwen21 env):
  python pe_rewrite_tiktok.py run --split train --limit 20
  python pe_rewrite_tiktok.py assemble
  python pe_rewrite_tiktok.py qc --count 5
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

EXP_ROOT = Path(__file__).resolve().parents[1]


def _find_repo_root() -> Path:
    # The server keeps experiments under a family directory, the local clone keeps
    # them flat, so the repo root is found by marker rather than by a fixed depth.
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src' / 'pose_edit' / 'prompt_contract.py').is_file():
            return parent
    raise RuntimeError('no src/pose_edit above this script; run it inside the repo')


REPO_ROOT = _find_repo_root()
for candidate in (str(REPO_ROOT), str(EXP_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from src.pose_edit.common import atomic_json, sha256_file
from src.pose_edit.prompt_contract import (IMAGE_REFERENCE, PE_SAMPLING,
    parse_pe_answer, referenced_images)

SCHEMA = 'tiktok_pei2i_prompt_rewrite_v1'
SKELETON_LABEL = 'the DWPose-style skeleton detected on the target photograph'
ROLES = ('source_A', 'target_dwpose', 'target_B')
ROLE_WORDS = {
    'source_A': 'the source image',
    'target_dwpose': 'the target joint arrangement described in this instruction',
    'target_B': 'the target pose described in this instruction',
}
PAIRS_ROOT = REPO_ROOT / 'datasets/TikTokDataset/TikTok_v3/manifests/pairs_strict_v1'
# The stored body block is DWPose's 18-point order, NOT COCO-17:
# 0 nose, 1 neck, 2 Rshoulder, 3 Relbow, 4 Rwrist, 5 Lshoulder, 6 Lelbow, 7 Lwrist,
# 8 Rhip, 9 Rknee, 10 Rankle, 11 Lhip, 12 Lknee, 13 Lankle, 14 Reye, 15 Leye, 16 Rear, 17 Lear.
# (add/tools/dwpose_mimicmotion_official/dwpose/util.py draws limbs with 1-based limbSeq.)
LOWER_BODY = (8, 9, 10, 11, 12, 13)
LEG_TRIPLES = {'right': (8, 9, 10), 'left': (11, 12, 13)}
PE_MODEL = Path('/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I')

# PE writes "the joint arrangement defined by <image2>"; substituting the role word first
# would leave "the joint arrangement defined by the target joint arrangement ...". The
# lead-in is dropped while the reference token is still literal, so matching is anchored.
GUIDE_LEADIN = re.compile(
    r'\b(?:the|a|an|this|that)\s+(?:colored\s+|black\s+|reference\s+|estimated\s+|provided\s+|'
    r'given\s+|2D\s+)?(?:skeleton|pose skeleton|skeleton guide|skeletal guide|skeletal map|'
    r'pose guide|joint guide|joint map|joint arrangement|joint configuration|body configuration|'
    r'pose estimate|limb lines|skeleton limb lines)\s*(?:defined|encoded|specified|described|'
    r'shown|visible)?\s*(?:by|in|from|of|as|that shows|showing|which shows)?\s*'
    r'(?:the|a|an|this|that)?\s*(?=<image[_\s]*2>|\bimage[_\s]*2\b)', re.I)
REF2 = re.compile(r'<image[_\s]*2>|\bimage[_\s]*2\b', re.I)
REF3 = re.compile(r'<image[_\s]*3>|\bimage[_\s]*3\b', re.I)
# The training consumer never receives the target photograph, so clauses that point the
# reader at it ("transfer the pose from the target photograph") must name the described
# pose instead. Possessive forms ("never copy the target's identity") are left alone:
# they forbid something the reader cannot do, which stays harmless.
TARGET_PHOTO = re.compile(r'\bthe\s+(?:target|reference)\s+(?:photograph|photo|frame|image|picture)s?\b', re.I)
TRAILING_GUIDE = re.compile(r'(the target joint arrangement described in this instruction)\s+'
    r'(?:skeleton|skeletal guide|guide|map|image|picture|diagram)\b', re.I)

SYSTEM_SUFFIX = (
    '\n\nTIKTOK POSE-TRANSFER OUTPUT REFERENCE CONTRACT. This takes precedence over '
    'the general multi-image reference rules above for the final rewritten_prompt only. '
    'You receive three images and may refer to all three in your internal reasoning: '
    '<image1> is the source photograph, <image2> is the pose skeleton, and <image3> is the target photograph. '
    'The downstream editor will receive only the source photograph and the pose skeleton. '
    'Therefore the final rewritten_prompt MUST NOT contain <image3> or the words "image 3" / "image3", '
    'or any reference to the target photograph, even in preservation or negative object clauses. '
    'Synthesize and translate everything needed from the skeleton (<image2>) and target photograph (<image3>) '
    'into explicit body, head, gaze, expression and hand action in words. '
    'Refer only to the source as <image1> and the skeleton as <image2>; '
    'never cite image 3 in rewritten_prompt. '
    'Keep the official three-field JSON contract, wh_ratio="", ratio_follow="<image1>". '
    'Exclude target-only props in plain words, without citing their source image.'
)

INSTRUCTION = (
    'Write a precise English editing instruction for single-person pose transfer on the '
    'TikTok dataset. '
    '<image1> is the source photograph: retain its person identity, face, hairstyle, glasses, '
    'clothing design, colors, material, accessories, shoes, held or worn objects and photographic scene. '
    '<image2> is ' + SKELETON_LABEL + ', drawn in the target frame at the target position '
    'and scale; black regions are not a scene, and the line colors carry no appearance. '
    'Use it for the visible joint arrangement. '
    '<image3> is the target photograph: use it together with the skeleton in <image2> to accurately understand '
    'and describe the target body pose, orientation, head direction, facial expression and observable hand configuration. '
    'Never copy identity, hair, clothes, jewelry, colors, background or objects from <image3>. '
    'The target photograph may clarify head direction and hands when the skeleton is inaccurate; '
    'do not invent invisible details and do not request anatomy that neither reference shows. '
    'GROUNDING ON WHAT IS ACTUALLY PHOTOGRAPHED: the detector extrapolates joints it never '
    'saw, so this skeleton can contain hips, knees, ankles or feet that lie outside the '
    'person visible in the target photograph (a bust shot cropped at the thighs still gets a '
    'full leg skeleton drawn). A joint that has no counterpart visible in the target '
    'photograph is not evidence. State leg, knee, ankle, foot, hip and lower-body wording '
    'only for the parts actually visible in the target photograph; when the lower body is '
    'cropped out, describe only the visible head, torso, arm and hand configuration and say '
    'nothing about the leg stance, foot placement or lower-body proportions. Never copy a '
    'suggested stance from the drawn leg lines alone. '
    'The task transfers pose only: preserve every object already present on the source '
    'person or in the source scene, and do not add any object that appears only in the '
    'target photograph nor a replacement for it. '
    'Organize the instruction around source appearance preservation, the explicit target '
    'body/head/expression/hand action, the skeletal joint arrangement, the original '
    'photographic style, and the actual edit. '
    'Describe arm and hand placement using viewer image-left/image-right only; never '
    'assign anatomical left/right labels, which can conflict with image directions. '
    'Do not enumerate the target clothes or other target appearance in negative clauses. '
    'Avoid guessing ethnicity, age, brands or hidden anatomy. '
    'The downstream editor will receive only the source photograph (<image1>) and the skeleton (<image2>), '
    'and cannot see <image3>. Therefore, describe the pose explicitly in words based on the skeleton and target photograph. '
    'Write one detailed paragraph of roughly 200 to 320 English words. '
    'Keep the source canvas, framing and overall illumination, allowing only natural cloth folds and pose-dependent shadows. '
    'Do not render skeleton lines or their colors. '
    'CRITICAL CONTRACT: In rewritten_prompt refer to the source as <image1> and the skeleton as <image2>. '
    'DO NOT include "<image3>" or the words "image 3" / "image3" anywhere in rewritten_prompt. '
    'Never name the target photograph, target photo or target frame as something the editor can look at in rewritten_prompt: '
    'the editor sees only the source (<image1>) and the skeleton (<image2>), so attribute the action to this instruction '
    'or to the skeleton. Return the official JSON contract with rewritten_prompt, wh_ratio="", ratio_follow="<image1>".'
)


def compile_words(text: str) -> str:
    """Training form: image references become role words (the VLM sees only A)."""
    def replace(match):
        index = int(match.group(1) or match.group(2))
        if index == 3:
            raise ValueError('prompt refers to image 3 (target photograph), which is not '
                'available downstream; describe the action in words instead')
        if index not in (1, 2):
            raise ValueError(f'unknown image reference {index}')
        return ROLE_WORDS['source_A' if index == 1 else 'target_dwpose']

    # Anchor on the literal <image2> first, then substitute the remaining references.
    cleaned = text
    while True:
        shorter = GUIDE_LEADIN.sub('', cleaned)
        if shorter == cleaned:
            break
        cleaned = shorter
    if not REF2.search(cleaned) and REF2.search(text):
        raise ValueError('cleaning removed the skeleton reference itself')
    cleaned = TARGET_PHOTO.sub('the described target pose', cleaned)
    compiled = IMAGE_REFERENCE.sub(replace, cleaned)
    # "the joint arrangement of the <image2> skeleton" leaves a dangling noun behind.
    compiled = TRAILING_GUIDE.sub(r'\1', compiled)
    compiled = re.sub(r'\b(\w+)\s+\1\b', r'\1', compiled, flags=re.I)
    compiled = re.sub(r'\s+([,.;:])', r'\1', ' '.join(compiled.split()))
    # Stripping a sentence-initial lead-in can expose a lowercase role word after ". ".
    compiled = re.sub(r'(^|[.!?]\s+)(the|this)\s',
        lambda match: match.group(1) + match.group(2).capitalize() + ' ', compiled)
    if not compiled or re.search(r'<image|\bimage[_\s]*\d+', compiled, re.I):
        raise ValueError('unresolved image reference')
    return compiled


def compile_inference(text: str) -> str:
    """Workflow form: A is <image1>, the skeleton is <image2>."""
    def replace(match):
        index = int(match.group(1) or match.group(2))
        if index == 1:
            return '<image1>'
        if index == 2:
            return '<image2>'
        if index == 3:
            raise ValueError('prompt refers to image 3 (target photograph), which the '
                'generator cannot see')
        raise ValueError(f'unknown image reference {index}')

    compiled = IMAGE_REFERENCE.sub(replace, text)
    compiled = TARGET_PHOTO.sub('the described target pose', ' '.join(compiled.split()))
    if not compiled or re.search(r'<image(?![12]>)[^>]*>', compiled, re.I):
        raise ValueError('unresolved image reference')
    contract = (
        'Edit the single person in <image1> into the described pose. '
        '<image1> supplies all identity, hairstyle, face, outfit, accessories, held '
        'objects and scene appearance. '
        '<image2> is only a colored skeletal pose guide; use its joint positions, never '
        'its line colors as appearance. '
        'Do not add any object that is not already present in <image1>. '
    )
    return (contract + compiled + ' Preserve the source person, their clothing, objects '
        'and setting. Render a natural photograph without skeleton lines, diagrams or '
        'pose-guide colors.')


def dwpose_quality(row: dict) -> dict:
    """Per-pair condition-image reliability, read from the existing DWPose JSON.

    dwpose_v1 renders every body keypoint, so a frame can show joints the detector
    scored at or below 0.3. 38% of the target frames in pairs_strict_v1 carry at least
    four such fabricated joints, which the prompt alone cannot reveal.
    """
    path = row.get('target_dwpose_json')
    if not path or not Path(path).is_file():
        return {'dwpose_json_found': False}
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    scores = [float(score) for score in (record.get('scores') or [])][:18]
    return {'dwpose_json_found': True,
        'body_mean_conf': round(float(record.get('body_mean_conf') or 0.0), 4),
        'body_valid_03': int(record.get('body_valid_03') or 0),
        'low_conf_drawn_body_pts': sum(1 for score in scores if score <= 0.3),
        'detected_persons': int(record.get('detected_persons') or 0),
        'hand_mean_conf': round(float(record.get('hand_mean_conf') or 0.0), 4),
        'face_mean_conf': round(float(record.get('face_mean_conf') or 0.0), 4)}


def leg_chain_coherent(kps: list, scores: list, hip: int, knee: int, ankle: int):
    """Is the knee on the hip-ankle segment?  None when any of the three is low-confidence.

    Measured on 900 sampled target frames with the correct DWPose triples: right 85.2%,
    left 92.9% coherent.  A failing chain therefore flags that one frame, not a systematic
    storage bug -- an earlier version of this helper used COCO-17 indices and appeared to
    find a 99% broken left leg.  It had selected hip, ankle and *left eye*.
    """
    if len(kps) <= ankle or len(scores) <= ankle:
        return None
    if min(scores[hip], scores[knee], scores[ankle]) <= 0.3:
        return None
    (hx, hy), (kx, ky), (ax, ay) = kps[hip], kps[knee], kps[ankle]
    dx, dy = ax - hx, ay - hy
    length2 = dx * dx + dy * dy
    if length2 < 1e-8:
        return False
    t = ((kx - hx) * dx + (ky - hy) * dy) / length2
    if not -0.1 <= t <= 1.1:
        return False
    px, py = hx + t * dx, hy + t * dy
    return ((kx - px) ** 2 + (ky - py) ** 2) ** 0.5 < 0.25 * length2 ** 0.5


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def row_inputs(row: dict) -> list[tuple[str, Path]]:
    return [('source_A', Path(row['source_rgb'])), ('target_dwpose', Path(row['target_dwpose'])),
            ('target_B', Path(row['target_rgb']))]


def select_rows(rows: list[dict], limit: int, seed: int, mode: str) -> list[dict]:
    if mode == 'head':
        return rows[:limit]
    pool = sorted(rows, key=lambda row: row['pair_id'])
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[index] for index in sorted(order[:limit])]


def run(args) -> None:
    out_root = Path(args.out)
    records_dir = out_root / 'records' / args.split
    raw_dir = out_root / 'raw' / args.split
    records_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pairs_root / f'{args.split}_pairs.jsonl')
    if args.pair_id:
        wanted = set(args.pair_id)
        rows = [row for row in rows if row['pair_id'] in wanted]
        if len(rows) != len(wanted):
            raise SystemExit(f'pair_id not found in {args.split}: {sorted(wanted - {r["pair_id"] for r in rows})}')
        if args.shard_count > 1:
            rows = rows[args.shard_index::args.shard_count]
    elif args.limit:
        if args.shard_count > 1:
            rows = rows[args.shard_index::args.shard_count]
        rows = select_rows(rows, args.limit, args.seed, args.sample_mode)
    elif args.shard_count > 1:
        rows = rows[args.shard_index::args.shard_count]
    pending = []
    for row in rows:
        record_path = records_dir / f'{row["pair_id"]}.json'
        if record_path.is_file() and not args.overwrite:
            existing = json.loads(record_path.read_text(encoding='utf-8'))
            if existing.get('status') == 'complete':
                continue
        pending.append(row)
    print(f'PLAN split={args.split} selected={len(rows)} pending={len(pending)}', flush=True)
    if args.dry_run:
        for row in pending[:20]:
            missing = [str(path) for _, path in row_inputs(row) if not path.is_file()]
            print(f'  {row["pair_id"]} ' + (f'MISSING {missing}' if missing else 'ok'), flush=True)
        return

    model_dir = Path(args.pe_model)
    system_prompt = (model_dir / 'system_prompt.txt').read_text(encoding='utf-8')
    effective_system = system_prompt + SYSTEM_SUFFIX
    started_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir),
        dtype=torch.bfloat16).to('cuda').eval()
    # The PE config pins mamba_ssm_dtype=float32, and flash-linear-attention's chunk kernel
    # refuses float32, so the fast path silently stays disabled. Overriding it at runtime is
    # the only way to test whether that is what keeps decode at ~18 tok/s.
    if args.mamba_dtype:
        changed = []
        for config in (model.config, getattr(model.config, 'text_config', None)):
            if config is not None and hasattr(config, 'mamba_ssm_dtype'):
                changed.append(f'{getattr(config, "mamba_ssm_dtype")}->{args.mamba_dtype}')
                config.mamba_ssm_dtype = args.mamba_dtype
        print(f'MAMBA_DTYPE {args.mamba_dtype} changed={changed}', flush=True)
    print(f'PE_LOADED seconds={time.perf_counter() - started_load:.1f}', flush=True)

    # sha256 of the two prompt-contract texts pins what this batch was asked to do.
    contract_identity = {
        'system_prompt_sha256': sha256_file(model_dir / 'system_prompt.txt'),
        'system_suffix_sha256': hashlib.sha256(SYSTEM_SUFFIX.encode()).hexdigest(),
        'effective_system_sha256': hashlib.sha256(effective_system.encode()).hexdigest(),
        'instruction_sha256': hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
        'compiler_sha256': hashlib.sha256(
            (inspect.getsource(compile_words) + inspect.getsource(compile_inference)
             + json.dumps(ROLE_WORDS, sort_keys=True)).encode()).hexdigest(),
    }
    done = failed = 0
    for position, row in enumerate(pending, start=1):
        pair_id = row['pair_id']
        sampling = dict(PE_SAMPLING, eos_token_id=[248044, 248046])
        if args.no_thinking:
            sampling['max_new_tokens'] = 2048
        record = {'schema': SCHEMA, 'pair_id': pair_id, 'split': args.split,
            'sequence_id': row['sequence_id'], 'source_stem': row['source_stem'],
            'target_stem': row['target_stem'], 'protocol': 'pei2i_ab3',
            'dwpose_quality': dwpose_quality(row),
            'enable_thinking': not args.no_thinking, 'max_image_side': args.max_image_side,
            'mamba_ssm_dtype': args.mamba_dtype or 'config_default',
            'shard': f'{args.shard_index}/{args.shard_count}',
            'input_roles': list(ROLES), 'sampling': sampling, 'model_dir': str(model_dir),
            **contract_identity}
        attempt_started = time.perf_counter()
        try:
            images, inputs_meta = [], []
            for index, (role, path) in enumerate(row_inputs(row), start=1):
                with Image.open(path) as opened:
                    image = opened.convert('RGB')
                if args.max_image_side and max(image.size) > args.max_image_side:
                    scale = args.max_image_side / max(image.size)
                    image = image.resize((max(1, round(image.width * scale)),
                        max(1, round(image.height * scale))), Image.LANCZOS)
                images.append(image)
                inputs_meta.append({'index': index, 'role': role, 'path': str(path),
                    'sha256': sha256_file(path), 'size': list(image.size)})
            torch.cuda.reset_peak_memory_stats()
            attempts = []
            for attempt in range(2):
                seed = args.pe_seed + attempt * 1000 + (int(hashlib.md5(pair_id.encode()).hexdigest()[:6], 16) % 10000)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                retry = '' if not attempts else ('\nYour previous answer was rejected: '
                    + attempts[-1]['error'] + '. Write the action in explicit words without '
                    'any reference to image 3 or the target photograph. Refer only to <image1> and <image2>. '
                    'Keep the official JSON format.')
                messages = [{'role': 'system', 'content': [{'type': 'text', 'text': effective_system}]},
                    {'role': 'user', 'content': [*[{'type': 'image', 'image': image} for image in images],
                        {'type': 'text', 'text': INSTRUCTION + retry}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=not args.no_thinking,
                    return_dict=True, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**inputs, **sampling)
                tokens = generated[0, inputs['input_ids'].shape[1]:]
                raw = processor.decode(tokens, skip_special_tokens=True)
                raw_path = raw_dir / f'{pair_id}_attempt{attempt + 1}.txt'
                # A 15-day unattended run must not lose a finished generation because the
                # evidence directory disappeared underneath it.
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw, encoding='utf-8')
                evidence = {'attempt': attempt + 1, 'seed': seed, 'raw_path': str(raw_path),
                    'raw_sha256': sha256_file(raw_path), 'output_tokens': int(len(tokens))}
                try:
                    answer = parse_pe_answer(raw)
                    rewritten = answer['rewritten_prompt'].strip()
                    compiled = compile_words(rewritten)
                    inference_form = compile_inference(rewritten)
                except (ValueError, KeyError) as error:
                    evidence.update(status='rejected', error=str(error))
                    attempts.append(evidence)
                    continue
                evidence['status'] = 'accepted'
                attempts.append(evidence)
                break
            else:
                raise RuntimeError('PE failed the JSON/image-role contract twice')
            record.update(status='complete', inputs=inputs_meta, attempts=attempts,
                seed=seed, attempt=attempt + 1, raw_path=str(raw_path),
                raw_sha256=evidence['raw_sha256'], rewritten_prompt=rewritten,
                compiled_prompt=compiled, compiled_prompt_training=compiled,
                compiled_prompt_inference=inference_form,
                compiled_prompt_sha256=hashlib.sha256(compiled.encode()).hexdigest(),
                compiled_prompt_inference_sha256=hashlib.sha256(inference_form.encode()).hexdigest(),
                tokens=int(len(tokens)), seconds=time.perf_counter() - attempt_started,
                peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
            done += 1
            print(f'[{position}/{len(pending)}] OK {pair_id} tokens={record["tokens"]} '
                f'words={len(compiled.split())} seconds={record["seconds"]:.1f}', flush=True)
        except Exception as error:
            record.update(status='failed', error=str(error),
                seconds=time.perf_counter() - attempt_started)
            failed += 1
            print(f'[{position}/{len(pending)}] FAIL {pair_id} {error}', flush=True)
        records_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(records_dir / f'{pair_id}.json', record)
    print(f'RUN_COMPLETE split={args.split} ok={done} failed={failed}', flush=True)


def load_records(records_dir: Path) -> list[dict]:
    if not records_dir.is_dir():
        return []
    return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(records_dir.glob('*.json'))]


def assemble(args) -> None:
    out_root = Path(args.out)
    prompts_dir = out_root / 'prompts'
    prompts_dir.mkdir(parents=True, exist_ok=True)
    summary = {'schema': SCHEMA, 'pairs_root': str(args.pairs_root), 'splits': {}}
    for split in ('train', 'val', 'test'):
        records_dir = out_root / 'records' / split
        if not records_dir.is_dir():
            continue
        rows = [row for row in load_records(records_dir) if row.get('status') == 'complete']
        failed = [row for row in load_records(records_dir) if row.get('status') != 'complete']
        # Join each record with its DWPose detection so the bank itself carries the
        # reliability split; this works for records written before the field existed.
        manifest = {item['pair_id']: item for item in read_rows(args.pairs_root / f'{split}_pairs.jsonl')}
        lower_hist = Counter()
        chain_hist = {'left': Counter(), 'right': Counter()}
        for row in rows:
            detection_path = (manifest.get(row['pair_id']) or {}).get('target_dwpose_json')
            low = None
            row['left_leg_chain_coherent'] = row['right_leg_chain_coherent'] = None
            if detection_path and Path(detection_path).is_file():
                detection = json.loads(Path(detection_path).read_text(encoding='utf-8'))
                scores = [float(score) for score in (detection.get('scores') or [])]
                kps = detection.get('keypoints_norm') or []
                if len(scores) > max(LOWER_BODY):
                    low = sum(1 for index in LOWER_BODY if scores[index] <= 0.3)
                    lower_hist[low] += 1
                for side, (hip, knee, ankle) in LEG_TRIPLES.items():
                    coherent = leg_chain_coherent(kps, scores, hip, knee, ankle)
                    row[f'{side}_leg_chain_coherent'] = coherent
                    chain_hist[side]['unjudgeable' if coherent is None else str(coherent)] += 1
            row['lower_body_extrapolated_joints'] = low
        text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        path = prompts_dir / f'{split}.jsonl'
        path.write_text(text, encoding='utf-8')
        lengths = [len(row['compiled_prompt'].split()) for row in rows]
        quality = [row.get('dwpose_quality') or {} for row in rows]
        low_conf = [item.get('low_conf_drawn_body_pts', 0) for item in quality]
        confs = [item['body_mean_conf'] for item in quality if item.get('dwpose_json_found')]
        summary['splits'][split] = {'available_pairs': len(read_rows(args.pairs_root / f'{split}_pairs.jsonl')),
            'complete': len(rows), 'failed': len(failed),
            'manifest': str(path),
            'manifest_sha256': hashlib.sha256(text.encode()).hexdigest() if text else None,
            'words_min': min(lengths) if lengths else None,
            'words_median': int(np.median(lengths)) if lengths else None,
            'words_max': max(lengths) if lengths else None,
            'distinct_prompts': len({row['compiled_prompt'] for row in rows}),
            'skeleton_4plus_low_conf_joints': sum(1 for count in low_conf if count >= 4),
            'lower_body_extrapolated_histogram': {str(key): value for key, value in sorted(lower_hist.items())},
            'leg_chain_histogram': {side: dict(counts) for side, counts in chain_hist.items()},
            'skeleton_median_body_mean_conf': round(float(np.median(confs)), 4) if confs else None,
            'failed_pair_ids': sorted(row.get('pair_id', '') for row in failed)[:20]}
        for row in failed:
            print(f'FAILED {split}/{row.get("pair_id")}: {row.get("error")}', flush=True)
    atomic_json(out_root / 'prompts' / 'assemble_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def qc(args) -> None:
    out_root = Path(args.out)
    by_split = {}
    for split in args.splits.split(','):
        rows = [row for row in load_records(out_root / 'records' / split)
            if row.get('status') == 'complete']
        random.Random(args.seed).shuffle(rows)
        if rows:
            by_split[split] = rows
    if not by_split:
        raise SystemExit(f'no complete records under {out_root}/records -- run `run` first')
    # Spread the budget over the splits that actually have data: dividing by every
    # requested split silently shrinks the sample when a split has not started yet.
    per_split = max(1, round(args.count / len(by_split))) if args.count else args.per_split
    picked: list[tuple[str, dict]] = []
    for split, rows in by_split.items():
        picked.extend(list(zip([split] * len(rows), rows))[:per_split])
    lines = ['# PE 重写提示词抽样', '']
    panels = []
    height, label = 360, 22
    for split, row in picked:
        lines += [f"## {row['pair_id']} ({split}) seq {row['sequence_id']} "
                  f"src {row['source_stem']} -> tgt {row['target_stem']}", '',
            '**训练用（无图号）**', '', row['compiled_prompt'], '',
            '**推理用（A + 骨架）**', '', row['compiled_prompt_inference'], '',
            '输入：', *[f"- {item['role']}: {item['path']}" for item in row['inputs']], '']
        images = [Image.open(item['path']).convert('RGB') for item in row['inputs']]
        tiles = [image.resize((max(1, round(image.width * height / image.height)), height))
            for image in images]
        panels.append((row['pair_id'], tiles))
    if not panels:
        raise SystemExit(f'no complete records under {out_root}/records -- run `run` first')
    report = out_root / 'qc'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'prompt_samples.md').write_text('\n'.join(lines), encoding='utf-8')
    gap = 12
    sheet_width = sum(tile.width for tile in panels[0][1]) + gap * (len(panels[0][1]) - 1)
    sheet = Image.new('RGB', (sheet_width, (height + label) * len(panels) + gap * (len(panels) - 1)),
        (250, 250, 250))
    try:
        from PIL import ImageDraw
        drawer = ImageDraw.Draw(sheet)
    except ImportError:
        drawer = None
    for row_index, (pair_id, tiles) in enumerate(panels):
        y = row_index * (height + label + gap)
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, y + label))
            x += tile.width + gap
        if drawer is not None:
            drawer.text((2, y), pair_id, fill=(20, 20, 20))
    sheet.save(report / 'inputs_grid.png')
    print(f'QC_PAIRS={len(picked)} report={report}', flush=True)


def stats(args) -> None:
    """Summarise throughput and output length of one records tree."""
    out_root = Path(args.out)
    rows = [row for split in ('train', 'val', 'test')
        for row in load_records(out_root / 'records' / split)] if (out_root / 'records').is_dir() \
        else [json.loads(path.read_text(encoding='utf-8'))
            for path in sorted((out_root / 'records').glob('*.json'))]
    complete = [row for row in rows if row.get('status') == 'complete']
    failed = [row for row in rows if row.get('status') != 'complete']
    seconds = [row['seconds'] for row in complete]
    tokens = [row['tokens'] for row in complete]
    words = [len(row['compiled_prompt'].split()) for row in complete]
    summary = {'out': str(out_root), 'records': len(rows), 'complete': len(complete),
        'failed': len(failed), 'thinking': sorted({str(row.get('enable_thinking', 'unknown')) for row in complete}),
        'max_image_side': sorted({str(row.get('max_image_side', 'unknown')) for row in complete}),
        'seconds_mean': round(float(np.mean(seconds)), 1) if seconds else None,
        'seconds_min': round(min(seconds), 1) if seconds else None,
        'seconds_max': round(max(seconds), 1) if seconds else None,
        'tokens_mean': int(np.mean(tokens)) if tokens else None,
        'words_mean': int(np.mean(words)) if words else None,
        'words_min': min(words) if words else None,
        'words_max': max(words) if words else None,
        'retried_pairs': sum(1 for row in complete if len(row.get('attempts', [])) > 1),
        'errors': sorted({row.get('error', '')[:120] for row in failed}),
        'hours_per_1000_pairs': round(float(np.mean(seconds)) / 3.6, 1) if seconds else None}
    path = out_root / 'stats.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EXP_ROOT)
    sub = parser.add_subparsers(dest='command', required=True)

    run_p = sub.add_parser('run')
    run_p.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    run_p.add_argument('--pairs-root', type=Path, default=PAIRS_ROOT)
    run_p.add_argument('--pe-model', type=Path, default=PE_MODEL)
    run_p.add_argument('--limit', type=int)
    run_p.add_argument('--pair-id', nargs='+')
    run_p.add_argument('--sample-mode', choices=('random', 'head'), default='random')
    run_p.add_argument('--seed', type=int, default=20261014)
    run_p.add_argument('--pe-seed', type=int, default=42)
    run_p.add_argument('--overwrite', action='store_true')
    run_p.add_argument('--dry-run', action='store_true')
    run_p.add_argument('--max-image-side', type=int, default=0,
        help='Downscale the longest input side before feeding PE (0 keeps native size)')
    run_p.add_argument('--no-thinking', action='store_true',
        help='Disable the PE thinking pass; much faster, quality unverified')
    run_p.add_argument('--shard-index', type=int, default=0)
    run_p.add_argument('--shard-count', type=int, default=1)
    run_p.add_argument('--mamba-dtype', default=None,
        help='Override the PE config mamba_ssm_dtype (e.g. bfloat16) so the FLA chunk kernel engages')

    as_p = sub.add_parser('assemble')
    as_p.add_argument('--pairs-root', type=Path, default=PAIRS_ROOT)

    qc_p = sub.add_parser('qc')
    qc_p.add_argument('--splits', default='train,val,test')
    qc_p.add_argument('--per-split', type=int, default=3)
    qc_p.add_argument('--count', type=int, default=None)
    qc_p.add_argument('--seed', type=int, default=7)

    st_p = sub.add_parser('stats')

    args = parser.parse_args()
    if args.command == 'qc' and args.count:
        args.per_split = max(1, round(args.count / len(args.splits.split(','))))
    {'run': run, 'assemble': assemble, 'qc': qc, 'stats': stats}[args.command](args)


if __name__ == '__main__':
    main()

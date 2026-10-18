#!/usr/bin/env python3
"""Ground-truth-free scoring for the workflow, written next to (never on top of) evaluate.py.

Run with the workflow interpreter, e.g.
    /home/shangguanrz/nihaochen/envs/qwen21/bin/python evaluate2.py --all
    /home/shangguanrz/nihaochen/envs/qwen21/bin/python evaluate2.py --run hard_05 --fit-source

evaluate.py answers "did the body land on the requested skeleton", using SMPLest-X on both
sides. This adds the four things it cannot answer, without needing an edited reference:

  pose      the same guide-versus-output comparison re-read by DWPose, an architecturally
            different estimator, plus the out-of-frame accounting that inflates the existing
            PCK and a strict variant that charges dropped joints as misses;
  shape     limb-length ratios, which a pose change must not alter, source versus generated;
  identity  ArcFace cosine between the source face and the generated face;
  scene     background and person region MAE/PSNR/SSIM against the source image, and what
            happened to the other people in the frame.

Artifacts go to ``evaluation/eval2.json``, ``evaluation/eval2_overlay.png`` and
``evaluation/stage2.json``. ``eval.json`` and ``stage.json`` belong to evaluate.py; this
script only reads them. Runs whose generated image is byte-identical to an already scored
one are recorded as ``duplicate`` instead of being counted twice.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import no_gt_metrics as M
from src.pose_edit.common import (atomic_json, finish_stage, make_request, model_signature,
    read_json, sha256_file, stage_reasons)
from src.pose_edit.geometry import render_poseguide

ROOT = Path(__file__).resolve().parent
CODE = ['workflow/evaluate2.py', 'src/evaluation/no_gt_metrics.py', 'src/pose_edit/common.py',
    'src/pose_edit/geometry.py']
DWPOSE_DIR = REPO_ROOT / 'models' / 'DWPose'
ANTELOPE_DIR = REPO_ROOT / 'models' / 'antelopev2'
STAGE_NAME = 'evaluate2'


def source_image(run_dir):
    for name in ('A_rgb.png', 'A_reference.png'):
        path = run_dir / 'inputs' / name
        if path.is_file():
            return path
    return None


def points134(array):
    return np.asarray(array, dtype=float).reshape(M.DWSLOT_COUNT, 2)


def read_pixels(path, size=None):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if size is not None and image.size != tuple(size):
            image = image.resize(tuple(size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64), image.size


def guide_in_final_coords(run_dir, final_size):
    """The requested skeleton in generated-image pixels, plus how much of it left the frame."""
    with np.load(run_dir / 'geometry' / 'B_keypoints.npz') as arrays:
        canvas = points134(arrays['keypoints_generation_px'])
        valid_canvas = np.asarray(arrays['valid_generation'], dtype=bool)
    preprocess = read_json(run_dir / 'inputs' / 'source_preprocess.json')
    offset = np.asarray(preprocess['content_box'][:2], dtype=float)
    guide = canvas - offset
    valid = valid_canvas & M.validity(guide, *final_size)
    return guide, valid, {'offset_from_canvas': offset.tolist(),
        'guide_joints_canvas': int(valid_canvas.sum()),
        'guide_joints_final': int(valid.sum()),
        'guide_joints_out_of_frame': int((valid_canvas & ~valid).sum())}


def open_dwpose():
    """ONNX sessions for the second reader, pinned to CPU on purpose.

    The CUDA execution provider in this onnxruntime build cannot load (missing
    ``libcublasLt.so.13``) and floods stderr with retries; the four-arm evaluator this
    metric is meant to cross-check against also runs on CPUExecutionProvider, so CPU keeps
    the two readers comparable and the scores reproducible across machines.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ['CPUExecutionProvider']
    detector = ort.InferenceSession(str(DWPOSE_DIR / 'yolox_l.onnx'), sess_options=options,
        providers=providers)
    pose = ort.InferenceSession(str(DWPOSE_DIR / 'dw-ll_ucoco_384.onnx'), sess_options=options,
        providers=providers)
    return detector, pose


def open_identity():
    """ArcFace is optional: the SMPLest-X interpreter has no insightface build."""
    try:
        from src.evaluation.joint_pilot_metrics import IdentityMetric
        return IdentityMetric(), None
    except Exception as error:
        return None, f'{type(error).__name__}: {error}'


def identity_block(identity, source_path, final_path, unavailable=None):
    if identity is None:
        return {'face_detected_source': None, 'face_detected_generated': None,
            'identity_cosine': None, 'status': 'skipped',
            'reason': unavailable or 'identity model unavailable'}
    with Image.open(source_path) as source, Image.open(final_path) as generated:
        embedding_source = identity.embedding(source.convert('RGB'))
        embedding_generated = identity.embedding(generated.convert('RGB'))
    cosine = None
    if embedding_source is not None and embedding_generated is not None:
        cosine = M.num(float(np.dot(embedding_source, embedding_generated)))
    return {'face_detected_source': embedding_source is not None,
        'face_detected_generated': embedding_generated is not None,
        'identity_cosine': cosine,
        'limits': 'ArcFace falls off with yaw and occlusion, so this is comparable only '
            'within one pair or against a calibrated ceiling; not across different people.'}


def ssim_metric():
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    return structural_similarity


def second_reader(path, sessions):
    with Image.open(path) as image:
        return M.dwpose134_on_image(image, *sessions)


def preservation_block(source_path, final_path, points_source, points_generated,
        torso_source, torso_generated, lpips_model=None, device='cpu'):
    generated_pixels, size = read_pixels(final_path)
    source_pixels, _ = read_pixels(source_path, size=size)
    mask_source = M.dilate(M.mask_from_limbs(points_source, size, torso=torso_source),
        0.30 * (torso_source or 40.0))
    mask_generated = M.dilate(M.mask_from_limbs(points_generated, size, torso=torso_generated),
        0.30 * (torso_generated or 40.0))
    metrics = M.region_metrics(source_pixels, generated_pixels, mask_source, mask_generated,
        ssim_function=ssim_metric())
    metrics['mask_kind'] = 'limb_hull_v1'
    metrics['source_resized_to'] = list(size)
    metrics['mask_area_ratio_source'] = M.num(float(mask_source.mean()))
    metrics['mask_area_ratio_generated'] = M.num(float(mask_generated.mean()))
    if lpips_model is not None:
        import torch

        both = mask_source | mask_generated
        outside = ~both
        def score(region):
            if not region.any():
                return None
            top, bottom = np.flatnonzero(region.any(axis=1))[[0, -1]]
            left, right = np.flatnonzero(region.any(axis=0))[[0, -1]]
            crop_a = source_pixels[top:bottom + 1, left:right + 1]
            crop_b = generated_pixels[top:bottom + 1, left:right + 1]
            def tensor(pixels):
                array = np.asarray(pixels, dtype=np.float32) / 127.5 - 1.0
                return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                return M.num(float(lpips_model(tensor(crop_a), tensor(crop_b)).item()))
        metrics['background_lpips'] = score(outside)
        metrics['person_lpips'] = score(both)
    return metrics


def open_lpips():
    """LPIPS is optional: the package is installed here but the weights may not be cached."""
    try:
        import torch
        from lpips import LPIPS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return LPIPS(net='alex').to(device).eval(), device
    except Exception as error:
        print(f'EVAL2 LPIPS_DISABLED {type(error).__name__}: {error}', flush=True)
        return None, 'cpu'


def beta_block(run_dir, source_params):
    generated_params_path = run_dir / 'evaluation' / 'final_smplx.pt'
    if source_params is None:
        # A previous `--fit-source` run with the geometry interpreter already cached the fit.
        cache = run_dir / 'evaluation' / 'source_smplx.pt'
        if cache.is_file():
            import torch
            source_params = torch.load(cache, map_location='cpu', weights_only=False)
    if source_params is None or not generated_params_path.is_file():
        return {'status': 'skipped',
            'reason': 'needs evaluation/final_smplx.pt (run evaluate.py) and a source fit '
                '(evaluate2.py --fit-source with the geometry interpreter)'}
    import torch
    generated = torch.load(generated_params_path, map_location='cpu', weights_only=False)
    block = M.beta_distance(source_params['betas'], generated['betas'])
    block['status'] = 'computed'
    return block


def fit_source(run_dir, config):
    """Optional SMPLest-X fit of A, for the beta metric and a mesh-accurate person mask."""
    if 'smplest_root' not in config:
        return None, 'config.json has no smplest_root'
    try:
        from src.pose_edit.estimator import estimate_single
        import torch
    except Exception as error:
        return None, f'estimator unavailable in this interpreter ({type(error).__name__})'
    cache = run_dir / 'evaluation' / 'source_smplx.pt'
    if cache.is_file():
        return torch.load(cache, map_location='cpu', weights_only=False), 'cache_hit'
    settings = {key: config[key] for key in
        ('smplest_root', 'smplest_checkpoint', 'person_detection_confidence')}
    try:
        params, staging = estimate_single(run_dir / 'inputs' / 'A_rgb.png',
            run_dir / 'evaluation', settings, config['smplest_root'])
    except Exception as error:
        return None, f'SMPLest-X failed on the source image: {type(error).__name__}: {error}'
    import shutil
    shutil.copyfile(staging['estimator_params'], cache)
    return params, 'fitted'


def subject_index(boxes, points):
    """Which detected source box the transferred person is, by skeleton-centre distance."""
    centre = np.nanmean(np.asarray(points[:14], dtype=float), axis=0)
    if not np.isfinite(centre).all():
        return 0
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    return int(np.argmin(np.linalg.norm(centres - centre, axis=1)))


def rescale(points, factor):
    """Keypoints/boxes from the source photo into generated-image pixels (see M.scale_points)."""
    return M.scale_points(points, factor)


def evaluate_run(run_dir, sessions, identity, config, fit=False, perceptual=None,
        perceptual_device='cpu', identity_reason=None):
    started = time.perf_counter()
    final_path = run_dir / 'generation' / 'final_image.png'
    keypoints = run_dir / 'geometry' / 'B_keypoints.npz'
    source_path = source_image(run_dir)
    missing = [label for label, ok in (('generation/final_image.png', final_path.is_file()),
        ('geometry/B_keypoints.npz', keypoints.is_file()),
        ('inputs source image', bool(source_path and source_path.is_file()))) if not ok]
    if missing:
        return {'status': 'incomplete', 'missing': missing, 'run': run_dir.name}
    stage_dir = run_dir / 'evaluation'
    stage_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(final_path) as image:
        size = image.size
    with Image.open(source_path) as image:
        source_size = image.size
    factor = (size[0] / source_size[0], size[1] / source_size[1])
    guide, guide_valid, guide_meta = guide_in_final_coords(run_dir, size)
    points_generated, scores_generated, boxes_generated = second_reader(final_path, sessions)
    points_source, scores_source, boxes_source = second_reader(source_path, sessions)
    if points_generated is None or points_source is None:
        return {'status': 'no_person_detected', 'run': run_dir.name,
            'detected': {'source': int(len(boxes_source)), 'generated': int(len(boxes_generated))}}
    points_source = rescale(points_source, factor)
    boxes_source = rescale(boxes_source, factor) if len(boxes_source) else boxes_source

    torso_generated = M.torso_px(points_generated)
    torso_source = M.torso_px(points_source)
    torso_guide = M.torso_px(guide)
    result = {'schema': 'single_person_no_gt_evaluation_v1', 'run': run_dir.name,
        'final_image': str(final_path), 'final_size': list(size),
        'source_image': str(source_path), 'source_size': list(source_size),
        'source_resample_factor': [M.num(value) for value in factor], 'guide': guide_meta,
        'torso_px': {'guide': M.num(torso_guide),
            'source_at_generated_size': M.num(torso_source),
            'generated': M.num(torso_generated)},
        # Both ratios matter and they answer different questions: guide is "did the generator
        # land the body at the requested size" (the scale-anchor failure mode the workflow
        # README measured at 2.35x on hard_05), source is "did the person keep their size".
        'scale_fidelity': {'torso_ratio_generated_over_guide': M.num(
            torso_generated / torso_guide if torso_guide else None),
            'torso_ratio_generated_over_source': M.num(
                torso_generated / torso_source if torso_source else None)},
        'persons_detected': {'source': int(len(boxes_source)),
            'generated': int(len(boxes_generated))},
        'pose_generated_vs_guide': M.pose_error_stats(guide, guide_valid, points_generated,
            M.validity(points_generated, *size), size=size, normaliser=torso_guide)}
    # Did the body move at all? A generator that reproduces the source pose scores 0 against
    # the guide while looking perfectly healthy on every preservation metric, so the two
    # readings have to sit side by side to be interpretable.
    result['pose_generated_vs_source'] = M.pose_error_stats(points_source,
        M.validity(points_source, *size), points_generated, M.validity(points_generated, *size),
        size=size, normaliser=torso_guide)

    first = run_dir / 'evaluation' / 'final_keypoints.npz'
    if first.is_file():
        with np.load(first) as arrays:
            smplx_points = points134(arrays['dwpose134_px'])
        result['pose_generated_vs_guide_smplestx'] = M.pose_error_stats(guide, guide_valid,
            smplx_points, M.validity(smplx_points, *size), size=size, normaliser=torso_generated)
        result['estimator_agreement'] = M.estimator_agreement(smplx_points, points_generated,
            size, normaliser=torso_guide)
    else:
        result['pose_generated_vs_guide_smplestx'] = {'status': 'skipped',
            'reason': 'run evaluate.py to get the SMPLest-X estimate'}
        result['estimator_agreement'] = {'status': 'skipped',
            'reason': 'run evaluate.py to get the SMPLest-X estimate'}

    ratios_source = M.limb_ratios(points_source)
    ratios_generated = M.limb_ratios(points_generated)
    result['shape'] = {'source': ratios_source, 'generated': ratios_generated,
        'drift': M.ratio_drift(ratios_source, ratios_generated)}
    result['identity'] = identity_block(identity, source_path, final_path,
        unavailable=identity_reason)
    result['preservation'] = preservation_block(source_path, final_path, points_source,
        points_generated, torso_source, torso_generated, lpips_model=perceptual,
        device=perceptual_device)
    if len(boxes_source) > 1:
        result['nonsubject'] = M.nonsubject_regions(boxes_source, boxes_generated,
            subject_index(boxes_source, points_source))
    source_params = None
    if fit:
        source_params, note = fit_source(run_dir, config)
        result['source_fit'] = {'status': note}
    result['beta'] = beta_block(run_dir, source_params)

    overlay = Image.blend(Image.open(final_path).convert('RGB'),
        render_poseguide(np.where(guide_valid[:, None], guide, np.nan), size), 0.4)
    Image.blend(overlay, render_poseguide(points_generated, size), 0.4).save(
        run_dir / 'evaluation' / 'eval2_overlay.png')
    result['status'] = 'complete'
    result['elapsed_seconds'] = round(time.perf_counter() - started, 2)
    return result


def request_for(run_dir, config, fit):
    paths = {}
    for key, path in (('final_image', run_dir / 'generation' / 'final_image.png'),
            ('B_keypoints', run_dir / 'geometry' / 'B_keypoints.npz'),
            ('source_preprocess', run_dir / 'inputs' / 'source_preprocess.json')):
        if path.is_file():
            paths[key] = sha256_file(path)
    source = source_image(run_dir)
    if source:
        paths['source_image'] = sha256_file(source)
    identity = {'settings': {'fit_source': bool(fit),
            'second_reader': 'dwpose-ll-ucoco-384', 'detector_input': 'original_size'},
        'inputs_sha256': paths,
        'model_identity': {'dwpose_detector': model_signature(DWPOSE_DIR / 'yolox_l.onnx'),
            'dwpose_pose': model_signature(DWPOSE_DIR / 'dw-ll_ucoco_384.onnx'),
            'arcface': model_signature(ANTELOPE_DIR / 'glintr100.onnx'),
            'scrfd': model_signature(ANTELOPE_DIR / 'scrfd_10g_bnkps.onnx')},
        'code': {name: sha256_file(REPO_ROOT / name) for name in CODE}}
    return make_request(STAGE_NAME, identity, run_dir,
        {'run_dir': str(run_dir), 'final_image': final_image_of(run_dir)})


def final_image_of(run_dir):
    return run_dir / 'generation' / 'final_image.png'


def summary_of(result):
    second = (result.get('pose_generated_vs_guide') or {}).get('summary') or {}
    agreement = result.get('estimator_agreement') or {}
    preservation = result.get('preservation') or {}
    drift = (result.get('shape') or {}).get('drift') or {}
    values = {'second_reader_body_pck@0.2': second.get('body_pck@0.2'),
        'second_reader_body_pck@0.2_strict': second.get('body_pck@0.2_strict'),
        'torso_ratio_guide': (result.get('scale_fidelity') or {}).get(
            'torso_ratio_generated_over_guide'),
        'torso_ratio_source': (result.get('scale_fidelity') or {}).get(
            'torso_ratio_generated_over_source'),
        'pose_change_mpjpe_norm': ((result.get('pose_generated_vs_source') or {}).get(
            'summary') or {}).get('body_mpjpe_norm'),
        'identity_cosine': result['identity'].get('identity_cosine'),
        'background_mae': preservation.get('background_mae'),
        'person_mae': preservation.get('person_mae'),
        'shape_max_rel_change': drift.get('max_rel_change'),
        'view_max_rel_change': drift.get('view_max_rel_change'),
        'estimator_agree@0.2': agreement.get('body_agree@0.2'),
        'estimator_verdict': agreement.get('verdict')}
    return {key: M.num(value) if isinstance(value, (int, float, np.floating)) or value is None
        else value for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=[], help='run name under runs/')
    parser.add_argument('--all', action='store_true', help='every run holding a generated image')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--fit-source', action='store_true',
        help='also fit A with SMPLest-X (run with the geometry interpreter) for the beta metric')
    parser.add_argument('--no-lpips', action='store_true',
        help='skip the perceptual background/person distance (LPIPS weights not cached)')
    parser.add_argument('--fit-only', action='store_true',
        help='only fit the source image with SMPLest-X and cache evaluation/source_smplx.pt; '
            'use the geometry interpreter, which has no insightface or lpips')
    args = parser.parse_args()
    config = read_json(ROOT / 'config.json')
    runs_dir = ROOT / 'runs'
    names = sorted(path.parent.parent.name for path in runs_dir.glob('*/generation/final_image.png')) \
        if args.all else list(args.run)
    if not names:
        parser.error('pass --run NAME (repeatable) or --all')

    if args.fit_only:
        for name in names:
            params, note = fit_source(runs_dir / name, config)
            print(f'EVAL2 FIT {name} {note}', flush=True)
        return

    sessions = open_dwpose()
    identity, identity_reason = open_identity()
    perceptual, perceptual_device = (None, 'cpu') if args.no_lpips else open_lpips()
    seen, scored, cached = {}, 0, 0
    for name in names:
        run_dir = runs_dir / name
        final_path = final_image_of(run_dir)
        if not final_path.is_file():
            print(f'EVAL2 SKIPPED {name} no generation/final_image.png', flush=True)
            continue
        digest = sha256_file(final_path)
        record = run_dir / 'evaluation' / 'stage2.json'
        if digest in seen:
            atomic_json(record, {'schema': 'single_person_no_gt_stage_v1', 'status': 'duplicate',
                'duplicate_of': seen[digest], 'image_sha256': digest})
            print(f'EVAL2 DUPLICATE {name} identical image to {seen[digest]}', flush=True)
            continue
        seen[digest] = name
        request = request_for(run_dir, config, args.fit_source)
        if record.is_file() and not args.force:
            reasons = stage_reasons(record, request, run_dir)
            if not reasons:
                cached += 1
                print(f'EVAL2 CACHE_HIT {name}', flush=True)
                continue
        try:
            result = evaluate_run(run_dir, sessions, identity, config, fit=args.fit_source,
                perceptual=perceptual, perceptual_device=perceptual_device,
                identity_reason=identity_reason)
            if result.get('status') != 'complete':
                print(f'EVAL2 SKIPPED {name} {result.get("status")} '
                    f'{"".join(result.get("missing") or [])}', flush=True)
                continue
            atomic_json(run_dir / 'evaluation' / 'eval2.json', result)
            artifacts = {'eval2': run_dir / 'evaluation' / 'eval2.json'}
            overlay = run_dir / 'evaluation' / 'eval2_overlay.png'
            if overlay.is_file():
                artifacts['overlay'] = overlay
            summary = summary_of(result)
            finish_stage(record, request, artifacts, {'summary': summary}, run_dir)
        except Exception as error:
            # A concurrent session may rename or prune a run directory mid-sweep; one lost
            # run must not abort the rest of the set.
            print(f'EVAL2 FAILED {name} {type(error).__name__}: {error}', flush=True)
            continue
        scored += 1
        print('EVAL2_COMPLETE ' + name + ' ' + ' '.join(f'{key}={value:.3f}'
            for key, value in summary.items() if isinstance(value, float)), flush=True)
    print(f'EVAL2 SCORED {scored} new, {cached} cached, {len(seen)} unique images of '
        f'{len(names)} runs', flush=True)


if __name__ == '__main__':
    main()

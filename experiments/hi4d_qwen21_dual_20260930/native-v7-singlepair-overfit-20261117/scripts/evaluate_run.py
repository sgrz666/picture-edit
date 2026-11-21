"""Offline pilot evaluation: fixed V7 loss, SDPose images, or CPU face identity.

Use qwen21 for fixed-probe / identity-metrics, the SDPose environment for images.
Every invocation requires a new report directory; rendered inputs are read only.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import hashlib
import json
import math
import sys
import numpy as np

PROJECT = Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0, str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))
BODY21_COMMON = (16, 18, 20, 15, 17, 19, 1, 4, 7, 0, 3, 6)
BODY18_COMMON = tuple(range(2, 14))
SIGMAS = (.2, .5, .8)
NOISE_SEED = 24042


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def reference(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            sha.update(chunk)
    return {'path': str(path), 'sha256': sha.hexdigest()}


def write_json(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf8') as file:
        file.write(serialized)


def new_report(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def generation_records(paths):
    records = []
    for path in paths:
        report = read(path)
        if not report.get('records'):
            raise ValueError('Generation report has no records: ' + str(path))
        for record in report['records']:
            if reference(record['path'])['sha256'] != record['sha256']:
                raise ValueError('Generation SHA differs: ' + record['path'])
            records.append({'generation_report': reference(path), 'checkpoint': report.get('checkpoint'), 'record': record})
    return records


def valid_points(points, scores, hw):
    points = np.asarray(points)
    return (np.isfinite(points).all(-1) & np.isfinite(scores) & (np.asarray(scores) >= .30)
            & (points[..., 0] >= 0) & (points[..., 0] < hw[1])
            & (points[..., 1] >= 0) & (points[..., 1] < hw[0]))


def summarize(errors, normalized):
    return None if not len(errors) else {'valid_joints': len(errors), 'mpjpe_px': float(np.mean(errors)),
        'nme': float(np.mean(normalized)), **{f'pck@{v}': float(np.mean(np.asarray(normalized) <= v)) for v in (.1, .2, .5)}}


def pose_metrics(requested, projected_valid, people, hw, *, raw_count):
    """Score only the 12 anatomically identical Body21 / Body18 joints."""
    requested = np.asarray(requested, dtype=float)
    request_valid = np.asarray(projected_valid, bool) & valid_points(requested, np.ones((2, 21)), hw)
    result = {'raw_persons_detected': int(raw_count), 'fg_persons_detected': len(people),
              'assignment': None, 'all': None, 'persons': [None, None], 'reason': None}
    if len(people) != 2:
        return {**result, 'reason': 'foreground_person_count_not_two'}
    detected = np.stack([person['points'] for person in people])
    detected_valid = np.stack([valid_points(p['points'], p['scores'], hw) for p in people])
    if not request_valid[:, [0, 1]].all() or not detected_valid[:, [8, 11]].all():
        return {**result, 'reason': 'hip_center_unavailable'}
    hips = requested[:, [0, 1]].mean(1)
    detected_hips = detected[:, [8, 11]].mean(1)
    costs = [float(np.linalg.norm(hips - detected_hips[list(order)], axis=-1).sum()) for order in ((0, 1), (1, 0))]
    assignment = [0, 1] if costs[0] <= costs[1] else [1, 0]
    result.update(assignment=assignment, ambiguous=abs(costs[0] - costs[1]) < 40, assignment_costs_px=costs,
                  hip_gap_error_px=abs(float(np.linalg.norm(hips[0] - hips[1]) - np.linalg.norm(detected_hips[0] - detected_hips[1]))))
    gap = float(np.linalg.norm(hips[0] - hips[1]))
    result['hip_gap_error_rel'] = result['hip_gap_error_px'] / gap if gap > 0 else None
    all_errors, all_normalized = [], []
    for slot, person in enumerate(assignment):
        if not request_valid[slot, [0, 1, 15, 16]].all():
            result['persons'][slot] = {'reason': 'requested_torso_unavailable'}
            continue
        torso = float(np.linalg.norm(requested[slot, [15, 16]].mean(0) - hips[slot]))
        if torso <= 0:
            result['persons'][slot] = {'reason': 'requested_torso_degenerate'}
            continue
        keep = request_valid[slot, BODY21_COMMON] & detected_valid[person, BODY18_COMMON]
        errors = np.linalg.norm(requested[slot, BODY21_COMMON] - detected[person, BODY18_COMMON], axis=-1)[keep]
        normalized = errors / torso
        ratio = (float(np.linalg.norm(detected[person, [2, 5]].mean(0) - detected_hips[person])) / torso
                 if detected_valid[person, [2, 5]].all() else None)
        result['persons'][slot] = {**(summarize(errors, normalized) or {'valid_joints': 0, 'reason': 'no_common_joints'}),
                                   'torso_length_px': torso, 'torso_scale_ratio': ratio}
        all_errors.extend(errors.tolist()); all_normalized.extend(normalized.tolist())
    result['all'] = summarize(all_errors, all_normalized)
    return result


def background_metrics(source, generated, valid, source_masks, projected):
    if source.shape != generated.shape or source.shape[:2] != np.asarray(valid).squeeze().shape:
        raise ValueError('Source/generated/valid Canvas grids differ; implicit resizing forbidden')
    hw = source.shape[:2]
    foreground = np.asarray(source_masks).reshape(-1, *hw).any(0) | np.asarray(projected).reshape(-1, *hw).any(0)
    keep = np.asarray(valid, bool).squeeze() & ~foreground
    delta = source.astype(float)[keep] - generated.astype(float)[keep]
    mse = float(np.mean(delta**2)) if delta.size else 0
    return {'valid_pixels': int(keep.sum()), 'mae_rgb_0_255': float(np.abs(delta).mean()) if delta.size else None,
            'psnr_db': 10 * math.log10(255**2 / mse) if mse > 0 else None}


def gpu_budget(fraction):
    import torch
    if not 0 < fraction <= .90:
        raise ValueError('Evaluation GPU fraction must be in (0, 0.90]')
    free, total = torch.cuda.mem_get_info()
    if fraction * total > free - 2 * 1024**3:
        raise RuntimeError('Requested GPU cap plus 2GiB margin exceeds free memory')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return {'memory_fraction': fraction, 'free_before_bytes': free, 'total_gpu_bytes': total}


def fixed_probe(args, config):
    import torch
    from src.pose_control.qwen21_full.native_v7.training import TrainingConfigV7, DataPlanV7, NativeBackendV7
    from src.pose_control.qwen21_full.native_v62.training.data import route_options
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
    from src.pose_control.qwen21_full.native_v62.training.io import verify
    cfg = TrainingConfigV7.from_dict(config)
    if cfg.stage.phase != 'S3' or len(cfg.stage.routes) != 1 or cfg.stage.routes[0].branch != 'joint':
        raise ValueError('Fixed comparison requires the actual final S3 JOINT objective/config')
    if args.initial_checkpoint is None:
        raise ValueError('--initial-checkpoint required for fixed-probe')
    checkpoints = {'initial': reference(args.initial_checkpoint), 'trained': verify(cfg.initialization)}
    records, provenance = {}, {}
    for label, checkpoint in checkpoints.items():
        current = replace(cfg, initialization=checkpoint)
        data = DataPlanV7(current)
        saved = (load_committed(checkpoint, schema=cfg.schema)[0] if label == 'trained'
                 else torch.load(checkpoint['path'], map_location='cpu', mmap=True, weights_only=True))
        if label == 'trained' and saved['training_state']['global_update'] != 300:
            raise ValueError('Pilot final checkpoint must contain 300 real optimizer updates')
        if label == 'trained' and any(saved['training_state']['config'][key] != cfg.record()[key]
                for key in ('stage', 'sources', 'model_manifest', 'cache_identity', 'paper_loss', 'qwen_condition_profile')):
            raise ValueError('Trained checkpoint objective/data/model differs from final generation config')
        backend = NativeBackendV7(current, data, saved=saved, purpose='grad')
        backend.runtime.eval()
        evaluators = backend.ensure_evaluators(backend.required_evaluators)
        route = current.stage.routes[0]
        provenance[label] = {'checkpoint': checkpoint, 'backend': backend.identity, 'objective': current.record(),
                             'runtime_training': backend.runtime.training, 'vae_tiling': bool(backend.pipe.vae.use_tiling)}
        records[label] = []
        with torch.no_grad():
            for source, dataset in data.datasets.items():
                for index in range(len(dataset)):
                    sample, batch, pair = backend.pair(source, index)
                    options, base = route_options(route, sample['native_scene_kind'])
                    generator = torch.Generator(device=backend.device).manual_seed(NOISE_SEED)
                    noise = torch.randn(pair['target_latent'].shape, device=backend.device, dtype=pair['target_latent'].dtype, generator=generator)
                    for sigma_value in SIGMAS:
                        entry = {'route': route.name, 'sigma': sigma_value, 'status': data.status[(route.name, source, index)]}
                        counts = backend.window_counts([entry], {route.name: route}, update=4)
                        sigma = torch.tensor([sigma_value], device=backend.device)
                        prediction, target, coverage, noisy = backend.forward(batch, pair, route, sigma, noise, options, base)
                        loss, parts = backend.compute_loss(prediction, target, coverage, batch['training_supervision_v5'].to(backend.device),
                            noisy, sigma, (batch['tgt_image'].to(backend.device).float() + 1) * .5, route.loss, counts, 4, evaluators,
                            spatial=batch.get('spatial_weights_v5'))
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite fixed objective')
                        values = {key: float(value.detach()) if torch.is_tensor(value) else value for key, value in parts.items()}
                        records[label].append({'source_id': source, 'pair_id': sample['pair_metadata']['pair_id'], 'sigma': sigma_value, 'parts': values})
                        del prediction, target, coverage, noisy, loss, parts
                    del sample, batch, pair, noise, sigma, evaluators
                    evaluators = tuple(backend.evaluators)
        del backend, saved, data, evaluators
        gc.collect(); torch.cuda.empty_cache()
    deltas = []
    for initial, trained in zip(records['initial'], records['trained'], strict=True):
        if any(initial[key] != trained[key] for key in ('source_id', 'pair_id', 'sigma')):
            raise ValueError('Probe rows do not align')
        deltas.append({key: trained[key] for key in ('source_id', 'pair_id', 'sigma')} | {'trained_minus_initial':
            {key: value - initial['parts'][key] for key, value in trained['parts'].items()
             if type(value) in (int, float) and type(initial['parts'].get(key)) in (int, float)}})
    return {'kind': 'fixed_noise_joint_objective', 'sigmas': SIGMAS, 'noise_seed_per_sample': NOISE_SEED,
            'auxiliary_update': 4, 'optimizer_created': False, 'provenance': provenance, 'records': records, 'deltas': deltas}


def identity_summary(matrix):
    diagonal = [float(matrix[p, p]) for p in range(2)]
    return {'name': 'target_aligned_source_identity_conflates_placement_and_identity', 'source_to_generated_2x2': matrix.tolist(),
            'diagonal': diagonal, 'diagonal_mean': float(np.mean(diagonal)), 'swap_margin_on_mean': .02,
            'swap_suspected': bool(matrix[0, 1] + matrix[1, 0] > matrix[0, 0] + matrix[1, 1] + .04)}


def aligned_identity(sample, generated, qualification, model):
    import torch
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.face_loss_alignment import align_face_rgb
    source_canvas = CanvasSpec.from_record(sample['source_canvas'])
    target_canvas = CanvasSpec.from_record(sample['target_canvas'])
    source = ((sample['src_image'].float() + 1) * .5)[None].expand(2, -1, -1, -1)
    pixels = torch.from_numpy(generated.copy()).permute(2, 0, 1).float()[None].expand(2, -1, -1, -1) / 255
    source_points = source_canvas.pixel_points(np.asarray(qualification['source']['landmarks5_original']))
    target_points = target_canvas.pixel_points(np.asarray(qualification['target']['landmarks5_original']))
    with torch.no_grad():
        source_features = torch.nn.functional.normalize(model(align_face_rgb(source, torch.tensor(source_points).float()) * 2 - 1), dim=-1)
        generated_features = torch.nn.functional.normalize(model(align_face_rgb(pixels, torch.tensor(target_points).float()) * 2 - 1), dim=-1)
    indices = sample['local_condition_v4'].source_indices[0]
    matrix = (source_features[indices] @ generated_features.T).cpu().numpy()
    return identity_summary(matrix)


def identity_metrics(args, config):
    from PIL import Image
    from src.data.canvas_contract import CanvasSpec
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v62.training.io import verify
    from src.pose_control.qwen21_full.native_v5.losses import frozen_evaluators
    if not args.prepared or not args.identity_model or not args.generation_report:
        raise ValueError('identity-metrics requires --prepared, --identity-model, --generation-report')
    model_ref = reference(args.identity_model)
    model, _ = frozen_evaluators(model_ref['path'], None, device='cpu', sha=[model_ref['sha256'], None])
    qualifications = {row['pair_id']: row for row in read(args.prepared)['qualification']}
    datasets = {source['source_id']: NativeFullDatasetV62(verify(source['manifest'])['path'], load_target=True,
                data_profile=source['data_profile']) for source in config['sources']}
    results, controls = [], {}
    for item in generation_records(args.generation_report):
        record = item['record']; dataset = datasets[record['source_id']]
        sample = dataset[next(i for i, row in enumerate(dataset.rows) if row['pair_id'] == record['pair_id'])]
        pixels = np.asarray(Image.open(record['path']).convert('RGB'))
        if pixels.shape[:2] != CanvasSpec.from_record(sample['target_canvas']).canvas_hw:
            raise ValueError('Identity input differs from target Canvas')
        key = (record['source_id'], record['pair_id'])
        if key not in controls:
            qualification = qualifications[record['pair_id']]
            # Target RGB is an evaluation-only positive control, never a generation condition.
            target = ((sample['tgt_image'].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
            gt_identity = aligned_identity(sample, target, qualification, model)
            preparation = np.asarray(qualification['source_target_similarity'])
            controls[key] = {'source_id': key[0], 'pair_id': key[1], 'gt_identity': gt_identity,
                'preparation_source_to_gt_2x2': preparation.tolist(),
                'diagonal_delta_vs_preparation': (np.asarray(gt_identity['diagonal']) - preparation.diagonal()).tolist(),
                'source_indices': sample['local_condition_v4'].source_indices[0].tolist(),
                'target_person_ids': sample['local_condition_v4'].metadata[0]['target_person_ids'],
                'pair_metadata': sample['pair_metadata'], 'target_canvas': sample['target_canvas']}
        results.append({**item, 'identity': aligned_identity(sample, pixels, qualifications[record['pair_id']], model)})
    return {'kind': 'target_aligned_source_identity', 'identity_model': model_ref,
            'prepared': reference(args.prepared), 'positive_controls': list(controls.values()), 'records': results}


def image_metrics(args, config):
    import cv2
    import torch
    from PIL import Image, ImageDraw
    from src.data.canvas_contract import CanvasSpec
    from src.evaluation import sdpose
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v62.training.io import verify
    if not args.generation_report:
        raise ValueError('--generation-report required for image-metrics')
    records = generation_records(args.generation_report)
    datasets = {source['source_id']: NativeFullDatasetV62(verify(source['manifest'])['path'], load_target=False,
                data_profile=source['data_profile']) for source in config['sources']}
    runtime = sdpose.load_runtime(device=args.pose_device)
    results = []
    for number, item in enumerate(records):
        record = item['record']; dataset = datasets[record['source_id']]
        index = next(i for i, row in enumerate(dataset.rows) if row['pair_id'] == record['pair_id'])
        sample = dataset[index]
        canvas = CanvasSpec.from_record(sample['target_canvas'])
        if CanvasSpec.from_record(sample['source_canvas']).metadata() != canvas.metadata():
            raise ValueError('Background comparison requires identical Source/Target Canvas')
        image = Image.open(record['path']).convert('RGB'); pixels = np.asarray(image)
        if pixels.shape[:2] != canvas.canvas_hw or list(image.size) != list(record['size']):
            raise ValueError('Generated pixels differ from recorded native Canvas')
        requested = np.stack([sample['joint_uv' + suffix].numpy() for suffix in ('', '_b')])
        projected_valid = np.stack([sample['projection_valid' + suffix].numpy() for suffix in ('', '_b')])
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        people, raw = sdpose.detect_people(bgr, runtime, order='geometry', sort_by='score', return_stats=True, min_area=args.min_area)
        pose = pose_metrics(requested, projected_valid, people, canvas.canvas_hw, raw_count=raw)
        pose['detections'] = [{'box_xyxy': person['box_xyxy'], 'box_score': person['box_score'],
            'confident_points': [{'slot134': int(joint), 'xy': person['points'][joint].tolist(), 'score': float(person['scores'][joint])}
                for joint in np.flatnonzero(valid_points(person['points'], person['scores'], canvas.canvas_hw))]} for person in people]
        region_counts = []
        for support in sample['dual_depth'].projected[0, :, 0].numpy():
            yy, xx = np.nonzero(support)
            if not len(xx):
                region_counts.append(None); continue
            left, right, top, bottom = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
            dx, dy = int(.2 * (right - left)), int(.2 * (bottom - top))
            crop = bgr[max(0, top - dy):min(bgr.shape[0], bottom + dy), max(0, left - dx):min(bgr.shape[1], right + dx)]
            region_people, region_raw = sdpose.detect_people(crop, runtime, order='geometry', sort_by='score', return_stats=True, min_area=args.min_area)
            region_counts.append({'foreground': len(region_people), 'raw': region_raw})
        pose.update(region_count=region_counts, pose_basis='full_image_common12', common_body21_slots=BODY21_COMMON)
        source = ((sample['src_image'].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        background = background_metrics(source, pixels, sample['valid_image_mask'].numpy(),
                                        sample['source_person_mask'].numpy(), sample['dual_depth'].projected.numpy())
        overlay = image.copy(); draw = ImageDraw.Draw(overlay)
        for slot in range(2):
            for xy in requested[slot, BODY21_COMMON][projected_valid[slot, BODY21_COMMON]]:
                if np.isfinite(xy).all():
                    x, y = xy; draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline='white', width=2)
        for person, detected in enumerate(people):
            slot = pose['assignment'].index(person) if pose['assignment'] else person
            keep = valid_points(detected['points'], detected['scores'], canvas.canvas_hw)
            for x, y in np.asarray(detected['points'])[list(BODY18_COMMON)][keep[list(BODY18_COMMON)]]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=('lime', 'dodgerblue')[slot % 2])
        overlay_path = args.report_dir / f'overlay_{number:02d}.png'
        with overlay_path.open('xb') as file:
            overlay.save(file, format='PNG')
        results.append({**item, 'pose': pose, 'background': background, 'overlay': reference(overlay_path),
                        'target_canvas': sample['target_canvas'], 'pair_metadata': sample['pair_metadata']})
        print(json.dumps({'image_metrics_progress': number + 1, 'total': len(records), 'source_id': record['source_id'],
                          'variant': record['variant'], 'fg_persons_detected': len(people)}), flush=True)
    return {'kind': 'generated_image_metrics', 'pose_backend': 'sdpose_yolo11x_wholebody_geometry', 'pose_device': str(runtime.device),
            'pose_code': reference(sdpose.__file__), 'pose_spec': reference(Path(sdpose.__file__).parents[2] / 'docs/SDPOSE_POSE_EVAL_SPEC.md'),
            'pose_checkpoint_dir': str(runtime.ckpt_dir), 'yolo': reference(runtime.ckpt_dir / 'yolo11x.pt'),
            'score_threshold': .30, 'foreground_min_area_px2': args.min_area, 'records': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('fixed-probe', 'image-metrics', 'identity-metrics'), required=True)
    parser.add_argument('--config', type=Path, required=True, help='SHA-bound final-generation.json with the S3 JOINT objective')
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--generation-report', type=Path, action='append')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--identity-model', type=Path)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    parser.add_argument('--pose-device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--min-area', type=float, default=0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_area) or args.min_area < 0:
        parser.error('--min-area must be finite and nonnegative')
    new_report(args.report_dir)
    try:
        cpu = args.action == 'identity-metrics' or (args.action == 'image-metrics' and args.pose_device == 'cpu')
        budget = {'device': 'cpu'} if cpu else gpu_budget(args.memory_fraction)
        config = read(args.config)
        result = {'fixed-probe': fixed_probe, 'image-metrics': image_metrics, 'identity-metrics': identity_metrics}[args.action](args, config)
        result.update(config=reference(args.config), config_record=config, execution=budget, implementation=reference(__file__))
        write_json(args.report_dir / 'evaluation.json', result)
        print(json.dumps({'report': str(args.report_dir / 'evaluation.json'), 'action': args.action}))
    except Exception as error:
        write_json(args.report_dir / 'failure.json', {'action': args.action, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Known-geometry one-pair evaluation; truth is read only by the evaluator.

The native pipeline is loaded once. All default image comparisons use 512 RGB,
40 steps and seeds 42/43/44. Optional regression probes have their own explicit
step budget. Existing PNGs are reused only after configuration and image SHA checks.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image, ImageDraw

from src.evaluation.qwen21_dual_v8_metrics import (
    DWPosePeople, Dinov2Appearance, EvaluationTarget, PersonDetection,
    aggregate_quality, appearance_binding, array_sha256, assign_by_geometry,
    background_error, boxes_from_masks, file_sha256, generated_person_masks,
    generated_mesh_quality, heatmap_observations, mesh_contact_proxy, pck,
)


def configuration_sha256(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def expected_backbone_sha256(checkpoint_payload):
    """Require the trained checkpoint's declaration of its native tensor state."""
    value = checkpoint_payload.get('metadata', {}).get('backbone_sha256')
    if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
        raise ValueError('evaluation checkpoint requires a declared backbone_sha256')
    return value


def write_json(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def planned_runs(seeds, short_prompt, detailed_prompt):
    return [dict(key=f'{method}_{kind}_seed{seed}', method=method, prompt_kind=kind,
                 prompt=prompt, seed=seed, target_geometry_known=method == 'dual_v8')
            for seed in seeds for method in ('native', 'dual_v8')
            for kind, prompt in (('short', short_prompt), ('detailed', detailed_prompt))]


def reusable_output(path, expected_configuration_sha256):
    path = Path(path)
    if not path.exists():
        return None
    record_path = path.with_suffix('.json')
    if not record_path.exists():
        raise ValueError(f'existing image has no configuration record: {path}')
    record = json.loads(record_path.read_text())
    if record.get('configuration_sha256') != expected_configuration_sha256:
        raise ValueError(f'existing output configuration differs: {path}; choose a new output directory or --overwrite')
    if record.get('output_sha256') != file_sha256(path):
        raise ValueError(f'existing output image SHA differs: {path}')
    return record


def validate_rescore_destination(original, destination):
    original, destination = Path(original).resolve(), Path(destination).resolve()
    if original == destination or original in destination.parents:
        raise ValueError('rescore output must be outside the original evaluation directory')
    if destination.exists():
        raise ValueError('rescore requires a new output directory')


def validate_rescore_record(image_path, expected_configuration):
    """Allow scoring code changes while preserving the original generation claim."""
    image_path = Path(image_path).resolve()
    record_path = image_path.with_suffix('.json')
    record = json.loads(record_path.read_text())
    original = record.get('configuration')
    if not isinstance(original, dict) or configuration_sha256(original) != record.get('configuration_sha256'):
        raise ValueError(f'rescore original configuration digest mismatch: {record_path}')
    actual_conditions = {key: value for key, value in original.items() if key != 'code_sha256'}
    expected_conditions = {key: value for key, value in expected_configuration.items() if key != 'code_sha256'}
    if actual_conditions != expected_conditions:
        different = sorted(key for key in set(actual_conditions) | set(expected_conditions)
            if key not in actual_conditions or key not in expected_conditions or actual_conditions[key] != expected_conditions[key])
        raise ValueError(f'rescore generation conditions differ: {different}')
    if Path(record.get('output', '')).resolve() != image_path or record.get('output_sha256') != file_sha256(image_path):
        raise ValueError(f'rescore original PNG SHA/path mismatch: {image_path}')
    with Image.open(image_path) as image:
        if image.size != (512, 512) or image.format != 'PNG':
            raise ValueError('rescore requires original 512 generated PNGs')
    provenance = dict(original_record=str(record_path), original_record_sha256=file_sha256(record_path),
        original_output=str(image_path), original_output_sha256=record['output_sha256'],
        original_configuration=original, original_configuration_sha256=record['configuration_sha256'],
        original_generation_code_sha256=original.get('code_sha256'), generation_configuration_preserved=True)
    return record, provenance


def validate_rescore_generation_code(metadata, project=PROJECT):
    """Older PNGs have an aggregate code hash; verify their frozen core contract."""
    project = Path(project)
    saved = metadata.get('run_contract', {}).get('code_sha256')
    if not isinstance(saved, dict):
        raise ValueError('rescore requires checkpoint generation source hashes')
    paths = list((project / 'src/pose_control/qwen21_full').rglob('*.py'))
    paths += [project / 'src/data/hi4d_smplx_dataset.py']
    paths += [project / name for name in ('src/__init__.py', 'src/data/__init__.py', 'src/pose_control/__init__.py') if (project / name).exists()]
    current = {path.relative_to(project).as_posix(): file_sha256(path) for path in sorted(set(paths))}
    relevant_saved = {name: digest for name, digest in saved.items()
        if name.startswith('src/pose_control/qwen21_full/') or name in current}
    if current != relevant_saved:
        raise ValueError('rescore generation source differs from the checkpoint frozen contract')
    return current


def _scoring_provenance():
    paths = list((PROJECT / 'src/evaluation').rglob('*.py'))
    paths += [Path(__file__), PROJECT / 'scripts/eval/multihmr_rgb_worker.py']
    versions = _runtime_versions()
    for name in ('onnxruntime', 'insightface'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return dict(code_sha256={path.relative_to(PROJECT).as_posix(): file_sha256(path)
        for path in sorted(set(paths)) if path.is_file()}, runtime_versions=versions)


def _rescore_origin(original, run_plan, checkpoint_sha, checkpoint_step, backbone_sha):
    original = Path(original).resolve()
    run_path, probes_path = original / 'run.json', original / 'probes.json'
    run = json.loads(run_path.read_text())
    saved_checkpoint = run.get('checkpoint', {})
    if any(saved_checkpoint.get(name) != value for name, value in
           (('sha256', checkpoint_sha), ('payload_step', checkpoint_step), ('backbone_sha256', backbone_sha))):
        raise ValueError('rescore original checkpoint binding differs')
    required = {str((original / (spec['key'] + '.json')).resolve()) for spec in run_plan}
    declared = run.get('output_records', [])
    if len(declared) != len(required) or {str(Path(path).resolve()) for path in declared} != required:
        raise ValueError('rescore original run does not match the requested methods/prompts/seeds')
    probes = json.loads(probes_path.read_text())
    if probes.get('configuration_sha256') != configuration_sha256(probes.get('configuration', {})):
        raise ValueError('rescore original probes configuration digest mismatch')
    if probes.get('results') != run.get('probes'):
        raise ValueError('rescore original probe results disagree with run.json')
    for result in probes['results'].values():
        if not isinstance(result, dict):
            continue
        for name, path in result.items():
            if isinstance(path, str) and path.endswith('.png') and (name == 'output' or name.endswith('_output')):
                artifact = Path(path).resolve()
                if original not in artifact.parents:
                    raise ValueError('rescore original probe output escaped its directory')
                if artifact.with_suffix('.json').is_file():
                    record = json.loads(artifact.with_suffix('.json').read_text())
                    digest = configuration_sha256(record['configuration'])
                    reusable_output(artifact, digest)
                elif name == 'output' and result.get('output_sha256') == file_sha256(artifact):
                    pass
                else:
                    raise ValueError('rescore original probe PNG has no verified provenance')
    provenance = dict(original_directory=str(original), original_run_sha256=file_sha256(run_path),
        original_probes_file_sha256=file_sha256(probes_path), original_probes_configuration=probes['configuration'],
        original_probes_configuration_sha256=probes['configuration_sha256'],
        interpretation='original generation and probe evidence; no new generation or probe calls')
    return run, probes, provenance


def image_difference(first, second):
    first, second = np.asarray(first.convert('RGB')), np.asarray(second.convert('RGB'))
    if first.shape != second.shape:
        raise ValueError('probe images have different sizes')
    delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return dict(pixel_exact=bool(np.array_equal(first, second)),
                max_absolute_channel_error=int(delta.max()), mean_absolute_channel_error=float(delta.mean()),
                different_pixels=int(np.any(delta, axis=-1).sum()))


def target_replacement_probe(generate_fn, pipe, generation, evaluation_target, *, adapter=None,
                             baseline=None, output=None, **generation_kwargs):
    """Rerun the identical generation object after replacing actual evaluator GT.

    Neither the original nor the changed EvaluationTarget is a generate argument.
    A saved main comparison may supply the baseline; the replacement run is fresh.
    """
    before = evaluation_target.sha256
    fresh_calls = 2 if baseline is None else 1
    if baseline is None:
        baseline = generate_fn(pipe, generation, adapter=adapter, **generation_kwargs)
    # Save the original condition identity before changing the evaluator's data.
    changed_evaluator_target = evaluation_target.changed_in_memory()
    after = changed_evaluator_target.sha256
    repeated = generate_fn(pipe, generation, adapter=adapter, **generation_kwargs)
    if output is not None:
        repeated.save(output)
    result = image_difference(baseline, repeated)
    result.update(evaluator_target_before_sha256=before, evaluator_target_after_sha256=after,
                  evaluator_rgb_and_masks_changed=before != after,
                  fresh_generation_calls=fresh_calls, generation_object_reused=True,
                  target_object_passed_to_generate=False)
    return result


def _array(tensor):
    return tensor.detach().float().cpu().numpy()


def _panel(image, title, width=256):
    canvas = Image.new('RGB', (width, width + 44), 'white')
    canvas.paste(image.convert('RGB').resize((width, width)), (0, 44))
    ImageDraw.Draw(canvas).multiline_text((5, 4), title, fill='black', spacing=2)
    return canvas


def _board(path, panels, columns=4, width=256):
    rows = (len(panels) + columns - 1) // columns
    board = Image.new('RGB', (columns * width, rows * (width + 44)), 'white')
    for index, (image, title) in enumerate(panels):
        board.paste(_panel(image, title, width), ((index % columns) * width, (index // columns) * (width + 44)))
    board.save(path)


def _heatmap(value):
    value = np.asarray(value, dtype=np.float32)
    value = np.nan_to_num(value).clip(0, 1)
    return Image.fromarray(np.stack([255 * value, 255 * np.sqrt(value), 255 * (1 - value)], -1).astype(np.uint8))


def export_controls(generation, output):
    """Pure geometry routing and diagnostic images; no evaluator truth input."""
    import torch
    import torch.nn.functional as F
    from src.pose_control.qwen21_full.dual.routing import compute_routing
    geometry = generation.geometry
    geo, app, contact = compute_routing(geometry, (512, 512))
    mask, visible = geometry.amodal_mask, geometry.visible_owner
    valid_person = geometry.person_valid[..., None, None, None]
    active = (mask > 0) & valid_person
    a, b = active[:, 0], active[:, 1]
    da, db = geometry.depth[:, 0], geometry.depth[:, 1]
    overlap = a & b
    tie = overlap & torch.isclose(da, db, atol=1e-5, rtol=1e-5)
    owner_a = ((a & ~b) | (overlap & (da < db) & ~tie)).float() + .5 * tie
    owner_b = ((b & ~a) | (overlap & (db < da) & ~tie)).float() + .5 * tie
    owner = torch.stack((owner_a, owner_b), 1)
    expanded = F.max_pool2d(owner.flatten(0, 1), 17, 1, 8).reshape_as(owner)
    depth = torch.where(owner > 0, geometry.depth, torch.full_like(geometry.depth, float('inf')))
    near = -F.max_pool2d(-depth.flatten(0, 1), 17, 1, 8).reshape_as(depth)
    nearest = near.amin(1, keepdim=True)
    winners = (expanded > 0) & torch.isfinite(near) & torch.isclose(near, nearest, atol=1e-5, rtol=0)
    u = winners.float() / winners.sum(1, keepdim=True).clamp_min(1) * valid_person
    panels = []
    maps = dict(u=_array(u[0, :, 0]), v=_array(app[0, :, :, 0]).reshape(2, 512, 512),
                W_geometry=_array(geo[0, :, :, 0]).reshape(2, 512, 512), W_contact=_array(contact[0, :, 0]).reshape(512, 512))
    for actor in range(2):
        normal = ((_array(geometry.normal[0, actor]).transpose(1, 2, 0) + 1) * .5 * _array(mask[0, actor, 0])[..., None])
        panels.append((Image.fromarray((normal.clip(0, 1) * 255).astype(np.uint8)), f'actor {actor} amodal normal'))
        depth_value = _array(geometry.depth[0, actor, 0]); valid = depth_value > 0
        normalized = np.zeros_like(depth_value)
        if valid.any():
            normalized[valid] = (depth_value[valid] - depth_value[valid].min()) / max(float(np.ptp(depth_value[valid])), 1e-8)
        panels.extend([(_heatmap(normalized), f'actor {actor} metric-depth relative display'),
                       (_heatmap(_array(geometry.pose[0, actor]).max(0)), f'actor {actor} body25 confidence')])
        for name in ('u', 'v', 'W_geometry'):
            path = output / f'control_actor{actor}_{name}.png'
            _heatmap(maps[name][actor]).save(path)
            panels.append((_heatmap(maps[name][actor]), f'actor {actor} {name} actual weight'))
    panels.append((_heatmap(maps['W_contact']), 'W_contact actual weight'))
    _heatmap(maps['W_contact']).save(output / 'control_W_contact.png')
    np.savez_compressed(output / 'control_weights.npz', **maps)
    _board(output / 'controls_board.png', panels)
    return panels


def save_residual_heatmaps(adapter, output):
    maps = getattr(adapter, 'last_residual_maps', None)
    if not maps:
        return dict(available=False, reason='runtime_did_not_capture_spatial_residual_maps')
    panels, metadata = [], []
    for layer, record in sorted(maps.items(), key=lambda entry: int(entry[0])):
        value = _array(record['rms']) if hasattr(record['rms'], 'detach') else np.asarray(record['rms'])
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError('pilot residual map must be [1,H,W] or [H,W]')
        maximum = float(value.max()); sigma = float(record['sigma']); timestep = float(record['timestep'])
        panels.append((_heatmap(value / max(maximum, 1e-12)), f'L{layer} final total delta RMS\nmax={maximum:.4g} sigma={sigma:.4g}'))
        metadata.append(dict(layer=int(layer), maximum=maximum, sigma=sigma, timestep=timestep,
                             target_hw=list(record['target_hw']), normalization='per_layer_map_max_for_display'))
    _board(output, panels, columns=3)
    return dict(available=True, path=str(output), maps=metadata,
                interpretation='last_actual_forward_total_residual_not_branch_maps')


def _code_signature():
    paths = list((PROJECT / 'src/pose_control/qwen21_full').rglob('*.py'))
    paths += [Path(__file__), PROJECT / 'src/evaluation/qwen21_dual_v8_metrics.py', PROJECT / 'src/data/hi4d_smplx_dataset.py']
    worker = PROJECT / 'scripts/eval/multihmr_rgb_worker.py'
    if worker.is_file():
        paths.append(worker)
    return configuration_sha256({path.relative_to(PROJECT).as_posix(): file_sha256(path) for path in sorted(set(paths))})


def _runtime_versions():
    """Use the training run contract's package set without loading model weights."""
    result = {}
    for package in ('torch', 'torchvision', 'diffusers', 'transformers', 'numpy', 'scipy', 'pillow'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _native_code_sha256(model_path, pipe=None):
    """Hash actual component class files, or resolve those classes for reuse.

    The metrics-only path reads local class declarations and imports definitions;
    it never calls from_pretrained or creates a native model instance.
    """
    names = ('pipeline', 'transformer', 'vae', 'text_encoder')
    classes = {}
    if pipe is not None:
        classes = {'pipeline': type(pipe)}
        classes.update({name: type(getattr(pipe, name)) for name in names[1:] if getattr(pipe, name, None) is not None})
    else:
        index_path = Path(model_path) / 'model_index.json'
        index = json.loads(index_path.read_text()) if index_path.is_file() else {}
        declarations = dict(pipeline=['diffusers', index.get('_class_name')],
                            **{name: index.get(name) for name in names[1:]})
        for name, declaration in declarations.items():
            if not isinstance(declaration, (tuple, list)) or len(declaration) != 2 or not all(isinstance(value, str) for value in declaration):
                continue
            try:
                classes[name] = getattr(importlib.import_module(declaration[0]), declaration[1])
            except (ImportError, AttributeError):
                continue
    hashes = dict.fromkeys(names)
    for name, component_class in classes.items():
        try:
            hashes[name] = file_sha256(inspect.getfile(component_class))
        except (TypeError, OSError):
            pass
    return hashes


def _revision():
    result = subprocess.run(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _wholebody_truth(root, truth, explicit=None):
    from src.data.hi4d_smplx_dataset import letterbox_affine
    metadata = truth.metadata
    path = explicit or (root / 'research/hi4d_smplx_pilot_20261011/observations' /
                        f"{metadata.get('action', 'hug14')}_{int(metadata.get('target_frame', 62)):06d}_cam{int(metadata.get('camera', 4))}.npz")
    if not Path(path).is_file():
        return None, None, dict(available=False, reason='wholebody_observation_file_unavailable', path=str(path))
    with np.load(path, allow_pickle=False) as values:
        affine = letterbox_affine(int(values['width']), int(values['height']))
        xy = values['xy_native'] * affine[0, 0] + affine[:2, 2]
        scores = values['scores'].copy()
    return xy, scores, dict(available=True, path=str(path), sha256=file_sha256(path),
        provenance='actual_camera_dwpose_observations_not_fitted_hand_prior', confidence_threshold=.3)


def _face_embeddings(metric, rgb, masks):
    embeddings, failures = [None, None], [None, None]
    from src.data.hi4d_smplx_dataset import source_person_crops
    masks = np.asarray(masks)
    for actor in range(2):
        try:
            if (masks[actor] > .5).sum() < 16:
                raise ValueError('insufficient_generated_or_source_person_pixels')
            other = masks[1 - actor]
            # The crop helper validates both slots. An unused placeholder lies
            # outside this actor, preserving every scored exclusive pixel.
            if ((other > .5) & (masks[actor] <= .5)).sum() < 16:
                outside = np.flatnonzero(masks[actor].reshape(-1) <= .5)
                if len(outside) < 16:
                    raise ValueError('no_exclusive_space_for_missing_other_person')
                other = np.zeros_like(other)
                other.reshape(-1)[outside[:16]] = 1
            values, _, _ = source_person_crops(rgb, np.stack([masks[actor], other]))
            crop = Image.fromarray(values[0])
            embeddings[actor] = metric.embedding(crop)
            if embeddings[actor] is None:
                failures[actor] = 'face_not_detected'
        except Exception as exc:
            failures[actor] = f'{type(exc).__name__}: {exc}'
    return embeddings, failures


def _detector_calibration(metric, target_rgb, target_boxes, target_xy, target_scores, person_uids):
    """Diagnose the detector on evaluator GT; never repair generated detections."""
    result = dict(evaluator_only=True, input='actual_evaluation_target_rgb',
        input_rgb_sha256=array_sha256(target_rgb), used_to_modify_generated_detections=False,
        detected_people=None, assignment=None, undetected_person_uids=None, failure=None)
    try:
        if metric is None:
            raise ValueError('dwpose_metric_unavailable')
        detections = metric.detect(Image.fromarray(target_rgb))
        assignment = assign_by_geometry(detections, target_boxes, target_xy, target_scores)
        result.update(detected_people=len(detections), assignment=assignment,
            detections=[detection.as_dict() for detection in detections],
            undetected_person_uids=[uid for uid, index in zip(person_uids, assignment['indices']) if index < 0])
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
    return result


class GeneratedMeshMetric:
    """Optional local Multi-HMR, called only on generated PNGs after Qwen unload."""
    def __init__(self, root, geometry_manifest, device='cuda'):
        import torch
        from src.data.hi4d_smplx_dataset import smplx_part_labels
        repo = root / 'add/tools/multi-hmr'
        weight = repo / 'models/multiHMR/multiHMR_896_L.pt'
        local_dino = repo.parent / 'dinov2_e1277af'
        if not weight.is_file() or not local_dino.is_dir():
            raise FileNotFoundError('existing local Multi-HMR weights and dinov2 source are required; downloads are disabled')
        camera_record = Path(geometry_manifest).with_suffix('.json')
        information = json.loads(camera_record.read_text())
        self.K = np.asarray(information['K512'], np.float32)
        previous_cwd = Path.cwd()
        os.environ['MULTIHMR_DINOV2_REPO'] = str(local_dino)
        os.environ['XFORMERS_DISABLED'] = '1'
        sys.path.insert(0, str(repo))
        try:
            os.chdir(repo)
            from demo_smplx_only import load_model, open_image, forward_model
            self.model = load_model('multiHMR_896_L', device=torch.device(device)).eval()
        finally:
            os.chdir(previous_cwd)
        self.open_image, self.forward_model, self.device = open_image, forward_model, device
        body = self.model.smpl_layer['neutral_10'].bm_x
        self.labels = smplx_part_labels(body.lbs_weights.detach().cpu().numpy())
        self.metadata = dict(model='multiHMR_896_L', weight_sha256=file_sha256(weight),
            part_partition_sha256=array_sha256(self.labels), camera_source=str(camera_record),
            generated_rgb_only=True, units='predicted_camera_metres_proxy')

    def measure(self, image_path, target_boxes, target_xy, target_scores, endpoints, output):
        import torch
        image, _ = self.open_image(str(image_path), self.model.img_size, device=torch.device(self.device))
        K = self.K.copy(); K[:2] *= self.model.img_size / 512
        humans = self.forward_model(self.model, image, torch.tensor(K, device=self.device)[None], det_thresh=.2, nms_kernel_size=3)
        converted = [{key: _array(value) for key, value in human.items() if hasattr(value, 'detach')} for human in humans]
        arrays = {}
        for index, human in enumerate(converted):
            for key in ('v3d', 'j3d', 'j2d', 'shape', 'rotvec', 'transl_pelvis'):
                if key in human:
                    arrays[f'person_{index}_{key}'] = human[key]
        result = generated_mesh_quality(converted, self.labels, self.K, self.model.img_size,
            target_boxes, target_xy, target_scores, endpoints)
        np.savez_compressed(output, actor_assignment=np.asarray(result['actor_assignment']), **arrays)
        result['reconstruction_path'] = str(output)
        return result


def run_mesh_worker(args, outputs, geometry_path):
    """Invoke one isolated batch process, then authenticate its raw exports."""
    from src.pose_control.qwen21_full.dual.schema import SMPLX_JOINT_TO_PART
    python = args.mesh_python.resolve()
    if not python.is_file():
        raise ValueError(f'--mesh-python executable is missing: {python}')
    work = args.output / 'multihmr_raw'
    if work.exists():
        work = args.output / f'multihmr_raw_{time.time_ns()}'
    work.mkdir()
    camera_path = Path(geometry_path).with_suffix('.json')
    camera = json.loads(camera_path.read_text())
    request_rows = []
    for record in outputs:
        origin = record.get('generation_provenance')
        if origin is None:
            source_record = Path(record['output']).with_suffix('.json')
            snapshot = work / (record['configuration']['key'] + '_generation_record.json')
            shutil.copyfile(source_record, snapshot)
            original_record_sha = file_sha256(snapshot)
        else:
            original_record_sha = origin['original_record_sha256']
        request_rows.append(dict(key=record['configuration']['key'], image=str(Path(record['output']).resolve()),
            image_sha256=record['output_sha256'], original_configuration_sha256=record['configuration_sha256'],
            original_record_sha256=original_record_sha, output_npz=str((work / (record['configuration']['key'] + '.npz')).resolve())))
    request = dict(schema_version='dual_v8_multihmr_request_v1', root=str(args.root.resolve()), K512=camera['K512'],
        geometry_sidecar_sha256=file_sha256(camera_path), joint_to_part=list(SMPLX_JOINT_TO_PART), outputs=request_rows)
    request_path = work / 'request.json'; write_json(request_path, request)
    worker = PROJECT / 'scripts/eval/multihmr_rgb_worker.py'
    expected_worker_sha = file_sha256(worker)
    completed = subprocess.run([str(python), str(worker), '--request', str(request_path.resolve()), '--device', args.mesh_device])
    if completed.returncode != 0:
        raise RuntimeError(f'Multi-HMR worker initialization/batch failed (exit {completed.returncode}); inspect its traceback and rerun in a new rescore directory')
    result_path = work / 'results.json'; result = json.loads(result_path.read_text())
    provenance = result.get('provenance', {})
    if result.get('schema_version') != 'dual_v8_multihmr_result_v1' or provenance.get('request_sha256') != file_sha256(request_path) or provenance.get('worker_source_sha256') != expected_worker_sha:
        raise ValueError('Multi-HMR worker result request/source provenance mismatch')
    body_path = Path(provenance['body_partition_path']).resolve()
    if work.resolve() not in body_path.parents or file_sha256(body_path) != provenance['body_partition_sha256']:
        raise ValueError('Multi-HMR worker body partition SHA/path mismatch')
    with np.load(body_path, allow_pickle=False) as body:
        lbs, labels = body['lbs_weights'], body['part_labels'].copy()
        if lbs.shape != (10475, 55) or not np.isfinite(lbs).all() or not np.array_equal(labels, np.asarray(SMPLX_JOINT_TO_PART)[lbs.argmax(1)]):
            raise ValueError('Multi-HMR worker SMPL-X dominant-LBS partition mismatch')
    rows = result.get('records', [])
    if len(rows) != len(request_rows) or {row.get('key') for row in rows} != {row['key'] for row in request_rows}:
        raise ValueError('Multi-HMR worker batch is incomplete or duplicated')
    by_key, raw = {row['key']: row for row in rows}, {}
    for expected in request_rows:
        row = by_key[expected['key']]
        bindings = dict(original_output_sha256=expected['image_sha256'],
            original_configuration_sha256=expected['original_configuration_sha256'], original_record_sha256=expected['original_record_sha256'],
            request_sha256=provenance['request_sha256'], worker_provenance=provenance)
        if any(row.get(name) != value for name, value in bindings.items()):
            raise ValueError('Multi-HMR raw output original generation binding mismatch')
        sidecar = Path(expected['output_npz']).with_suffix('.json')
        if json.loads(sidecar.read_text()) != row:
            raise ValueError('Multi-HMR raw sidecar differs from the authenticated batch manifest')
        if row.get('failure') is not None:
            raw[expected['key']] = dict(failure=row['failure'], part_pair_distances=None, raw_worker_record=row)
            continue
        path = Path(row['output_npz']).resolve()
        if path != Path(expected['output_npz']) or file_sha256(path) != row['output_npz_sha256']:
            raise ValueError('Multi-HMR raw NPZ SHA/path mismatch')
        with np.load(path, allow_pickle=False) as arrays:
            count = int(arrays['person_count']); image_size = int(arrays['image_size'])
            if count != row['reconstructed_people'] or image_size != provenance['image_size'] or not np.array_equal(arrays['part_labels'], labels) or not np.array_equal(arrays['K512'], np.asarray(request['K512'], np.float32)):
                raise ValueError('Multi-HMR raw camera/topology/person count mismatch')
            humans = [{name: arrays[f'person_{index}_{name}'].copy() for name in ('v3d', 'j3d', 'j2d')}
                      for index in range(count)]
        raw[expected['key']] = dict(humans=humans, labels=labels, K512=np.asarray(request['K512']), image_size=image_size,
            reconstruction_path=str(path), reconstruction_sha256=row['output_npz_sha256'], raw_worker_record=row,
            raw_sidecar_path=str(sidecar.resolve()), raw_sidecar_sha256=file_sha256(sidecar))
    provenance.update(result_manifest=str(result_path.resolve()), result_manifest_sha256=file_sha256(result_path))
    return raw, provenance


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--evaluation-target', '--supervision', dest='evaluation_target', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/home/shangguanrz/project/pic-edit'))
    p.add_argument('--model', type=Path)
    p.add_argument('--switch-generation', type=Path)
    p.add_argument('--wholebody-observations', type=Path)
    p.add_argument('--dino', type=Path, default=Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code/models/dinov2-giant'))
    p.add_argument('--metric-device', default='cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--cfg', type=float, default=1.)
    p.add_argument('--no-kv-cache', action='store_true')
    p.add_argument('--geometry-strength', type=float, default=1.)
    p.add_argument('--appearance-strength', type=float, default=1.)
    p.add_argument('--contact-strength', type=float, default=1.)
    p.add_argument('--max-ratio', type=float)
    p.add_argument('--probe-seed', type=int, default=42)
    p.add_argument('--probe-steps', type=int, default=4)
    p.add_argument('--target-replacement-check', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--geometry-switch-check', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--zero-strength-check', action='store_true')
    p.add_argument('--cache-check', action='store_true')
    p.add_argument('--cfg-probes', type=float, nargs='*', default=[])
    p.add_argument('--native-checksum', action='store_true')
    p.add_argument('--face-metrics', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-dino', action='store_true')
    p.add_argument('--mesh-metrics', action='store_true')
    p.add_argument('--mesh-python', type=Path, help='existing isolated Multi-HMR Python; load once in a batch subprocess')
    p.add_argument('--mesh-device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--rescore-from', type=Path, help='rescore authenticated original PNGs into a new directory without Qwen generation')
    p.add_argument('--metrics-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p


def main():
    args = _parser().parse_args()
    if args.steps < 1 or args.probe_steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('positive steps and distinct seeds are required')
    if args.rescore_from is not None:
        if args.metrics_only or args.overwrite:
            raise ValueError('--rescore-from is explicit scoring-only and requires a new directory; do not combine --metrics-only/--overwrite')
        validate_rescore_destination(args.rescore_from, args.output)
    scoring_only = args.metrics_only or args.rescore_from is not None
    if args.mesh_python is not None:
        if not args.mesh_python.is_file():
            raise ValueError('--mesh-python must name an existing isolated executable')
        args.mesh_metrics = True
    import torch
    from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset, letterbox_affine
    from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
    from src.pose_control.qwen21_full.dual.precision import precision_policy
    from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint
    from src.pose_control.qwen21_full.dual.runtime import DETAILED_HUG_PROMPT, generate, load_native, prepared_for_qwen, tensor_model_sha256
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.model or args.root / 'models/Qwen-Image-2.1'
    dataset = Hi4DSMPLXDataset(args.generation)
    if len(dataset) != 1:
        raise ValueError('approved pilot evaluation requires exactly one paired sample')
    generation = dataset.generation()
    row = dataset.rows[0]
    source_rgb = (_array(generation.source.rgb[0]).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    source_masks = _array(generation.source.masks[0, :, 0])
    target_boxes = boxes_from_masks(_array(generation.geometry.amodal_mask[0, :, 0]))
    target_xy, target_scores = heatmap_observations(_array(generation.geometry.pose[0])[:, :24])
    strengths = ControlStrengths(args.geometry_strength, args.appearance_strength, args.contact_strength, args.max_ratio)
    run_plan = planned_runs(args.seeds, generation.prompts[0], DETAILED_HUG_PROMPT)
    code_sha = _code_signature()
    dependencies = dict(generation_sha256=file_sha256(args.generation),
        source_rgb_sha256=file_sha256(row['source_rgb']), source_masks_sha256=[file_sha256(path) for path in row['source_masks']],
        source_features_sha256=file_sha256(row['source_features']), geometry_sha256=file_sha256(row['target_geometry']),
        code_sha256=code_sha, model_id=str(model_path.resolve()), resolution=512,
        precision_policy=precision_policy())
    checkpoint_sha = file_sha256(args.checkpoint)
    common = dict(steps=args.steps, cfg=args.cfg, use_kv_cache=not args.no_kv_cache)
    outputs, probes, adapter, pipe = [], {}, None, None
    checkpoint_payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    checkpoint_step = checkpoint_payload.get('step')
    if checkpoint_payload.get('metadata', {}).get('architecture_version') != 'dual_v8' or not isinstance(checkpoint_step, int) or checkpoint_step <= 0:
        raise ValueError('evaluation requires a newly trained dual_v8 checkpoint with a positive payload step')
    expected_backbone = expected_backbone_sha256(checkpoint_payload)
    dependencies['backbone_sha256'] = expected_backbone
    rescore_provenance, original_run, original_probes = None, None, None
    if args.rescore_from is not None:
        metadata = checkpoint_payload['metadata']
        verified_code = validate_rescore_generation_code(metadata)
        if metadata.get('model_id') != str(model_path.resolve()) or metadata.get('model_config_sha256') != file_sha256(model_path / 'transformer/config.json'):
            raise ValueError('rescore current native model/config differs from the checkpoint binding')
        original_run, original_probes, rescore_provenance = _rescore_origin(args.rescore_from, run_plan,
            checkpoint_sha, checkpoint_step, expected_backbone)
        rescore_provenance['verified_frozen_generation_source_sha256'] = verified_code
    del checkpoint_payload
    before_checksum = None
    if not scoring_only:
        pipe = load_native(model_path)
        before_checksum = tensor_model_sha256(pipe.transformer)
        if before_checksum != expected_backbone:
            raise ValueError('actual native transformer backbone_sha256 differs from the trained checkpoint declaration')
        adapter, checkpoint_metadata, payload = load_checkpoint(args.checkpoint, device='cuda')
        checkpoint_step = payload['step']; del payload
        if checkpoint_metadata['model_id'] != str(model_path.resolve()):
            raise ValueError('checkpoint native model differs from --model')
        adapter.eval(); adapter.capture_residual_maps = True
    dependencies.update(runtime_versions=_runtime_versions(), native_code_sha256=_native_code_sha256(model_path, pipe))
    scoring_provenance = _scoring_provenance()
    control_panels = export_controls(generation, args.output)

    def configuration_for(spec, value, settings=None, active_strengths=None):
        settings = common if settings is None else settings
        active_strengths = strengths if active_strengths is None else active_strengths
        uses_adapter = spec['method'] == 'dual_v8'
        configuration = dict(spec)
        configuration.update(dependencies)
        configuration.update(settings)
        configuration.update(geometry_sha256=value.geometry.geometry_hash,
            checkpoint_sha256=checkpoint_sha if uses_adapter else None,
            checkpoint_step=checkpoint_step if uses_adapter else None,
            strengths=vars(active_strengths) if uses_adapter else None,
            target_geometry_known=uses_adapter and not active_strengths.disabled)
        return configuration

    def execute(spec, value, *, settings=None, active_strengths=None, force=False):
        settings = common if settings is None else settings
        active_strengths = strengths if active_strengths is None else active_strengths
        uses_adapter = spec['method'] == 'dual_v8'
        configuration = configuration_for(spec, value, settings, active_strengths)
        digest = configuration_sha256(configuration)
        image_path = args.output / (spec['key'] + '.png')
        record = None if args.overwrite or force else reusable_output(image_path, digest)
        if record is None:
            if scoring_only:
                raise ValueError(f'--metrics-only requires a matching saved output: {image_path}')
            started = time.time()
            image = generate(pipe, value, adapter=adapter if uses_adapter else None,
                             strengths=active_strengths, **settings)
            image.save(image_path)
            residual = save_residual_heatmaps(adapter, image_path.with_name(image_path.stem + '_residuals.png')) if uses_adapter and not active_strengths.disabled else dict(available=False, reason='control_disabled')
            record = dict(configuration=configuration, configuration_sha256=digest,
                precision_policy=getattr(pipe,'last_dual_v8_precision_policy',None) or precision_policy(),
                output=str(image_path), output_sha256=file_sha256(image_path), checkpoint_step=configuration['checkpoint_step'],
                target_geometry_known=configuration['target_geometry_known'], elapsed_seconds=time.time() - started,
                diagnostics=adapter.last_diagnostics if uses_adapter and not active_strengths.disabled else None,
                native_inference_diagnostics=getattr(pipe, 'last_dual_v8_diagnostics', None),
                adapter_inference_diagnostics=getattr(adapter, 'last_inference_diagnostics', None) if uses_adapter else None,
                residual_heatmaps=residual, code_revision=_revision())
            write_json(image_path.with_suffix('.json'), record)
            print(json.dumps(dict(generated=spec['key'], elapsed_seconds=record['elapsed_seconds'])), flush=True)
        else:
            print(json.dumps(dict(reused=spec['key'])), flush=True)
        return record, Image.open(image_path).convert('RGB')

    for spec in run_plan:
        value = replace(generation, prompts=(spec['prompt'],), seeds=(spec['seed'],))
        if args.rescore_from is None:
            record, _ = execute(spec, value)
        else:
            source_image = args.rescore_from / (spec['key'] + '.png')
            record, origin = validate_rescore_record(source_image, configuration_for(spec, value))
            destination = args.output / source_image.name
            shutil.copyfile(source_image, destination)
            record.update(output=str(destination.resolve()), generation_provenance=origin)
            if file_sha256(destination) != origin['original_output_sha256']:
                raise ValueError('rescore copied PNG differs from original generation')
        outputs.append(record)

    # Truth is first loaded here, after all main generation calls have finished.
    truth = EvaluationTarget.read(args.evaluation_target, generation_manifest=args.generation)
    if 'target_geometry_sha256' in truth.metadata and truth.metadata['target_geometry_sha256'] != generation.geometry.geometry_hash:
        raise ValueError('evaluation target geometry SHA does not match the supplied generation geometry')
    if 'person_uids' in truth.metadata and tuple(truth.metadata['person_uids']) != generation.source.person_uids[0]:
        raise ValueError('evaluation target person UIDs do not match the source actor correspondence')
    if original_run is not None:
        original_target = original_run.get('evaluation_target', {})
        if original_target.get('sha256') != truth.sha256 or original_target.get('strict_supervision_metadata') != truth.metadata:
            raise ValueError('rescore evaluator data differs from the original strict supervision binding')
        shutil.copyfile(args.rescore_from / 'run.json', args.output / 'original_run.json')
        shutil.copyfile(args.rescore_from / 'probes.json', args.output / 'original_probes.json')
        probes = original_probes['results']
    probe_input = replace(generation, seeds=(args.probe_seed,))
    probe_spec = dict(key=f'dual_v8_short_seed{args.probe_seed}', method='dual_v8', prompt_kind='short', prompt=probe_input.prompts[0], seed=args.probe_seed)
    switch_path = args.switch_generation or args.generation.with_name('generation_talk_control.json')
    probe_config = dict(common=common, code_sha256=code_sha, checkpoint_sha256=checkpoint_sha,
                        dependencies=dependencies, probe_seed=args.probe_seed, strengths=vars(strengths),
                        evaluator_target_sha256=truth.sha256, probe_steps=args.probe_steps, cfg_probes=args.cfg_probes,
                        switch_generation_sha256=file_sha256(switch_path) if args.geometry_switch_check else None)
    probe_digest = configuration_sha256(probe_config)
    saved_probes = args.output / 'probes.json'
    old = json.loads(saved_probes.read_text()) if saved_probes.exists() else {}
    if args.rescore_from is None and old.get('configuration_sha256') == probe_digest and not args.overwrite:
        probes = old.get('results', {})
    if not scoring_only:
        baseline_record, baseline = execute(probe_spec, probe_input)
        if args.target_replacement_check and 'target_replacement' not in probes:
            path = args.output / 'probe_target_replacement.png'
            probes['target_replacement'] = target_replacement_probe(generate, pipe, probe_input, truth, adapter=adapter,
                baseline=baseline, output=path, strengths=strengths, **common)
            probes['target_replacement'].update(output=str(path), output_sha256=file_sha256(path), steps=args.steps, seed=args.probe_seed)
            write_json(saved_probes, dict(configuration_sha256=probe_digest, configuration=probe_config, results=probes))
        if args.geometry_switch_check and 'geometry_switch' not in probes:
            switch_dataset = Hi4DSMPLXDataset(switch_path)
            if 'talk14_000032' not in Path(switch_dataset.rows[0]['target_geometry']).stem:
                raise ValueError('approved control switch must use the existing actual talk14 frame32 fit')
            switched = switch_dataset.generation(prompt=probe_input.prompts[0], seed=args.probe_seed)
            if switched.geometry.geometry_hash == probe_input.geometry.geometry_hash:
                raise ValueError('geometry switch requires a different actual fitted geometry')
            if array_sha256(_array(switched.source.rgb), _array(switched.source.masks)) != array_sha256(_array(probe_input.source.rgb), _array(probe_input.source.masks)) or switched.source.person_uids != probe_input.source.person_uids:
                raise ValueError('geometry switch must use the same source frame, masks and UID correspondence')
            if file_sha256(switch_dataset.rows[0]['source_features']) != dependencies['source_features_sha256']:
                raise ValueError('geometry switch must use the exact same source feature bank')
            switched_record, switched_image = execute(dict(probe_spec, key='probe_actual_talk32_geometry'), switched)
            hashes = []
            for value in (probe_input, switched):
                state = prepared_for_qwen(adapter, value.to('cuda', torch.float32))
                hashes.append(array_sha256(_array(state.dense), _array(state.nodes), _array(state.contacts))); del state
            probes['geometry_switch'] = dict(image_difference(baseline, switched_image),
                geometry_bank_changed=hashes[0] != hashes[1], geometry_bank_sha256=hashes,
                original_geometry_sha256=probe_input.geometry.geometry_hash, switched_geometry_sha256=switched.geometry.geometry_hash,
                switch_manifest=str(switch_path), switched_output=switched_record['output'],
                switched_condition='actual_fitted_talk32_geometry_not_hug34', same_source_prompt_seed=True)
            write_json(saved_probes, dict(configuration_sha256=probe_digest, configuration=probe_config, results=probes))
        bounded = dict(common, steps=args.probe_steps)
        if args.zero_strength_check and 'all_controls_zero' not in probes:
            native_record, native = execute(dict(probe_spec, key='probe_native_control_zero', method='native'), probe_input, settings=bounded)
            zero_record, zero = execute(dict(probe_spec, key='probe_adapter_control_zero'), probe_input,
                settings=bounded, active_strengths=ControlStrengths(0., 0., 0.))
            probes['all_controls_zero'] = dict(image_difference(native, zero), steps=args.probe_steps,
                native_output=native_record['output'], adapter_zero_output=zero_record['output'])
        if args.cache_check and 'cache_on_off' not in probes:
            on_record, on = execute(dict(probe_spec, key='probe_cache_on'), probe_input, settings=dict(bounded, use_kv_cache=True))
            off_record, off = execute(dict(probe_spec, key='probe_cache_off'), probe_input, settings=dict(bounded, use_kv_cache=False))
            probes['cache_on_off'] = dict(image_difference(on, off), steps=args.probe_steps,
                cache_on_output=on_record['output'], cache_off_output=off_record['output'],
                interpretation='decoded_pixel_error_reported_no_unverified_bf16_tolerance_claim')
        for scale in args.cfg_probes:
            key = f'cfg_{scale:g}'
            if key not in probes:
                record, image = execute(dict(probe_spec, key='probe_' + key), probe_input, settings=dict(bounded, cfg=scale))
                probes[key] = dict(output=record['output'], cfg=scale, steps=args.probe_steps,
                                   decoded_rgb_finite=bool(np.isfinite(np.asarray(image)).all()), diagnostics=record['diagnostics'],
                                   actual_velocity_diagnostics=record.get('native_inference_diagnostics'))
        if before_checksum is not None and args.native_checksum:
            after_checksum = tensor_model_sha256(pipe.transformer)
            probes['native_weight_checksum'] = dict(expected=expected_backbone, before=before_checksum, after=after_checksum,
                matches_checkpoint=before_checksum == expected_backbone, unchanged=before_checksum == after_checksum)
        write_json(saved_probes, dict(configuration_sha256=probe_digest, configuration=probe_config, results=probes))
    del pipe, adapter
    gc.collect(); torch.cuda.empty_cache()

    # Optional metric models are loaded after native inference releases GPU memory.
    metric_metadata, model_failures = {}, {}
    pose_model, dino_model, face_model = None, None, None
    try:
        pose_model = DWPosePeople(args.root / 'models')
        metric_metadata['dwpose'] = pose_model.metadata
    except Exception as exc:
        model_failures['dwpose'] = f'{type(exc).__name__}: {exc}'
    if not args.skip_dino:
        try:
            dino_model = Dinov2Appearance(args.dino, args.metric_device)
            metric_metadata['dinov2'] = dino_model.metadata
        except Exception as exc:
            model_failures['dinov2'] = f'{type(exc).__name__}: {exc}'
    if args.face_metrics:
        try:
            from src.evaluation.joint_pilot_metrics import IdentityMetric
            for name in ('glintr100.onnx', 'scrfd_10g_bnkps.onnx'):
                if not (args.root / 'models/antelopev2' / name).is_file():
                    raise FileNotFoundError(args.root / 'models/antelopev2' / name)
            face_model = IdentityMetric(args.root / 'models')
            metric_metadata['arcface'] = dict(model='antelopev2/glintr100', provider='CPUExecutionProvider',
                recognition_sha256=file_sha256(args.root / 'models/antelopev2/glintr100.onnx'),
                detector_sha256=file_sha256(args.root / 'models/antelopev2/scrfd_10g_bnkps.onnx'))
        except Exception as exc:
            model_failures['arcface'] = f'{type(exc).__name__}: {exc}'
    source_embeddings, source_dino_failures = [None, None], ['metric_unavailable'] * 2
    if dino_model is not None:
        source_embeddings, source_dino_failures, source_crops = dino_model.encode(source_rgb, source_masks)
        for actor, crop in enumerate(source_crops):
            if crop is not None:
                crop.save(args.output / f'eval_source_actor{actor}_crop.png')
    source_faces, source_face_failures = _face_embeddings(face_model, source_rgb, source_masks) if face_model is not None else ([None, None], ['metric_unavailable'] * 2)
    whole_xy, whole_scores, whole_metadata = _wholebody_truth(args.root, truth, args.wholebody_observations)
    metric_metadata['wholebody_truth'] = whole_metadata
    metric_metadata['dwpose_evaluator_gt_calibration'] = _detector_calibration(pose_model, truth.rgb,
        target_boxes, target_xy, target_scores, generation.source.person_uids[0])
    original_size = Image.open(row['source_rgb']).size
    affine = letterbox_affine(*original_size)
    yy, xx = np.mgrid[:512, :512]
    valid_canvas = (xx >= affine[0, 2]) & (xx < affine[0, 2] + original_size[0] * affine[0, 0]) & (yy >= affine[1, 2]) & (yy < affine[1, 2] + original_size[1] * affine[1, 1])
    for record in outputs:
        path = Path(record['output']); image = Image.open(path).convert('RGB')
        failures = {}
        try:
            detections = pose_model.detect(image) if pose_model is not None else []
        except Exception as exc:
            detections = []; failures['dwpose_detection'] = f'{type(exc).__name__}: {exc}'
        assignment = assign_by_geometry(detections, target_boxes, target_xy, target_scores)
        masks = generated_person_masks(detections, assignment['indices'])
        embeddings, dino_failures, crops = dino_model.encode(np.asarray(image), masks) if dino_model is not None else ([None, None], ['metric_unavailable'] * 2, [None, None])
        faces, face_failures = _face_embeddings(face_model, np.asarray(image), masks) if face_model is not None else ([None, None], ['metric_unavailable'] * 2)
        people = []
        for actor, index in enumerate(assignment['indices']):
            detection = detections[index] if index >= 0 else None
            body_detection = None if detection is None else PersonDetection(detection.box, detection.xy[:24], detection.scores[:24])
            whole_result = pck(detection, whole_xy[actor], whole_scores[actor]) if whole_xy is not None else dict(eligible_joints=0, detected_joints=0, pck_002=None, pck_005=None, pck_002_hits=0, pck_005_hits=0, failure='wholebody_truth_unavailable')
            people.append(dict(person_uid=generation.source.person_uids[0][actor], detection_index=index,
                               body=pck(body_detection, target_xy[actor], target_scores[actor]), wholebody=whole_result))
            if crops[actor] is not None:
                crops[actor].save(path.with_name(path.stem + f'_actor{actor}_crop.png'))
        detected_count = len(detections) if pose_model is not None and 'dwpose_detection' not in failures else None
        quality = dict(detected_people=detected_count, third_people=None if detected_count is None else max(detected_count - 2, 0),
            assignment=assignment, detections=[detected.as_dict() for detected in detections], people=people,
            appearance=appearance_binding(source_embeddings, embeddings), face_identity=appearance_binding(source_faces, faces, metric_kind='arcface_face_identity'),
            background=background_error(np.asarray(image), source_rgb, source_masks, truth.person_masks, valid_canvas=valid_canvas),
            generated_appearance_mask_provenance='generated_rgb_dwpose_boxes_partitioned_by_detected_body_joints_proxy',
            source_dino_failures=source_dino_failures, generated_dino_failures=dino_failures,
            source_face_failures=source_face_failures, generated_face_failures=face_failures,
            metric_model_failures=model_failures, failures=failures,
            mesh=dict(failure='mesh_metrics_not_requested', part_pair_distances=None))
        quality['occlusion'] = dict(coarse_generated_depth_order=None, expected_geometry_depth_order=None,
                                    failure='requires_generated_mesh_reconstruction')
        record['quality'] = quality
        record['evaluation_target_sha256'] = truth.sha256
        write_json(path.with_suffix('.json'), record)
        preview = image.copy(); draw = ImageDraw.Draw(preview)
        for index, detected in enumerate(detections):
            draw.rectangle(tuple(detected.box), outline='red', width=2)
            draw.text(tuple(detected.box[:2]), f'detection {index}', fill='red')
        panels = [(Image.fromarray(source_rgb), 'source RGB only for generation'),
                  (Image.fromarray(truth.rgb), 'evaluation GT only; not a reference'),
                  (image, f"{record['configuration']['method']} {record['configuration']['prompt_kind']}\nknown target geometry={record['target_geometry_known']}"),
                  (preview, f"generated DWPose people={len(detections)}")]
        for actor, crop in enumerate(crops):
            if crop is not None:
                panels.append((crop, f'geometry assigned actor {actor} crop proxy'))
        panels += control_panels[:2]
        _board(path.with_name(path.stem + '_board.png'), panels)
        print(json.dumps(dict(scored=path.stem, people=len(detections), assigned=assignment['indices'])), flush=True)
    del dino_model, face_model, pose_model
    gc.collect(); torch.cuda.empty_cache()
    if args.mesh_metrics:
        mesh_model, worker_meshes = None, None
        if args.mesh_python is not None:
            worker_meshes, mesh_provenance = run_mesh_worker(args, outputs, row['target_geometry'])
            metric_metadata['multihmr'] = mesh_provenance
        else:
            try:
                mesh_model = GeneratedMeshMetric(args.root, row['target_geometry'])
                metric_metadata['multihmr'] = mesh_model.metadata
            except Exception as exc:
                model_failures['multihmr'] = f'{type(exc).__name__}: {exc}'
        endpoints = generation.geometry.contact_endpoints[0][generation.geometry.contact_valid[0]].cpu().numpy()
        for record in outputs:
            path = Path(record['output'])
            try:
                if worker_meshes is not None:
                    raw = worker_meshes[record['configuration']['key']]
                    if raw.get('failure') is not None:
                        mesh = dict(raw)
                    else:
                        mesh = generated_mesh_quality(raw['humans'], raw['labels'], raw['K512'], raw['image_size'],
                            target_boxes, target_xy, target_scores, endpoints)
                        mesh.update({name: raw[name] for name in ('reconstruction_path', 'reconstruction_sha256', 'raw_worker_record', 'raw_sidecar_path', 'raw_sidecar_sha256')})
                else:
                    mesh = mesh_model.measure(path, target_boxes, target_xy, target_scores, endpoints,
                            path.with_name(path.stem + '_generated_mesh.npz')) if mesh_model is not None else dict(failure=model_failures['multihmr'], part_pair_distances=None)
            except Exception as exc:
                mesh = dict(failure=f'{type(exc).__name__}: {exc}', part_pair_distances=None)
            record['quality']['mesh'] = mesh
            means = mesh.get('mean_camera_z_by_actor')
            expected_depth = _array(generation.geometry.depth[0, :, 0])
            expected_mask = _array(generation.geometry.amodal_mask[0, :, 0]) > .5
            expected_means = [float(expected_depth[actor][expected_mask[actor]].mean()) for actor in range(2)]
            predicted_order = None if means is None else int(np.sign(means[1] - means[0]))
            expected_order = int(np.sign(expected_means[1] - expected_means[0]))
            record['quality']['occlusion'] = dict(coarse_generated_depth_order=predicted_order,
                expected_geometry_depth_order=expected_order,
                coarse_depth_order_agreement=None if predicted_order is None else predicted_order == expected_order,
                failure=mesh.get('failure'), interpretation='mean_camera_depth_proxy_not_pixelwise_visible_owner')
            write_json(path.with_suffix('.json'), record)
        del mesh_model
    scoring_provenance.update(metric_models=metric_metadata,
        scoring_configuration=dict(evaluation_target_sha256=truth.sha256, mesh_metrics=args.mesh_metrics,
            mesh_python=str(args.mesh_python.resolve()) if args.mesh_python is not None else None,
            mesh_device=args.mesh_device, metric_device=args.metric_device, face_metrics=args.face_metrics,
            skip_dino=args.skip_dino, dino_path=str(args.dino.resolve()),
            wholebody_observations_sha256=file_sha256(args.wholebody_observations) if args.wholebody_observations is not None else None))
    scoring_provenance['scoring_configuration_sha256'] = configuration_sha256(scoring_provenance['scoring_configuration'])
    for record in outputs:
        record['scoring_provenance'] = dict(scoring_provenance, generated_rgb_sha256=record['output_sha256'],
            generation_configuration_sha256=record['configuration_sha256'])
        write_json(Path(record['output']).with_suffix('.json'), record)
    groups = {}
    for method in ('native', 'dual_v8'):
        for kind in ('short', 'detailed'):
            group = [record['quality'] for record in outputs if record['configuration']['method'] == method and record['configuration']['prompt_kind'] == kind]
            groups[f'{method}_{kind}'] = aggregate_quality(group)
    summary = dict(task='one_pair_known_target_geometry_pilot_not_generalization',
        precision_policy=precision_policy(),
        runtime_versions=dependencies['runtime_versions'], native_code_sha256=dependencies['native_code_sha256'],
        configuration=vars(args), checkpoint=dict(path=str(args.checkpoint), sha256=checkpoint_sha, payload_step=checkpoint_step,
            backbone_sha256=expected_backbone, actual_backbone_sha256=before_checksum),
        evaluation_target=dict(manifest=str(args.evaluation_target), sha256=truth.sha256, never_passed_to_generate=True,
            strict_supervision_metadata=truth.metadata),
        output_records=[str(Path(record['output']).with_suffix('.json')) for record in outputs],
        groups=groups, probes=probes, metric_metadata=metric_metadata, metric_model_failures=model_failures,
        scoring_provenance=scoring_provenance, rescore_provenance=rescore_provenance,
        limitations=['DINOv2 is an appearance/clothing proxy; generated crop masks are DWPose spatial proxies.',
                    'ArcFace failures and missing people remain NA with explicit denominators.',
                    'Multi-HMR contact and surface proximity use estimated camera-space geometry and do not establish signed penetration.',
                    'A single fitted training pair cannot establish identity or interaction generalization.'])
    summary['hard_probe_failures'] = []
    for key in ('target_replacement', 'all_controls_zero'):
        if key in probes and not probes[key].get('pixel_exact'):
            summary['hard_probe_failures'].append(key)
    if 'geometry_switch' in probes and (probes['geometry_switch'].get('pixel_exact') or not probes['geometry_switch'].get('geometry_bank_changed')):
        summary['hard_probe_failures'].append('geometry_switch')
    if 'native_weight_checksum' in probes and not probes['native_weight_checksum'].get('unchanged'):
        summary['hard_probe_failures'].append('native_weight_checksum')
    summary['configuration'] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / 'run.json', summary)
    panels = [(Image.fromarray(source_rgb), 'source: talk32'), (Image.fromarray(truth.rgb), 'evaluator GT: hug62')]
    panels += [(Image.open(record['output']), f"{record['configuration']['method']} {record['configuration']['prompt_kind']} seed={record['configuration']['seed']}\nknown geometry={record['target_geometry_known']}") for record in outputs]
    _board(args.output / 'comparison_board.png', panels)
    print(json.dumps(dict(run_json=str(args.output / 'run.json'), comparison_board=str(args.output / 'comparison_board.png'), groups=groups)), flush=True)


if __name__ == '__main__':
    main()

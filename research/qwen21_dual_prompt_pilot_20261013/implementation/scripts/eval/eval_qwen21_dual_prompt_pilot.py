#!/usr/bin/env python3
"""Evaluate one preselected true-test pair with two finished prompt adapters.

All twelve fixed comparisons finish under a global inference guard before GT
supervision is opened. Existing generated-RGB evaluators score the saved PNGs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image
import torch

from src.evaluation.qwen21_dual_prompt_pilot import (
    configuration_sha256, model_state_sha256, planned_runs, read_external_pose,
    run_generation_phase, source_canvas_mask, validate_test_dataset, validate_training_code, validate_training_contracts, write_json,
)
from src.evaluation.qwen21_dual_v8_metrics import (
    DWPosePeople, Dinov2Appearance, EvaluationTarget, PersonDetection,
    aggregate_quality, appearance_binding, array_sha256, assign_by_geometry,
    background_error, boxes_from_masks, file_sha256, generated_mesh_quality,
    generated_person_masks, pck,
)
from scripts.eval.eval_qwen21_dual_v8 import (
    GeneratedMeshMetric, _array, _board, _detector_calibration, _face_embeddings,
    _native_code_sha256, _runtime_versions, _scoring_provenance, run_mesh_worker,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new directory; existing results are never overwritten')
    p.add_argument('--root', type=Path, default=Path('/home/shangguanrz/project/pic-edit'))
    p.add_argument('--model', type=Path)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--cfg', type=float, default=1.)
    p.add_argument('--no-kv-cache', action='store_true')
    p.add_argument('--dino', type=Path, default=Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code/models/dinov2-giant'))
    p.add_argument('--metric-device', default='cpu')
    p.add_argument('--mesh-python', type=Path, help='existing isolated Multi-HMR environment; loaded once after Qwen unload')
    p.add_argument('--mesh-device', default='cuda', choices=['cuda', 'cpu'])
    return p


def _evaluation_code_provenance():
    paths = [PROJECT / 'src/evaluation/qwen21_dual_prompt_pilot.py',
        PROJECT / 'src/evaluation/qwen21_dual_v8_metrics.py', Path(__file__),
        PROJECT / 'src/evaluation/joint_pilot_metrics.py',
        PROJECT / 'scripts/eval/eval_qwen21_dual_v8.py', PROJECT / 'scripts/eval/multihmr_rgb_worker.py']
    paths += list((PROJECT / 'src/evaluation/dwpose_onnx').glob('*.py'))
    return {path.relative_to(PROJECT).as_posix(): file_sha256(path) for path in sorted(set(paths)) if path.is_file()}


def score_outputs(args, outputs, truth, generation, row, target_xy, target_scores, pose_metadata, source_assets):
    """Metrics consume generated PNGs and evaluator truth after inference ends."""
    source_rgb = (_array(generation.source.rgb[0]).transpose(1, 2, 0) * 255).round().astype(np.uint8)
    source_masks = _array(generation.source.masks[0, :, 0])
    target_boxes = boxes_from_masks(_array(generation.geometry.amodal_mask[0, :, 0]))
    uids = generation.source.person_uids[0]
    endpoints = _array(generation.geometry.contact_endpoints[0][generation.geometry.contact_valid[0]]).astype(int)
    model_metadata, model_failures = dict(target_pose=pose_metadata), {}
    pose_model, dino_model, face_model = None, None, None
    try:
        pose_model = DWPosePeople(args.root / 'models')
        model_metadata['dwpose'] = pose_model.metadata
    except Exception as exc:
        model_failures['dwpose'] = f'{type(exc).__name__}: {exc}'
    try:
        dino_model = Dinov2Appearance(args.dino, args.metric_device)
        model_metadata['dinov2'] = dino_model.metadata
    except Exception as exc:
        model_failures['dinov2'] = f'{type(exc).__name__}: {exc}'
    try:
        from src.evaluation.joint_pilot_metrics import IdentityMetric
        for name in ('glintr100.onnx', 'scrfd_10g_bnkps.onnx'):
            if not (args.root / 'models/antelopev2' / name).is_file():
                raise FileNotFoundError(args.root / 'models/antelopev2' / name)
        face_model = IdentityMetric(args.root / 'models')
        model_metadata['arcface'] = dict(model='antelopev2/glintr100', provider='CPUExecutionProvider',
            weights={name: file_sha256(args.root / 'models/antelopev2' / name)
                for name in ('glintr100.onnx', 'scrfd_10g_bnkps.onnx')})
    except Exception as exc:
        model_failures['arcface'] = f'{type(exc).__name__}: {exc}'
    source_embeddings, source_dino_failures = [None, None], ['metric_unavailable'] * 2
    if dino_model is not None:
        source_embeddings, source_dino_failures, crops = dino_model.encode(source_rgb, source_masks)
        for actor, crop in enumerate(crops):
            if crop is not None:
                crop.save(args.output / f'source_actor{actor}_appearance_crop.png')
    source_faces, source_face_failures = _face_embeddings(face_model, source_rgb, source_masks) if face_model else ([None, None], ['metric_unavailable'] * 2)
    model_metadata['evaluator_gt_detector_calibration'] = _detector_calibration(pose_model,
        truth.rgb, target_boxes, target_xy[:, :24], target_scores[:, :24], uids)
    valid_canvas = source_canvas_mask(source_assets)
    for record in outputs:
        path = Path(record['output'])
        if file_sha256(path) != record['output_sha256']:
            raise ValueError('generated PNG content changed before scoring')
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        failures = {}
        try:
            detections = pose_model.detect(image) if pose_model is not None else []
        except Exception as exc:
            detections = []
            failures['dwpose_detection'] = f'{type(exc).__name__}: {exc}'
        assignment = assign_by_geometry(detections, target_boxes, target_xy[:, :24], target_scores[:, :24])
        masks = generated_person_masks(detections, assignment['indices'])
        embeddings, dino_failures, crops = dino_model.encode(np.asarray(image), masks) if dino_model is not None else ([None, None], ['metric_unavailable'] * 2, [None, None])
        faces, face_failures = _face_embeddings(face_model, np.asarray(image), masks) if face_model else ([None, None], ['metric_unavailable'] * 2)
        people = []
        for actor, index in enumerate(assignment['indices']):
            detected = detections[index] if index >= 0 else None
            body = None if detected is None else PersonDetection(detected.box, detected.xy[:24], detected.scores[:24])
            people.append(dict(person_uid=uids[actor], actor_slot='AB'[actor], detection_index=index,
                body=pck(body, target_xy[actor, :24], target_scores[actor, :24]),
                wholebody=pck(detected, target_xy[actor], target_scores[actor])))
            if crops[actor] is not None:
                crops[actor].save(path.with_name(path.stem + f'_actor{actor}_appearance_crop.png'))
        count = len(detections) if pose_model is not None and not failures.get('dwpose_detection') else None
        rgb = np.asarray(image)
        gt_background = background_error(rgb, truth.rgb, source_masks, truth.person_masks, valid_canvas=valid_canvas)
        gt_background['rgb_reference'] = 'evaluator_target_rgb_opened_after_all_generation'
        gt_background['canvas_provenance'] = 'source_native_size_and_recorded_affine; letterbox_padding_excluded'
        quality = dict(detected_people=count, third_people=None if count is None else max(count - 2, 0),
            assignment=assignment, detections=[detected.as_dict() for detected in detections], people=people,
            appearance=appearance_binding(source_embeddings, embeddings),
            face_identity=appearance_binding(source_faces, faces, metric_kind='arcface_face_identity'),
            background=gt_background,
            source_background_preservation=background_error(rgb, source_rgb, source_masks, truth.person_masks, valid_canvas=valid_canvas),
            generated_appearance_mask_provenance='generated_rgb_dwpose_boxes_partitioned_by_detected_body_joints_proxy',
            source_dino_failures=source_dino_failures, generated_dino_failures=dino_failures,
            source_face_failures=source_face_failures, generated_face_failures=face_failures,
            metric_model_failures=dict(model_failures), failures=failures,
            mesh=dict(failure='pending_generated_rgb_reconstruction', part_pair_distances=None,
                expected_contact_pairs=len(endpoints)),
            occlusion=dict(failure='pending_generated_rgb_reconstruction'))
        record['quality'] = quality
        record['evaluation_target_sha256'] = truth.sha256
        write_json(path.with_suffix('.json'), record)
    del pose_model, dino_model, face_model
    gc.collect()
    torch.cuda.empty_cache()

    # Always attempt actual generated-RGB mesh reconstruction. Failure remains
    # visible as NA, and control meshes never substitute generated predictions.
    mesh_model, worker_meshes = None, None
    try:
        if args.mesh_python is not None:
            worker_meshes, metadata = run_mesh_worker(args, outputs, row['target_geometry'])
        else:
            mesh_model = GeneratedMeshMetric(args.root, row['target_geometry'], device=args.mesh_device)
            metadata = mesh_model.metadata
        model_metadata['multihmr'] = metadata
    except Exception as exc:
        model_failures['multihmr'] = f'{type(exc).__name__}: {exc}'
    depth = _array(generation.geometry.depth[0, :, 0])
    amodal = _array(generation.geometry.amodal_mask[0, :, 0]) > .5
    expected_means = [float(depth[actor][amodal[actor]].mean()) for actor in range(2)]
    expected_order = int(np.sign(expected_means[1] - expected_means[0]))
    for record in outputs:
        path = Path(record['output'])
        try:
            if worker_meshes is not None:
                raw = worker_meshes[record['configuration']['key']]
                if raw.get('failure') is not None:
                    mesh = dict(raw)
                else:
                    mesh = generated_mesh_quality(raw['humans'], raw['labels'], raw['K512'], raw['image_size'],
                        target_boxes, target_xy[:, :24], target_scores[:, :24], endpoints)
                    mesh.update({name: raw[name] for name in ('reconstruction_path', 'reconstruction_sha256',
                        'raw_worker_record', 'raw_sidecar_path', 'raw_sidecar_sha256')})
            elif mesh_model is not None:
                mesh = mesh_model.measure(path, target_boxes, target_xy[:, :24], target_scores[:, :24],
                    endpoints, path.with_name(path.stem + '_generated_mesh.npz'))
                if mesh.get('reconstruction_path'):
                    mesh['reconstruction_sha256'] = file_sha256(mesh['reconstruction_path'])
            else:
                mesh = dict(failure=model_failures['multihmr'], part_pair_distances=None)
        except Exception as exc:
            mesh = dict(failure=f'{type(exc).__name__}: {exc}', part_pair_distances=None)
        mesh['expected_contact_pairs'] = len(endpoints)
        mesh['signed_penetration'] = None
        mesh['control_mesh_substituted'] = False
        record['quality']['mesh'] = mesh
        record['quality']['metric_model_failures'] = dict(model_failures)
        means = mesh.get('mean_camera_z_by_actor')
        predicted_order = None if means is None else int(np.sign(means[1] - means[0]))
        record['quality']['occlusion'] = dict(coarse_generated_depth_order=predicted_order,
            expected_geometry_depth_order=expected_order,
            coarse_depth_order_agreement=None if predicted_order is None else predicted_order == expected_order,
            failure=mesh.get('failure'), interpretation='mean_camera_depth_proxy_not_pixelwise_visible_owner')
        write_json(path.with_suffix('.json'), record)
    del mesh_model
    gc.collect()
    torch.cuda.empty_cache()
    groups = {}
    for method in ('native', 'dual'):
        for kind in ('short', 'structured'):
            quality = [record['quality'] for record in outputs
                if record['configuration']['method'] == method and record['configuration']['prompt_kind'] == kind]
            aggregate = aggregate_quality(quality)
            aggregate['actor_detection_coverage'] = {slot: dict(expected_outputs=len(quality),
                detected_outputs=sum(row['assignment']['indices'][actor] >= 0 for row in quality),
                missing_outputs=sum(row['assignment']['indices'][actor] < 0 for row in quality))
                for actor, slot in enumerate('AB')}
            contacts = [entry for row in quality for entry in (row['mesh'].get('part_pair_distances') or [])]
            aggregate['contact_coverage'] = dict(expected_pairs=len(quality) * len(endpoints),
                scored_pairs=sum(entry.get('distance_over_mean_height') is not None for entry in contacts),
                failed_outputs=sum(bool(row['mesh'].get('failure')) for row in quality),
                signed_penetration_available=False)
            groups[method + '_' + kind] = aggregate
    return dict(groups=groups, metric_metadata=model_metadata, metric_model_failures=model_failures)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise ValueError('choose a new output directory; existing results are preserved')
    if args.steps != 40 or args.cfg != 1.:
        raise ValueError('approved final test requires 40 steps and CFG1')
    planned_runs(args.seeds, '', '')  # Validate fixed seeds before loading models.
    if args.mesh_python is not None and not args.mesh_python.is_file():
        raise ValueError('--mesh-python must name the existing isolated executable')
    args.output = args.output.resolve()
    bundle = validate_test_dataset(args.data_dir)
    contracts = validate_training_contracts(args.training_root, bundle)
    from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset
    from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint
    from src.pose_control.qwen21_full.dual.precision import precision_policy
    from src.pose_control.qwen21_full.dual.runtime import generate, load_native
    torch.set_num_threads(8)
    datasets = {kind: Hi4DSMPLXDataset(bundle['manifests']['test'][kind]) for kind in ('short', 'structured')}
    generations = {kind: dataset.generation() for kind, dataset in datasets.items()}
    row = datasets['short'].rows[0]
    selected = bundle['test_sample']
    if (file_sha256(row['target_geometry']) != selected.get('target_geometry_sha256') or
            file_sha256(Path(row['target_geometry']).with_suffix('.json')) != selected.get('target_geometry_sidecar_sha256')):
        raise ValueError('test geometry/sidecar differs from the preselected dataset audit')
    target_xy, target_scores, pose_metadata = read_external_pose(row['target_geometry'])
    model_path = (args.model or args.root / 'models/Qwen-Image-2.1').resolve()
    for contract in contracts.values():
        if contract['metadata'].get('model_id') != str(model_path):
            raise ValueError('checkpoint native model differs from current --model')
        if contract['metadata'].get('model_config_sha256') != file_sha256(model_path / 'transformer/config.json'):
            raise ValueError('checkpoint native transformer configuration differs from current --model')
    pipe = load_native(model_path)
    adapters = {}
    try:
        actual_backbone = model_state_sha256(pipe.transformer)
        if actual_backbone != contracts['short']['metadata']['backbone_sha256']:
            raise ValueError('actual native weights do not match the completed training backbone SHA')
        actual_native_code = _native_code_sha256(model_path, pipe)
        if any(contract['metadata'].get('native_code_sha256') != actual_native_code for contract in contracts.values()):
            raise ValueError('actual native component source differs from training contract')
        for kind, contract in contracts.items():
            adapters[kind], _, payload = load_checkpoint(contract['path'], device='cuda')
            del payload
        checkpoints = {kind: dict(path=str(value['path']), sha256=value['sha256'], step=value['step'],
            best_validation=value['best_validation'], completion_sha256=value['completion_sha256'],
            root_completion_sha256=value['root_completion_sha256']) for kind, value in contracts.items()}
        provenance = dict(dataset_sha256=bundle['dataset_sha256'], split_manifest=bundle['split_manifest'],
            selection_lock=bundle['selection_lock'], sample_id=bundle['sample_ids']['test'][0], pair_holdout=bundle['pair_ids'],
            identity_scope='pair holdout; global cross-pair actor IDs were not proven',
            generation_manifest_sha256={kind: file_sha256(bundle['manifests']['test'][kind]) for kind in datasets},
            source_rgb_sha256=file_sha256(row['source_rgb']),
            source_mask_sha256=[file_sha256(path) for path in row['source_masks']],
            source_features_sha256=file_sha256(row['source_features']), person_uids=row['person_uids'],
            geometry_sha256=file_sha256(row['target_geometry']),
            geometry_sidecar_sha256=file_sha256(Path(row['target_geometry']).with_suffix('.json')),
            checkpoints=checkpoints, model_id=str(model_path), actual_backbone_sha256=actual_backbone,
            precision_policy=precision_policy(), runtime_versions=_runtime_versions(),
            native_code_sha256=actual_native_code,
            verified_training_code_sha256=contracts['short']['verified_training_code_sha256'],
            evaluation_code_sha256=_evaluation_code_provenance(),
            text_provenance=dict(short='action label only; no target geometry descriptions',
                structured='source actor anchors, action label and reliable external target geometry descriptions; no target RGB'))
        plan = planned_runs(args.seeds, generations['short'].prompts[0], generations['structured'].prompts[0])
        records, audit = run_generation_phase(generate, pipe, generations, adapters, plan, args.output,
            provenance, steps=args.steps, cfg=args.cfg, use_kv_cache=not args.no_kv_cache)
        if any(file_sha256(contract['path']) != contract['sha256'] for contract in contracts.values()):
            raise ValueError('best checkpoint content changed during final inference')
        if file_sha256(bundle['dataset_path']) != bundle['dataset_sha256'] or _evaluation_code_provenance() != provenance['evaluation_code_sha256']:
            raise ValueError('frozen dataset or runtime source changed during final inference')
        for contract in contracts.values():
            validate_training_code(contract['metadata'])
    except BaseException as exc:
        if args.output.exists() and not (args.output / 'evaluation_failed.json').exists():
            write_json(args.output / 'evaluation_failed.json', dict(status='failed', phase='generation', error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        del pipe
        adapters.clear()
        gc.collect()
        torch.cuda.empty_cache()
    # This is the first operation that opens test supervision and GT RGB/masks.
    try:
        supervision_path = bundle['manifests']['test']['supervision']
        if file_sha256(supervision_path) != bundle['metadata']['splits']['test']['supervision_sha256']:
            raise ValueError('test supervision manifest SHA differs from frozen dataset binding')
        truth = EvaluationTarget.read(bundle['manifests']['test']['supervision'],
            generation_manifest=bundle['manifests']['test']['short'])
        metrics = score_outputs(args, records, truth, generations['short'], row, target_xy, target_scores,
            pose_metadata, source_assets=selected['source_assets'])
        scoring_provenance = _scoring_provenance()
        scoring_provenance['prompt_pilot_evaluator_sha256'] = file_sha256(Path(__file__))
        for record in records:
            record['scoring_provenance'] = dict(scoring_provenance, generated_rgb_sha256=record['output_sha256'])
            write_json(Path(record['output']).with_suffix('.json'), record)
        source_rgb = (_array(generations['short'].source.rgb[0]).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        panels = [(Image.fromarray(source_rgb), 'same source RGB for all generation'),
            (Image.fromarray(truth.rgb), 'test GT: opened only after 12 generations')]
        panels += [(Image.open(record['output']), f"{record['configuration']['method']} {record['configuration']['prompt_kind']} seed={record['configuration']['seed']}\ndense control={record['explicit_target_geometry_control']}; geometry text={record['prompt_geometry_descriptions']}") for record in records]
        board = args.output / 'comparison_board.png'
        _board(board, panels, columns=4)
        summary = dict(status='success', task='one_true_test_pair_prompt_pilot',
            test_sample=bundle['test_sample'], generation_provenance=provenance,
            inference_guard=audit, target_supervision=dict(path=str(bundle['manifests']['test']['supervision']),
                sha256=file_sha256(bundle['manifests']['test']['supervision']),
                target_sha256=truth.sha256, opened_after_all_12_generations=True, never_passed_to_generate=True),
            output_records=[dict(path=str(Path(record['output']).with_suffix('.json')),
                sha256=file_sha256(Path(record['output']).with_suffix('.json')),
                png_sha256=record['output_sha256']) for record in records],
            comparison_board=dict(path=str(board), sha256=file_sha256(board)), **metrics,
            limitations=['One held-out pair cannot establish overall generalization or the two-person quality threshold.',
                'Both structured prompts describe reliable external target geometry; only adapter comparisons receive explicit dense/graph/contact geometry controls.',
                'Pair holdout does not prove global actor identity disjointness across pairs.',
                'DINO is a clothing/appearance proxy; generated person crops are DWPose spatial proxies.',
                'Missing actors/faces/meshes stay NA and coverage denominators are explicit.',
                'Multi-HMR contact is unsigned estimated camera-space proximity; signed penetration is unavailable.'])
        write_json(args.output / 'run.json', summary)
        write_json(args.output / 'evaluation_complete.json', dict(status='success', generated_outputs=12,
            run_sha256=file_sha256(args.output / 'run.json'), inference_guard_sha256=file_sha256(args.output / 'inference_guard.json'),
            comparison_board_sha256=file_sha256(board)))
        print(json.dumps(dict(run_json=str(args.output / 'run.json'), comparison_board=str(board), groups=metrics['groups'])), flush=True)
    except BaseException as exc:
        write_json(args.output / 'evaluation_failed.json', dict(status='failed', phase='scoring', error=f'{type(exc).__name__}: {exc}', generated_outputs=len(records)))
        raise


if __name__ == '__main__':
    main()

"""CPU 身份审计：真实 SCRFD 五点对齐，geometry 唯一匹配到稳定人物槽位。

生成脸不用 GT 五点裁切，不按外观或左右顺序分配身份。完整复用
prepare_assets.match_faces 的双人资格门禁；检测或匹配失败记录 null。
Source 使用已资格的真实五点，同一 frozen TorchScript；另评两个真实 GT。
--baseline-report / --final-report 只选择 native；输出目录必须是新目录。
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def embed(pixels, points, model):
    import torch
    from src.pose_control.qwen21_full.face_loss_alignment import align_face_rgb
    images = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()[None].expand(len(points), -1, -1, -1) / 255
    with torch.no_grad():
        value = model(align_face_rgb(images, torch.tensor(points).float()) * 2 - 1)
        value = value[-1] if isinstance(value, (tuple, list)) else value
        return torch.nn.functional.normalize(value.flatten(1).float(), dim=-1)


def measure_case(pixels, canvas, expected, source_features, detector, model, matcher, evaluation):
    import cv2
    from src.data.canvas_contract import CanvasSpec
    boxes, points = detector.detect(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    boxes = np.asarray(boxes).reshape(-1, 5)
    result = {'raw_faces_detected': len(boxes), 'selected_detections': None, 'identity': None, 'reason': None,
        'expected_face_boxes_canvas_xyxy': (np.asarray(expected) * np.array([canvas.canvas_hw[1], canvas.canvas_hw[0]] * 2)).tolist(),
        'detections': [{'box_xyxy': box[:4].tolist(), 'confidence': float(box[4]),
                        'landmarks5_canvas_pixels': None if points is None else points[index].tolist()} for index, box in enumerate(boxes)]}
    try:
        if points is None:
            raise ValueError('SCRFD supplied no real five-point landmarks')
        # SCRFD receives the already padded Canvas. Its output is Canvas pixels;
        # use the identity transform when calling the original-image matcher.
        detector_canvas = CanvasSpec(canvas.canvas_hw, canvas.canvas_hw, canvas.canvas_hw)
        selected = matcher(boxes, expected, detector_canvas)
        landmarks = np.asarray(points)[selected]
        features = embed(pixels, landmarks, model)
        matrix = (source_features @ features.T).cpu().numpy()
        identity = evaluation.identity_summary(matrix)
        identity['name'] = 'source_to_detected_face_identity_geometry_slot_matched'
        result.update(selected_detections=selected, selected_landmarks5_canvas_pixels=landmarks.tolist(), identity=identity)
    except ValueError as error:
        result['reason'] = str(error)
    return result


def rgb(sample, key):
    return ((sample[key].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def original_rgb_references(row, evaluation):
    from src.pose_control.qwen21_full.native_v62.training.io import verify
    if 'source_rgb' not in row:
        manifest = Path(verify(row['v4_manifest'])['path'])
        row = next(value for line in manifest.read_text().splitlines() if line.strip()
                   and (value := json.loads(line))['pair_id'] == row['pair_id'])
    return {role: verify(row[role + '_rgb']) for role in ('source', 'target')}


def run(args, evaluation, preparer):
    evaluation.new_report(args.report_dir)
    try:
        import onnxruntime as ort
        from PIL import Image
        from insightface import model_zoo
        from src.data.canvas_contract import CanvasSpec
        from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
        from src.pose_control.qwen21_full.native_v62.training.io import verify
        from src.pose_control.qwen21_full.native_v5.losses import frozen_evaluators
        config = evaluation.read(args.config)
        model_ref = evaluation.reference(args.identity_model)
        model, _ = frozen_evaluators(model_ref['path'], None, device='cpu', sha=[model_ref['sha256'], None])
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        detector = model_zoo.get_model(str(args.detector), providers=['CPUExecutionProvider'], sess_options=options)
        detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=.5)
        qualifications = {row['pair_id']: row for row in evaluation.read(args.prepared)['qualification']}
        samples, sources = {}, []
        for source in config['sources']:
            manifest = verify(source['manifest'])
            dataset = NativeFullDatasetV62(manifest['path'], load_target=True, data_profile=source['data_profile'])
            for index in range(len(dataset)):
                sample = dataset[index]; pair_id = dataset.rows[index]['pair_id']
                source_canvas = CanvasSpec.from_record(sample['source_canvas'])
                points = source_canvas.pixel_points(np.asarray(qualifications[pair_id]['source']['landmarks5_original']))
                indices = sample['local_condition_v4'].source_indices[0]
                features = embed(rgb(sample, 'src_image'), points, model)[indices]
                refs = original_rgb_references(dataset.rows[index], evaluation)
                samples[(source['source_id'], pair_id)] = (sample, features, refs)
                sources.append({'source_id': source['source_id'], 'pair_id': pair_id, 'typed_manifest': manifest,
                    'original_rgb': refs, 'source_canvas': sample['source_canvas'], 'target_canvas': sample['target_canvas'],
                    'source_landmarks5_canvas_pixels': points.tolist(), 'source_indices': indices.tolist(),
                    'target_person_ids': sample['local_condition_v4'].metadata[0]['target_person_ids']})
        cases = [(phase, item) for phase, path in (('baseline', args.baseline_report), ('final', args.final_report))
                 for item in evaluation.generation_records([path]) if item['record']['variant'] == 'native']
        records = []
        for phase, item in cases:
            record = item['record']; key = (record['source_id'], record['pair_id'])
            sample, features, refs = samples[key]
            canvas = CanvasSpec.from_record(sample['target_canvas'])
            pixels = np.asarray(Image.open(record['path']).convert('RGB'))
            if pixels.shape[:2] != canvas.canvas_hw:
                raise ValueError('Generated image differs from the exact target Canvas')
            result = measure_case(pixels, canvas, sample['local_condition_v4'].face_boxes[0].numpy(),
                                  features, detector, model, preparer.match_faces, evaluation)
            records.append({'phase': phase, 'source_id': key[0], 'pair_id': key[1],
                'image': evaluation.reference(record['path']), 'generation_record': item, **result})
        controls = []
        for (source, pair_id), (sample, features, refs) in samples.items():
            pixels = rgb(sample, 'tgt_image'); canvas = CanvasSpec.from_record(sample['target_canvas'])
            result = measure_case(pixels, canvas, sample['local_condition_v4'].face_boxes[0].numpy(),
                                  features, detector, model, preparer.match_faces, evaluation)
            controls.append({'source_id': source, 'pair_id': pair_id, 'image': refs['target'],
                'canvas_rgb_uint8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(), **result})
        report = {'kind': 'detected_fivepoint_source_identity_audit', 'execution': {'device': 'cpu', 'onnx_provider': 'CPUExecutionProvider'},
            'matching_policy': {'confidence_min': .5, 'box_min_side_px': 24, 'iou_min': .15,
                'center_margin_fraction': .25, 'second_best_assignment_margin': .02, 'both_people_required': True},
            'interpretation': 'Generated five points come from SCRFD, not GT. Geometry qualifies slot correspondence; detection/matching failures remain unavailable. This audit reduces target-alignment placement effects but still depends on face visibility/detection and is not an identity ground-truth score.',
            'detector': evaluation.reference(args.detector), 'identity_model': model_ref,
            'prepared': evaluation.reference(args.prepared), 'config': evaluation.reference(args.config),
            'implementation': evaluation.reference(__file__), 'matcher_implementation': evaluation.reference(args.preparer),
            'alignment_implementation': evaluation.reference(Path(__import__('src.pose_control.qwen21_full.face_loss_alignment', fromlist=['x']).__file__)),
            'sources': sources, 'records': records, 'positive_controls': controls}
        evaluation.write_json(args.report_dir / 'evaluation.json', report)
        return report
    except Exception as error:
        evaluation.write_json(args.report_dir / 'failure.json', {'error': repr(error), 'device': 'cpu'})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU detected-five-point identity audit; never overwrites reports.')
    for name in ('config', 'baseline-report', 'final-report', 'prepared', 'identity-model', 'detector', 'report-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, default=Path(__file__).with_name('evaluate_run.py'))
    parser.add_argument('--preparer', type=Path, default=Path(__file__).with_name('prepare_assets.py'))
    args = parser.parse_args(argv)
    evaluation = load_module(args.evaluator, 'detected_identity_evaluation')
    preparer = load_module(args.preparer, 'detected_identity_preparation')
    result = run(args, evaluation, preparer)
    print(json.dumps({'report': str(args.report_dir / 'evaluation.json'), 'records': len(result['records']), 'controls': len(result['positive_controls'])}))


if __name__ == '__main__':
    main()

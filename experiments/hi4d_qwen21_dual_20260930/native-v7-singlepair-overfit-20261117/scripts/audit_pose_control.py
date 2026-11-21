"""真实 Target GT 的 CPU 姿态正对照；不设分数通过阈值。

复用所指定 evaluate_run.py 的 pose_metrics，保持共同12关节、geometry序、
置信度0.30、请求torso归一化和髋心分配。每个typed样本仅跑一次全图检测，
不裁区域，不生成图像，不改训练或既有产物。白圈为请求投影，绿/蓝点为检测。

在既有SDPose环境运行（报告目录必须新建）：
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
CUDA_HOME=/home/shangguanrz/nihaochen/envs/qwen21 python audit_pose_control.py \
  --config /absolute/S0-ready.json --evaluator /absolute/evaluate_run-cpu-pose-v1.py \
  --report-dir /absolute/pose-gt-control-cpu-v1

GT分数反映几何拟合、投影、遮挡和估计器误差，不是生成模型的质量分数，
也不能从生成图的误差中直接减去。失败检测保留null指标及原因。
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import numpy as np


def run(args, evaluation):
    evaluation.new_report(args.report_dir)
    try:
        import cv2
        from PIL import Image, ImageDraw
        from src.data.canvas_contract import CanvasSpec
        from src.evaluation import sdpose
        from src.pose_control.qwen21_full.body_pose import JOINT_NAMES
        from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
        from src.pose_control.qwen21_full.native_v62.training.io import verify
        config = evaluation.read(args.config)
        runtime = sdpose.load_runtime(device='cpu')
        if str(runtime.device) != 'cpu':
            raise ValueError('GT control requires the actual CPU runtime')
        records = []
        for source in config['sources']:
            manifest = verify(source['manifest'])
            dataset = NativeFullDatasetV62(manifest['path'], load_target=True, data_profile=source['data_profile'])
            for index in range(len(dataset)):
                sample = dataset[index]
                target_canvas = CanvasSpec.from_record(sample['target_canvas'])
                pixels = ((sample['tgt_image'].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
                if pixels.shape[:2] != target_canvas.canvas_hw:
                    raise ValueError('Typed Target RGB differs from its canonical Canvas')
                requested = np.stack([sample['joint_uv' + suffix].numpy() for suffix in ('', '_b')])
                valid = np.stack([sample['projection_valid' + suffix].numpy() for suffix in ('', '_b')])
                people, raw = sdpose.detect_people(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), runtime,
                    order='geometry', sort_by='score', return_stats=True, min_area=0.)
                pose = evaluation.pose_metrics(requested, valid, people, target_canvas.canvas_hw, raw_count=raw)
                pose['pose_basis'] = 'true_target_gt_full_image_common12'
                overlay = Image.fromarray(pixels); draw = ImageDraw.Draw(overlay)
                for slot in range(2):
                    keep = valid[slot] & evaluation.valid_points(requested[slot], np.ones(21), target_canvas.canvas_hw)
                    for joint in evaluation.BODY21_COMMON:
                        if keep[joint]:
                            x, y = requested[slot, joint]; draw.ellipse((x-4, y-4, x+4, y+4), outline='white', width=2)
                detections = []
                for person, detected in enumerate(people):
                    slot = pose['assignment'].index(person) if pose['assignment'] else person
                    keep = evaluation.valid_points(detected['points'], detected['scores'], target_canvas.canvas_hw)
                    for joint in evaluation.BODY18_COMMON:
                        if keep[joint]:
                            x, y = detected['points'][joint]; draw.ellipse((x-2, y-2, x+2, y+2), fill=('lime', 'dodgerblue')[slot % 2])
                    detections.append({'box_xyxy': detected['box_xyxy'], 'box_score': float(detected['box_score']),
                        'valid_points134': int(keep.sum()), 'confident_points': [{'slot134': int(j),
                        'xy': detected['points'][j].tolist(), 'score': float(detected['scores'][j])} for j in np.flatnonzero(keep)]})
                path = args.report_dir / f'overlay_{len(records):02d}.png'
                with path.open('xb') as file:
                    overlay.save(file, format='PNG')
                records.append({'source_id': source['source_id'], 'pair_id': dataset.rows[index]['pair_id'],
                    'typed_manifest': manifest, 'source_canvas': sample['source_canvas'], 'target_canvas': sample['target_canvas'],
                    'pair_metadata': sample['pair_metadata'], 'source_indices': sample['local_condition_v4'].source_indices[0].tolist(),
                    'target_person_ids': sample['local_condition_v4'].metadata[0]['target_person_ids'],
                    'pose': pose, 'detections': detections, 'overlay': evaluation.reference(path),
                    'requested_common12': [[{'body21': joint, 'xy': requested[slot, joint].tolist()
                        if np.isfinite(requested[slot, joint]).all() else None, 'projection_valid': bool(valid[slot, joint])}
                        for joint in evaluation.BODY21_COMMON] for slot in range(2)]})
        report = {'kind': 'true_target_gt_pose_control', 'pose_backend': 'sdpose_yolo11x_wholebody_geometry',
            'execution': {'device': str(runtime.device), 'dtype': str(runtime.dtype),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
                'mkl_num_threads': os.environ.get('MKL_NUM_THREADS')}, 'score_threshold': .30, 'full_image_only': True,
            'interpretation': 'Measures GT projected-geometry versus detector differences, including fitting/occlusion/estimator error; no acceptance threshold or baseline error subtraction.',
            'config': evaluation.reference(args.config), 'config_record': config, 'evaluator': evaluation.reference(args.evaluator),
            'implementation': evaluation.reference(__file__), 'pose_code': evaluation.reference(sdpose.__file__),
            'pose_checkpoint_dir': str(runtime.ckpt_dir), 'yolo': evaluation.reference(runtime.ckpt_dir / 'yolo11x.pt'),
            'common_mapping': [{'body21': joint, 'name': JOINT_NAMES[joint], 'sdpose134_geometry': point}
                for joint, point in zip(evaluation.BODY21_COMMON, evaluation.BODY18_COMMON)], 'records': records}
        evaluation.write_json(args.report_dir / 'evaluation.json', report)
        return report
    except Exception as error:
        evaluation.write_json(args.report_dir / 'failure.json', {'error': repr(error), 'device': 'cpu'})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path, default=Path(__file__).with_name('evaluate_run.py'))
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('native_pilot_evaluation', args.evaluator)
    evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)
    result = run(args, evaluation)
    print(json.dumps({'report': str(args.report_dir / 'evaluation.json'), 'records': len(result['records']), 'device': 'cpu'}))


if __name__ == '__main__':
    main()

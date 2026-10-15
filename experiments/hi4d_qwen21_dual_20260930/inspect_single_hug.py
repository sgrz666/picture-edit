from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from eval_hi4d_qwen21_dual import DualPoseMetric, _target_keypoints
from src.evaluation.dwpose_onnx.onnxdet import inference_detector
from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset


roots = [Path('/home/shangguanrz/project/pic-edit/experiments/qwen21_hi4d_dual') / name / 'images'
         for name in ('single_hug_eval', 'single_hug_eval_seed43')]
manifest = Path('/tmp/qwen21_hi4d_single_hug_train.jsonl')
dataset = Hi4DQwen21Dataset(manifest, '/home/shangguanrz/nihaochen/qwen21_hi4d_controls')
sample = dataset[0]
row = sample['row']
boxes = []
for side in ('a', 'b'):
    mask = sample[f'human_mask_{side}'][0].numpy() > .5
    yy, xx = np.nonzero(mask)
    boxes.append(np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]) / 512)
print('target_boxes', boxes)
print('relations', int(sample['contact_valid_mask'].sum()), 'raster_max', float(sample['contact_raster'].max()))
for side in ('A', 'B'):
    _, scores = _target_keypoints(row[f'target_dwpose_{side}'], row['native_width'], row['native_height'])
    print(f'target_{side}_valid_joints', int((scores > .3).sum()))
pose = DualPoseMetric()
for root in roots:
    for mode in ('off', 'geometry', 'full'):
        image = Image.open(root / f'0000_{mode}.png')
        detected = pose.detect(image, boxes)
        bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        print(root.parent.name, mode, 'raw_detections', len(inference_detector(pose.det, bgr)),
              [(None if d is None else (int((d[1] > .3).sum()), float(d[1].max()))) for d in detected])

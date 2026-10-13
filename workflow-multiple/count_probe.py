#!/usr/bin/env python3
"""Why a two-person picture can be counted as one body: inspect the detector's own boxes.

Whole-image person count is the strictest gate the dual evaluation has, which makes it worth
checking rather than trusting. This replays the same YOLOX session and postprocessing the
evaluation uses and prints the class-0 boxes that survive at three NMS thresholds, so a merged
embrace can be told apart from a genuinely missing body. Run it with the geometry interpreter:

    add/envs/smplestx_bw/bin/python count_probe.py pairs/hug_pair00_084_c4_r000/B_target.png \
        runs/<name>/generation/final_image.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(__file__).resolve().parent

from src.evaluation.dwpose_onnx import onnxdet  # noqa: E402

NMS_THRESHOLDS = (0.45, 0.7, 0.95)
SCORE_CUTOFF = 0.3


def boxes_for(session, image, nms_thr):
    """The detector's person boxes in original-image pixels, at one NMS threshold."""
    input_shape = (640, 640)
    prepared, ratio = onnxdet.preprocess(image, input_shape)
    output = session.run(None, {session.get_inputs()[0].name: prepared[None, :, :, :]})
    predictions = onnxdet.demo_postprocess(output[0], input_shape)[0]
    boxes, scores = predictions[:, :4], predictions[:, 4:5] * predictions[:, 5:]
    corners = np.ones_like(boxes)
    corners[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    corners[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    corners[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    corners[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    corners /= ratio
    detections = onnxdet.multiclass_nms(corners, scores, nms_thr=nms_thr, score_thr=0.05)
    kept = [index for index in range(len(detections))
        if int(detections[index, 5]) == 0 and detections[index, 4] > SCORE_CUTOFF]
    return [{'score': float(detections[index, 4]),
        'box_xyxy': [float(value) for value in detections[index, :4]]} for index in kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path, nargs='+')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    args = parser.parse_args()
    session = ort.InferenceSession(json.loads(args.config.read_text())['dwpose_det_model'],
        providers=['CPUExecutionProvider'])
    for path in args.image:
        image = cv2.imread(str(path))
        if image is None:
            print(f'UNREADABLE {path}')
            continue
        print(f'== {path.name} {image.shape[1]}x{image.shape[0]}')
        for nms_thr in NMS_THRESHOLDS:
            boxes = boxes_for(session, image, nms_thr)
            print(f'   nms_thr={nms_thr}: {len(boxes)} person boxes above score '
                f'{SCORE_CUTOFF} ' + str([[round(value) for value in box['box_xyxy']]
                    for box in boxes]))
    print('the evaluation uses nms_thr=0.45; a wider threshold that only produces near-duplicate '
        'boxes over the same body means the pair was merged by the detector, not that the '
        'generator drew one person')


if __name__ == '__main__':
    main()

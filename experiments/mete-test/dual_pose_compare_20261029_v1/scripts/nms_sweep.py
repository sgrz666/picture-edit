import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import save_new


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.


def greedy_nms(boxes, scores, threshold):
    order = np.argsort(-np.asarray(scores, float))
    kept = []
    for index in order:
        if all(iou(boxes[index], boxes[other]) <= threshold for other in kept):
            kept.append(int(index))
    return sorted(kept)


def distinct_people(boxes, gt_boxes, threshold=.5):
    """Greedy one-to-one matching of predicted boxes to the two GT bodies."""
    pairs = sorted(((iou(box, gt), i, j) for i, box in enumerate(boxes)
                    for j, gt in enumerate(gt_boxes)), reverse=True)
    used_person, used_gt = set(), set()
    for overlap, i, j in pairs:
        if overlap < threshold:
            continue
        if i in used_person or j in used_gt:
            continue
        used_person.add(i)
        used_gt.add(j)
    return len(used_gt)


def sweep(frames, thresholds):
    rows = []
    for threshold in thresholds:
        two, duplicate, any_people, exactly_two, capped_two = 0, 0, 0, 0, 0
        histogram = {}
        for frame in frames:
            boxes = frame['raw_boxes']
            kept = greedy_nms(boxes, frame['raw_scores'], threshold) if boxes else []
            chosen = [boxes[i] for i in kept]
            histogram[str(len(chosen))] = histogram.get(str(len(chosen)), 0) + 1
            matched = distinct_people(chosen, frame['gt_boxes']) if chosen else 0
            two += matched == 2
            exactly_two += len(chosen) == 2 and matched == 2
            any_people += matched >= 1
            duplicate += len(chosen) >= 2 and matched <= 1
            top = sorted(kept, key=lambda i: -frame['raw_scores'][i])[:2]
            capped_two += distinct_people([boxes[i] for i in top], frame['gt_boxes']) == 2
        rows.append({'nms_thr': threshold, 'frames': len(frames),
                     'frames_with_two_distinct_people': two,
                     'two_distinct_rate': two / len(frames) if frames else None,
                     'frames_with_exactly_two_matching': exactly_two,
                     'capped_two_distinct_frames': capped_two,
                     'det_box_histogram': dict(sorted(histogram.items())),
                     'frames_with_any_person': any_people,
                     'duplicate_person_candidates': duplicate})
    return rows


def load_frames(output, request_path, mode):
    request = json.loads(Path(request_path).read_text())
    records = {item['id']: item for item in request['records']}
    frames = []
    for path in sorted((Path(output) / mode).glob('*.json')):
        saved = json.loads(path.read_text())
        detection = saved.get('detection') or {}
        if not detection.get('raw_boxes'):
            continue
        record = records[saved['id']]
        frames.append({'id': saved['id'], 'gt_boxes': record['gt_boxes'],
                       'raw_boxes': detection['raw_boxes'], 'raw_scores': detection['raw_scores'],
                       'recorded_post_nms': detection.get('post_nms_boxes')})
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--mode', default='sapiens_detr070')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[.3, .4, .5, .6, .7, .8, .9])
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    frames = load_frames(request['out_dir'], args.request, args.mode)
    rows = sweep(frames, args.thresholds)
    # Self-check: the recorded post-NMS boxes at 0.7 must be reproduced from the raw ones.
    check = greedy_nms(frames[0]['raw_boxes'], frames[0]['raw_scores'], .7) if frames else []
    save_new(args.out / 'nms_sweep.json',
             {'mode': args.mode, 'frames': len(frames), 'thresholds': rows,
              'reproduction_check': {'first_frame_kept': check}})
    for row in rows:
        print(json.dumps(row))


if __name__ == '__main__':
    main()

"""Measure visible source/output bodies with the same existing DWPose model on CPU.

Writes only diagnostics. Does not regenerate images or change any run.
"""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

WORKFLOW = Path('/home/shangguanrz/project/pic-edit/workflow')
ASSETS = WORKFLOW.parent / 'add/tools/dwpose_mimicmotion_official'
OUT = WORKFLOW / 'diagnostics/2026-10-13-scale'
sys.path.insert(0, str(ASSETS))
from dwpose.onnxdet import inference_detector
from dwpose.onnxpose import inference_pose


def session(name):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(ASSETS / 'models' / name), sess_options=options,
        providers=['CPUExecutionProvider'])


def iou(boxes, anchor):
    lo = np.maximum(boxes[:, :2], anchor[:2])
    hi = np.minimum(boxes[:, 2:], anchor[2:])
    area = np.prod(np.maximum(0, hi - lo), axis=1)
    return area / (np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        + np.prod(anchor[2:] - anchor[:2]) - area + 1e-8)


def measure(path, anchor, detector, pose):
    image = cv2.imread(str(path))
    boxes = inference_detector(detector, image)
    if len(boxes) == 0:
        raise RuntimeError(f'No person detected in {path}')
    overlaps = iou(boxes, anchor)
    selected = int(np.argmax(overlaps))
    p, scores = inference_pose(pose, boxes[selected:selected+1], image)
    p, scores = p[0], scores[0]
    neck, pelvis = p[[5, 6]].mean(axis=0), p[[11, 12]].mean(axis=0)
    face = p[23:40]
    face_scores = scores[23:40]
    visible_face = face[face_scores >= 0.3]
    return {'path':str(path), 'selected_bbox':boxes[selected].tolist(),
        'subject_iou_to_A_box':float(overlaps[selected]), 'persons_detected':int(len(boxes)),
        'torso_px':float(np.linalg.norm(neck-pelvis)),
        'shoulder_width_px':float(np.linalg.norm(p[5]-p[6])),
        'neck_canvas':neck.tolist(), 'pelvis_canvas':pelvis.tolist(),
        'torso_landmark_scores':scores[[5,6,11,12]].tolist(),
        'face_contour_height_px':float(np.ptp(visible_face[:,1])) if len(visible_face)>=5 else None,
        'ankle_scores':scores[[15,16]].tolist()}, p, scores


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    detector = session('yolox_l.onnx')
    pose = session('dw-ll_ucoco_384.onnx')
    names = ['hard_01','hard_02','hard_03','hard_04','hard_05','medium_01','medium_02',
        'hard_05_torso','hard_05_manual']
    results = []
    source_cache = {}
    for name in names:
        root = WORKFLOW / 'runs' / name
        camera = json.loads((root/'geometry/B_camera.json').read_text())
        meta = camera['source_letterbox']
        anchor = np.asarray(camera['detections']['A']['bbox_xyxy']) * np.tile(meta['scale_xy'],2)
        anchor += np.tile(meta['offset_xy'],2)
        source_name = 'hard_05' if name.startswith('hard_05') else name
        if source_name not in source_cache:
            source_cache[source_name] = measure(root/'inputs/A_reference.png',anchor,detector,pose)
        a, ap, asc = source_cache[source_name]
        g, gp, gsc = measure(root/'generation/final_canvas.png',anchor,detector,pose)
        guide = np.load(root/'geometry/B_keypoints.npz')['keypoints_generation_px']
        guide_torso = float(np.linalg.norm(guide[1]-np.nanmean(guide[[8,11]],axis=0)))
        row = {'run':name,'source':a,'generated':g,
            'output_to_A_torso':g['torso_px']/a['torso_px'],
            'guide_to_A_torso':guide_torso/a['torso_px'],
            'desired_A_torso_scale':a['torso_px']/guide_torso,
            'estimate_kind':'DWPose observed 2D; pose/viewpoint affect projected lengths; not 3D anthropometry'}
        np.savez_compressed(OUT/f'{name}_visible_pose.npz',source_raw133=ap,source_scores=asc,
            generated_raw133=gp,generated_scores=gsc)
        results.append(row)
        (OUT/'source_pose_measurements.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(name,'A/output torso',round(a['torso_px'],1),round(g['torso_px'],1),
            'output/A',round(row['output_to_A_torso'],3),
            'guide/A',round(row['guide_to_A_torso'],3),flush=True)

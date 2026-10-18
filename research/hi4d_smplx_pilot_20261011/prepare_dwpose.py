"""Existing DWPose model on SMPL person regions, then calibrated triangulation."""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from geometry import project, associate, triangulate


def reorder(keypoints, scores):
    # Same reorder and inserted neck as the installed official Wholebody class.
    info = np.concatenate([keypoints, scores[..., None]], axis=-1)
    neck = info[:, [5, 6]].mean(axis=1)
    neck[:, 2] = np.logical_and(info[:, 5, 2] > .3, info[:, 6, 2] > .3)
    info = np.insert(info, 17, neck, axis=1)
    info[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = info[:, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
    return info[..., :2], info[..., 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/shangguanrz/project/pic-edit')
    p.add_argument('--workdir', required=True)
    a = p.parse_args()
    root, work = Path(a.root), Path(a.workdir)
    repo = root / 'add/tools/dwpose_mimicmotion_official'
    sys.path.insert(0, str(repo))
    from dwpose.onnxpose import inference_pose
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    session = ort.InferenceSession(str(repo / 'models/dw-ll_ucoco_384.onnx'), sess_options=opts, providers=['CPUExecutionProvider'])
    out = work / 'observations'
    out.mkdir(exist_ok=True)
    started = time.time()
    records = []
    match_dw = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    match_smpl = [17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7]
    for action, frame in [('talk14', 32), ('hug14', 62)]:
        raw = root / 'datasets/Hi4D_pilot_v1/raw/pair14' / action
        smpl = np.load(raw / 'smpl' / f'{frame:06d}.npz')
        cams = np.load(raw / 'cameras/rgb_cameras.npz')
        all_xy, all_scores, assignments = [], [], []
        for c, camera in enumerate(cams['ids']):
            image_path = work / 'inputs/pair14' / action / 'images' / str(camera) / f'{frame:06d}.jpg'
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            boxes = []
            for actor in range(2):
                uv = project(smpl['verts'][actor], cams['intrinsics'][c], cams['extrinsics'][c])
                lo, hi = uv.min(0), uv.max(0)
                pad = .08 * (hi - lo)
                lo = np.maximum(lo - pad, 0)
                hi = np.minimum(hi + pad, [width - 1, height - 1])
                boxes.append(np.r_[lo, hi])
            xy, scores = reorder(*inference_pose(session, np.stack(boxes), image))
            reference = np.stack([project(j[match_smpl], cams['intrinsics'][c], cams['extrinsics'][c]) for j in smpl['joints_3d']])
            assignment, errors = associate(xy[:, match_dw], scores[:, match_dw], reference, max_error=65)
            ordered_xy = np.zeros((2, 134, 2), dtype=np.float32)
            ordered_scores = np.zeros((2, 134), dtype=np.float32)
            for actor, candidate in enumerate(assignment):
                if candidate >= 0:
                    ordered_xy[actor] = xy[candidate]
                    ordered_scores[actor] = scores[candidate]
            np.savez_compressed(out / f'{action}_{frame:06d}_cam{camera}.npz', xy_native=ordered_xy, scores=ordered_scores, actor_assignment=assignment, assignment_error_px=errors, boxes_native=np.stack(boxes), K=cams['intrinsics'][c], extrinsics=cams['extrinsics'][c], width=width, height=height)
            all_xy.append(ordered_xy)
            all_scores.append(ordered_scores)
            assignments.append(assignment)
            print('DWPOSE', action, int(camera), 'assignment', assignment.tolist(), 'errors', errors.round(2).tolist(), flush=True)
        all_xy, all_scores = np.stack(all_xy), np.stack(all_scores)
        xyz, valid, reproj, nviews = [], [], [], []
        for actor in range(2):
            points, ok, error, nv = triangulate(all_xy[:, actor], all_scores[:, actor], cams['intrinsics'], cams['extrinsics'])
            # Face contours describe view silhouettes, not fixed anatomical 3D points.
            ok[24:41] = False
            for region, anchor, radius in [(slice(24, 92), 15, .25), (slice(92, 113), 20, .3), (slice(113, 134), 21, .3)]:
                ok[region] &= np.linalg.norm(points[region] - smpl['joints_3d'][actor, anchor], axis=-1) < radius
            points[~ok] = np.nan
            error[~ok] = np.nan
            nv[~ok] = 0
            xyz.append(points); valid.append(ok); reproj.append(error); nviews.append(nv)
        np.savez_compressed(out / f'{action}_{frame:06d}_triangulated.npz', xyz_world=np.stack(xyz), valid=np.stack(valid), median_reprojection_px=np.stack(reproj), inlier_views=np.stack(nviews), xy_native=all_xy, scores=all_scores, camera_ids=cams['ids'])
        rec = {'action': action, 'frame': frame, 'valid_3d_points': [int(v.sum()) for v in valid], 'valid_left_hand': [int(v[92:113].sum()) for v in valid], 'valid_right_hand': [int(v[113:134].sum()) for v in valid], 'face_contour_3d': 'invalid: view dependent', 'assignments': np.stack(assignments).tolist()}
        records.append(rec)
        print(json.dumps(rec), flush=True)
    (out / 'metadata.json').write_text(json.dumps({'records': records, 'elapsed_seconds': time.time() - started, 'model': str(repo / 'models/dw-ll_ucoco_384.onnx'), 'person_regions': 'Hi4D SMPL projections; used only for offline annotation fitting', 'provider': session.get_providers()}, indent=2))


if __name__ == '__main__':
    main()

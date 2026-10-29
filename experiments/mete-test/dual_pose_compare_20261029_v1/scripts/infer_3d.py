#!/usr/bin/env python3
"""Four 3D benchmark arms; model imports are deferred until uncached work exists."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

BODY_NAMES = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
              'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
              'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
MODES = ('multihmr', 'multihmr_fov60', 'smplest_gtbox', 'smplest_yolox')
SCHEMA = 'dual_pose_benchmark_v1'
GIB = 1024 ** 3


def finite_array(value, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite array of shape {shape}, got {value.shape}')
    return value


def affine_for_resize(native_wh, resized_wh, canvas_wh):
    native = finite_array(native_wh, (2,))
    resized = finite_array(resized_wh, (2,))
    canvas = finite_array(canvas_wh, (2,))
    if np.any(native <= 0) or np.any(resized <= 0) or np.any(canvas < resized):
        raise ValueError('Invalid resize dimensions')
    sx, sy = resized / native
    px, py = [round(float(v) / 2) for v in canvas - resized]
    return np.array([[sx, 0, px], [0, sy, py], [0, 0, 1]], dtype=np.float64)


def apply_affine(xy, affine):
    xy = finite_array(xy)
    affine = finite_array(affine, (3, 3))
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('xy must be Nx2')
    homogeneous = np.column_stack([xy, np.ones(len(xy))]) @ affine.T
    if np.any(homogeneous[:, 2] == 0):
        raise ValueError('Invalid homogeneous denominator')
    return finite_array(homogeneous[:, :2] / homogeneous[:, 2:])


def project_points(xyz, k, eps=0.):
    xyz = finite_array(xyz)
    k = finite_array(k, (3, 3))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError('Expected Nx3 camera joints and pinhole K')
    depth = xyz[:, 2:] + eps
    if np.any(depth <= 0):
        raise ValueError('Nonpositive camera depth')
    return finite_array((xyz[:, :2] / depth) @ k[:2, :2].T + k[:2, 2])


def crop_camera(bbox_xywh, focal, principal, input_body_shape):
    x, y, w, h = finite_array(bbox_xywh, (4,))
    body_h, body_w = finite_array(input_body_shape, (2,))
    if min(w, h, body_w, body_h) <= 0:
        raise ValueError('Invalid crop dimensions')
    scale = np.array([w / body_w, h / body_h])
    f = finite_array(focal, (2,)) * scale
    c = finite_array(principal, (2,)) * scale + [x, y]
    return np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])


def camera_policy(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    shared = mode.startswith('multihmr')
    return {'scene_camera_available': shared, 'privileged_camera': mode == 'multihmr',
            'camera_basis': 'shared_scene_camera' if shared else 'per_person_crop_camera',
            'absolute_3d_comparable': mode == 'multihmr',
            'inter_person_3d_available': shared}


def to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return finite_array(value)


def build_instance(xyz, xy, names, bbox=None, score=1.):
    xyz = finite_array(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('xyz must be Nx3')
    xy = finite_array(xy, (len(xyz), 2))
    names = list(names[:len(xyz)])
    if len(names) != len(xyz) or len(set(names)) != len(names):
        raise ValueError('Joint names must uniquely label every joint')
    try:
        indices = [names.index(name) for name in BODY_NAMES]
        pelvis = xyz[names.index('pelvis')]
    except ValueError as exc:
        raise ValueError('Missing named SMPL-X body joint or pelvis') from exc
    score = float(finite_array(score).reshape(()))
    if not 0 <= score <= 1:
        raise ValueError('Score must be in [0, 1]')
    return {'bbox': None if bbox is None else finite_array(bbox, (4,)).tolist(),
            'xy': xy[indices].tolist(), 'scores': [score] * 12, 'joint_names': list(BODY_NAMES),
            'xyz': xyz[indices].tolist(), 'pelvis': pelvis.tolist(),
            'extra_xyz': xyz.tolist(), 'extra_joint_names': names,
            'extra_xy': xy.tolist(), 'extra_layout': 'official_smplx_joint_names'}


def verify_mesh(reconstructed, expected):
    reconstructed = finite_array(reconstructed)
    expected = finite_array(expected, reconstructed.shape)
    error = float(np.max(np.abs(reconstructed - expected)))
    if error >= 1e-4:
        raise ValueError(f'SMPL-X reconstructed mesh mismatch: {error:.9g} m')
    return error


def multihmr_forward(model, tensor, camera):
    return model(tensor.float(), K=camera.float(), is_training=False,
                 det_thresh=.2, nms_kernel_size=3)


def multihmr_instances(humans, names, affine):
    inverse = np.linalg.inv(finite_array(affine, (3, 3)))
    return [build_instance(to_numpy(h['j3d']), apply_affine(to_numpy(h['j2d']), inverse),
                           names, score=to_numpy(h['scores']).reshape(())) for h in humans]


def reconstruct_smplest(out, layer, torch):
    mapping = {'global_orient': 'smplx_root_pose', 'body_pose': 'smplx_body_pose',
               'left_hand_pose': 'smplx_lhand_pose', 'right_hand_pose': 'smplx_rhand_pose',
               'jaw_pose': 'smplx_jaw_pose', 'betas': 'smplx_shape',
               'expression': 'smplx_expr', 'transl': 'cam_trans'}
    params = {key: out[source].detach().float().cpu() for key, source in mapping.items()}
    with torch.no_grad():
        body = layer(**params, leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3))
    error = verify_mesh(to_numpy(body.vertices), to_numpy(out['smplx_mesh_cam']))
    return to_numpy(body.joints)[0], error


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strict_loads(raw):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON literal: {value}')
    return json.loads(raw, parse_constant=invalid)


def read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read()
    return strict_loads(raw), hashlib.sha256(raw).hexdigest()


def write_json_xb(path, payload):
    raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def validate_request(request):
    if request.get('schema') != 'dual_pose_benchmark_v1':
        raise ValueError('Expected dual_pose_benchmark_v1 request')
    if request.get('body_joint_names') != BODY_NAMES:
        raise ValueError('body_joint_names must match the ordered body12 contract')
    for key in ('project_root', 'out_dir'):
        if not isinstance(request.get(key), str) or not Path(request[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    records = request.get('records')
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    seen = set()
    for record in records:
        identifier = record.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{16}', identifier):
            raise ValueError('Record id must be the first 16 lowercase SHA256 digits')
        if identifier in seen:
            raise ValueError(f'Duplicate id: {identifier}')
        seen.add(identifier)


def verify_input(record):
    expected = record.get('image_sha256', '')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Invalid image SHA256')
    if record['id'] != expected[:16]:
        raise ValueError('Record id does not match image SHA256')
    if not isinstance(record.get('image'), str) or not Path(record['image']).is_absolute():
        raise ValueError('image must be an absolute path')
    if file_sha256(record['image']) != expected:
        raise ValueError(f'Image SHA256 mismatch: {record["image"]}')


def cached_ok(path, record, mode, request_sha):
    if not path.exists():
        return False
    old, _ = read_json(path)
    if (old.get('image_sha256'), old.get('request_sha256')) != (record['image_sha256'], request_sha):
        raise FileExistsError(f'Different fingerprint; refusing overwrite: {path}')
    if old.get('status') != 'ok':
        raise FileExistsError(f'Existing non-ok result; refusing rerun: {path}')
    if old.get('id') != record['id'] or old.get('mode') != mode or old.get('schema') != SCHEMA:
        raise FileExistsError(f'Invalid cached result; refusing overwrite: {path}')
    return True


def resolve_boxes(request, record, mode, request_sha):
    if mode == 'smplest_gtbox':
        boxes = finite_array(record['gt_boxes'], (2, 4))
        provenance = {'bbox_source': 'gt_boxes', 'privileged_boxes': True}
    else:
        path = Path(request['out_dir']) / 'dwpose_auto' / f'{record["id"]}.json'
        detection, digest = read_json(path)
        expected = {'id': record['id'], 'mode': 'dwpose_auto', 'status': 'ok',
                    'image_sha256': record['image_sha256'], 'request_sha256': request_sha}
        if any(detection.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Stale or failed dwpose_auto result: {path}')
        instances = detection['instances']
        if detection.get('det_count') != len(instances):
            raise ValueError('dwpose_auto det_count differs from instances')
        boxes = finite_array([person['bbox'] for person in instances], (len(instances), 4)) if instances else np.empty((0, 4))
        provenance = {'bbox_source': 'dwpose_auto', 'privileged_boxes': False,
                      'bbox_source_path': str(path), 'bbox_source_sha256': digest}
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Expected positive ordered XYXY boxes')
    return boxes.tolist(), provenance


def require_gpu(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU inference is not supported')
    free, _ = torch.cuda.mem_get_info('cuda:0')
    if free < 32 * GIB:
        raise RuntimeError(f'GPU startup requires 32 GiB free; available {free / GIB:.2f} GiB')
    return free / GIB


@contextmanager
def model_directory(root):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def load_smplest_config(root, out_dir, config_class):
    root, out_dir = Path(root), Path(out_dir)
    config_path = root / 'pretrained_models/smplest_x_h/config_base.py'
    checkpoint = root / 'pretrained_models/smplest_x_h/smplest_x_h.pth.tar'
    human_models = root / 'human_models/human_model_files'
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not human_models.is_dir():
        raise FileNotFoundError(human_models)
    cfg = config_class.load_config(str(config_path))
    exp_name = f'dual_pose_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex}'
    cfg.update_config({'model': {'pretrained_model_path': str(checkpoint),
                                 'human_model_path': str(human_models)},
                       'log': {'exp_name': exp_name, 'log_dir': str(out_dir / 'logs' / exp_name),
                               'output_dir': None, 'model_dir': None, 'result_dir': None}})
    cfg.prepare_log()
    return cfg


class FrameError(RuntimeError):
    def __init__(self, message, det_count):
        super().__init__(message)
        self.det_count = det_count


class MultiHMRBackend:
    def __init__(self, request, mode, torch):
        root = Path(request['project_root'])
        repo = root / 'add/tools/multi-hmr'
        checkpoint = repo / 'models/multiHMR/multiHMR_896_L.pt'
        for path in (checkpoint, repo / 'models/smpl_mean_params.npz',
                     repo / 'models/smplx/SMPLX_NEUTRAL.npz'):
            if not path.is_file():
                raise FileNotFoundError(path)
        os.environ['MULTIHMR_DINOV2_REPO'] = str(root / 'add/tools/dinov2_e1277af')
        os.environ['XFORMERS_DISABLED'] = '1'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        sys.path.insert(0, str(repo))
        with model_directory(repo):
            from demo_smplx_only import load_model, open_image, get_camera_parameters
            self.model = load_model('multiHMR_896_L', device=torch.device('cuda:0')).float().eval()
        from smplx.joint_names import JOINT_NAMES
        self.names = JOINT_NAMES
        self.open_image = open_image
        self.get_camera_parameters = get_camera_parameters
        self.torch, self.mode, self.device = torch, mode, 'cuda:0'
        self.size = int(self.model.img_size)
        self.provenance = {'model': 'multiHMR_896_L', 'checkpoint': str(checkpoint),
                           'det_thresh': .2, 'nms_kernel_size': 3, 'inference_dtype': 'float32',
                           'autocast': False, 'score_semantics': 'person_detection_score_repeated_body12'}

    def predict(self, record):
        from PIL import Image, ImageOps
        torch = self.torch
        with Image.open(record['image']) as image:
            native_wh = image.size
            resized_wh = ImageOps.contain(image, (self.size, self.size)).size
        affine = affine_for_resize(native_wh, resized_wh, (self.size, self.size))
        if self.mode == 'multihmr':
            k_native = finite_array(record['K'], (3, 3))
            camera = torch.as_tensor(affine @ k_native, dtype=torch.float32, device=self.device)[None]
        else:
            camera = self.get_camera_parameters(self.size, fov=60, device=torch.device(self.device)).float()
            k_native = np.linalg.inv(affine) @ to_numpy(camera)[0]
        tensor, _ = self.open_image(record['image'], self.size, device=torch.device(self.device))
        with torch.no_grad():
            humans = multihmr_forward(self.model, tensor, camera)
        try:
            instances = multihmr_instances(humans, self.names, affine)
        except Exception as exc:
            raise FrameError(str(exc), len(humans)) from exc
        return {'instances': instances, 'provenance': {
            'K_native': k_native.tolist(), 'K_model': to_numpy(camera)[0].tolist(),
            'affine_native_to_model': affine.tolist(), 'native_wh': list(native_wh),
            'resized_wh': list(resized_wh), 'fov_degrees': 60 if self.mode.endswith('fov60') else None,
            'pixel_basis': 'original_file_pixels_no_exif_transpose'}}


class SmplestBackend:
    def __init__(self, request, mode, torch):
        root = Path(request['project_root']) / 'add/tools/SMPLest-X'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        sys.path.insert(0, str(root))
        from main.config import Config
        from human_models.human_models import SMPLX
        from main.base import Tester
        from utils.data_utils import load_img, process_bbox, generate_patch_image
        from torchvision.transforms import ToTensor
        from smplx.joint_names import JOINT_NAMES
        self.cfg = load_smplest_config(root, Path(request['out_dir']), Config)
        human = SMPLX(self.cfg.model.human_model_path)
        self.layer = human.layer['neutral'].cpu().eval()
        self.mapping = list(human.joint_idx)
        self.names = JOINT_NAMES
        tester = Tester(self.cfg)
        tester._make_model()
        self.model = tester.model
        self.torch, self.device = torch, 'cuda:0'
        self.load_img, self.process_bbox = load_img, process_bbox
        self.generate_patch_image, self.transform = generate_patch_image, ToTensor()
        self.provenance = {'model': 'SMPLest-X_H', 'checkpoint': self.cfg.model.pretrained_model_path,
                           'human_model_path': self.cfg.model.human_model_path,
                           'bbox_ratio': float(getattr(self.cfg.data, 'bbox_ratio', 1.25)),
                           'inference_dtype': 'float32', 'retarget_cam': False,
                           'score_semantics': 'constant_one_no_joint_confidence',
                           'allowed_3d_comparison': 'root_aligned_only', 'units': 'meters'}

    def predict(self, record):
        boxes = record['_boxes']
        instances = []
        if not boxes:
            return {'instances': [], 'provenance': record['_bbox_provenance']}
        try:
            image = self.load_img(record['image'])
            height, width = image.shape[:2]
            cfg = self.cfg
            for xyxy in boxes:
                x, y, right, bottom = xyxy
                bbox = self.process_bbox(bbox=np.array([x, y, right - x, bottom - y]),
                                         img_width=width, img_height=height,
                                         input_img_shape=cfg.model.input_img_shape,
                                         ratio=getattr(cfg.data, 'bbox_ratio', 1.25))
                if bbox is None:
                    raise ValueError('process_bbox rejected a person box')
                bbox = finite_array(bbox, (4,))
                patch, _, _ = self.generate_patch_image(cvimg=image, bbox=bbox, scale=1., rot=0.,
                                                        do_flip=False, out_shape=cfg.model.input_img_shape)
                tensor = (self.transform(patch.astype(np.float32)) / 255).to(self.device)[None]
                with self.torch.no_grad():
                    out = self.model({'img': tensor}, {}, {}, 'test')
                xyz, mesh_error = reconstruct_smplest(out, self.layer, self.torch)
                k = crop_camera(bbox, cfg.model.focal, cfg.model.princpt, cfg.model.input_body_shape)
                xy = project_points(xyz, k, eps=1e-4)
                person = build_instance(xyz, xy, self.names, bbox=xyxy)
                person.update(K=k.tolist(), processed_bbox_xywh=bbox.tolist(),
                              camera_basis='per_person_crop_camera', scene_camera_available=False,
                              mesh_max_error_m=mesh_error,
                              extra_keypoints=xy[self.mapping].tolist(),
                              extra_keypoint_names=[self.names[i] for i in self.mapping],
                              extra_keypoints_layout='smplestx_137')
                instances.append(person)
        except Exception as exc:
            raise FrameError(str(exc), len(boxes)) from exc
        return {'instances': instances, 'provenance': record['_bbox_provenance']}


def make_backend(request, mode):
    sys.dont_write_bytecode = True
    import torch
    free_gib = require_gpu(torch)
    backend_type = MultiHMRBackend if mode.startswith('multihmr') else SmplestBackend
    backend = backend_type(request, mode, torch)
    backend.provenance.update(script_sha256=file_sha256(Path(__file__).resolve()),
                              torch_version=torch.__version__, numpy_version=np.__version__,
                              gpu_name=torch.cuda.get_device_name('cuda:0'), startup_free_gib=free_gib,
                              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return backend


def timed_prediction(backend, record, warmup):
    cuda, device = backend.torch.cuda, backend.device
    if warmup:
        backend.predict(record)
    cuda.synchronize(device)
    cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = backend.predict(record)
    cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1000
    return result, {'latency_ms': latency, 'peak_allocated_gib': cuda.max_memory_allocated(device) / GIB,
                    'peak_reserved_gib': cuda.max_memory_reserved(device) / GIB, 'device': str(device)}


def frame_payload(record, mode, request_sha, instances=None, metrics=None, provenance=None, error=None):
    policy = camera_policy(mode)
    payload = {'schema': SCHEMA, 'id': record['id'], 'mode': mode,
               'image_sha256': record.get('image_sha256'), 'request_sha256': request_sha,
               'status': 'error' if error is not None else 'ok',
               'det_count': getattr(error, 'det_count', None) if error is not None else len(instances),
               'scene_camera_available': policy['scene_camera_available'],
               'camera_basis': policy['camera_basis'], 'instances': [] if error is not None else instances,
               'latency_ms': None, 'peak_allocated_gib': None, 'peak_reserved_gib': None,
               'device': None, 'provenance': {'units': 'meters', **(provenance or {}), **policy}}
    if metrics:
        payload.update(metrics)
    if error is not None:
        payload['error'] = f'{type(error).__name__}: {error}'
    return payload


def record_error(path, record, mode, request_sha, error, provenance=None):
    print(f'[{mode}/{record["id"]}] {type(error).__name__}: {error}', file=sys.stderr)
    if path.exists():
        return
    try:
        write_json_xb(path, frame_payload(record, mode, request_sha, error=error, provenance=provenance))
    except Exception as save_error:
        print(f'Cannot save error without overwrite: {path}: {save_error}', file=sys.stderr)


def run_batch(request_path, mode, limit=None, backend_factory=None):
    if mode not in MODES or (limit is not None and limit < 0):
        raise ValueError('Invalid mode or negative limit')
    request, request_sha = read_json(Path(request_path))
    validate_request(request)
    records = request['records'] if limit is None else request['records'][:limit]
    pending, failed = [], False
    for record in records:
        path = Path(request['out_dir']) / mode / f'{record["id"]}.json'
        try:
            verify_input(record)
            if cached_ok(path, record, mode, request_sha):
                continue
            work = dict(record)
            if mode.startswith('smplest'):
                work['_boxes'], work['_bbox_provenance'] = resolve_boxes(request, record, mode, request_sha)
            elif mode == 'multihmr':
                k = finite_array(record['K'], (3, 3))
                if not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0:
                    raise ValueError('Invalid native camera K')
            pending.append((work, path))
        except Exception as exc:
            failed = True
            record_error(path, record, mode, request_sha, exc)
    if not pending:
        return int(failed)
    try:
        backend = (backend_factory or make_backend)(request, mode)
    except Exception as exc:
        for record, path in pending:
            record_error(path, record, mode, request_sha, exc)
        return 1
    warmed = False
    for record, path in pending:
        provenance = dict(backend.provenance)
        det_count = None
        try:
            # Recheck immediately before inference in case startup took a long time.
            verify_input(record)
            if cached_ok(path, record, mode, request_sha):
                continue
            result, metrics = timed_prediction(backend, record, warmup=not warmed)
            det_count = len(result['instances'])
            warmed = warmed or not mode.startswith('smplest') or bool(record['_boxes'])
            provenance.update(result.get('provenance', {}), warmup_passes=1,
                              latency_scope='steady_state_decode_preprocess_inference_cpu_geometry_excludes_hash_json_init')
            payload = frame_payload(record, mode, request_sha, result['instances'], metrics, provenance)
            write_json_xb(path, payload)
        except Exception as exc:
            failed = True
            if det_count is not None and not isinstance(exc, FrameError):
                exc = FrameError(f'{type(exc).__name__}: {exc}', det_count)
            record_error(path, record, mode, request_sha, exc, backend.provenance)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=MODES)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    try:
        return run_batch(args.request.resolve(), args.mode, args.limit)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

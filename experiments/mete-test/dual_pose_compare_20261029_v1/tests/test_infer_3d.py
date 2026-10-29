import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'infer_3d.py'
BODY = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.value.astype(np.float32))

    def numpy(self):
        return self.value

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, index):
        return Tensor(self.value[index])

    def __truediv__(self, other):
        return Tensor(self.value / other)


class MemoryFiles:
    def __init__(self, files):
        self.files = {str(Path(k)): v for k, v in files.items()}
        self.writes = []

    def open(self, path, mode='r', **kwargs):
        name = str(path)
        if mode == 'rb':
            if name not in self.files:
                raise FileNotFoundError(name)
            return io.BytesIO(self.files[name])
        if mode != 'xb':
            raise AssertionError(f'Unexpected file mode: {mode}')
        if name in self.files:
            raise FileExistsError(name)
        self.writes.append((name, mode))
        owner = self

        class Writer(io.BytesIO):
            def close(self):
                if not self.closed:
                    owner.files[name] = self.getvalue()
                super().close()

        return Writer()

    def install(self, stack):
        stack.enter_context(mock.patch.object(Path, 'open', lambda p, *a, **k: self.open(p, *a, **k)))
        stack.enter_context(mock.patch.object(Path, 'exists', lambda p: str(p) in self.files))
        stack.enter_context(mock.patch.object(Path, 'mkdir'))


class Infer3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.available = SCRIPT.is_file()
        if cls.available:
            spec = importlib.util.spec_from_file_location('infer_3d_under_test', SCRIPT)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(self.available, 'infer_3d implementation does not exist yet')
        self.m = self.module
        self.names = ['pelvis', 'unused'] + BODY[::-1]
        self.xyz = np.arange(len(self.names) * 3, dtype=float).reshape(-1, 3) / 10
        self.xyz[:, 2] += 3
        self.xy = self.xyz[:, :2] * 100
        self.root = SCRIPT.parents[1]
        self.image = self.root / 'input' / 'frame.jpg'
        self.image_bytes = b'input-image-for-hash-only'
        self.sha = hashlib.sha256(self.image_bytes).hexdigest()
        self.record = {'id': self.sha[:16], 'image': str(self.image),
                       'image_sha256': self.sha, 'K': [[900, 0, 500], [0, 910, 333], [0, 0, 1]],
                       'gt_boxes': [[10, 20, 110, 220], [200, 40, 260, 180]]}
        self.request_path = self.root / 'request.json'
        self.request = {'schema': 'dual_pose_benchmark_v1', 'project_root': str(self.root),
                        'out_dir': str(self.root / 'output'), 'body_joint_names': BODY,
                        'records': [self.record]}

    def instance(self, shift=0):
        return self.m.build_instance(self.xyz + shift, self.xy, self.names)

    def fs(self):
        raw = json.dumps(self.request).encode()
        return MemoryFiles({self.request_path: raw, self.image: self.image_bytes}), hashlib.sha256(raw).hexdigest()

    def result_path(self, mode='multihmr', record=None):
        return Path(self.request['out_dir']) / mode / ((record or self.record)['id'] + '.json')

    def fake_backend(self):
        cuda = mock.Mock()
        cuda.max_memory_allocated.return_value = 2 * 1024 ** 3
        cuda.max_memory_reserved.return_value = 3 * 1024 ** 3
        return SimpleNamespace(torch=SimpleNamespace(cuda=cuda), device='cuda:0',
                               provenance={'test_backend': True},
                               predict=mock.Mock(return_value={'instances': [self.instance(), self.instance(1)],
                                                               'provenance': {}}))

    def test_non_square_intrinsics_and_inverse_affine(self):
        affine = self.m.affine_for_resize((1001, 667), (896, 597), (896, 896))
        self.assertNotEqual(affine[0, 0], affine[1, 1])
        self.assertEqual(affine[1, 2], 150)
        k = np.asarray(self.record['K'], float)
        native = self.m.project_points(self.xyz, k)
        model = self.m.project_points(self.xyz, affine @ k)
        np.testing.assert_allclose(self.m.apply_affine(native, affine), model)
        np.testing.assert_allclose(self.m.apply_affine(model, np.linalg.inv(affine)), native)
        portrait = self.m.affine_for_resize((667, 1001), (597, 896), (896, 896))
        self.assertEqual(portrait[0, 2], 150)

    def test_padding_matches_actual_pillow_pixels(self):
        from PIL import Image, ImageOps
        for size in [(1001, 667), (667, 1001), (1920, 1080), (2048, 2048)]:
            image = Image.new('RGB', size, (255, 255, 255))
            resized = ImageOps.contain(image, (896, 896))
            padded = np.asarray(ImageOps.pad(resized, (896, 896)))
            ys, xs = np.where(padded[:, :, 0] > 0)
            affine = self.m.affine_for_resize(size, resized.size, (896, 896))
            np.testing.assert_array_equal(affine[:2, 2], [xs.min(), ys.min()])

    def test_named_body_mapping_and_pelvis(self):
        person = self.instance()
        self.assertEqual(person['joint_names'], BODY)
        np.testing.assert_array_equal(person['xyz'], self.xyz[[self.names.index(n) for n in BODY]])
        np.testing.assert_array_equal(person['pelvis'], self.xyz[0])
        self.assertEqual(np.shape(person['xy']), (12, 2))
        self.assertEqual(np.shape(person['scores']), (12,))
        self.assertIsNone(person['bbox'])

    def test_volume_translation_does_not_change_root_relative_joints(self):
        a = self.instance()
        b = self.m.build_instance(self.xyz + [9, -3, 5], self.xy, self.names)
        np.testing.assert_allclose(np.asarray(a['xyz']) - a['pelvis'],
                                   np.asarray(b['xyz']) - b['pelvis'], atol=1e-12)
        self.assertFalse(np.allclose(a['pelvis'], b['pelvis']))

    def test_missing_names_and_nonfinite_geometry_fail(self):
        with self.assertRaises(ValueError):
            self.m.build_instance(self.xyz, self.xy, ['absent'] * len(self.names))
        xyz = self.xyz.copy()
        xyz[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.m.build_instance(xyz, self.xy, self.names)

    def test_crop_cameras_are_per_person_and_use_processed_xywh(self):
        k1 = self.m.crop_camera([10, 20, 100, 200], [5000, 6000], [96, 128], [256, 192])
        k2 = self.m.crop_camera([200, 40, 60, 140], [5000, 6000], [96, 128], [256, 192])
        np.testing.assert_allclose(k1, [[5000 / 192 * 100, 0, 60], [0, 6000 / 256 * 200, 120], [0, 0, 1]])
        self.assertFalse(np.array_equal(k1, k2))
        xy = self.m.project_points(np.array([[1, 2, 4.]]), k1, eps=1e-4)
        np.testing.assert_allclose(xy, [[k1[0, 0] / 4.0001 + 60, k1[1, 1] * 2 / 4.0001 + 120]])
        for mode in ['smplest_gtbox', 'smplest_yolox']:
            policy = self.m.camera_policy(mode)
            self.assertFalse(policy['scene_camera_available'])
            self.assertEqual(policy['camera_basis'], 'per_person_crop_camera')
        self.assertTrue(self.m.camera_policy('multihmr')['privileged_camera'])
        self.assertFalse(self.m.camera_policy('multihmr_fov60')['privileged_camera'])

    def test_mesh_tolerance_is_strict(self):
        mesh = np.zeros((1, 10, 3))
        self.assertLess(self.m.verify_mesh(mesh, mesh + 9e-5), 1e-4)
        for delta in [1e-4, 1e-3, np.nan]:
            with self.assertRaises(ValueError):
                self.m.verify_mesh(mesh, mesh + delta)

    def test_multihmr_keeps_zero_and_more_than_two_people(self):
        human = {'j3d': Tensor(self.xyz), 'j2d': Tensor(self.xy), 'scores': Tensor(.75)}
        for count in [0, 2, 3]:
            instances = self.m.multihmr_instances([human] * count, self.names, np.eye(3))
            self.assertEqual(len(instances), count)
            if count:
                self.assertEqual(instances[0]['scores'], [.75] * 12)

    def test_direct_multihmr_call_is_float32_with_fixed_threshold(self):
        model = mock.Mock(return_value=[])
        self.m.multihmr_forward(model, Tensor(np.zeros(1, np.float16)), Tensor(np.eye(3, dtype=np.float16)))
        args, kwargs = model.call_args
        self.assertEqual(args[0].value.dtype, np.float32)
        self.assertEqual(kwargs['K'].value.dtype, np.float32)
        self.assertEqual(kwargs['det_thresh'], .2)
        self.assertEqual(kwargs['nms_kernel_size'], 3)
        self.assertFalse(kwargs['is_training'])

    def test_reconstruction_passes_camera_translation_and_zero_eye_poses(self):
        keys = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose',
                'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans']
        out = {key: Tensor(np.ones((1, 3)) * i) for i, key in enumerate(keys)}
        out['smplx_mesh_cam'] = Tensor(np.zeros((1, 10, 3)))
        body = SimpleNamespace(vertices=Tensor(np.zeros((1, 10, 3))), joints=Tensor(self.xyz[None]))
        layer = mock.Mock(return_value=body)
        torch = SimpleNamespace(no_grad=contextlib.nullcontext, zeros=lambda *s: Tensor(np.zeros(s)))
        joints, error = self.m.reconstruct_smplest(out, layer, torch)
        np.testing.assert_array_equal(joints, self.xyz.astype(np.float32))
        self.assertEqual(error, 0.)
        kwargs = layer.call_args.kwargs
        np.testing.assert_array_equal(kwargs['transl'].numpy(), out['cam_trans'].numpy())
        np.testing.assert_array_equal(kwargs['leye_pose'].numpy(), [[0, 0, 0]])
        np.testing.assert_array_equal(kwargs['right_hand_pose'].numpy(), out['smplx_rhand_pose'].numpy())

    def test_smplest_predict_exports_both_people_and_separate_cameras(self):
        backend = self.m.SmplestBackend.__new__(self.m.SmplestBackend)
        backend.torch = SimpleNamespace(no_grad=contextlib.nullcontext)
        backend.device = 'cpu'
        backend.names = self.names
        backend.mapping = list(range(len(self.names)))
        backend.cfg = SimpleNamespace(model=SimpleNamespace(input_img_shape=[512, 384],
                                      input_body_shape=[256, 192], focal=[5000, 5000], princpt=[96, 128]),
                                      data=SimpleNamespace(bbox_ratio=1.2))
        backend.load_img = lambda _: np.zeros((400, 500, 3), np.float32)
        backend.process_bbox = mock.Mock(side_effect=lambda **kw: kw['bbox'])
        backend.generate_patch_image = mock.Mock(return_value=(np.full((2, 2, 3), 255.), None, None))
        backend.transform = lambda image: Tensor(image.transpose(2, 0, 1))
        backend.model = mock.Mock(side_effect=[{'person': 0}, {'person': 1}])
        backend.layer = object()
        record = dict(self.record, _boxes=self.record['gt_boxes'], _bbox_provenance={})
        with mock.patch.object(self.m, 'reconstruct_smplest', side_effect=[(self.xyz, 0.), (self.xyz + 1, 0.)]):
            result = backend.predict(record)
        self.assertEqual(backend.model.call_count, 2)
        self.assertEqual(len(result['instances']), 2)
        self.assertNotEqual(result['instances'][0]['K'], result['instances'][1]['K'])
        self.assertNotEqual(result['instances'][0]['pelvis'], result['instances'][1]['pelvis'])
        self.assertEqual(backend.process_bbox.call_args.kwargs['ratio'], 1.2)
        self.assertFalse(backend.generate_patch_image.call_args.kwargs['do_flip'])
        for call in backend.model.call_args_list:
            self.assertEqual(call.args[3], 'test')
            self.assertEqual(float(call.args[0]['img'].numpy().max()), 1.)

    def test_smplest_config_paths_and_logs_never_point_to_shared_outputs(self):
        cfg = mock.Mock()
        config = mock.Mock()
        config.load_config.return_value = cfg
        with mock.patch.object(Path, 'is_file', return_value=True), mock.patch.object(Path, 'is_dir', return_value=True):
            self.m.load_smplest_config(self.root, Path(self.request['out_dir']), config)
            first = cfg.update_config.call_args.args[0]
            self.m.load_smplest_config(self.root, Path(self.request['out_dir']), config)
            second = cfg.update_config.call_args.args[0]
        self.assertNotEqual(first['log']['exp_name'], second['log']['exp_name'])
        self.assertTrue(Path(first['log']['log_dir']).is_relative_to(Path(self.request['out_dir']) / 'logs'))
        for key in ['output_dir', 'model_dir', 'result_dir']:
            self.assertIsNone(first['log'][key])
        self.assertTrue(Path(first['model']['human_model_path']).is_absolute())
        self.assertTrue(Path(first['model']['pretrained_model_path']).is_absolute())

    def test_strict_json_uses_exclusive_create_and_serializes_before_open(self):
        fs = MemoryFiles({})
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            with self.assertRaises(ValueError):
                self.m.write_json_xb(self.result_path(), {'x': float('nan')})
            self.assertEqual(fs.writes, [])
            self.m.write_json_xb(self.result_path(), {'x': 1})
            with self.assertRaises(FileExistsError):
                self.m.write_json_xb(self.result_path(), {'x': 2})
        self.assertEqual(json.loads(fs.files[str(self.result_path())]), {'x': 1})
        self.assertEqual(fs.writes[0][1], 'xb')

    def test_gpu_guard_requires_32_gib_without_allocating_or_killing(self):
        cuda = mock.Mock()
        cuda.is_available.return_value = True
        cuda.mem_get_info.return_value = (31 * 1024 ** 3, 85 * 1024 ** 3)
        with self.assertRaisesRegex(RuntimeError, '32'):
            self.m.require_gpu(SimpleNamespace(cuda=cuda))
        cuda.empty_cache.assert_not_called()
        cuda.mem_get_info.return_value = (32 * 1024 ** 3, 85 * 1024 ** 3)
        self.m.require_gpu(SimpleNamespace(cuda=cuda))
        cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError):
            self.m.require_gpu(SimpleNamespace(cuda=cuda))

    def test_timing_warms_then_synchronizes_and_resets_peaks(self):
        backend = self.fake_backend()
        events = []
        backend.predict.side_effect = lambda _: events.append('predict') or {'instances': [], 'provenance': {}}
        backend.torch.cuda.synchronize.side_effect = lambda *_: events.append('sync')
        backend.torch.cuda.reset_peak_memory_stats.side_effect = lambda *_: events.append('reset')
        with mock.patch.object(self.m.time, 'perf_counter', side_effect=[10., 10.25]):
            _, metrics = self.m.timed_prediction(backend, self.record, warmup=True)
        self.assertEqual(events, ['predict', 'sync', 'reset', 'predict', 'sync'])
        self.assertEqual(metrics['latency_ms'], 250.)
        self.assertEqual(metrics['peak_allocated_gib'], 2.)
        self.assertEqual(metrics['peak_reserved_gib'], 3.)

    def test_batch_success_then_same_fingerprint_skips_model(self):
        fs, request_sha = self.fs()
        backend = self.fake_backend()
        factory = mock.Mock(return_value=backend)
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 0)
            saved = fs.files[str(self.result_path())]
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 0)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(backend.predict.call_count, 2)
        self.assertEqual(saved, fs.files[str(self.result_path())])
        result = json.loads(saved)
        self.assertEqual(result['schema'], self.request['schema'])
        self.assertEqual(result['request_sha256'], request_sha)
        self.assertEqual(result['det_count'], 2)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['scene_camera_available'])
        self.assertTrue(result['provenance']['privileged_camera'])

    def test_bad_image_hash_is_recorded_before_backend_and_not_retried(self):
        fs, _ = self.fs()
        fs.files[str(self.image)] = b'changed'
        factory = mock.Mock()
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 1)
            result = json.loads(fs.files[str(self.result_path())])
            self.assertEqual(result['status'], 'error')
            self.assertIn('SHA256', result['error'])
            fs.files[str(self.image)] = self.image_bytes
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 1)
        factory.assert_not_called()
        self.assertEqual(len(fs.writes), 1)

    def test_different_fingerprint_refuses_overwrite_and_backend(self):
        fs, _ = self.fs()
        old = json.dumps({'status': 'ok', 'image_sha256': self.sha, 'request_sha256': 'old'}).encode()
        fs.files[str(self.result_path())] = old
        factory = mock.Mock()
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 1)
        factory.assert_not_called()
        self.assertEqual(fs.files[str(self.result_path())], old)

    def test_existing_ok_still_requires_current_input_hash(self):
        fs, _ = self.fs()
        backend = self.fake_backend()
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=lambda *_: backend), 0)
            saved = fs.files[str(self.result_path())]
            fs.files[str(self.image)] = b'corrupt'
            factory = mock.Mock()
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 1)
        factory.assert_not_called()
        self.assertEqual(fs.files[str(self.result_path())], saved)

    def test_error_frame_does_not_drop_later_frames(self):
        other = dict(self.record, image=str(self.root / 'input' / 'other.jpg'))
        other['image_sha256'] = hashlib.sha256(b'other').hexdigest()
        other['id'] = other['image_sha256'][:16]
        self.request['records'].append(other)
        fs, _ = self.fs()
        fs.files[other['image']] = b'other'
        backend = self.fake_backend()
        backend.predict.side_effect = [RuntimeError('frame failure'), {'instances': [], 'provenance': {}},
                                       {'instances': [], 'provenance': {}}]
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=lambda *_: backend), 1)
        self.assertEqual(json.loads(fs.files[str(self.result_path())])['status'], 'error')
        self.assertEqual(json.loads(fs.files[str(self.result_path(record=other))])['det_count'], 0)

    def test_startup_failure_records_each_frame_without_inference(self):
        fs, _ = self.fs()
        factory = mock.Mock(side_effect=RuntimeError('requires 32 GiB free'))
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=factory), 1)
        result = json.loads(fs.files[str(self.result_path())])
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['det_count'])
        self.assertIsNone(result['latency_ms'])

    def test_nan_model_output_becomes_strict_error_json(self):
        fs, _ = self.fs()
        backend = self.fake_backend()
        backend.predict.return_value['instances'][0]['xy'][0][0] = float('nan')
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', backend_factory=lambda *_: backend), 1)
        raw = fs.files[str(self.result_path())]
        self.assertNotIn(b'NaN', raw)
        self.assertEqual(json.loads(raw)['status'], 'error')
        self.assertEqual(json.loads(raw)['det_count'], 2)

    def test_multihmr_backend_uses_true_k_or_default_fov60(self):
        from PIL import Image
        for mode in ['multihmr', 'multihmr_fov60']:
            backend = self.m.MultiHMRBackend.__new__(self.m.MultiHMRBackend)
            backend.torch = SimpleNamespace(no_grad=contextlib.nullcontext, device=str, float32=np.float32,
                                           as_tensor=lambda x, **kw: Tensor(np.asarray(x, dtype=kw['dtype'])))
            backend.mode, backend.device, backend.size = mode, 'cpu', 896
            backend.names = self.names
            backend.open_image = mock.Mock(return_value=(Tensor(np.zeros((1, 3, 896, 896))), None))
            default_k = np.array([[777., 0, 448], [0, 777, 448], [0, 0, 1]])
            backend.get_camera_parameters = mock.Mock(return_value=Tensor(default_k[None]))
            affine = self.m.affine_for_resize((1001, 667), (896, 597), (896, 896))
            human = {'j3d': Tensor(self.xyz), 'j2d': Tensor(self.m.apply_affine(self.xy, affine)),
                     'scores': Tensor(.9)}
            backend.model = mock.Mock(return_value=[human, human, human])
            with mock.patch.object(Image, 'open', return_value=Image.new('RGB', (1001, 667))):
                result = backend.predict(self.record)
            actual_k = backend.model.call_args.kwargs['K'].numpy()[0]
            expected_k = affine @ self.record['K'] if mode == 'multihmr' else default_k
            np.testing.assert_allclose(actual_k, expected_k, rtol=1e-6)
            np.testing.assert_allclose(result['instances'][0]['xy'],
                                       self.xy[[self.names.index(n) for n in BODY]], atol=1e-4)
            self.assertEqual(len(result['instances']), 3)
            if mode == 'multihmr':
                backend.get_camera_parameters.assert_not_called()
            else:
                backend.get_camera_parameters.assert_called_once_with(896, fov=60, device='cpu')

    def test_smplest_empty_boxes_never_call_person_model(self):
        backend = self.m.SmplestBackend.__new__(self.m.SmplestBackend)
        backend.model = mock.Mock()
        result = backend.predict(dict(self.record, _boxes=[], _bbox_provenance={}))
        self.assertEqual(result['instances'], [])
        backend.model.assert_not_called()

    def test_second_person_failure_keeps_detection_count_and_is_not_partial_ok(self):
        fs, _ = self.fs()
        backend = self.fake_backend()
        backend.predict.side_effect = self.m.FrameError('second person reconstruction failed', 2)
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'smplest_gtbox',
                                             backend_factory=lambda *_: backend), 1)
        result = json.loads(fs.files[str(self.result_path('smplest_gtbox'))])
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['det_count'], 2)
        self.assertEqual(result['instances'], [])
        self.assertFalse(result['scene_camera_available'])
        self.assertEqual(result['provenance']['camera_basis'], 'per_person_crop_camera')

    def test_yolox_missing_stale_and_zero_detections(self):
        fs, request_sha = self.fs()
        path = self.result_path('dwpose_auto')
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            with self.assertRaises(FileNotFoundError):
                self.m.resolve_boxes(self.request, self.record, 'smplest_yolox', request_sha)
            detection = {'id': self.record['id'], 'mode': 'dwpose_auto', 'status': 'ok',
                         'image_sha256': self.sha, 'request_sha256': request_sha, 'det_count': 0, 'instances': []}
            fs.files[str(path)] = json.dumps(detection).encode()
            boxes, source = self.m.resolve_boxes(self.request, self.record, 'smplest_yolox', request_sha)
            self.assertEqual(boxes, [])
            self.assertEqual(source['bbox_source'], 'dwpose_auto')
            detection['image_sha256'] = 'stale'
            fs.files[str(path)] = json.dumps(detection).encode()
            with self.assertRaises(ValueError):
                self.m.resolve_boxes(self.request, self.record, 'smplest_yolox', request_sha)

    def test_gtbox_requires_two_ordered_xyxy_boxes(self):
        boxes, _ = self.m.resolve_boxes(self.request, self.record, 'smplest_gtbox', 'sha')
        self.assertEqual(boxes, self.record['gt_boxes'])
        for boxes in [[], [[10, 20, 5, 200], [20, 30, 100, 150]]]:
            record = dict(self.record, gt_boxes=boxes)
            with self.assertRaises(ValueError):
                self.m.resolve_boxes(self.request, record, 'smplest_gtbox', 'sha')

    def test_limit_zero_does_not_load_backend(self):
        fs, _ = self.fs()
        factory = mock.Mock()
        with contextlib.ExitStack() as stack:
            fs.install(stack)
            self.assertEqual(self.m.run_batch(self.request_path, 'multihmr', limit=0, backend_factory=factory), 0)
        factory.assert_not_called()
        self.assertEqual(fs.writes, [])

    def test_request_validation_rejects_duplicate_ids_and_wrong_joint_order(self):
        for change in [{'records': [self.record, self.record]}, {'body_joint_names': BODY[::-1]},
                       {'schema': 'wrong'}, {'records': [dict(self.record, id='../unsafe')]}]:
            request = dict(self.request, **change)
            with self.assertRaises(ValueError):
                self.m.validate_request(request)


if __name__ == '__main__':
    unittest.main()

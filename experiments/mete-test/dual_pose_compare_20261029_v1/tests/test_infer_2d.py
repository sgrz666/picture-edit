import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "infer_2d.py"
META = Path(os.environ.get(
    "SAPIENS_META_PATH",
    "/data/shangguanrz/models/meta/sapiens2-code/sapiens/pose/configs/_base_/keypoints308.py",
))
NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SAPIENS_IDS = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
BOXES = [[1, 2, 20, 40], [10, 5, 30, 50]]
infer = None
if SCRIPT.is_file():
    spec = importlib.util.spec_from_file_location("infer_2d", SCRIPT)
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)


class MemoryFiles:
    def __init__(self, files):
        self.files = {str(Path(k)): v for k, v in files.items()}

    def read(self, path):
        try:
            return self.files[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path)) from None

    def open(self, path, mode):
        assert mode == "xb", mode
        key = str(path)
        if key in self.files:
            raise FileExistsError(key)
        self.files[key] = b""
        files = self.files

        class Writer(io.BytesIO):
            def fileno(self):
                return 123

            def close(self):
                files[key] = self.getvalue()
                super().close()

        return Writer()

    def __enter__(self):
        self.patches = [
            patch.object(Path, "read_bytes", lambda p: self.read(p)),
            patch.object(Path, "exists", lambda p: str(p) in self.files),
            patch.object(Path, "mkdir"),
            patch.object(Path, "open", lambda p, mode: self.open(p, mode)),
            patch("os.fsync"),
        ]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *args):
        for item in reversed(self.patches):
            item.stop()


class InferTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(infer, "infer_2d.py implementation is missing")

    def models(self, boxes=BOXES):
        return SimpleNamespace(
            torch=None, device="cpu", keypoint_count=133,
            indices=list(range(5, 17)), provenance={"timing": "CPU test"},
            detect=Mock(return_value=(boxes, {"source": "test"})),
            pose=Mock(return_value=(np.zeros((len(boxes), 133, 2)),
                                    np.ones((len(boxes), 133)))),
        )

    def request_files(self):
        root = SCRIPT.parents[1]
        request = {
            "schema": "dual_pose_benchmark_v1", "project_root": str(root),
            "out_dir": str(root / "output"), "body_joint_names": NAMES,
            "records": [],
        }
        files = {}
        for index in range(3):
            raw = f"frame-{index}".encode()
            digest = hashlib.sha256(raw).hexdigest()
            image = str(root / f"frame-{index}.jpg")
            files[image] = raw
            request["records"].append({
                "id": digest[:16], "image": image, "image_sha256": digest,
                "category": "interaction", "pair": "pair00", "action": "hug",
                "camera_id": 1, "K": np.eye(3).tolist(), "gt_boxes": BOXES,
            })
        path = root / "request.json"
        files[str(path)] = json.dumps(request).encode()
        return request, path, files

    def batch(self, files, request_path, model, mode="dwpose_auto", limit=None):
        cv2 = SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda *a: np.zeros((8, 8, 3)))
        factory = Mock(return_value=model)
        with MemoryFiles(files) as memory, patch.dict(sys.modules, {"cv2": cv2}), \
                patch.object(infer, "Models", factory), patch("sys.stderr", io.StringIO()):
            exit_code = infer.run(request_path, mode, limit)
        return exit_code, factory, memory.files

    def test_import_does_not_load_gpu_libraries(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("onnxruntime", sys.modules)

    def test_dwpose_sessions_use_cpu_and_at_most_four_threads(self):
        request, _, _ = self.request_files()
        ort = SimpleNamespace(SessionOptions=Mock(side_effect=SimpleNamespace),
                              InferenceSession=Mock(),
                              ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"))
        module = SimpleNamespace(inference_detector=Mock(return_value=BOXES),
                                 inference_pose=Mock())
        with patch.dict(sys.modules, {"onnxruntime": ort}), \
                patch.object(infer, "load_module", return_value=module):
            model = infer.Models(request, "dwpose_auto", "b" * 64)
            self.assertEqual(model.detect({}, object())[0].tolist(), BOXES)
            self.assertEqual(ort.InferenceSession.call_count, 2)
            for call in ort.InferenceSession.call_args_list:
                self.assertEqual(call.kwargs["providers"], ["CPUExecutionProvider"])
                options = call.kwargs["sess_options"]
                self.assertLessEqual(options.intra_op_num_threads, 4)
                self.assertEqual(options.inter_op_num_threads, 1)
            ort.InferenceSession.reset_mock()
            model = infer.Models(request, "dwpose_gtbox", "b" * 64)
            model.detect({"gt_boxes": BOXES}, object())
            self.assertEqual(ort.InferenceSession.call_count, 1)

    def test_sapiens_gtbox_passes_both_people_in_one_batch(self):
        from contextlib import nullcontext

        request, _, _ = self.request_files()
        names = {i: {"name": f"joint_{i}"} for i in range(308)}
        names.update({i: {"name": n} for i, n in zip(SAPIENS_IDS, NAMES)})
        meta = {"keypoint_info": names}
        prediction = Mock()
        prediction.cpu.return_value.numpy.return_value = np.zeros((2, 308, 2, 2))
        model = Mock(return_value=prediction)
        model.cfg.codec = {"type": "UDPHeatmap", "input_size": (768, 1024)}
        model.pipeline.side_effect = lambda info: {
            "inputs": np.zeros((1, 3, 2, 2)),
            "data_samples": {"meta": {"input_size": np.array([2, 2]),
                                       "bbox_center": np.array([[10, 20]]),
                                       "bbox_scale": np.array([[2, 2]])}},
        }
        model.data_preprocessor.side_effect = lambda data: data
        codec = Mock()
        codec.decode.return_value = (np.ones((1, 308, 2)), np.ones((1, 308)))
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.mem_get_info.return_value = (10 * 1024 ** 3, 20 * 1024 ** 3)
        torch = SimpleNamespace(cuda=cuda, no_grad=nullcontext,
                                cat=Mock(side_effect=lambda arrays, dim: np.concatenate(arrays, axis=dim)))
        init = Mock(return_value=model)
        modules = {
            "torch": torch,
            "sapiens.pose.models": SimpleNamespace(init_model=init),
            "sapiens.pose.datasets": SimpleNamespace(parse_pose_metainfo=Mock(return_value=meta),
                                                     UDPHeatmap=Mock(return_value=codec)),
        }
        with patch.dict(sys.modules, modules), patch.object(infer, "load_sapiens_metadata", return_value=meta):
            backend = infer.Models(request, "sapiens_gtbox", "b" * 64)
            xy, scores = backend.pose(object(), np.array(BOXES))
        init.assert_called_once()
        self.assertEqual(model.call_count, 1)
        self.assertEqual(model.call_args.args[0].shape[0], 2)
        self.assertEqual(np.asarray(xy).shape, (2, 308, 2))
        self.assertEqual(np.asarray(scores).shape, (2, 308))
        self.assertEqual(np.asarray(xy)[0, 0].tolist(), [10, 20])
        self.assertEqual(backend.provenance["gpu_free_before_load_gib"], 10)
        self.assertEqual(model.cfg.codec["type"], "UDPHeatmap")

    @unittest.skipUnless(META.is_file(), "official table available on sg only")
    def test_real_official_dataset_has_wrist_62_41_and_rejects_coco(self):
        tables = runpy.run_path(str(META))
        self.assertEqual(infer.load_sapiens_metadata(META), tables["dataset_info"])
        self.assertEqual(infer.sapiens_body_indices(tables["dataset_info"]), SAPIENS_IDS)
        parsed = {"keypoint_id2name": {
            i: item["name"] for i, item in tables["dataset_info"]["keypoint_info"].items()
        }}
        self.assertEqual(infer.sapiens_body_indices(parsed), SAPIENS_IDS)
        with self.assertRaises(ValueError):
            infer.sapiens_body_indices(tables["coco_wholebody_info"])

    def test_padded_coco_mapping_is_rejected(self):
        names = {i: f"joint_{i}" for i in range(308)}
        names.update(dict(zip(range(5, 17), NAMES)))
        with self.assertRaises(ValueError):
            infer.sapiens_body_indices({"keypoint_id2name": names})

    def test_normalizes_dwpose_and_sapiens_shapes_without_dropping_raw_points(self):
        for count, indices in [(133, list(range(5, 17))), (308, SAPIENS_IDS)]:
            xy = np.arange(2 * count * 2).reshape(2, count, 2)
            scores = np.full((2, count), 0.75)
            for points, conf in [(xy, scores), (xy[:, None], scores[:, None]),
                                 (xy, scores[..., None])]:
                rows = infer.normalise_instances(BOXES, points, conf, indices, count)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["joint_names"], NAMES)
                self.assertEqual(rows[1]["xy"], xy[1, indices].tolist())
                self.assertEqual(rows[0]["scores"], [0.75] * 12)
                self.assertEqual(rows[0]["extra_keypoints"], xy[0].tolist())
                self.assertEqual(rows[1]["extra_scores"], scores[1].tolist())
                json.dumps(rows, allow_nan=False)
            one = infer.normalise_instances(BOXES[0], xy[0], scores[0], indices, count)
            self.assertEqual(one[0]["bbox"], BOXES[0])

    def test_bad_shapes_and_nan_are_errors(self):
        for xy in [np.zeros((1, 133, 2)), np.zeros((2, 132, 2)),
                   np.full((2, 133, 2), np.nan)]:
            with self.assertRaises(ValueError):
                infer.normalise_instances(BOXES, xy, np.ones((2, 133)), list(range(5, 17)), 133)

    def test_zero_boxes_never_call_pose(self):
        model = self.models([])
        result = infer.infer_frame(model, {}, object())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["det_count"], 0)
        self.assertEqual(result["instances"], [])
        self.assertIsNone(result["peak_allocated_gib"])
        self.assertIsNone(result["peak_reserved_gib"])
        model.pose.assert_not_called()

    def test_pose_error_keeps_actual_detection_count(self):
        model = self.models()
        model.pose.side_effect = RuntimeError("pose failed")
        result = infer.infer_frame(model, {}, object())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["det_count"], 2)
        self.assertIn("pose failed", result["error"])
        self.assertEqual(result["instances"], [])

    def test_cuda_timing_and_peaks_are_per_frame(self):
        model = self.models()
        model.torch = Mock()
        cuda = model.torch.cuda
        cuda.max_memory_allocated.return_value = 2 * 1024 ** 3
        cuda.max_memory_reserved.return_value = 3 * 1024 ** 3
        model.device = "cuda:0"
        result = infer.infer_frame(model, {}, object())
        self.assertEqual(result["peak_allocated_gib"], 2)
        self.assertEqual(result["peak_reserved_gib"], 3)
        cuda.reset_peak_memory_stats.assert_called_once_with("cuda:0")
        self.assertEqual(cuda.synchronize.call_count, 2)
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_detr_records_raw_and_post_nms_070_without_fallback(self):
        nms = Mock(return_value=[1])
        boxes, audit = infer.postprocess_detr(BOXES, [0.6, 0.9], nms)
        self.assertEqual(boxes.tolist(), [BOXES[1]])
        self.assertEqual(nms.call_args.args[1], 0.7)
        self.assertEqual(audit["bbox_thr"], 0.3)
        self.assertEqual(audit["raw_boxes"], BOXES)
        self.assertEqual(audit["post_nms_boxes"], [BOXES[1]])
        self.assertEqual(audit["raw_scores"], [0.6, 0.9])
        nms.reset_mock()
        boxes, audit = infer.postprocess_detr([], [], nms)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(audit["post_nms_boxes"], [])
        nms.assert_not_called()

    def test_exclusive_write_and_resume_reject_different_fingerprint_or_error(self):
        record = {"id": "a" * 16, "image_sha256": "a" * 64}
        expected = infer.fingerprint(record, "dwpose_auto", "b" * 64)
        path = SCRIPT.parent / "virtual.json"
        row = {**expected, "status": "ok"}
        with MemoryFiles({}) as memory:
            self.assertFalse(infer.should_skip(path, expected))
            infer.write_json_exclusive(path, row)
            saved = memory.read(path)
            self.assertTrue(infer.should_skip(path, expected))
            with self.assertRaises(FileExistsError):
                infer.write_json_exclusive(path, row)
            for key in expected:
                with self.assertRaises(FileExistsError):
                    infer.should_skip(path, {**expected, key: "different"})
            self.assertEqual(saved, memory.read(path))
            memory.files[str(path)] = json.dumps({**row, "status": "error"}).encode()
            with self.assertRaises(FileExistsError):
                infer.should_skip(path, expected)
            memory.files[str(path)] = b"{partial"
            with self.assertRaises(FileExistsError):
                infer.should_skip(path, expected)

    def test_strict_json_is_validated_before_file_creation(self):
        path = SCRIPT.parent / "virtual.json"
        with MemoryFiles({}) as memory:
            with self.assertRaises(ValueError):
                infer.write_json_exclusive(path, {"value": float("nan")})
            self.assertNotIn(str(path), memory.files)

    def test_cached_yolox_uses_every_bbox_and_refuses_missing_or_stale_file(self):
        record = {"id": "a" * 16, "image_sha256": "a" * 64}
        out = SCRIPT.parent / "virtual_output"
        path = out / "dwpose_auto" / (record["id"] + ".json")
        row = {**infer.fingerprint(record, "dwpose_auto", "b" * 64),
               "status": "ok", "det_count": 2,
               "instances": [{"bbox": b} for b in BOXES]}
        with MemoryFiles({}) as memory:
            with self.assertRaises(FileNotFoundError):
                infer.cached_yolox(out, record, "b" * 64)
            memory.files[str(path)] = json.dumps(row).encode()
            boxes, audit = infer.cached_yolox(out, record, "b" * 64)
            self.assertEqual(boxes.tolist(), BOXES)
            self.assertEqual(audit["source_sha256"], hashlib.sha256(memory.read(path)).hexdigest())
            with self.assertRaises(ValueError):
                infer.cached_yolox(out, record, "c" * 64)
            memory.files[str(path)] = json.dumps({**row, "det_count": 0, "instances": []}).encode()
            self.assertEqual(infer.cached_yolox(out, record, "b" * 64)[0].shape, (0, 4))

    def test_batch_loads_once_limit_and_resume_load_nothing(self):
        request, path, files = self.request_files()
        code, factory, saved = self.batch(files, path, self.models(), limit=2)
        self.assertEqual(code, 0)
        factory.assert_called_once()
        for rec in request["records"][:2]:
            target = Path(request["out_dir"]) / "dwpose_auto" / (rec["id"] + ".json")
            row = json.loads(saved[str(target)])
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["det_count"], 2)
            self.assertEqual(row["request_sha256"], hashlib.sha256(files[str(path)]).hexdigest())
            self.assertIn("provenance", row)
        code, factory, resumed = self.batch(saved, path, self.models(), limit=2)
        self.assertEqual(code, 0)
        factory.assert_not_called()
        self.assertEqual(saved, resumed)

    def test_image_hash_failure_is_recorded_and_batch_exits_nonzero(self):
        request, path, files = self.request_files()
        rec = request["records"][0]
        files[rec["image"]] = b"changed"
        model = self.models()
        code, factory, saved = self.batch(files, path, model)
        self.assertEqual(code, 1)
        self.assertEqual(model.pose.call_count, 2)
        factory.assert_called_once()
        target = Path(request["out_dir"]) / "dwpose_auto" / (rec["id"] + ".json")
        row = json.loads(saved[str(target)])
        self.assertEqual(row["status"], "error")
        self.assertIn("sha256", row["error"])

    def test_existing_conflict_is_not_overwritten(self):
        request, path, files = self.request_files()
        rec = request["records"][0]
        target = str(Path(request["out_dir"]) / "dwpose_auto" / (rec["id"] + ".json"))
        files[target] = b'{"status":"ok","request_sha256":"wrong"}'
        code, factory, saved = self.batch(files, path, self.models(), limit=1)
        self.assertEqual(code, 1)
        self.assertEqual(saved[target], files[target])
        factory.assert_not_called()

    def test_failed_model_load_is_not_retried_for_each_frame(self):
        request, path, files = self.request_files()
        cv2 = SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda *a: np.zeros((8, 8, 3)))
        with MemoryFiles(files) as memory, patch.dict(sys.modules, {"cv2": cv2}), \
                patch.object(infer, "Models", side_effect=RuntimeError("load failed")) as factory, \
                patch("sys.stderr", io.StringIO()):
            self.assertEqual(infer.run(path, "dwpose_auto"), 1)
        factory.assert_called_once()
        for rec in request["records"]:
            target = str(Path(request["out_dir"]) / "dwpose_auto" / (rec["id"] + ".json"))
            self.assertIn("load failed", json.loads(memory.files[target])["error"])

    def test_gtbox_is_two_person_and_request_joint_order_is_fixed(self):
        request, _, _ = self.request_files()
        infer.validate_request(request)
        request["body_joint_names"] = list(reversed(NAMES))
        with self.assertRaises(ValueError):
            infer.validate_request(request)
        request["body_joint_names"] = NAMES
        request["records"][0]["gt_boxes"] = BOXES[:1]
        with self.assertRaises(ValueError):
            infer.validate_request(request)


if __name__ == "__main__":
    unittest.main(verbosity=2)

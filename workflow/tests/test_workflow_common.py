import importlib
import os
import tempfile
import unittest
from pathlib import Path


class StageCacheTests(unittest.TestCase):
    def setUp(self):
        try:
            self.common = importlib.import_module('pose_transfer.common')
        except ModuleNotFoundError:
            self.fail('workflow cache implementation is missing')

    def record(self, root, outputs, request):
        run_dir = root / 'run'
        stage_dir = run_dir / 'generation'
        stage_dir.mkdir(parents=True, exist_ok=True)
        files = {name: stage_dir / name for name in outputs}
        for name, path in files.items():
            path.write_bytes(outputs[name])
        request = self.common.make_request('generate', request, run_dir, files)
        self.common.finish_stage(stage_dir / 'stage.json', request, files,
            {'seed': 42}, run_dir)
        return run_dir, stage_dir / 'stage.json', request

    def test_reuse_checks_request_and_actual_artifact_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, record, request = self.record(root, {'final_image.png': b'first-result'},
                {'settings': {'steps': 30}})
            self.assertTrue(self.common.stage_valid(record, request, run_dir))
            self.assertFalse(self.common.stage_valid(record, self.common.make_request('generate',
                {'settings': {'steps': 40}}, run_dir, {}), run_dir))
            (run_dir / 'generation' / 'final_image.png').write_bytes(b'changed-result')
            self.assertFalse(self.common.stage_valid(record, request, run_dir))

    def test_reasons_name_the_artifact_that_was_edited_after_the_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, record, request = self.record(root,
                {'generation_prompt.txt': b'from prompt engineering'},
                {'settings': {'seed': 42}})
            (run_dir / 'generation' / 'generation_prompt.txt').write_bytes(b'hand rewritten')
            reasons = self.common.stage_reasons(record, request, run_dir)
            self.assertEqual(len(reasons), 1)
            self.assertIn('generation_prompt.txt', reasons[0])
            self.assertIn('content changed after the stage wrote it', reasons[0])

    def test_reasons_name_the_identity_field_that_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, record, request = self.record(root, {'final_image.png': b'x'},
                {'settings': {'seed': 42}, 'code': {'run.py': 'aaa'}})
            moved = self.common.make_request('generate',
                {'settings': {'seed': 43}, 'code': {'run.py': 'bbb'}}, run_dir, {})
            reasons = self.common.stage_reasons(record, moved, run_dir)
            joined = '\n'.join(reasons)
            self.assertIn('settings.seed changed', joined)
            self.assertIn('workflow code changed: run.py', joined)

    def test_record_is_relocatable_because_paths_are_not_fingerprinted(self):
        """Renaming a run or moving the checkout must not invalidate verified content."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, record, request = self.record(root, {'final_image.png': b'bytes'},
                {'settings': {'seed': 42}})
            self.common.atomic_json(record, {**self.common.read_json(record),
                'outputs': {'final_image.png': dict(self.common.read_json(record)['outputs']
                    ['final_image.png'], path='now/stale/absolute/path.png')}})
            renamed = root / 'renamed_run'
            run_dir.rename(renamed)
            request_at_new_path = self.common.make_request('generate',
                self.common.request_identity(request), renamed, {})
            self.assertTrue(self.common.stage_valid(renamed / 'generation/stage.json',
                request_at_new_path, renamed))

    def test_input_content_change_alters_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'A.png'
            source.write_bytes(b'first')
            first = self.common.fingerprint({'A': self.common.sha256_file(source)})
            source.write_bytes(b'other')
            second = self.common.fingerprint({'A': self.common.sha256_file(source)})
            self.assertNotEqual(first, second)

    def test_canvas_and_letterbox_share_one_coordinate_transform(self):
        from PIL import Image
        size = self.common.canvas_size((1280, 2196), 1024)
        self.assertEqual(size, (768, 1344))
        image, meta = self.common.letterbox(Image.new('RGB', (1280, 2196)), size)
        self.assertEqual(image.size, size)
        x0, y0, x1, y1 = meta['content_box']
        self.assertEqual((x1 - x0, y1 - y0), tuple(meta['resized_size']))
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 1280 / 2196, places=3)

    def test_rounded_canvas_is_stable_under_pipeline_condition_resizing(self):
        import math
        for original in ((760, 1280), (514, 1280), (1280, 760), (1280, 514)):
            with self.subTest(original=original):
                w, h = self.common.canvas_size(original, 1024)
                ratio = w / h
                expected = (round(math.sqrt(1024 ** 2 * ratio) / 32) * 32,
                    round(math.sqrt(1024 ** 2 / ratio) / 32) * 32)
                self.assertEqual((w, h), expected)

    @unittest.skipUnless(os.name == 'posix', 'Estimator runs on the Linux GPU host')
    def test_estimator_root_is_local_but_weights_are_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, sandbox = root / 'source', root / 'run' / 'estimator'
            (source / 'main').mkdir(parents=True)
            (source / 'scripts').mkdir()
            (source / 'main/inference.py').write_text('original inference')
            (source / 'main/config.py').write_text('original config')
            (source / 'scripts/run_smplestx_params_only.py').write_text('original wrapper')
            for name in ('models', 'utils', 'human_models', 'datasets', 'configs', 'pretrained_models'):
                (source / name).mkdir()
            self.common.create_estimator_sandbox(source, sandbox)
            self.assertFalse((sandbox / 'main/inference.py').is_symlink())
            self.assertEqual((sandbox / 'main/inference.py').resolve().parent.parent, sandbox)
            self.assertEqual((sandbox / 'scripts/run_smplestx_params_only.py').resolve().parents[1], sandbox)
            self.assertEqual((sandbox / 'pretrained_models').resolve(), source / 'pretrained_models')
            for name in ('demo', 'outputs'):
                self.assertFalse((sandbox / name).is_symlink())
                self.assertFalse((source / name).exists())


if __name__ == '__main__':
    unittest.main()

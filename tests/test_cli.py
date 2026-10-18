import importlib
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.name == 'posix', 'Workflow locking runs on Linux')
class InputFailureTests(unittest.TestCase):
    def test_corrupt_source_updates_existing_run_and_latest_to_failed(self):
        from PIL import Image, UnidentifiedImageError
        run = importlib.import_module('run')
        common = importlib.import_module('pose_transfer.common')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'A.jpg', root / 'B.png'
            a.write_bytes(b'not a valid JPEG')
            Image.new('RGB', (100, 150)).save(b)
            config = run.ROOT / 'config.json'
            (root / 'pose_transfer').mkdir()
            for name in ('pose_transfer/common.py', 'run.py'):
                shutil.copyfile(run.ROOT / name, root / name)
            common.atomic_json(root / 'runs/demo/run.json', {'status': 'complete'})
            common.atomic_json(root / 'latest_run.json', {'status': 'complete'})
            argv = ['run.py', '--config', str(config), '--a', str(a), '--b', str(b)]
            with patch.object(run, 'ROOT', root), patch.object(sys, 'argv', argv):
                with self.assertRaises(UnidentifiedImageError):
                    run.main()
            manifest = common.read_json(root / 'runs/demo/run.json')
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['active_stage'], 'inputs')
            self.assertEqual(common.read_json(root / 'latest_run.json')['status'], 'failed')

    def test_verify_failure_does_not_overwrite_existing_run_or_latest(self):
        from PIL import Image
        run = importlib.import_module('run')
        common = importlib.import_module('pose_transfer.common')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'A.jpg', root / 'B.png'
            a.write_bytes(b'not a valid JPEG')
            Image.new('RGB', (100, 150)).save(b)
            config = run.ROOT / 'config.json'
            (root / 'pose_transfer').mkdir()
            for name in ('pose_transfer/common.py', 'run.py'):
                shutil.copyfile(run.ROOT / name, root / name)
            manifest_path, latest_path = root / 'runs/demo/run.json', root / 'latest_run.json'
            common.atomic_json(manifest_path, {'status': 'complete', 'finished_at_unix': 123})
            common.atomic_json(latest_path, {'status': 'complete', 'finished_at_unix': 123})
            before = manifest_path.read_bytes(), latest_path.read_bytes()
            argv = ['run.py', '--config', str(config), '--a', str(a), '--b', str(b), '--verify']
            with patch.object(run, 'ROOT', root), patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(RuntimeError, 'Prepared inputs'):
                    run.main()
            self.assertEqual(before, (manifest_path.read_bytes(), latest_path.read_bytes()))


class PromptSelectionTests(unittest.TestCase):
    """A hand-written prompt is an explicit input; PE's own output must survive it."""

    def setUp(self):
        self.run = importlib.import_module('run')
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / 'prompt').mkdir(parents=True)
        self.compiled = self.root / 'prompt' / 'pe_compiled_prompt.txt'
        self.compiled.write_text('Edit the person in <image1> following <image2>.\n',
            encoding='utf-8')

    def tearDown(self):
        self.directory.cleanup()

    def select(self, manual=None, verify=False):
        return self.run.select_generation_prompt(self.root,
            {'pe_compiled_prompt': str(self.compiled)}, manual, verify)

    def test_pe_output_is_the_default_generation_prompt(self):
        paths = self.select()
        self.assertEqual(Path(paths['generation_prompt']).read_text(encoding='utf-8'),
            self.compiled.read_text(encoding='utf-8'))
        self.assertEqual(json.loads(
            Path(paths['generation_prompt_provenance']).read_text(encoding='utf-8'))['source'],
            'pe_compiled')

    def test_manual_prompt_replaces_the_selection_without_touching_pe_output(self):
        manual = self.root / 'hand_written.txt'
        manual.write_text('Pose-only edit of <image1> using the joints in <image2>.\n',
            encoding='utf-8')
        before = self.compiled.read_bytes()
        paths = self.select(manual)
        self.assertEqual(Path(paths['generation_prompt']).read_text(encoding='utf-8'),
            manual.read_text(encoding='utf-8'))
        self.assertEqual(self.compiled.read_bytes(), before)
        provenance = json.loads(
            Path(paths['generation_prompt_provenance']).read_text(encoding='utf-8'))
        self.assertEqual(provenance['source'], 'manual')
        self.assertEqual(provenance['source_path'], str(manual))

    def test_switching_back_to_pe_output_is_visible_in_the_provenance(self):
        manual = self.root / 'hand_written.txt'
        manual.write_text('Manual <image1> <image2>.\n', encoding='utf-8')
        self.select(manual)
        paths = self.select()
        self.assertEqual(Path(paths['generation_prompt']).read_text(encoding='utf-8'),
            self.compiled.read_text(encoding='utf-8'))
        self.assertEqual(json.loads(
            Path(paths['generation_prompt_provenance']).read_text(encoding='utf-8'))['source'],
            'pe_compiled')

    def test_prompt_naming_an_image_the_generator_never_sees_is_rejected(self):
        manual = self.root / 'bad.txt'
        manual.write_text('Follow the skeleton in <image3>.\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.select(manual)

    def test_empty_prompt_is_rejected(self):
        manual = self.root / 'empty.txt'
        manual.write_text('   \n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.select(manual)

    def test_verify_never_writes_a_missing_selection(self):
        with self.assertRaisesRegex(RuntimeError, 'does not hold the selected prompt'):
            self.select(verify=True)


if __name__ == '__main__':
    unittest.main()

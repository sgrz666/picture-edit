import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import core


class FullTrainingContracts(unittest.TestCase):
    def test_full_training_reads_all_pairs_instead_of_pilot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'pair_id': str(i), 'split': 'train'} for i in range(5)]
            core.write_jsonl(root / 'train.jsonl', rows)
            core.write_jsonl(root / 'pilot_train.jsonl', rows[:2])
            selected = core.load_training_rows({'training_manifest': 'train.jsonl', 'expected_train_pairs': 5}, root)
            self.assertEqual(selected, rows)

    def test_pilot_default_stays_backward_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            core.write_jsonl(Path(temp) / 'pilot_train.jsonl', [{'pair_id': 'a', 'split': 'train'}])
            self.assertEqual(core.load_training_rows({}, temp)[0]['pair_id'], 'a')

    def test_train_rejects_validation_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            core.write_jsonl(Path(temp) / 'train.jsonl', [{'pair_id': 'a', 'split': 'val'}])
            with self.assertRaisesRegex(ValueError, 'train'):
                core.load_training_rows({'training_manifest': 'train.jsonl'}, temp)

    def test_full_training_rejects_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            core.write_jsonl(Path(temp) / 'train.jsonl', [{'pair_id': 'a', 'split': 'train'}])
            with self.assertRaisesRegex(ValueError, 'expected'):
                core.load_training_rows({'training_manifest': 'train.jsonl', 'expected_train_pairs': 9520}, temp)

    def test_full_cache_includes_full_train_and_excludes_test(self):
        config = {'training_manifest': 'train.jsonl',
                  'cache_manifests': ['train.jsonl', 'pilot_val.jsonl', 'diagnostics.jsonl']}
        self.assertEqual(core.cache_manifest_names(config)[0], 'train.jsonl')
        with self.assertRaisesRegex(ValueError, 'test'):
            core.cache_manifest_names({**config, 'cache_manifests': ['train.jsonl', 'test.jsonl']})

    def test_one_epoch_visits_every_sample_once_and_is_reproducible(self):
        order = core.training_permutation(9520, 42, 0)
        self.assertEqual(len(set(order)), 9520)
        self.assertEqual(sorted(order), list(range(9520)))
        self.assertEqual(order, core.training_permutation(9520, 42, 0))
        self.assertNotEqual(order, core.training_permutation(9520, 42, 1))


if __name__ == '__main__':
    unittest.main()

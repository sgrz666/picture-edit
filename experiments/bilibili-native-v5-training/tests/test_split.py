"""Dataset partition contracts; no model, CUDA or training is involved."""
import importlib.util
from pathlib import Path
import unittest
import tempfile


class SplitContracts(unittest.TestCase):
    def load(self):
        path = Path(__file__).parents[1] / "scripts/split_dataset.py"
        self.assertTrue(path.is_file(), "The reproducible split tool is not implemented")
        spec = importlib.util.spec_from_file_location("bilibili_split", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_video_partition_is_stable_and_has_39_5_5_clips(self):
        module = self.load()
        clips = ["BV%03d" % i for i in range(49)]
        mapping = module.assign_clips(clips, 42)
        self.assertEqual(mapping, module.assign_clips(reversed(clips), 42))
        self.assertEqual([list(mapping.values()).count(s) for s in module.SPLITS], [39, 5, 5])

    def test_rows_are_all_preserved_and_pair_endpoints_follow_the_video(self):
        module = self.load()
        frames = [{"frame_id": "a", "clip_id": "A", "loss_mask": {"hand_a": False}},
                  {"frame_id": "b", "clip_id": "A"}]
        pairs = [{"pair_id": "p", "clip_id": "A", "source_frame_id": "a", "target_frame_id": "b"}]
        f, p = module.partition(frames, pairs, {"A": "train"})
        self.assertEqual(sum(map(len, f.values())), 2)
        self.assertEqual(sum(map(len, p.values())), 1)
        self.assertEqual(f["train"][0]["loss_mask"], frames[0]["loss_mask"])
        self.assertNotIn("split", frames[0])
        self.assertEqual(p["train"][0]["split"], "train")

    def test_cross_video_pair_and_duplicate_ids_are_rejected(self):
        module = self.load()
        frames = [{"frame_id": "a", "clip_id": "A"}, {"frame_id": "b", "clip_id": "B"}]
        pair = {"pair_id": "p", "clip_id": "A", "source_frame_id": "a", "target_frame_id": "b"}
        with self.assertRaises(ValueError):
            module.partition(frames, [pair], {"A": "train", "B": "val"})
        with self.assertRaises(ValueError):
            module.partition([frames[0], frames[0]], [], {"A": "train"})

    def test_verify_reads_all_three_splits_and_checks_pixel_leakage(self):
        module = self.load()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "frames").mkdir()
            (root / "pairs").mkdir()
            frames = [{"frame_id": s, "clip_id": s, "pixel_sha256": s} for s in module.SPLITS]
            pairs = [{"pair_id": s, "clip_id": s, "source_frame_id": s, "target_frame_id": s} for s in module.SPLITS]
            module.write(root / "frames/frames.jsonl", frames, True)
            module.write(root / "pairs/pairs.jsonl", pairs, True)
            for i, s in enumerate(module.SPLITS):
                (root / s).mkdir()
                module.write(root / s / "frames.jsonl", [{**frames[i], "split": s, "group_id": s, "original_split": None}], True)
                module.write(root / s / "pairs.jsonl", [{**pairs[i], "split": s, "group_id": s, "original_split": None}], True)
                module.write(root / s / "groups.json", {s: "bilibili:" + s})
            self.assertTrue(module.verify(root, root / "result.json")["passed"])


if __name__ == "__main__":
    unittest.main()

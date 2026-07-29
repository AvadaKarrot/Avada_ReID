import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.caption_store import CaptionStore
from data.records import image_only_sample
from engine.batch import normalize_batch


class DataContractTest(unittest.TestCase):
    def test_target_sample_has_no_caption(self):
        sample = image_only_sample("query/a.jpg", 1, 2, 0)
        self.assertFalse(sample.has_caption)
        self.assertEqual(sample.captions, ())

    def test_caption_store_rejects_target_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "split": "query",
                        "image_path": "query/a.jpg",
                        "captions": ["not allowed"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                CaptionStore.from_jsonl(path)

    def test_legacy_train_tuple_preserves_paths_and_dataset_ids(self):
        batch = (
            torch.randn(2, 3, 16, 8),
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            ("train/a.jpg", "train/b.jpg"),
            (3, 3),
        )
        normalized = normalize_batch(batch)
        self.assertEqual(normalized["image_paths"], batch[3])
        self.assertEqual(normalized["dataset_ids"], batch[4])
        self.assertIsNone(normalized["captions"])
        self.assertIsNone(normalized["caption_mask"])

    def test_legacy_test_tuple_does_not_treat_dataset_ids_as_captions(self):
        batch = (
            torch.randn(2, 3, 16, 8),
            torch.tensor([1, 2]),
            (0, 1),
            torch.tensor([0, 1]),
            ("query/a.jpg", "gallery/b.jpg"),
            (4, 4),
        )
        normalized = normalize_batch(batch)
        self.assertTrue(torch.equal(normalized["camids"], batch[3]))
        self.assertEqual(normalized["image_paths"], batch[4])
        self.assertEqual(normalized["dataset_ids"], batch[5])
        self.assertIsNone(normalized["captions"])


if __name__ == "__main__":
    unittest.main()

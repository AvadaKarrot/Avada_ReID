import json
import os.path as osp
import tempfile
import unittest
from pathlib import Path

import torch

from data.caption_store import CaptionStore
from data.collate import caption_collate_fn
from data.datasets.image.cuhk03 import _rebase_split_paths
from data.records import image_only_sample
from engine.batch import normalize_batch


class DataContractTest(unittest.TestCase):
    def test_cuhk03_split_paths_are_rebased_to_active_dataset_root(self):
        records = [
            ("/old/machine/cuhk03/images_detected/1_001_1_01.png", 7, 0)
        ]
        rebased = _rebase_split_paths(records, "/datasets/cuhk03/images_detected")
        self.assertEqual(
            rebased,
            [
                (
                    osp.join(
                        "/datasets/cuhk03/images_detected",
                        "1_001_1_01.png",
                    ),
                    7,
                    0,
                )
            ],
        )

    def test_target_sample_has_no_caption(self):
        sample = image_only_sample("query/a.jpg", 1, 2, 0)
        self.assertFalse(sample.has_caption)
        self.assertEqual(sample.captions, ())

    def test_caption_store_ignores_target_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "market1501",
                        "split": "query",
                        "image_path": "query/a.jpg",
                        "captions": ["not allowed"],
                    }
                ),
                encoding="utf-8",
            )
            store = CaptionStore.from_jsonl(path)
            self.assertEqual(len(store), 0)
            self.assertEqual(store.get("query/a.jpg", "market1501"), ())

    def test_caption_store_binds_relative_train_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "market1501",
                        "split": "train",
                        "image_path": "bounding_box_train/a.jpg",
                        "captions": ["red shirt", "dark trousers"],
                    }
                ),
                encoding="utf-8",
            )
            store = CaptionStore.from_jsonl(path)
            records = [
                (
                    "/datasets/market/bounding_box_train/a.jpg",
                    1,
                    2,
                    0,
                )
            ]
            bound = store.bind(records, "market1501")
            self.assertEqual(
                bound[0][4],
                ("red shirt", "dark trousers"),
            )

    def test_caption_store_masks_missing_source_caption(self):
        store = CaptionStore({})
        records = [("/datasets/market/a.jpg", 1, 2, 0)]
        with self.assertRaises(ValueError):
            store.bind(records, "market1501", missing_policy="error")
        self.assertEqual(
            store.bind(records, "market1501", missing_policy="mask")[0][4],
            (),
        )

    def test_caption_collate_preserves_dataset_ids_and_mask(self):
        batch = [
            (
                torch.zeros(3, 8, 4),
                1,
                2,
                "train/a.jpg",
                3,
                "red shirt",
            ),
            (
                torch.ones(3, 8, 4),
                2,
                3,
                "train/b.jpg",
                4,
                "",
            ),
        ]
        collated = caption_collate_fn(batch)
        self.assertEqual(collated["dataset_ids"], (3, 4))
        self.assertEqual(collated["captions"], ("red shirt", ""))
        self.assertTrue(
            torch.equal(
                collated["caption_mask"],
                torch.tensor([True, False]),
            )
        )

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

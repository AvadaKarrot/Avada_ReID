import json
import os.path as osp
import tempfile
import unittest
from pathlib import Path

import torch

from data.caption_store import CaptionStore
from data.collate import NaFlexCollator, caption_collate_fn
from data.datasets.image.cuhk03 import _rebase_split_paths
from data.records import image_only_sample
from engine.batch import normalize_batch


class DataContractTest(unittest.TestCase):
    def test_naflex_collate_keeps_mask_and_spatial_shape_with_metadata(self):
        class FakeProcessor:
            def __call__(self, images, return_tensors, max_num_patches):
                self.images = images
                self.return_tensors = return_tensors
                self.max_num_patches = max_num_patches
                return {
                    "pixel_values": torch.randn(2, 4, 12),
                    "pixel_attention_mask": torch.tensor(
                        [[1, 1, 0, 0], [1, 1, 1, 1]]
                    ),
                    "spatial_shapes": torch.tensor([[2, 1], [2, 2]]),
                }

        processor = FakeProcessor()
        collator = NaFlexCollator(
            processor, max_num_patches=4, caption=True
        )
        batch = [
            (object(), 1, 2, "a.jpg", 3, "red shirt"),
            (object(), 4, 5, "b.jpg", 6, ""),
        ]

        collated = collator(batch)

        self.assertEqual(processor.return_tensors, "pt")
        self.assertEqual(processor.max_num_patches, 4)
        self.assertEqual(
            set(collated["images"]),
            {"pixel_values", "pixel_attention_mask", "spatial_shapes"},
        )
        self.assertEqual(collated["images"]["pixel_values"].shape, (2, 4, 12))
        self.assertTrue(
            torch.equal(
                collated["caption_mask"], torch.tensor([True, False])
            )
        )
        self.assertEqual(collated["dataset_ids"], (3, 6))

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

    def test_caption_store_can_index_explicit_full_source_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            records = [
                {
                    "dataset": "market1501",
                    "split": "train",
                    "image_path": "bounding_box_train/a.jpg",
                    "captions": ["train caption"],
                },
                {
                    "dataset": "market1501",
                    "split": "query",
                    "image_path": "query/b.jpg",
                    "captions": ["query caption"],
                },
                {
                    "dataset": "market1501",
                    "split": "gallery",
                    "image_path": "bounding_box_test/c.jpg",
                    "captions": ["gallery caption"],
                },
            ]
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            store = CaptionStore.from_jsonl(
                path,
                allowed_splits=("train", "query", "gallery"),
            )

        self.assertEqual(len(store), 3)
        self.assertEqual(
            store.get("query/b.jpg", "market1501"),
            ("query caption",),
        )
        self.assertEqual(
            store.get("bounding_box_test/c.jpg", "market1501"),
            ("gallery caption",),
        )

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

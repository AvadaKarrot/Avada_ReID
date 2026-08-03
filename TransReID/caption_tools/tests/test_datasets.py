from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CAPTION_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPTION_TOOLS))

from datasets import iter_caption_images, iter_source_train_images  # noqa: E402


class Market1501DatasetTests(unittest.TestCase):
    def _make_layout(self, root: Path) -> Path:
        dataset_dir = root / "market"
        for directory in (
            "bounding_box_train",
            "query",
            "bounding_box_test",
        ):
            (dataset_dir / directory).mkdir(parents=True)
        files = {
            "bounding_box_train": (
                "0001_c1s1_000001_00.jpg",
                "0000_c1s1_000002_00.jpg",
            ),
            "query": ("0002_c2s1_000003_00.jpg",),
            "bounding_box_test": (
                "0002_c3s1_000004_00.jpg",
                "0000_c3s1_000005_00.jpg",
                "-1_c3s1_000006_00.jpg",
            ),
        }
        for directory, names in files.items():
            for name in names:
                (dataset_dir / directory / name).write_bytes(name.encode())
        return dataset_dir

    def test_train_scope_keeps_historical_train_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            records = list(iter_source_train_images("market1501", str(root)))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].split, "train")
        self.assertEqual(
            records[0].image_path,
            "bounding_box_train/0001_c1s1_000001_00.jpg",
        )

    def test_full_scope_matches_combine_all_junk_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            records = list(
                iter_caption_images(
                    "market1501", str(root), split_scope="full"
                )
            )

        self.assertEqual(
            [record.split for record in records],
            ["train", "query", "gallery"],
        )
        self.assertEqual(
            [record.pid for record in records],
            [1, 2, 2],
        )
        first_tokens = {
            Path(record.image_path).stem.split("_")[0]
            for record in records
        }
        self.assertNotIn("0000", first_tokens)
        self.assertNotIn("-1", first_tokens)


class MSMT17V1DatasetTests(unittest.TestCase):
    def _make_v1_layout(self, root: Path) -> Path:
        version_dir = root / "msmt17" / "MSMT17_V1"
        (version_dir / "train" / "0000").mkdir(parents=True)
        (version_dir / "train" / "0001").mkdir(parents=True)
        (version_dir / "train" / "0000" / "zero.jpg").write_bytes(b"zero")
        (version_dir / "train" / "0001" / "one.png").write_bytes(b"one")
        return version_dir

    def test_v1_uses_list_train_and_keeps_pid_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "train" / "0001" / "val.jpg").write_bytes(b"val")
            (version_dir / "list_train.txt").write_text(
                "0000/zero.jpg 0\n0001/one.png 1\n",
                encoding="utf-8",
            )
            (version_dir / "list_val.txt").write_text(
                "0001/val.jpg 1\n",
                encoding="utf-8",
            )

            records = list(iter_source_train_images("msmt17", str(root)))

        self.assertEqual([record.pid for record in records], [0, 1])
        self.assertEqual(
            [record.image_path for record in records],
            [
                "MSMT17_V1/train/0000/zero.jpg",
                "MSMT17_V1/train/0001/one.png",
            ],
        )
        self.assertTrue(all(record.dataset == "msmt17" for record in records))
        self.assertTrue(all(record.split == "train" for record in records))
        self.assertNotIn("val.jpg", {record.image_path for record in records})

    def test_v1_all_scope_uses_every_manifest_and_ignores_unlisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "train" / "0001" / "val.jpg").write_bytes(b"val")
            (version_dir / "test" / "0002").mkdir(parents=True)
            (version_dir / "test" / "0002" / "query.jpg").write_bytes(b"query")
            (version_dir / "test" / "0002" / "gallery.jpg").write_bytes(b"gallery")
            (version_dir / "test" / "0002" / "unlisted.jpg").write_bytes(b"skip")
            (version_dir / "list_train.txt").write_text(
                "0000/zero.jpg 0\n0001/one.png 1\n", encoding="utf-8"
            )
            (version_dir / "list_val.txt").write_text(
                "0001/val.jpg 1\n", encoding="utf-8"
            )
            (version_dir / "list_query.txt").write_text(
                "0002/query.jpg 2\n", encoding="utf-8"
            )
            (version_dir / "list_gallery.txt").write_text(
                "0002/gallery.jpg 2\n", encoding="utf-8"
            )

            records = list(
                iter_caption_images("msmt17", str(root), split_scope="all")
            )

        self.assertEqual(
            [record.split for record in records],
            ["train", "train", "val", "query", "gallery"],
        )
        self.assertNotIn(
            "unlisted.jpg",
            {Path(record.image_path).name for record in records},
        )

    def test_v1_requires_list_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_v1_layout(root)

            with self.assertRaisesRegex(
                FileNotFoundError, "list_train\\.txt"
            ):
                list(iter_source_train_images("msmt17", str(root)))

    def test_v1_rejects_malformed_split_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "list_train.txt").write_text(
                "0000/zero.jpg\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "格式错误"):
                list(iter_source_train_images("msmt17", str(root)))

    def test_v1_rejects_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "list_train.txt").write_text(
                "0002/missing.jpg 2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "图片不存在"):
                list(iter_source_train_images("msmt17", str(root)))

    def test_v1_rejects_path_outside_train_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "outside.jpg").write_bytes(b"outside")
            (version_dir / "list_train.txt").write_text(
                "../outside.jpg 2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "图片路径越界"):
                list(iter_source_train_images("msmt17", str(root)))

    def test_v1_rejects_duplicate_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = self._make_v1_layout(root)
            (version_dir / "list_train.txt").write_text(
                "0000/zero.jpg 0\n0000/zero.jpg 0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "图片重复"):
                list(iter_source_train_images("msmt17", str(root)))


class CUHK03NewDetectedDatasetTests(unittest.TestCase):
    def _make_layout(self, root: Path) -> Path:
        dataset_dir = root / "cuhk03"
        images_dir = dataset_dir / "images_detected"
        images_dir.mkdir(parents=True)
        for name in ("train.png", "query.png", "gallery.png"):
            (images_dir / name).write_bytes(name.encode("utf-8"))
        split = [
            {
                "train": [["/stale/path/train.png", 10, 0]],
                "query": [["/stale/path/query.png", 20, 1]],
                "gallery": [["/stale/path/gallery.png", 20, 1]],
            }
        ]
        (dataset_dir / "splits_new_detected.json").write_text(
            __import__("json").dumps(split),
            encoding="utf-8",
        )
        return dataset_dir

    def test_train_scope_uses_only_detected_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            records = list(iter_source_train_images("cuhk03", str(root)))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].split, "train")
        self.assertEqual(records[0].image_path, "images_detected/train.png")

    def test_all_scope_retains_official_split_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_layout(root)
            records = list(
                iter_caption_images("cuhk03", str(root), split_scope="all")
            )

        self.assertEqual(
            [record.split for record in records],
            ["train", "query", "gallery"],
        )
        self.assertEqual(
            [record.image_path for record in records],
            [
                "images_detected/train.png",
                "images_detected/query.png",
                "images_detected/gallery.png",
            ],
        )


if __name__ == "__main__":
    unittest.main()

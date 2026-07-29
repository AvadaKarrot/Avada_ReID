from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_jsonl import _export_legacy_json, _iter_clean_records  # noqa: E402


def _record(split: str, image_path: str) -> dict[str, object]:
    return {
        "dataset": "example",
        "split": split,
        "image_path": image_path,
        "pid": 1,
        "captions": ["a person"],
        "quality_score": 0.0,
        "generator": "test",
        "prompt_version": "v2.4",
        "attributes": {},
    }


class ExportJsonlTests(unittest.TestCase):
    def test_accepts_all_official_splits_and_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "example.jsonl"
            rows = [
                _record("train", "train.jpg"),
                _record("val", "val.jpg"),
                _record("query", "query.jpg"),
                _record("gallery", "gallery.jpg"),
                _record("invalid", "invalid.jpg"),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            records, bad_lines, incomplete = _iter_clean_records(root)

        self.assertEqual(bad_lines, 0)
        self.assertEqual(incomplete, 1)
        self.assertEqual(
            [record["split"] for record in records],
            ["train", "val", "query", "gallery"],
        )

    def test_legacy_export_contains_train_only(self) -> None:
        records = [
            _record("train", "train.jpg"),
            _record("query", "query.jpg"),
            _record("gallery", "gallery.jpg"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = _export_legacy_json(records, root)
            payload = json.loads(
                (root / "example" / "train_caption_dict.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(counts, {"example": 1})
        self.assertEqual(payload, {"train.jpg": ["a person"]})


if __name__ == "__main__":
    unittest.main()

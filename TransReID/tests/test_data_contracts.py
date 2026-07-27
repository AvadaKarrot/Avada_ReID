import json
import tempfile
import unittest
from pathlib import Path

from data.caption_store import CaptionStore
from data.records import image_only_sample


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


if __name__ == "__main__":
    unittest.main()

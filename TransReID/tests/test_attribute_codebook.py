import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from data.attribute_store import AttributeStore
from semantic.codebook import (
    _consolidate_clusters,
    build_adaptive_codebook,
    build_phrase_bank,
    validate_codebook_artifact,
)
from semantic.artifact import save_torch_artifact, sha256_file
from tools.validate_attribute_codebook import validate_directory
from objectives.reid_objective import ReIDObjective


class FakeTextEncoder(torch.nn.Module):
    def forward(self, values):
        features = []
        for value in values:
            value = value.lower()
            feature = torch.tensor(
                [
                    float("dark" in value),
                    float("light" in value or "white" in value),
                    float("shirt" in value),
                    float("pants" in value),
                    float("shoe" in value),
                    float("bag" in value),
                    float("hair" in value),
                    1.0,
                ]
            )
            features.append(feature)
        return torch.stack(features)


def _record(image_path, *, dataset="market1501", split="train"):
    return {
        "dataset": dataset,
        "split": split,
        "image_path": image_path,
        "pid": 1,
        "quality_score": 0.8,
        "captions": ["a person"],
        "attributes": {
            "upper_clothing": {
                "visibility": "clear",
                "attributes": [" Dark   Shirt "],
            },
            "lower_clothing": {
                "visibility": "not_visible",
                "attributes": ["black pants"],
            },
            "footwear": {
                "visibility": "partial",
                "attributes": ["white shoes"],
            },
            "carried_items": ["black bag"],
            "accessories": [],
            "hair": ["short dark hair"],
            "distinctive_features": [],
        },
    }


class AttributeStoreTest(unittest.TestCase):
    def test_visibility_is_applied_when_jsonl_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        _record("train/a.jpg"),
                        _record("train/not_in_active_split.jpg"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            store = AttributeStore.from_jsonl(path)

        record = store.get("/datasets/market/train/a.jpg", "market1501")
        self.assertIsNotNone(record)
        self.assertEqual(record.phrases("upper_clothing"), ("dark shirt",))
        self.assertEqual(record.phrases("lower_clothing"), ())
        self.assertEqual(record.phrases("footwear"), ("white shoes",))
        self.assertEqual(record.phrases("accessories"), ())
        self.assertEqual(record.quality_score, 0.8)

    def test_camera_metadata_comes_from_dataset_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        _record("train/a.jpg"),
                        _record("train/not_in_active_split.jpg"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            store = AttributeStore.from_jsonl(path)
            store = store.bind_camera_metadata(
                [("/datasets/market/train/a.jpg", 1, 7, 0)],
                "market1501",
            )

        record = store.get("train/a.jpg", "market1501")
        self.assertEqual(record.camera_id, 7)
        self.assertEqual(len(store), 1)
        statistics = store.phrase_statistics(
            "upper_clothing", domain_mode="camera"
        )
        self.assertEqual(
            statistics["dark shirt"]["domains"],
            {"market1501:camera:7": 1},
        )

    def test_dataset_and_split_filters_prevent_target_leakage(self):
        records = [
            _record("train/a.jpg"),
            _record("train/b.jpg", dataset="msmt17"),
            _record("query/c.jpg", split="query"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            store = AttributeStore.from_jsonl(
                path, allowed_datasets=("market1501",)
            )
        self.assertEqual(len(store), 1)
        self.assertIsNone(store.get("train/b.jpg", "msmt17"))


class SemanticCodebookTest(unittest.TestCase):
    def test_codebook_weight_schedule_supports_delay_and_linear_decay(self):
        objective = ReIDObjective(
            attribute_codebook_weight=0.05,
            attribute_codebook_start_epoch=6,
            attribute_codebook_decay_start_epoch=15,
            attribute_codebook_final_weight=0.01,
        )
        objective.set_epoch(5, max_epochs=30)
        self.assertEqual(objective.current_attribute_codebook_weight(), 0.0)
        objective.set_epoch(6, max_epochs=30)
        self.assertEqual(objective.current_attribute_codebook_weight(), 0.05)
        objective.set_epoch(15, max_epochs=30)
        self.assertEqual(objective.current_attribute_codebook_weight(), 0.05)
        objective.set_epoch(30, max_epochs=30)
        self.assertAlmostEqual(
            objective.current_attribute_codebook_weight(), 0.01
        )

    def test_merge_rejects_large_compactness_degradation(self):
        embeddings = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.1, 0.9],
                ]
            ),
            dim=1,
        )
        assignments = torch.tensor([0, 0, 1, 1])
        rejected = _consolidate_clusters(
            embeddings,
            assignments,
            merge_similarity=0.0,
            max_compactness_drop=0.01,
        )
        accepted = _consolidate_clusters(
            embeddings,
            assignments,
            merge_similarity=0.0,
            max_compactness_drop=1.0,
        )
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(accepted), 1)

    def test_phrase_bank_and_codebook_are_valid(self):
        records = [
            _record("train/a.jpg"),
            _record("train/b.jpg"),
        ]
        records[1]["attributes"]["upper_clothing"]["attributes"] = [
            "light shirt"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            store = AttributeStore.from_jsonl(path)
        phrase_bank = build_phrase_bank(store, FakeTextEncoder(), batch_size=2)
        codebook = build_adaptive_codebook(
            phrase_bank,
            expected_domains=("market1501",),
            max_codes=8,
            merge_similarity=0.99,
            min_support=1,
        )
        validation = validate_codebook_artifact(codebook, phrase_bank)
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(
            phrase_bank["buckets"]["upper_clothing"]["phrases"],
            ("dark shirt", "light shirt"),
        )
        self.assertGreaterEqual(
            validation["summary"]["upper_clothing"]["codes"], 1
        )

    def test_directory_validator_rejects_target_in_source_manifest(self):
        records = [_record("train/a.jpg")]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            caption_path = directory / "captions.jsonl"
            caption_path.write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8"
            )
            store = AttributeStore.from_jsonl(caption_path)
            phrase_bank = build_phrase_bank(store, FakeTextEncoder())
            codebook = build_adaptive_codebook(
                phrase_bank,
                expected_domains=("market1501",),
                max_codes=8,
                min_support=1,
            )
            phrase_path = directory / "phrase_bank.pt"
            codebook_path = directory / "codebook.pt"
            save_torch_artifact(phrase_bank, phrase_path)
            save_torch_artifact(codebook, codebook_path)
            manifest = {
                "source_datasets": ["market1501"],
                "domain_mode": "dataset",
                "caption_file": str(caption_path),
                "caption_sha256": sha256_file(caption_path),
                "artifacts": {
                    "phrase_bank": {
                        "path": phrase_path.name,
                        "sha256": sha256_file(phrase_path),
                    },
                    "codebook": {
                        "path": codebook_path.name,
                        "sha256": sha256_file(codebook_path),
                    },
                },
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            valid = validate_directory(
                directory,
                forbidden_datasets=("msmt17",),
                verify_caption_file=True,
            )
            invalid = validate_directory(
                directory,
                forbidden_datasets=("market1501",),
            )

        self.assertTrue(valid["valid"], valid["errors"])
        self.assertFalse(invalid["valid"])
        self.assertIn("forbidden target datasets", invalid["errors"][0])


if __name__ == "__main__":
    unittest.main()
